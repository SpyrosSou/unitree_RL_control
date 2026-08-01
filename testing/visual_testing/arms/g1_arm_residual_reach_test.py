"""G1 arm RESIDUAL policy reach test — command specific (x,y,z) targets and watch the
IK-baseline + learned-residual policy reach them, on the same genuinely fixed base
(`fix_root_link=True`, no walking policy at all) the residual task trains on.

Adapted directly from ``g1_arm_reach_test.py`` (same red-sphere marker, same R-key /
``Se2Keyboard`` async "new target" pattern, same never-force-a-timeout-prompt
philosophy) — see that file's docstring for the full interaction-design rationale,
not repeated here. The one adaptation that matters: this task's action isn't
``current_pose + policy_delta``, it's ``ik_baseline_q + small_residual`` (see
``g1_arm_residual_env.py``'s module docstring), and the IK baseline is normally solved
only inside ``_reset_idx`` when a NEW RANDOM goal is sampled during training. This
script sets a SPECIFIC, user-chosen goal instead of a random one, so ``set_target()``
here must ALSO explicitly re-solve the IK baseline for that goal
(``inner_env._solve_ik_baseline_for_envs(...)``) — skipping this would leave the
residual correcting against a stale baseline from whatever the previous target was,
silently breaking the reach for the new one.

``--no_residual`` forces ``residual_action_scale`` to zero for a live, visual
"IK+gravity-feedforward alone, no learned correction" comparison — the same
``ik_baseline`` bucket concept ``validation/eval_arm_residual.py`` measures
quantitatively, here as a direct before/after you can watch.

Usage:
    conda activate isaac_g1_ik
    cd ~/Elm/Code/g1_locomotion

    python testing/visual_testing/arms/g1_arm_residual_reach_test.py \\
        --side left \\
        --checkpoint logs/rsl_rl/arms/residual_left/<run>/model_1999.pt \\
        --targets 0.3 0.2 1.0

    # Cycle through multiple targets
    python testing/visual_testing/arms/g1_arm_residual_reach_test.py \\
        --side left --checkpoint <ckpt> \\
        --targets 0.3 0.2 1.0  0.28 0.25 1.05  0.22 0.15 0.95

    # Visual A/B: IK alone, no learned correction at all
    python testing/visual_testing/arms/g1_arm_residual_reach_test.py \\
        --side left --checkpoint <ckpt> --targets 0.3 0.2 1.0 --no_residual

Target coordinate reference (robot-local frame, base at origin, trimmed box this task
trains on by default):
    x  forward     0.20 - 0.31 m
    y  lateral     left arm: 0.08 - 0.40 m   right arm: -0.40 - -0.08 m
    z  height      0.9 - 1.15 m  (ground = 0)
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 arm RESIDUAL-policy reach test with specific (x,y,z) targets.")
parser.add_argument("--side", type=str, default="left", choices=["left", "right"], help="Which arm to test.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained residual checkpoint (.pt).")
parser.add_argument("--targets", type=float, nargs="+", required=True,
                    help="Target positions as flat list of x y z triplets, e.g. 0.3 0.2 1.0  0.28 0.25 1.05")
parser.add_argument("--no_residual", action="store_true",
                    help="Force residual_action_scale=0 -- IK+gravity-feedforward alone, no learned "
                    "correction, for a direct visual A/B against the normal (residual-active) behavior.")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=None,
                    help="Override actor/critic hidden dims to match the checkpoint (default: "
                    "G1ArmResidualPPORunnerCfg's own [512, 256, 128]).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import os

import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import _GOAL_BOUNDS
from g1_locomotion.tasks.manager_based.g1_arm_residual.agents.rsl_rl_ppo_cfg import (
    G1ArmResidualLeftPPORunnerCfg,
    G1ArmResidualRightPPORunnerCfg,
)
from g1_locomotion.tasks.manager_based.g1_arm_residual.g1_arm_residual_env import (
    G1ArmResidualEnv,
    G1ArmResidualLeftEnvCfg_PLAY,
    G1ArmResidualRightEnvCfg_PLAY,
)

# ---------------------------------------------------------------------------
# Red-sphere goal marker (same look as g1_arm_reach_test.py)
# ---------------------------------------------------------------------------
RED_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ArmResidualTestTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)


def parse_targets(flat: list[float]) -> list[torch.Tensor]:
    if len(flat) % 3 != 0:
        raise ValueError(f"--targets must be triplets of x y z values, got {len(flat)} numbers.")
    return [torch.tensor(flat[i:i + 3], dtype=torch.float32) for i in range(0, len(flat), 3)]


def _in_bounds(t: torch.Tensor, bounds: dict) -> bool:
    return (
        bounds["x"][0] <= t[0].item() <= bounds["x"][1]
        and bounds["y"][0] <= t[1].item() <= bounds["y"][1]
        and bounds["z"][0] <= t[2].item() <= bounds["z"][1]
    )


def _prompt_next_target(targets_local: list, current_idx: int, side: str, device: torch.device) -> tuple[list, int]:
    bounds = _GOAL_BOUNDS[side]
    n = len(targets_local)
    if n > 1:
        next_default = (current_idx + 1) % n
        prompt = (
            f"\nReachable range: x {bounds['x']}  y {bounds['y']}  z {bounds['z']} (meters)\n"
            f"Next target (x y z)  [Enter = cycle to target {next_default + 1}/{n}]: "
        )
    else:
        prompt = (
            f"\nReachable range: x {bounds['x']}  y {bounds['y']}  z {bounds['z']} (meters)\n"
            "Next target (x y z)  [Enter = repeat current]: "
        )

    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return targets_local, current_idx

        if not line:
            new_idx = (current_idx + 1) % n
            return targets_local, new_idx

        parts = line.split()
        if len(parts) != 3:
            print("  Need exactly 3 numbers: x y z")
            continue
        try:
            t = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=device)
            if not _in_bounds(t, bounds):
                confirm = input(
                    f"  {t.tolist()} is outside the trained range "
                    f"(x {bounds['x']}, y {bounds['y']}, z {bounds['z']}) — "
                    "the policy has never seen this and may behave strangely. "
                    "Send anyway? [y/N] "
                ).strip().lower()
                if confirm != "y":
                    continue
            targets_local.append(t)
            return targets_local, len(targets_local) - 1
        except ValueError:
            print(f"  Could not parse numbers from: {line!r}")


def main():
    side = args_cli.side
    targets_local = parse_targets(args_cli.targets)  # robot-local frame

    print(f"\n[ArmResidualTest] Side       : {side}")
    print(f"[ArmResidualTest] Checkpoint : {args_cli.checkpoint}")
    print(f"[ArmResidualTest] Targets    : {[t.tolist() for t in targets_local]}")
    print(f"[ArmResidualTest] Residual   : {'DISABLED (--no_residual, IK+tau_ff alone)' if args_cli.no_residual else 'active'}")
    print("[ArmResidualTest] Press R at any time for a new target — the sim keeps running until you do.\n")

    # ------------------------------------------------------------------
    # Build environment (1 env, fixed base, no episode randomisation we care about
    # here — we override goal_positions/the IK baseline directly each target).
    # ------------------------------------------------------------------
    if side == "left":
        env_cfg = G1ArmResidualLeftEnvCfg_PLAY()
        agent_cfg = G1ArmResidualLeftPPORunnerCfg()
    else:
        env_cfg = G1ArmResidualRightEnvCfg_PLAY()
        agent_cfg = G1ArmResidualRightPPORunnerCfg()

    env_cfg.scene.num_envs = 1
    if args_cli.no_residual:
        env_cfg.residual_action_scale = 0.0

    inner_env = G1ArmResidualEnv(env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(inner_env)
    device = inner_env.device

    targets_local = [t.to(device) for t in targets_local]

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    if args_cli.hidden_dims is not None:
        agent_cfg.policy.actor_hidden_dims = list(args_cli.hidden_dims)
        agent_cfg.policy.critic_hidden_dims = list(args_cli.hidden_dims)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    # ------------------------------------------------------------------
    # Goal visualisation (red sphere)
    # ------------------------------------------------------------------
    goal_vis = VisualizationMarkers(RED_SPHERE_CFG)
    env_ids_0 = torch.tensor([0], device=device)

    def set_target(idx: int):
        """Push target[idx] to the environment, RE-SOLVE the IK baseline for it (the
        critical residual-task-specific step — see module docstring), and update
        markers. Also resets the arm joints to default first, same reasoning as
        g1_arm_reach_test.py's set_target(): every target attempt should start clean,
        not from wherever the previous (possibly bad/out-of-range) target left the arm.
        """
        default_pos = inner_env.robot.data.default_joint_pos[0:1, inner_env.arm_joint_indices_tensor]
        zero_vel = torch.zeros_like(default_pos)
        inner_env.robot.write_joint_state_to_sim(
            default_pos, zero_vel,
            joint_ids=inner_env.arm_joint_indices_tensor, env_ids=env_ids_0,
        )
        # 2026-08-01: re-seed the slew-limiter's persistent commanded-target state
        # (G1ArmResidualEnv._cmd_targets) to match the pose just teleported to above —
        # without this, _apply_action's rate limit clamps against wherever the
        # PREVIOUS target's residual left _cmd_targets, not this freshly-reset
        # position, so the arm would visibly crawl from the stale old target toward
        # the new one instead of starting clean. _reset_idx (the natural per-episode
        # reset path) already does this same re-seed; this script bypasses _reset_idx
        # by writing joint state directly, so it needs its own copy of that line.
        arm = inner_env._arm_groups[0]
        inner_env._cmd_targets[0:1] = inner_env.robot.data.default_joint_pos[0:1, arm["joint_tensor"]]

        origin = inner_env.scene.env_origins[0]
        t = targets_local[idx]
        world = t + origin
        inner_env.goal_positions[0, 0, :] = world

        # THE residual-specific step: without this, ik_baseline_q/ik_baseline_tauff
        # stay at whatever the PREVIOUS target (or the initial default-pose value)
        # left them at, and the residual would be correcting against the wrong
        # baseline entirely for this new goal.
        inner_env._solve_ik_baseline_for_envs(env_ids_0)

        inner_env._update_goal_markers(env_ids_0)
        goal_vis.visualize(world.unsqueeze(0))
        print(f"[ArmResidualTest] Target {idx + 1}/{len(targets_local)}: {t.tolist()}")

    # ------------------------------------------------------------------
    # Keyboard: R requests a new target (same async, non-blocking pattern as
    # g1_arm_reach_test.py / g1_full_demo.py).
    # ------------------------------------------------------------------
    from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

    new_target_requested = [False]
    keyboard = Se2Keyboard(Se2KeyboardCfg(sim_device=str(device)))
    keyboard.add_callback("R", lambda: new_target_requested.__setitem__(0, True))

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    current_idx = 0
    step_count = 0
    already_reported_reached = False
    set_target(current_idx)

    print("[ArmResidualTest] Running — press R for a new target, Ctrl+C to stop")

    while simulation_app.is_running():
        with torch.inference_mode():
            action = policy(obs)

        obs, _, dones, _ = env.step(action)
        step_count += 1

        ee = inner_env._get_ee_position(0)[0]
        dist = (inner_env.goal_positions[0, 0, :] - ee).norm().item()
        print(f"\r  step {step_count:5d}  |  {side}: {dist * 100:.1f} cm   ", end="", flush=True)

        reached = dist < inner_env.cfg.goal_threshold
        if reached and not already_reported_reached:
            print(f"\n  >> REACHED in {step_count} steps ({dist * 100:.1f} cm) — still holding, press R for a new target")
            already_reported_reached = True

        if dones[0] and not new_target_requested[0]:
            # Episode's own natural boundary reset the joints under us -- silently
            # reissue the SAME target (re-solving its baseline again is harmless/cheap,
            # a single solve) instead of forcing a prompt.
            step_count = 0
            already_reported_reached = False
            set_target(current_idx)
            continue

        if new_target_requested[0]:
            new_target_requested[0] = False
            print()
            targets_local, current_idx = _prompt_next_target(targets_local, current_idx, side, device)
            step_count = 0
            already_reported_reached = False
            set_target(current_idx)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
