"""
G1 arm reach test — command specific (x,y,z) targets and watch the policy reach them.

Deliberately not called "IK": the deployed policy is pure RL joint-space control, no
inverse kinematics involved (see 29dof_implementation_plan.md).

Red spheres mark each target position. The terminal prints distance-to-goal every step.
Targets never auto-advance on a timeout (2026-07-22 fix — the old --hold_steps timeout
forced a blocking input() prompt at a fixed step count regardless of whether the policy
had actually converged yet, freezing the sim mid-attempt for anything slower than the
default 150 steps/5s). Instead: press R at any time for a new target — this only sets a
flag from an async keyboard callback (isaaclab.devices.keyboard.Se2Keyboard, same pattern
testing/visual_testing/full_demo/g1_full_demo.py already uses), so the sim keeps stepping and
rendering continuously until you actually decide to stop and type, not on a forced
schedule. A REACHED message prints when the goal is hit, but the arm keeps holding it —
watch as long as you want. The episode's own natural boundary (300 steps/10s) still
resets joints via the env, but the current target is immediately reissued, so it's
invisible — no prompt, no interruption.

For --arm both, supply one triplet per target; it is applied to the LEFT arm and
mirrored in y for the RIGHT arm (e.g. 0.3 0.2 1.0 → left=(0.3,0.2,1.0), right=(0.3,-0.2,1.0)).

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Single arm
    python testing/visual_testing/arms/g1_arm_reach_test.py \\
        --arm left \\
        --checkpoint logs/rsl_rl/arms/left/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0

    # Cycle through multiple targets
    python testing/visual_testing/arms/g1_arm_reach_test.py \\
        --arm left \\
        --checkpoint logs/rsl_rl/arms/left/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0  0.4 0.3 1.1  0.2 0.15 0.95

    # Both arms (y is auto-mirrored for right arm)
    python testing/visual_testing/arms/g1_arm_reach_test.py \\
        --arm both \\
        --checkpoint logs/rsl_rl/arms/both/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0  0.4 0.3 1.1

Target coordinate reference (robot-local frame, base at origin):
    x  forward     reachable range  0.20 – 0.42 m
    y  lateral     left arm: 0.08 – 0.40 m   right arm: -0.40 – -0.08 m
    z  height      reachable range  0.9 – 1.15 m  (ground = 0)
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 arm reach test with specific (x,y,z) targets.")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right", "both"],
                    help="Which arm(s) to test.")
parser.add_argument("--checkpoint", type=str, default=None,
                help="Path to a trained model checkpoint (.pt file). "
                    "If omitted, testing/arm_testing/checkpoints.yaml is consulted.")
parser.add_argument("--targets", type=float, nargs="+", required=True,
                    help="Target positions as flat list of x y z triplets, e.g. 0.3 0.2 1.0  0.4 0.3 1.1")
parser.add_argument("--locked_wrist", action="store_true",
                    help="Load a G1-Arm-Left-LockedWrist-v0 checkpoint (5 controlled joints, "
                    "wrist_pitch/wrist_yaw held at default) instead of the standard 7-DOF arm. "
                    "--arm left only.")
parser.add_argument("--legacy32", action="store_true",
                    help="Evaluate a checkpoint trained before 2026-07-26's action_fb observation "
                    "addition (32-D obs) — e.g. the 200/20-gain reference. Required or the policy "
                    "load fails with a strict shape-mismatch error. --arm left only. Same flag as "
                    "validation/eval_arm.py's.")
parser.add_argument("--log_std", action="store_true",
                    help="Evaluate a checkpoint trained with noise_std_type='log' instead of the "
                    "default 'scalar' (e.g. best_combined). Required or the policy load fails with "
                    "a strict key-mismatch error ('std' missing, 'log_std' unexpected). --arm left "
                    "only. Same flag as validation/eval_arm.py's.")
parser.add_argument("--debug_passive_joints", action="store_true",
                    help="Print the current joint-angle deviation from default (degrees) for the "
                    "UNCONTROLLED opposite arm and the 3 waist joints every ~1s, alongside the "
                    "usual distance readout. Added 2026-07-25 to get hard numbers on visually "
                    "observed right-arm/waist motion during a left-arm-only test — none of these "
                    "joints are ever given a policy or code-driven target in this env (confirmed "
                    "via direct code read: _apply_action only writes arm_joint_indices_tensor and "
                    "the locked wrist joints, _reset_idx writes everything else straight to "
                    "default_joint_pos), so any visible motion here is real physics (gravity sag / "
                    "dynamic coupling through each joint's own finite PD stiffness), not an "
                    "action-routing bug — this flag lets you see the actual magnitude and time "
                    "shape (one-time settle vs. continuous) instead of just the visual impression.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import os
import yaml
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    G1ArmEnv,
    G1ArmLeftEnvCfg_Legacy32,
    G1ArmLeftEnvCfg_PLAY,
    G1ArmLeftLockedWristEnvCfg,
    G1ArmRightEnvCfg_PLAY,
    G1ArmBothEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmLeftAblationLogStdPPORunnerCfg,
    G1ArmLeftLockedWristPPORunnerCfg,
    G1ArmLeftPPORunnerCfg,
    G1ArmRightPPORunnerCfg,
    G1ArmBothPPORunnerCfg,
)


# ---------------------------------------------------------------------------
# YAML checkpoint helper
# ---------------------------------------------------------------------------

def _yaml_checkpoint(yaml_path: str, arm: str) -> str | None:
    """Read checkpoint path for *arm* from checkpoints.yaml (or return None)."""
    if not os.path.isfile(yaml_path):
        return None
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    entry = cfg.get(arm, {}) or {}
    ckpt = entry.get("checkpoint")
    if not ckpt:
        return None
    # Resolve relative paths against the repo root (parent of testing/arm_testing/)
    if not os.path.isabs(ckpt):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(yaml_path))))
        ckpt = os.path.join(repo_root, ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# Red-sphere goal marker
# ---------------------------------------------------------------------------
RED_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ArmTestTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)


def parse_targets(flat: list[float]) -> list[torch.Tensor]:
    """Convert flat [x0,y0,z0, x1,y1,z1, ...] list to a list of (3,) tensors."""
    if len(flat) % 3 != 0:
        raise ValueError(f"--targets must be triplets of x y z values, got {len(flat)} numbers.")
    return [torch.tensor(flat[i:i + 3], dtype=torch.float32) for i in range(0, len(flat), 3)]


def _in_bounds(t: torch.Tensor, bounds: dict) -> bool:
    return (
        bounds["x"][0] <= t[0].item() <= bounds["x"][1]
        and bounds["y"][0] <= t[1].item() <= bounds["y"][1]
        and bounds["z"][0] <= t[2].item() <= bounds["z"][1]
    )


def _prompt_next_target(targets_local: list, current_idx: int, arm: str, device: torch.device) -> tuple[list, int]:
    """Block and ask the user for the next target. Returns updated (targets_local, new_idx)."""
    # "both" mode takes the target in the LEFT arm's convention (mirrored for right —
    # see set_target()), so always validate/print against the left bounds here.
    bounds = _GOAL_BOUNDS["left" if arm != "right" else "right"]
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
            # cycle to next pre-set target (or repeat if only one)
            new_idx = (current_idx + 1) % n
            return targets_local, new_idx

        parts = line.split()
        if len(parts) != 3:
            print("  Need exactly 3 numbers: x y z")
            continue
        try:
            t = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=device)
            if not _in_bounds(t, bounds):
                # Out-of-range targets aren't just "harder" — the policy has never
                # trained on anything outside this box, so it can produce genuinely
                # erratic motion trying to reach one, not just fall a bit short.
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
    arm = args_cli.arm
    checkpoint_path = args_cli.checkpoint
    targets_local = parse_targets(args_cli.targets)   # robot-local frame

    # Fall back to checkpoints.yaml if no --checkpoint supplied
    if checkpoint_path is None:
        checkpoint_path = _yaml_checkpoint(
            os.path.join(os.path.dirname(__file__), "checkpoints.yaml"), arm
        )
    if checkpoint_path is None:
        raise ValueError(
            f"No checkpoint for arm='{arm}'. Either pass --checkpoint or fill in "
            "testing/arm_testing/checkpoints.yaml."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[ArmTest] Arm       : {arm}")
    print(f"[ArmTest] Checkpoint: {checkpoint_path}")
    print(f"[ArmTest] Targets   : {[t.tolist() for t in targets_local]}")
    print("[ArmTest] Press R at any time for a new target — the sim keeps running until you do.\n")

    # ------------------------------------------------------------------
    # Build environment  (1 env, no episode randomisation)
    # ------------------------------------------------------------------
    if arm == "left" and args_cli.locked_wrist:
        env_cfg = G1ArmLeftLockedWristEnvCfg()
        agent_cfg = G1ArmLeftLockedWristPPORunnerCfg()
    elif arm == "left" and args_cli.legacy32:
        env_cfg = G1ArmLeftEnvCfg_Legacy32()
        agent_cfg = G1ArmLeftAblationLogStdPPORunnerCfg() if args_cli.log_std else G1ArmLeftPPORunnerCfg()
    elif arm == "left":
        env_cfg = G1ArmLeftEnvCfg_PLAY()
        agent_cfg = G1ArmLeftAblationLogStdPPORunnerCfg() if args_cli.log_std else G1ArmLeftPPORunnerCfg()
    elif arm == "right":
        env_cfg = G1ArmRightEnvCfg_PLAY()
        agent_cfg = G1ArmRightPPORunnerCfg()
    else:  # both
        env_cfg = G1ArmBothEnvCfg_PLAY()
        agent_cfg = G1ArmBothPPORunnerCfg()

    env_cfg.scene.num_envs = 1

    inner_env = G1ArmEnv(env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(inner_env)
    device = inner_env.device

    # --debug_passive_joints bookkeeping: joints this env's own code never targets.
    passive_names = None
    if args_cli.debug_passive_joints:
        opposite_prefix = {"left": "right", "right": "left"}.get(arm)
        passive_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        if opposite_prefix is not None:
            passive_names += [
                f"{opposite_prefix}_shoulder_pitch_joint", f"{opposite_prefix}_shoulder_roll_joint",
                f"{opposite_prefix}_shoulder_yaw_joint", f"{opposite_prefix}_elbow_joint",
                f"{opposite_prefix}_wrist_roll_joint", f"{opposite_prefix}_wrist_pitch_joint",
                f"{opposite_prefix}_wrist_yaw_joint",
            ]
        passive_ids, passive_resolved_names = inner_env.robot.find_joints(passive_names)
        passive_ids_t = torch.tensor(passive_ids, dtype=torch.long, device=device)
        passive_default = inner_env.robot.data.default_joint_pos[0:1, passive_ids_t].clone()
        print(f"[ArmTest] --debug_passive_joints tracking: {passive_resolved_names}")

    # Move local targets to device
    targets_local = [t.to(device) for t in targets_local]

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    # ------------------------------------------------------------------
    # Goal visualisation (red spheres)
    # ------------------------------------------------------------------
    goal_vis = VisualizationMarkers(RED_SPHERE_CFG)

    def set_target(idx: int):
        """Push target[idx] to the environment and update markers.

        Also resets the arm joints to their default pose first — without this, the arm
        just keeps going from wherever it physically ended up after the *previous*
        target (this env is never reset between targets, only once at the very start).
        If that previous target was out of range, the arm could be sitting in an
        extreme, never-seen-in-training pose — and the policy would try to reach the
        *new* target starting from that bad, out-of-distribution state, which can look
        like erratic/nonsensical motion that has nothing to do with whether the new
        target itself is reasonable. Resetting first means every target attempt starts
        clean, so if something still looks wrong, it's actually about that target.
        """
        default_pos = inner_env.robot.data.default_joint_pos[0:1, inner_env.arm_joint_indices_tensor]
        zero_vel = torch.zeros_like(default_pos)
        inner_env.robot.write_joint_state_to_sim(
            default_pos, zero_vel,
            joint_ids=inner_env.arm_joint_indices_tensor, env_ids=torch.tensor([0], device=device),
        )

        origin = inner_env.scene.env_origins[0]
        t = targets_local[idx]

        if arm == "both":
            # Left arm: user-supplied y; right arm: y mirrored
            left_world  = torch.stack([t[0], t[1], t[2]]) + origin
            right_world = torch.stack([t[0], -t[1], t[2]]) + origin
            inner_env.goal_positions[0, 0, :] = left_world
            inner_env.goal_positions[0, 1, :] = right_world
            marker_positions = torch.stack([left_world, right_world])
            print(f"[ArmTest] Target {idx + 1}/{len(targets_local)}: "
                  f"left={t.tolist()}  right={[t[0].item(), -t[1].item(), t[2].item()]}")
        else:
            world = t + origin
            inner_env.goal_positions[0, 0, :] = world
            marker_positions = world.unsqueeze(0)
            print(f"[ArmTest] Target {idx + 1}/{len(targets_local)}: {t.tolist()}")

        inner_env._update_goal_markers(torch.tensor([0], device=device))
        goal_vis.visualize(marker_positions)

    # ------------------------------------------------------------------
    # Keyboard: R requests a new target — only sets a flag from an async carb
    # callback (non-blocking); the actual blocking input() runs from the main loop
    # below, only once per press, so the sim never freezes on a schedule the user
    # didn't ask for. Same pattern as g1_full_demo.py's _prompt_target/"T" key.
    # ------------------------------------------------------------------
    from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

    new_target_requested = [False]  # single-item list: mutable cell for the closure below
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

    print("[ArmTest] Running — press R for a new target, Ctrl+C to stop")

    while simulation_app.is_running():
        with torch.inference_mode():
            action = policy(obs)

        obs, _, dones, _ = env.step(action)
        step_count += 1

        # Distance(s) to current target(s) — overwrite same line
        dists = []
        for i in range(len(inner_env._arm_groups)):
            ee = inner_env._get_ee_position(i)[0]
            dists.append((inner_env.goal_positions[0, i, :] - ee).norm().item())

        label = ["left", "right"] if arm == "both" else [arm]
        status = "  ".join(f"{name}: {d * 100:.1f} cm" for name, d in zip(label, dists))
        print(f"\r  step {step_count:5d}  |  {status}   ", end="", flush=True)

        if args_cli.debug_passive_joints and step_count % 30 == 0:
            current = inner_env.robot.data.joint_pos[0:1, passive_ids_t]
            delta_deg = ((current - passive_default) * 180.0 / torch.pi).squeeze(0)
            deltas_str = "  ".join(
                f"{n}: {d:+.2f}deg" for n, d in zip(passive_resolved_names, delta_deg.tolist())
            )
            print(f"\n  [passive] step {step_count:5d} | {deltas_str}")

        reached = all(d < inner_env.cfg.goal_threshold for d in dists)
        if reached and not already_reported_reached:
            dist_cm = ", ".join(f"{d * 100:.1f} cm" for d in dists)
            print(f"\n  >> REACHED in {step_count} steps ({dist_cm}) — still holding, press R for a new target")
            already_reported_reached = True

        if dones[0] and not new_target_requested[0]:
            # Episode's own natural boundary (300 steps/10s) reset the joints under us —
            # silently reissue the same target instead of forcing a prompt, so this is
            # invisible to whoever's watching, not an interruption.
            step_count = 0
            already_reported_reached = False
            set_target(current_idx)
            continue

        if new_target_requested[0]:
            new_target_requested[0] = False
            print()  # end the \r line
            targets_local, current_idx = _prompt_next_target(
                targets_local, current_idx, arm, device
            )
            step_count = 0
            already_reported_reached = False
            set_target(current_idx)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
