"""
G1 arm mirror test — run a single-arm trained policy on either arm via y-mirroring.

Train once (e.g. left arm), deploy on either side without retraining.
When --trained_arm != --active_arm the script applies a bilateral-symmetry transform:
  * Observations: negate y of ee_pos/goal/error and sign-flip roll/yaw joint angles/vels.
  * Actions:      sign-flip roll/yaw delta commands before applying to the robot.

Note: the transform is an approximation based on bilateral symmetry. Reaching accuracy
may be slightly lower than a natively trained policy for the same arm.

Usage (from project root, conda activate isaac_g1_control):

    # Run a left-trained policy on the RIGHT arm (mirrored)
    python testing/visual_testing/arms/g1_arm_mirror_test.py \\
        --trained_arm left \\
        --active_arm right \\
        --checkpoint logs/rsl_rl/arms/left/<run>/model_5000.pt \\
        --targets 0.3 -0.2 1.0

    # Run a left-trained policy on the LEFT arm (no mirror, same as g1_arm_reach_test.py)
    python testing/visual_testing/arms/g1_arm_mirror_test.py \\
        --trained_arm left \\
        --active_arm left \\
        --checkpoint logs/rsl_rl/arms/left/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0

    # Run a right-trained policy on the LEFT arm (mirrored)
    python testing/visual_testing/arms/g1_arm_mirror_test.py \\
        --trained_arm right \\
        --active_arm left \\
        --checkpoint logs/rsl_rl/arms/right/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0

Target coordinates are always in the physical robot frame:
    x  forward     reachable 0.20 – 0.42 m
    y  lateral     left arm: +0.08 – +0.40 m   right arm: -0.40 – -0.08 m
    z  height      reachable 0.9 – 1.15 m  (ground = 0)

At each transition a prompt appears — type 'x y z' and Enter to jump to a new target,
or press Enter alone to cycle through the initial --targets list.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Run a single-arm trained G1 policy on either arm, with optional y-mirroring."
)
parser.add_argument(
    "--trained_arm", type=str, default="left", choices=["left", "right"],
    help="Which arm the checkpoint was trained on.",
)
parser.add_argument(
    "--active_arm", type=str, default="right", choices=["left", "right"],
    help="Which arm to physically move during this test.",
)
parser.add_argument("--checkpoint", type=str, default=None,
                help="Path to a single-arm trained .pt checkpoint file. "
                    "If omitted, testing/arm_testing/checkpoints.yaml is consulted "
                    "(key = --trained_arm).")
parser.add_argument(
    "--targets", type=float, nargs="+", required=True,
    help=(
        "Initial target positions as flat x y z triplets. "
        "Use the active arm's coordinate frame "
        "(left arm: y > 0, right arm: y < 0)."
    ),
)
parser.add_argument(
    "--hold_steps", type=int, default=150,
    help="Steps to hold each target before prompting for the next (default 150 ≈ 5 s at 30 Hz).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import os

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
import yaml
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmLeftPPORunnerCfg,  # same architecture for both arms — used for network shape only
)
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    G1ArmEnv,
    G1ArmLeftEnvCfg_PLAY,
    G1ArmRightEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import mirror_arm_actions, mirror_arm_obs
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

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
    if not os.path.isabs(ckpt):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(yaml_path))))
        ckpt = os.path.join(repo_root, ckpt)
    return ckpt


# ---------------------------------------------------------------------------
# Bilateral-symmetry mirror transform
# ---------------------------------------------------------------------------
#
# The actual mirror math (sign flips for roll/yaw joints and every y-component, see the
# 26-D layout in g1_arm_env.py's module docstring) now lives in
# g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry (mirror_arm_obs / mirror_arm_actions)
# instead of being duplicated here — that module is also what the arm policy is now
# actually *trained* with (RslRlSymmetryCfg data augmentation), so this script's runtime
# mirror is guaranteed to match what was trained, not a second hand-maintained copy that
# could silently drift out of sync with it (as happened when the observation layout grew
# from 17-D to 26-D in Phase 2 — this file's old hardcoded indices were still pointing at
# the pre-Phase-2 layout until this change).


def _mirror_obs(obs_dict: dict) -> dict:
    return {"policy": mirror_arm_obs(obs_dict["policy"])}


def _mirror_act(action: torch.Tensor) -> torch.Tensor:
    return mirror_arm_actions(action)


# ---------------------------------------------------------------------------
# Red-sphere goal marker
# ---------------------------------------------------------------------------
RED_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/ArmMirrorTestTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_targets(flat: list[float]) -> list[torch.Tensor]:
    if len(flat) % 3 != 0:
        raise ValueError(f"--targets must be triplets of x y z values, got {len(flat)} numbers.")
    return [torch.tensor(flat[i:i + 3], dtype=torch.float32) for i in range(0, len(flat), 3)]


def _prompt_next_target(
    targets_local: list, current_idx: int, device: torch.device
) -> tuple[list, int]:
    """Block and ask the user for the next target. Returns updated (targets_local, new_idx)."""
    n = len(targets_local)
    if n > 1:
        next_default = (current_idx + 1) % n
        prompt = f"\nNext target (x y z)  [Enter = cycle to target {next_default + 1}/{n}]: "
    else:
        prompt = "\nNext target (x y z)  [Enter = repeat current]: "

    while True:
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return targets_local, current_idx

        if not line:
            return targets_local, (current_idx + 1) % n

        parts = line.split()
        if len(parts) != 3:
            print("  Need exactly 3 numbers: x y z")
            continue
        try:
            t = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=device)
            targets_local.append(t)
            return targets_local, len(targets_local) - 1
        except ValueError:
            print(f"  Could not parse numbers from: {line!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    trained_arm = args_cli.trained_arm
    active_arm = args_cli.active_arm
    do_mirror = (trained_arm != active_arm)
    checkpoint_path = args_cli.checkpoint
    targets_local = parse_targets(args_cli.targets)
    hold_steps = args_cli.hold_steps

    # Fall back to checkpoints.yaml using the trained_arm key
    if checkpoint_path is None:
        checkpoint_path = _yaml_checkpoint(
            os.path.join(os.path.dirname(__file__), "checkpoints.yaml"), trained_arm
        )
    if checkpoint_path is None:
        raise ValueError(
            f"No checkpoint for trained_arm='{trained_arm}'. Either pass --checkpoint "
            "or fill in testing/arm_testing/checkpoints.yaml."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[MirrorTest] Trained arm : {trained_arm}")
    print(f"[MirrorTest] Active arm  : {active_arm}")
    print(f"[MirrorTest] Mirror      : {'YES — y-transform applied' if do_mirror else 'NO  — direct passthrough'}")
    print(f"[MirrorTest] Checkpoint  : {checkpoint_path}")
    print(f"[MirrorTest] Targets     : {[t.tolist() for t in targets_local]}")
    print(f"[MirrorTest] Hold steps  : {hold_steps}")
    print("[MirrorTest] Type 'x y z' + Enter at each prompt to set a new target.\n")

    # ------------------------------------------------------------------
    # Build environment for the ACTIVE arm (1 env)
    # ------------------------------------------------------------------
    env_cfg = G1ArmLeftEnvCfg_PLAY() if active_arm == "left" else G1ArmRightEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1

    inner_env = G1ArmEnv(env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(inner_env)
    device = inner_env.device
    targets_local = [t.to(device) for t in targets_local]

    # ------------------------------------------------------------------
    # Load policy — architecture is identical for left and right (17→5)
    # ------------------------------------------------------------------
    agent_cfg = G1ArmLeftPPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(device))
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    # ------------------------------------------------------------------
    # Goal visualisation
    # ------------------------------------------------------------------
    goal_vis = VisualizationMarkers(RED_SPHERE_CFG)

    def set_target(idx: int):
        origin = inner_env.scene.env_origins[0]
        t = targets_local[idx]
        world = t + origin
        inner_env.goal_positions[0, 0, :] = world
        inner_env._update_goal_markers(torch.tensor([0], device=device))
        goal_vis.visualize(world.unsqueeze(0))
        print(f"[MirrorTest] Target {idx + 1}/{len(targets_local)}: {t.tolist()}")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    current_idx = 0
    step_count = 0
    set_target(current_idx)

    print("[MirrorTest] Running — Ctrl+C to stop")

    while simulation_app.is_running():
        # Apply mirror transform to observations if needed
        obs_for_policy = _mirror_obs(obs) if do_mirror else obs

        with torch.inference_mode():
            action = policy(obs_for_policy)

        # Apply mirror transform to actions if needed
        if do_mirror:
            action = _mirror_act(action)

        obs, _, dones, _ = env.step(action)
        step_count += 1

        # Distance to target — overwrite same line
        ee = inner_env._get_ee_position(0)[0]
        dist = (inner_env.goal_positions[0, 0, :] - ee).norm().item()
        print(f"\r  step {step_count:5d}  |  {active_arm}: {dist * 100:.1f} cm   ",
              end="", flush=True)

        reached = dist < inner_env.cfg.goal_threshold
        timed_out = step_count >= hold_steps

        if reached or timed_out or dones[0]:
            print()
            if reached:
                print(f"  >> REACHED in {step_count} steps ({dist * 100:.1f} cm)")
            else:
                print(f"  >> Hold time reached ({step_count} steps)")

            targets_local, current_idx = _prompt_next_target(
                targets_local, current_idx, device
            )
            step_count = 0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
