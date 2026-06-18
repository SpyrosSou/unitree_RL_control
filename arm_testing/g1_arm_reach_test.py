"""
G1 arm IK reach test — command specific (x,y,z) targets and watch the policy reach them.

Red spheres mark each target position. The terminal prints distance-to-goal every step.
Targets are cycled automatically when reached (< 2 cm) or after --hold_steps steps.

For --arm both, supply one triplet per target; it is applied to the LEFT arm and
mirrored in y for the RIGHT arm (e.g. 0.3 0.2 1.0 → left=(0.3,0.2,1.0), right=(0.3,-0.2,1.0)).

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Single arm
    python arm_testing/g1_arm_reach_test.py \\
        --arm left \\
        --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0

    # Cycle through multiple targets
    python arm_testing/g1_arm_reach_test.py \\
        --arm left \\
        --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0  0.4 0.3 1.1  0.2 0.15 0.95

    # Both arms (y is auto-mirrored for right arm)
    python arm_testing/g1_arm_reach_test.py \\
        --arm both \\
        --checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_5000.pt \\
        --targets 0.3 0.2 1.0  0.4 0.3 1.1

Target coordinate reference (robot-local frame, base at origin):
    x  forward     reachable range  0.1 – 0.5 m
    y  lateral     left arm: 0.05 – 0.45 m   right arm: -0.05 – -0.45 m
    z  height      reachable range  0.9 – 1.2 m  (ground = 0)
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 arm IK reach test with specific (x,y,z) targets.")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right", "both"],
                    help="Which arm(s) to test.")
parser.add_argument("--checkpoint", type=str, default=None,
                help="Path to a trained model checkpoint (.pt file). "
                    "If omitted, arm_testing/checkpoints.yaml is consulted.")
parser.add_argument("--targets", type=float, nargs="+", required=True,
                    help="Target positions as flat list of x y z triplets, e.g. 0.3 0.2 1.0  0.4 0.3 1.1")
parser.add_argument("--hold_steps", type=int, default=150,
                    help="Steps to hold each target before auto-advancing (default 150 ≈ 5 s at 30 Hz).")
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
    G1ArmIKEnv,
    G1ArmIKLeftEnvCfg_PLAY,
    G1ArmIKRightEnvCfg_PLAY,
    G1ArmIKBothEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmIKLeftPPORunnerCfg,
    G1ArmIKRightPPORunnerCfg,
    G1ArmIKBothPPORunnerCfg,
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
    # Resolve relative paths against the repo root (parent of arm_testing/)
    if not os.path.isabs(ckpt):
        repo_root = os.path.dirname(os.path.dirname(yaml_path))
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


def _prompt_next_target(targets_local: list, current_idx: int, arm: str, device: torch.device) -> tuple[list, int]:
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
            # cycle to next pre-set target (or repeat if only one)
            new_idx = (current_idx + 1) % n
            return targets_local, new_idx

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


def main():
    arm = args_cli.arm
    checkpoint_path = args_cli.checkpoint
    targets_local = parse_targets(args_cli.targets)   # robot-local frame
    hold_steps = args_cli.hold_steps

    # Fall back to checkpoints.yaml if no --checkpoint supplied
    if checkpoint_path is None:
        checkpoint_path = _yaml_checkpoint(
            os.path.join(os.path.dirname(__file__), "checkpoints.yaml"), arm
        )
    if checkpoint_path is None:
        raise ValueError(
            f"No checkpoint for arm='{arm}'. Either pass --checkpoint or fill in "
            "arm_testing/checkpoints.yaml."
        )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n[ArmTest] Arm       : {arm}")
    print(f"[ArmTest] Checkpoint: {checkpoint_path}")
    print(f"[ArmTest] Targets   : {[t.tolist() for t in targets_local]}")
    print(f"[ArmTest] Hold steps: {hold_steps}")
    print(f"[ArmTest] When a target is reached/timed out, you will be prompted to type the next one.\n")

    # ------------------------------------------------------------------
    # Build environment  (1 env, no episode randomisation)
    # ------------------------------------------------------------------
    if arm == "left":
        env_cfg = G1ArmIKLeftEnvCfg_PLAY()
        agent_cfg = G1ArmIKLeftPPORunnerCfg()
    elif arm == "right":
        env_cfg = G1ArmIKRightEnvCfg_PLAY()
        agent_cfg = G1ArmIKRightPPORunnerCfg()
    else:  # both
        env_cfg = G1ArmIKBothEnvCfg_PLAY()
        agent_cfg = G1ArmIKBothPPORunnerCfg()

    env_cfg.scene.num_envs = 1

    inner_env = G1ArmIKEnv(env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(inner_env)
    device = inner_env.device

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
        """Push target[idx] to the environment and update markers."""
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
    # Run loop
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    current_idx = 0
    step_count = 0
    set_target(current_idx)

    print("[ArmTest] Running — Ctrl+C to stop")

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
        status = "  ".join(f"{l}: {d * 100:.1f} cm" for l, d in zip(label, dists))
        print(f"\r  step {step_count:5d}  |  {status}   ", end="", flush=True)

        reached = all(d < inner_env.cfg.goal_threshold for d in dists)
        timed_out = step_count >= hold_steps

        if reached or timed_out or dones[0]:
            print()  # end the \r line
            if reached:
                dist_cm = ", ".join(f"{d * 100:.1f} cm" for d in dists)
                print(f"  >> REACHED in {step_count} steps ({dist_cm})")
            else:
                print(f"  >> Hold time reached ({step_count} steps)")

            targets_local, current_idx = _prompt_next_target(
                targets_local, current_idx, arm, device
            )
            step_count = 0
            set_target(current_idx)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
