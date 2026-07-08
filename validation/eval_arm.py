# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic, fixed-seed evaluation for an arm-IK checkpoint.

Sweeps two buckets — wobble curriculum forced off vs. forced on — so a single run
answers both "can it actually reach goals" (the base sanity check) and "does the
synthetic base-tilt signal (Phase 2, see g1_arm_env.py) meaningfully hurt it". Reuses
``ArmMetricsCsvWrapper``, same as training, so numbers are directly comparable to a run's
own arm_detailed.csv. See validation/README.md for what the numbers mean and what "good"
looks like.

Built after a real regression this eval would have caught immediately: a first attempt at
restricting the arm's joint ranges made a large fraction of the goal workspace physically
unreachable, but this only showed up after actually training for 1500 iterations and
noticing `min_dist_to_goal_cm` had plateaued far from 0. Sweeping a fixed-seed rollout
against a *known* checkpoint (rather than reading training-time CSVs, which mix improving
and already-converged episodes together) gives a clean, repeatable "is this specific
checkpoint actually any good" answer — the same reasoning as the standing/walking evals.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_arm.py \\
        --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_4999.pt \\
        --headless

    python validation/eval_arm.py \\
        --checkpoint chosen_checkpoints/arm_left_latest.pt \\
        --num_envs 512 \\
        --steps_per_bucket 3000 \\
        --headless

Output:
    Prints a per-bucket summary table (success rate, mean/median/p90 distance to goal,
    mean reward) and writes the same summary, plus the raw per-episode rows (reusing
    ``ArmMetricsCsvWrapper``), under <checkpoint_dir>/arm_eval/.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

_BUCKETS = ["no_wobble", "with_wobble"]

parser = argparse.ArgumentParser(description="Fixed-seed eval for a G1 arm-IK checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained arm checkpoint (.pt).")
parser.add_argument(
    "--buckets", type=str, nargs="+", default=_BUCKETS, choices=_BUCKETS,
    help="Which buckets to evaluate.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs per bucket.")
parser.add_argument(
    "--steps_per_bucket", type=int, default=1500,
    help="Control steps to run per bucket (at 30 Hz; 1500 steps = 50 s, several episodes per env).",
)
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible rollouts.")
parser.add_argument(
    "--hidden_dims", type=int, nargs="+", default=None,
    help=(
        "Override actor/critic hidden dims to match the checkpoint being evaluated — "
        "must match what it was trained with or runner.load() fails on a shape mismatch. "
        "E.g. --hidden_dims 512 256 128 for the WideNet overnight-sweep experiment. "
        "Default: baseline [256, 128, 64] (matches G1ArmIKLeftPPORunnerCfg)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import csv
import math
import os

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import G1ArmIKLeftPPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import G1ArmIKEnv, G1ArmIKLeftEnvCfg_PLAY

# Reuse the exact same per-episode metrics wrapper training runs use — keeps the eval's
# numbers directly comparable to what you'd see in a run's arm_detailed.csv.
from g1_locomotion.utils.metrics_wrappers import ArmMetricsCsvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _summarize(csv_path: str) -> dict:
    if not os.path.isfile(csv_path):
        return {"episodes": 0}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"episodes": 0}
    n = len(rows)
    dists = sorted(float(r["min_dist_to_goal_cm"]) for r in rows)

    def pct(p: float) -> float:
        return dists[min(int(n * p), n - 1)]

    return {
        "episodes": n,
        "success_rate": sum(int(r["success"]) for r in rows) / n,
        "mean_reward": sum(float(r["mean_reward"]) for r in rows) / n,
        "mean_dist_cm": sum(dists) / n,
        "p50_dist_cm": pct(0.50),
        "p90_dist_cm": pct(0.90),
    }


def _build_base_env() -> G1ArmIKEnv:
    env_cfg = G1ArmIKLeftEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    return G1ArmIKEnv(env_cfg)


def _joint_range_utilization(base_env: G1ArmIKEnv, min_pos: torch.Tensor, max_pos: torch.Tensor) -> list[dict]:
    """Per-joint achieved range vs. the real hardware range it's clamped to.

    Answers "did training actually explore a healthy chunk of what the joint can do, or
    just tiny movements near the default pose" — not visible anywhere else, since nothing
    logs joint positions during training (only end-effector/goal distance). min_pos/
    max_pos are the min/max each joint reached across every env and every step in this
    bucket's rollout (a single aggregate over the whole rollout, not per-episode).
    """
    joint_names = base_env.robot.data.joint_names
    hw = base_env._arm_hw_limits.cpu()
    rows = []
    for i, idx in enumerate(base_env.arm_joint_indices):
        hw_lo, hw_hi = float(hw[i, 0]), float(hw[i, 1])
        hw_span = hw_hi - hw_lo
        achieved_lo, achieved_hi = float(min_pos[i]), float(max_pos[i])
        pct = (achieved_hi - achieved_lo) / hw_span * 100.0 if hw_span > 0 else 0.0
        rows.append({
            "joint": joint_names[idx],
            "hw_range_deg": (math.degrees(hw_lo), math.degrees(hw_hi)),
            "achieved_range_deg": (math.degrees(achieved_lo), math.degrees(achieved_hi)),
            "pct_of_hw_range_used": pct,
        })
    return rows


def _run_bucket(base_env: G1ArmIKEnv, name: str, eval_root: str) -> tuple[dict, list[dict]]:
    print(f"\n[Eval] --- Bucket '{name}' ---")

    # Wobble is a synthetic observation-only signal now (see g1_arm_env.py's Phase 2
    # notes) — forcing it on/off is just a cfg-field flip, no physical reconfiguration
    # and no rebuild risk, unlike the standing/walking evals' curriculum forcing.
    base_env.cfg.enable_root_wobble = True
    base_env.cfg.root_wobble_enable_step = 0 if name == "with_wobble" else 10**9

    bucket_dir = os.path.join(eval_root, name)
    os.makedirs(bucket_dir, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        env = ArmMetricsCsvWrapper(base_env, bucket_dir)
        device = base_env.device
        wrapped_env = RslRlVecEnvWrapper(env)

        agent_cfg = G1ArmIKLeftPPORunnerCfg()
        if args_cli.hidden_dims is not None:
            agent_cfg.policy.actor_hidden_dims = list(args_cli.hidden_dims)
            agent_cfg.policy.critic_hidden_dims = list(args_cli.hidden_dims)
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        n_joints = len(base_env.arm_joint_indices)
        min_pos = torch.full((n_joints,), float("inf"), device=device)
        max_pos = torch.full((n_joints,), float("-inf"), device=device)

        obs, _ = wrapped_env.reset()
        for _ in range(args_cli.steps_per_bucket):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
            joint_pos = base_env.robot.data.joint_pos[:, base_env.arm_joint_indices_tensor]
            min_pos = torch.minimum(min_pos, joint_pos.amin(dim=0))
            max_pos = torch.maximum(max_pos, joint_pos.amax(dim=0))

    joint_ranges = _joint_range_utilization(base_env, min_pos.cpu(), max_pos.cpu())

    # Close only this bucket's CSV file handles — not wrapped_env.close(), which would
    # cascade down and tear down base_env, breaking the next bucket.
    env._csv.close()
    return _summarize(env.csv_path), joint_ranges


def main():
    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    eval_root = os.path.join(checkpoint_dir, "arm_eval")
    os.makedirs(eval_root, exist_ok=True)

    print(f"[Eval] Checkpoint : {args_cli.checkpoint}")
    print(f"[Eval] Buckets    : {args_cli.buckets}")
    print(f"[Eval] num_envs   : {args_cli.num_envs}   steps_per_bucket: {args_cli.steps_per_bucket}")

    print("[Eval] Building simulation (once, reused across all buckets)...")
    base_env = _build_base_env()

    summary_lines = [
        "# Arm Eval",
        "",
        f"Checkpoint: `{args_cli.checkpoint}`",
        "",
        "| Bucket | Episodes | Success rate | Mean reward | Mean dist (cm) | Median dist (cm) | p90 dist (cm) |",
        "|---|---|---|---|---|---|---|",
    ]

    joint_range_lines = [
        "",
        "## Joint range utilization",
        "",
        "How much of the real hardware range each joint actually used during this eval —",
        "answers \"did it learn genuine reaching motion, or just tiny movements near the",
        "default pose that happen to land on nearby goals\". Low % isn't automatically bad",
        "(a joint might genuinely not need much range for this goal workspace), but a",
        "joint sitting at ~0% while others are well-exercised is worth a second look.",
        "",
        "| Bucket | Joint | HW range (deg) | Achieved range (deg) | % of HW range used |",
        "|---|---|---|---|---|",
    ]

    for name in args_cli.buckets:
        stats, joint_ranges = _run_bucket(base_env, name, eval_root)
        if stats["episodes"] == 0:
            print(f"[Eval] Bucket '{name}': no completed episodes — increase --steps_per_bucket.")
            summary_lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        print(
            f"[Eval] Bucket '{name}': episodes={stats['episodes']} "
            f"success_rate={stats['success_rate']:.2%} "
            f"mean_reward={stats['mean_reward']:.3f} "
            f"mean_dist_cm={stats['mean_dist_cm']:.2f} "
            f"p50_dist_cm={stats['p50_dist_cm']:.2f} "
            f"p90_dist_cm={stats['p90_dist_cm']:.2f}"
        )
        summary_lines.append(
            f"| {name} | {stats['episodes']} | {stats['success_rate']:.2%} | {stats['mean_reward']:.3f} "
            f"| {stats['mean_dist_cm']:.2f} | {stats['p50_dist_cm']:.2f} | {stats['p90_dist_cm']:.2f} |"
        )
        for jr in joint_ranges:
            hw_lo, hw_hi = jr["hw_range_deg"]
            ach_lo, ach_hi = jr["achieved_range_deg"]
            print(
                f"[Eval]   {jr['joint']:28s} hw=[{hw_lo:7.1f}, {hw_hi:6.1f}] "
                f"achieved=[{ach_lo:7.1f}, {ach_hi:6.1f}]  ({jr['pct_of_hw_range_used']:5.1f}% of hw range)"
            )
            joint_range_lines.append(
                f"| {name} | {jr['joint']} | {hw_lo:.1f}, {hw_hi:.1f} | {ach_lo:.1f}, {ach_hi:.1f} "
                f"| {jr['pct_of_hw_range_used']:.1f}% |"
            )

    base_env.close()

    summary_path = os.path.join(eval_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines + joint_range_lines) + "\n")
    print(f"\n[Eval] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
