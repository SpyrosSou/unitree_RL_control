# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic, fixed-seed evaluation for a G1 29dof arm RESIDUAL-RL checkpoint —
adapted from ``validation/eval_arm.py`` (same ``ArmMetricsCsvWrapper`` reuse, same
two-wobble-bucket structure, same summary/joint-range/reward-breakdown reporting), with
one addition specific to this task: an ``ik_baseline`` bucket that forces the residual
to exactly zero (``cfg.residual_action_scale = 0.0`` for that bucket only), so a single
eval run reports "IK alone" and "IK + residual" side by side — the actual question
Phase 4 exists to answer (see ``personal_development/residual_rl.md`` section 6,
"what done looks like"), not something the base arm eval had any concept of (it has no
baseline to compare against).

Usage:
    conda activate isaac_g1_ik
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_arm_residual.py \\
        --checkpoint logs/rsl_rl/arms/residual_left/<run>/model_2999.pt \\
        --headless

Output: same shape as eval_arm.py — a per-bucket summary table (success rate,
mean/median/p90 distance to goal, mean reward), joint-range utilization, reward
breakdown, all under ``<checkpoint_dir>/arm_residual_eval/``. The
``ik_baseline`` row is the number to compare Phase 2's own fixed-base result against
(mean achieved error ~16.65cm, 0% within 2cm/5cm on a 15-target sample) — if this eval's
``ik_baseline`` bucket lands anywhere near that, it confirms this eval is measuring the
same thing before checking whether ``residual_no_wobble``/``residual_with_wobble``
actually improved on it.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

_BUCKETS = ["ik_baseline", "residual_no_wobble", "residual_with_wobble"]

parser = argparse.ArgumentParser(description="Fixed-seed eval for a G1 29dof arm residual-RL checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained residual checkpoint (.pt).")
parser.add_argument("--side", type=str, default="left", choices=["left", "right"], help="Which arm the checkpoint was trained for.")
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
    "--goal_x_range", type=float, nargs=2, default=None, metavar=("X_MIN", "X_MAX"),
    help="Override the goal box's x-range for this eval only (y/z untouched). Default: "
    "None (use the checkpoint's own env cfg bounds, the trimmed 0.20-0.31 region).",
)
parser.add_argument(
    "--hidden_dims", type=int, nargs="+", default=None,
    help="Override actor/critic hidden dims to match the checkpoint — must match what "
    "it was trained with or runner.load() fails on a shape mismatch. Default: None "
    "(use G1ArmResidualPPORunnerCfg's own default, [512, 256, 128]).",
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
from g1_locomotion.tasks.manager_based.g1_arm_residual.agents.rsl_rl_ppo_cfg import (
    G1ArmResidualLeftPPORunnerCfg,
    G1ArmResidualRightPPORunnerCfg,
)
from g1_locomotion.tasks.manager_based.g1_arm_residual.g1_arm_residual_env import (
    G1ArmResidualEnv,
    G1ArmResidualLeftEnvCfg_PLAY,
    G1ArmResidualRightEnvCfg_PLAY,
)

_EnvCfgCls = G1ArmResidualLeftEnvCfg_PLAY if args_cli.side == "left" else G1ArmResidualRightEnvCfg_PLAY
_PPORunnerCfgCls = G1ArmResidualLeftPPORunnerCfg if args_cli.side == "left" else G1ArmResidualRightPPORunnerCfg

# Reuse the exact same per-episode metrics wrapper training runs use (and the base arm
# task's own eval uses) — keeps numbers directly comparable across all three. Generic
# over any arm-family DirectRLEnv (reads goal_positions/successes/arm_joint_indices,
# all inherited unchanged from G1ArmEnv), confirmed working against this task already
# (arm_detailed.csv/arm_summary.csv were produced automatically during this task's own
# training smoke test).
from g1_locomotion.utils.eval_meta import write_eval_meta
from g1_locomotion.utils.metrics_wrappers import ArmMetricsCsvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _summarize(csv_path: str, goal_threshold_cm: float) -> dict:
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
        # "success" (env's own definition, unchanged): reached AND held under
        # goal_threshold for goal_hold_steps CONSECUTIVE steps -- the number relevant
        # for a task that needs to stay put once it arrives.
        "success_rate": sum(int(r["success"]) for r in rows) / n,
        # 2026-07-28 (user request): a SEPARATE reach-only rate -- did min_dist_to_goal
        # ever drop under goal_threshold at all, regardless of whether it then held
        # there for the required consecutive-step count. Derived directly from the
        # existing min_dist_to_goal_cm column (no wrapper/env change needed for this).
        # This is the more representative number for continuous/path-following control
        # (a sequence of waypoints), where holding at any single point isn't the goal
        # the way it is for the current hold-in-place task -- "success_rate" alone
        # would understate how well the policy tracks a moving target, since it's
        # penalized here for a hold behavior that a path-following use case doesn't
        # even want.
        "reach_rate_no_hold": sum(1 for d in dists if d < goal_threshold_cm) / n,
        "mean_reward": sum(float(r["mean_reward"]) for r in rows) / n,
        "mean_dist_cm": sum(dists) / n,
        "p50_dist_cm": pct(0.50),
        "p90_dist_cm": pct(0.90),
        # final_dist_to_goal_cm (2026-07-28 metrics_wrappers.py addition, pulled over
        # from main): distinguishes "still converging when the episode timed out" from
        # "reached its best point then stalled/overshot" -- min_dist alone can't tell
        # these apart. Directly relevant here: the residual is meant to correct a
        # STEADY-STATE gap, so a bucket where final_dist stays close to min_dist (i.e.
        # it settles and stays) is a materially different, better result than one
        # where it merely dips near the goal in passing.
        "mean_final_dist_cm": sum(float(r["final_dist_to_goal_cm"]) for r in rows) / n,
    }


def _build_base_env() -> G1ArmResidualEnv:
    env_cfg = _EnvCfgCls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.goal_x_range is not None:
        env_cfg.goal_bounds_x_override = tuple(args_cli.goal_x_range)
    return G1ArmResidualEnv(env_cfg)


def _joint_range_utilization(base_env: G1ArmResidualEnv, min_pos: torch.Tensor, max_pos: torch.Tensor) -> list[dict]:
    """Per-joint achieved range vs. the real hardware range it's clamped to — same
    diagnostic as eval_arm.py's own version, unchanged reasoning."""
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


def _run_bucket(base_env: G1ArmResidualEnv, name: str, eval_root: str) -> tuple[dict, list[dict], dict]:
    print(f"\n[Eval] --- Bucket '{name}' ---")

    # ik_baseline: force the residual to exactly zero (regardless of what the policy
    # outputs) so this bucket measures the IK+gravity-feedforward baseline ALONE --
    # the number Phase 2 already characterized (~16.65cm mean achieved error on a
    # fixed base, see ik_arm_integration_plan.md's 2026-07-28 status block). Comparing
    # this bucket against the residual_* buckets in the SAME run, same seed, same goal
    # sampling is the actual test of whether training helped.
    base_env.cfg.residual_action_scale = 0.0 if name == "ik_baseline" else base_env._orig_residual_action_scale
    # Wobble: only meaningful once the residual is actually active (ik_baseline has no
    # policy-driven correction to be robust with in the first place, so wobble is
    # forced off for it, matching how it'd otherwise just add irrelevant noise to a
    # pure-baseline measurement).
    base_env.cfg.enable_root_wobble = name != "ik_baseline"
    base_env.cfg.root_wobble_enable_step = 0 if name == "residual_with_wobble" else 10**9

    bucket_dir = os.path.join(eval_root, name)
    os.makedirs(bucket_dir, exist_ok=True)

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        env = ArmMetricsCsvWrapper(base_env, bucket_dir)
        device = base_env.device
        wrapped_env = RslRlVecEnvWrapper(env)

        agent_cfg = _PPORunnerCfgCls()
        if args_cli.hidden_dims is not None:
            agent_cfg.policy.actor_hidden_dims = list(args_cli.hidden_dims)
            agent_cfg.policy.critic_hidden_dims = list(args_cli.hidden_dims)
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        n_joints = len(base_env.arm_joint_indices)
        min_pos = torch.full((n_joints,), float("inf"), device=device)
        max_pos = torch.full((n_joints,), float("-inf"), device=device)

        reward_breakdown_sums: dict[str, float] = {}
        reward_breakdown_count = 0

        obs, _ = wrapped_env.reset()
        for _ in range(args_cli.steps_per_bucket):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
            joint_pos = base_env.robot.data.joint_pos[:, base_env.arm_joint_indices_tensor]
            min_pos = torch.minimum(min_pos, joint_pos.amin(dim=0))
            max_pos = torch.maximum(max_pos, joint_pos.amax(dim=0))
            for key, value in base_env.extras.get("log", {}).items():
                reward_breakdown_sums[key] = reward_breakdown_sums.get(key, 0.0) + float(value)
            reward_breakdown_count += 1

    joint_ranges = _joint_range_utilization(base_env, min_pos.cpu(), max_pos.cpu())
    reward_breakdown = {
        k: v / reward_breakdown_count for k, v in reward_breakdown_sums.items()
    } if reward_breakdown_count > 0 else {}

    # Close only this bucket's CSV file handles — not wrapped_env.close(), which would
    # cascade down and tear down base_env, breaking the next bucket.
    env._csv.close()
    goal_threshold_cm = base_env.cfg.goal_threshold * 100.0
    return _summarize(env.csv_path, goal_threshold_cm), joint_ranges, reward_breakdown


def main():
    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    eval_root = os.path.join(checkpoint_dir, "arm_residual_eval")
    os.makedirs(eval_root, exist_ok=True)
    write_eval_meta(eval_root, args_cli, __file__)

    print(f"[Eval] Checkpoint : {args_cli.checkpoint}")
    print(f"[Eval] Side       : {args_cli.side}")
    print(f"[Eval] Buckets    : {args_cli.buckets}")
    if args_cli.goal_x_range is not None:
        print(f"[Eval] goal_x_range override: {tuple(args_cli.goal_x_range)}")
    print(f"[Eval] num_envs   : {args_cli.num_envs}   steps_per_bucket: {args_cli.steps_per_bucket}")

    print("[Eval] Building simulation (once, reused across all buckets)...")
    base_env = _build_base_env()
    # Remember the cfg's own residual_action_scale (before any bucket forces it to 0)
    # so residual_no_wobble/residual_with_wobble can restore it after ik_baseline.
    base_env._orig_residual_action_scale = base_env.cfg.residual_action_scale

    summary_lines = [
        "# Arm Residual Eval",
        "",
        f"Checkpoint: `{args_cli.checkpoint}`",
        "",
        "`ik_baseline` forces the residual to zero -- this is the IK+gravity-feedforward "
        "number alone, for direct comparison against the residual_* buckets in the same "
        "run/seed/goal-sampling. Compare `ik_baseline`'s mean_dist_cm against Phase 2's "
        "own fixed-base result (~16.65cm mean achieved error) as a sanity check that this "
        "eval is measuring the same thing before trusting the residual buckets' improvement.",
        "",
        "`success_rate` = reached AND held for goal_hold_steps consecutive steps (the "
        "hold-in-place task this eval currently tests). `reach_rate_no_hold` = ever got "
        "under goal_threshold at all, regardless of holding -- the more representative "
        "number for continuous/path-following control, where holding at any one point "
        "isn't the goal.",
        "",
        "| Bucket | Episodes | Success rate (reach+hold) | Reach rate (no hold) | Mean reward | Mean dist (cm) | Median dist (cm) | p90 dist (cm) | Mean final dist (cm) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    joint_range_lines = [
        "",
        "## Joint range utilization",
        "",
        "| Bucket | Joint | HW range (deg) | Achieved range (deg) | % of HW range used |",
        "|---|---|---|---|---|",
    ]

    reward_breakdown_lines = [
        "",
        "## Reward component breakdown",
        "",
        "| Bucket | Component | Mean value |",
        "|---|---|---|",
    ]

    for name in args_cli.buckets:
        stats, joint_ranges, reward_breakdown = _run_bucket(base_env, name, eval_root)
        if stats["episodes"] == 0:
            print(f"[Eval] Bucket '{name}': no completed episodes — increase --steps_per_bucket.")
            summary_lines.append(f"| {name} | 0 | — | — | — | — | — | — | — |")
            continue
        print(
            f"[Eval] Bucket '{name}': episodes={stats['episodes']} "
            f"success_rate={stats['success_rate']:.2%} "
            f"reach_rate_no_hold={stats['reach_rate_no_hold']:.2%} "
            f"mean_reward={stats['mean_reward']:.3f} "
            f"mean_dist_cm={stats['mean_dist_cm']:.2f} "
            f"p50_dist_cm={stats['p50_dist_cm']:.2f} "
            f"p90_dist_cm={stats['p90_dist_cm']:.2f} "
            f"mean_final_dist_cm={stats['mean_final_dist_cm']:.2f}"
        )
        summary_lines.append(
            f"| {name} | {stats['episodes']} | {stats['success_rate']:.2%} | {stats['reach_rate_no_hold']:.2%} "
            f"| {stats['mean_reward']:.3f} | {stats['mean_dist_cm']:.2f} | {stats['p50_dist_cm']:.2f} "
            f"| {stats['p90_dist_cm']:.2f} | {stats['mean_final_dist_cm']:.2f} |"
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

        if reward_breakdown:
            print(f"[Eval]   --- reward breakdown ({name}) ---")
            for key, value in sorted(reward_breakdown.items()):
                print(f"[Eval]   {key:35s} {value:+.4f}")
                reward_breakdown_lines.append(f"| {name} | {key} | {value:+.4f} |")

    base_env.close()

    summary_path = os.path.join(eval_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines + joint_range_lines + reward_breakdown_lines) + "\n")
    print(f"\n[Eval] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
