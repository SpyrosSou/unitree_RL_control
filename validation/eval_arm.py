# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic, fixed-seed evaluation for a G1 29dof arm-policy checkpoint (7-DOF,
position-only — see g1_arm_env.py's module docstring; deliberately not called "arm-IK",
the deployed policy is pure RL joint-space control).

Rewritten 2026-07-21 for the 29dof pivot — replaces the 23dof-era, 5-DOF version that
used to live at this exact path (preserved on the `23_dof` git branch) — see
`29dof_implementation_plan.md` Phase 3. Structurally unchanged: same two-bucket sweep
(wobble curriculum forced off vs. forced on) answering both "can it actually reach
goals" and "does the synthetic base-tilt signal meaningfully hurt it", same
`ArmMetricsCsvWrapper` reuse so numbers are directly comparable to a run's own
arm_detailed.csv. Only the imports/class names and the default network size (see
`--hidden_dims` below) changed for the new 7-DOF task.

Built after a real regression this eval would have caught immediately: a first attempt at
restricting the arm's joint ranges made a large fraction of the goal workspace physically
unreachable, but this only showed up after actually training for 1500 iterations and
noticing `min_dist_to_goal_cm` had plateaued far from 0. Sweeping a fixed-seed rollout
against a *known* checkpoint (rather than reading training-time CSVs, which mix improving
and already-converged episodes together) gives a clean, repeatable "is this specific
checkpoint actually any good" answer — the same reasoning as the walking eval.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_arm.py \\
        --checkpoint logs/rsl_rl/arms/left/<run>/model_4999.pt \\
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

parser = argparse.ArgumentParser(description="Fixed-seed eval for a G1 29dof arm-policy checkpoint.")
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
    "--goal_x_range", type=float, nargs=2, default=None, metavar=("X_MIN", "X_MAX"),
    help=(
        "Override the goal box's x-range for this eval only (y/z untouched) — e.g. "
        "--goal_x_range 0.35 0.42 to evaluate any checkpoint against just the elbow-"
        "extension stress region (2026-07-08, see known_issues.md), for a fair "
        "before/after comparison independent of which checkpoint was trained on it. "
        "Default: None (use the checkpoint's own G1ArmLeftEnvCfg_PLAY bounds)."
    ),
)
parser.add_argument(
    "--legacy32", action="store_true",
    help="Evaluate a checkpoint trained before 2026-07-26's action_fb observation "
    "addition (32-D obs) — e.g. the 200/20-gain reference or the 4-way ablation "
    "sweep's baseline/privileged_critic/log_std runs. Required or runner.load() fails "
    "with a strict shape-mismatch error (safe failure, not silent corruption). Not "
    "compatible with --locked_wrist (locked-wrist's own legacy dim isn't wired up).",
)
parser.add_argument(
    "--locked_wrist", action="store_true",
    help="Evaluate a G1-Arm-Left-LockedWrist-v0 checkpoint (5 controlled joints, "
    "28-D obs) instead of the standard 7-DOF/32-D one. Required or runner.load() "
    "fails with a strict shape-mismatch error (safe failure, not silent corruption — "
    "confirmed 2026-07-24 trying to load a locked-wrist checkpoint without this flag).",
)
parser.add_argument(
    "--tail_window_s", type=float, default=5.0,
    help="Window (seconds) for the tail-settle-rate metric: fraction of episodes whose "
    "distance-to-goal stayed under --goal_threshold for the ENTIRE last N seconds of the "
    "episode, computed from the existing per-second dist_to_goal_cm_t*s snapshot columns "
    "(no env/wrapper change). The direct 'hold capability' measure for a checkpoint trained "
    "with terminate_on_success=False (where the legacy success/hold-counter metric reads a "
    "structural 0%% — see arms_policy_finalisation.md); also meaningful for any other "
    "checkpoint. Clipped to the episode length if longer.",
)
parser.add_argument(
    "--integrated", action="store_true",
    help="Evaluate a G1-Arm-Left-Integrated-v0 checkpoint (2026-07-29 static-torque-"
    "ceiling fix: integrated action targets + env-local ee/goal obs, 46-D). Required "
    "or runner.load() fails with a strict shape-mismatch error (46-D vs 39-D obs — "
    "safe failure, not silent corruption). Implies the log_std/privileged-critic "
    "runner recipe (G1ArmLeftIntegratedPPORunnerCfg); do not combine with "
    "--legacy32/--locked_wrist/--log_std.",
)
parser.add_argument(
    "--integrated_no_term", action="store_true",
    help="Evaluate a G1-Arm-Left-IntegratedNoTerm-v0 checkpoint (2026-07-30: same "
    "46-D integrated-target env as --integrated, but trained with "
    "terminate_on_success=False to fix the dithering/hold-quality problem — see "
    "arms_policy_finalisation.md step 1). Uses the matching NoTerm PLAY env (also "
    "terminate_on_success=False) so the eval's tail-settle-rate metric always sees a "
    "full trailing window, not one cut short by an early success termination. Same "
    "network/runner recipe as --integrated (shares G1ArmLeftIntegratedPPORunnerCfg — "
    "terminate_on_success doesn't affect network shape). Do not combine with "
    "--integrated/--legacy32/--locked_wrist/--log_std.",
)
parser.add_argument(
    "--log_std", action="store_true",
    help="Evaluate a checkpoint trained with noise_std_type='log' (e.g. the "
    "ablation_log_std run) instead of the default 'scalar' parameterization. "
    "Required or runner.load() fails with a strict key-mismatch error (missing "
    "'std', unexpected 'log_std' — safe failure, not silent corruption; confirmed "
    "2026-07-26 trying to load a log_std checkpoint without this flag).",
)
parser.add_argument(
    "--hidden_dims", type=int, nargs="+", default=None,
    help=(
        "Override actor/critic hidden dims to match the checkpoint being evaluated — "
        "must match what it was trained with or runner.load() fails on a shape mismatch. "
        "Default: None (use G1ArmLeftPPORunnerCfg's own default, [512, 256, 128] — the "
        "23dof-era 'WideNet' network size is the default here, not an opt-in variant, "
        "see g1_arm_env.py's module docstring). Only pass this if evaluating a "
        "checkpoint trained with a different, non-default network size."
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
import re

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmLeftAblationLogStdPPORunnerCfg,
    G1ArmLeftIntegratedPPORunnerCfg,
    G1ArmLeftLockedWristPPORunnerCfg,
    G1ArmLeftPPORunnerCfg,
)
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    G1ArmEnv,
    G1ArmLeftEnvCfg_Legacy32,
    G1ArmLeftEnvCfg_PLAY,
    G1ArmLeftIntegratedEnvCfg_PLAY,
    G1ArmLeftIntegratedNoTermEnvCfg_PLAY,
    G1ArmLeftLockedWristEnvCfg,
)

if args_cli.locked_wrist:
    _EnvCfgCls = G1ArmLeftLockedWristEnvCfg
elif args_cli.legacy32:
    _EnvCfgCls = G1ArmLeftEnvCfg_Legacy32
elif args_cli.integrated_no_term:
    _EnvCfgCls = G1ArmLeftIntegratedNoTermEnvCfg_PLAY
elif args_cli.integrated:
    _EnvCfgCls = G1ArmLeftIntegratedEnvCfg_PLAY
else:
    _EnvCfgCls = G1ArmLeftEnvCfg_PLAY

if args_cli.locked_wrist:
    _PPORunnerCfgCls = G1ArmLeftLockedWristPPORunnerCfg
elif args_cli.integrated or args_cli.integrated_no_term:
    _PPORunnerCfgCls = G1ArmLeftIntegratedPPORunnerCfg
elif args_cli.log_std:
    _PPORunnerCfgCls = G1ArmLeftAblationLogStdPPORunnerCfg
else:
    _PPORunnerCfgCls = G1ArmLeftPPORunnerCfg

# Reuse the exact same per-episode metrics wrapper training runs use — keeps the eval's
# numbers directly comparable to what you'd see in a run's arm_detailed.csv.
from g1_locomotion.utils.eval_meta import write_eval_meta
from g1_locomotion.utils.metrics_wrappers import ArmMetricsCsvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Matches ArmMetricsCsvWrapper's per-second snapshot columns, e.g. "dist_to_goal_cm_t5.0s".
_TAIL_SNAPSHOT_RE = re.compile(r"^dist_to_goal_cm_t([0-9.]+)s$")


def _tail_settle_stats(rows: list[dict], goal_threshold_cm: float, tail_window_s: float) -> dict:
    """2026-07-30: 'did distance-to-goal stay under threshold for the ENTIRE last
    tail_window_s of the episode' -- computed purely from the existing per-second
    dist_to_goal_cm_t*s snapshot columns (metrics_wrappers.py's
    _DIST_SNAPSHOT_INTERVAL_S=1.0s cadence, already written to every arm_detailed.csv;
    no env/wrapper change). Motivated by (see arms_policy_finalisation.md): with
    terminate_on_success=False the legacy success/hold-counter metric reads a
    structural 0% (the CSV's own 'success' column is `terminated[env_id]`, and
    terminated is only ever True when terminate_on_success is on), and
    final_dist_to_goal_cm/settled_rate_* only check the single LAST instant -- a
    policy still oscillating right up to episode end would pass those but shouldn't
    count as 'holding'. This checks a whole trailing window instead of one instant.

    Episodes lacking a full tail_window_s of snapshots (ended before that window, e.g.
    an old checkpoint's early-success termination, or too few --steps_per_bucket steps)
    are excluded from the denominator and counted in tail_excluded -- reported
    separately rather than silently treated as pass or fail either way."""
    if not rows:
        return {"tail_settle_rate": 0.0, "tail_settle_window_s": 0.0, "tail_settle_excluded": 0}
    candidate_times = sorted(
        float(m.group(1)) for c in rows[0].keys() if (m := _TAIL_SNAPSHOT_RE.match(c))
    )
    if not candidate_times:
        return {"tail_settle_rate": 0.0, "tail_settle_window_s": 0.0, "tail_settle_excluded": len(rows)}
    # FIXED 2026-07-31: drop any snapshot column that's EMPTY FOR EVERY ROW before
    # picking the trailing window -- confirmed on a real run (arms/integrated_no_term/
    # 2026-07-30_19-57-17) that the nominal final slot (t == episode_length_s, e.g.
    # t20.0s for a 20s episode) is empty in 100% of rows: an episode truncates the
    # instant episode_length_buf hits its cap, one control step short of the exact
    # step-count boundary metrics_wrappers.py's _update_dist_snapshot needs to fire
    # that slot's "due" check -- a property of that existing wrapper, not a
    # per-checkpoint quirk. The original version anchored the window to this
    # structurally-dead column, so EVERY episode got excluded (denom=0, silently
    # reporting the 0.0 fallback) regardless of real hold quality -- confirmed: that
    # run's own Settled-<2cm/mean-dist/frac_envs_reached numbers show near-perfect
    # holding, so the correct tail-settle rate was nowhere near 0%. Only entirely-dead
    # columns are dropped (not per-row) -- a column populated for most-but-not-all
    # rows still participates, and rows still missing it individually are (correctly)
    # excluded per-row below.
    usable_times = [
        t for t in candidate_times
        if any(r[f"dist_to_goal_cm_t{round(t, 1)}s"] != "" for r in rows)
    ]
    if not usable_times:
        return {"tail_settle_rate": 0.0, "tail_settle_window_s": 0.0, "tail_settle_excluded": len(rows)}
    max_t = usable_times[-1]
    window_used = min(tail_window_s, max_t)
    tail_cols = [f"dist_to_goal_cm_t{round(t, 1)}s" for t in usable_times if t > max_t - window_used + 1e-6]

    settled = 0
    excluded = 0
    for r in rows:
        vals = [r[c] for c in tail_cols]
        if any(v == "" for v in vals):
            excluded += 1
            continue
        if all(float(v) < goal_threshold_cm for v in vals):
            settled += 1
    denom = len(rows) - excluded
    return {
        "tail_settle_rate": (settled / denom) if denom > 0 else 0.0,
        "tail_settle_window_s": window_used,
        "tail_settle_excluded": excluded,
    }


def _summarize(csv_path: str, goal_threshold_cm: float, tail_window_s: float) -> dict:
    if not os.path.isfile(csv_path):
        return {"episodes": 0}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"episodes": 0}
    n = len(rows)
    dists = sorted(float(r["min_dist_to_goal_cm"]) for r in rows)
    tail_stats = _tail_settle_stats(rows, goal_threshold_cm, tail_window_s)

    def pct(p: float) -> float:
        return dists[min(int(n * p), n - 1)]

    return {
        "episodes": n,
        # "success" (env's own definition, unchanged): reached AND held under
        # goal_threshold for goal_hold_steps CONSECUTIVE steps.
        "success_rate": sum(int(r["success"]) for r in rows) / n,
        # 2026-07-29, ported from ik_residuals (user request there 2026-07-28): a
        # SEPARATE reach-only rate -- did min_dist_to_goal ever drop under
        # goal_threshold at all, regardless of whether it then held there for the
        # required consecutive-step count. Derived directly from the existing
        # min_dist_to_goal_cm column, no wrapper/env change needed. More representative
        # for anything that doesn't need a hold-in-place (e.g. a path-following/
        # residual-correction use case) -- success_rate alone would understate tracking
        # quality by penalizing a hold behavior that use case doesn't even want.
        "reach_rate_no_hold": sum(1 for d in dists if d < goal_threshold_cm) / n,
        "mean_reward": sum(float(r["mean_reward"]) for r in rows) / n,
        "mean_dist_cm": sum(dists) / n,
        "p50_dist_cm": pct(0.50),
        "p90_dist_cm": pct(0.90),
        # final_dist_to_goal_cm (2026-07-28 metrics_wrappers.py addition): distinguishes
        # "still converging when the episode timed out" from "reached its best point
        # then stalled/overshot" -- min_dist alone can't tell these apart.
        "mean_final_dist_cm": sum(float(r["final_dist_to_goal_cm"]) for r in rows) / n,
        # ADDED 2026-07-30: "settled" rates -- fraction of episodes ENDING within
        # 2cm/3cm of the goal, the use-case-representative headline for a grasp task
        # ("get there and be there at the end"). Motivated by the Integrated run's
        # eval (see policy_status.md 2026-07-30): success_rate read 6.5% while 81% of
        # episodes ended <2cm and 100% ended <2.76cm -- the 15-CONSECUTIVE-step hold +
        # terminate_on_success makes success_rate an incentive artifact (completing a
        # hold is reward-negative, so a well-trained policy dithers at the boundary
        # and rarely "succeeds"), NOT a capability measure. Judge checkpoints by these
        # two columns and the dist columns; treat success_rate as a legacy indicator
        # until the env's termination/bonus design is reworked (see
        # arms_policy_finalisation.md step 1).
        "settled_rate_2cm": sum(1 for r in rows if float(r["final_dist_to_goal_cm"]) < 2.0) / n,
        "settled_rate_3cm": sum(1 for r in rows if float(r["final_dist_to_goal_cm"]) < 3.0) / n,
        # ADDED 2026-07-30: see _tail_settle_stats' own docstring -- the direct "hold
        # capability" measure (stayed under goal_threshold for the WHOLE trailing
        # tail_window_s, not just the final instant), for use with
        # terminate_on_success=False checkpoints where the legacy success column
        # structurally reads 0%.
        "tail_settle_rate": tail_stats["tail_settle_rate"],
        "tail_settle_window_s": tail_stats["tail_settle_window_s"],
        "tail_settle_excluded": tail_stats["tail_settle_excluded"],
    }


def _build_base_env() -> G1ArmEnv:
    env_cfg = _EnvCfgCls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.goal_x_range is not None:
        env_cfg.goal_bounds_x_override = tuple(args_cli.goal_x_range)
    return G1ArmEnv(env_cfg)


def _joint_range_utilization(base_env: G1ArmEnv, min_pos: torch.Tensor, max_pos: torch.Tensor) -> list[dict]:
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


def _run_bucket(base_env: G1ArmEnv, name: str, eval_root: str) -> tuple[dict, list[dict]]:
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

        # ADDED 2026-07-26: retroactive reward-component breakdown for ANY checkpoint —
        # base_env.extras["log"] (see g1_arm_env.py's _get_rewards) is populated every
        # step regardless of train/eval mode, so this works on old checkpoints too, not
        # just new training runs — averaged here the same way rsl_rl's own runner would.
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
    return _summarize(env.csv_path, goal_threshold_cm, args_cli.tail_window_s), joint_ranges, reward_breakdown


def main():
    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    eval_root = os.path.join(checkpoint_dir, "arm_eval")
    os.makedirs(eval_root, exist_ok=True)
    write_eval_meta(eval_root, args_cli, __file__)

    print(f"[Eval] Checkpoint : {args_cli.checkpoint}")
    print(f"[Eval] Buckets    : {args_cli.buckets}")
    if args_cli.goal_x_range is not None:
        print(f"[Eval] goal_x_range override: {tuple(args_cli.goal_x_range)}")
    print(f"[Eval] num_envs   : {args_cli.num_envs}   steps_per_bucket: {args_cli.steps_per_bucket}")

    print("[Eval] Building simulation (once, reused across all buckets)...")
    base_env = _build_base_env()

    summary_lines = [
        "# Arm Eval",
        "",
        f"Checkpoint: `{args_cli.checkpoint}`",
        "",
        "Headline columns: **Settled <2cm / <3cm** (fraction of episodes ENDING that close "
        "to the goal — the grasp-use-case measure) and **Tail-settle (Ws)** (fraction of "
        "episodes that stayed under goal_threshold for the ENTIRE last W seconds, W shown "
        "per-row — the direct HOLD-capability measure, distinguishing 'settled and still' "
        "from 'happened to be close at the very last instant'; computed from the existing "
        "per-second dist_to_goal_cm_t*s columns, see arms_policy_finalisation.md). Episodes "
        "ending before a full W-second window existed are excluded from that rate, not "
        "counted as pass or fail — see 'excl.' in the same cell. 'Success rate (reach+hold)' "
        "is the env's legacy 15-consecutive-steps-under-2cm + early-termination definition — "
        "known to be an INCENTIVE ARTIFACT for well-trained policies (completing the hold is "
        "reward-negative under terminate_on_success, so good policies dither at the boundary "
        "and score single digits while ending ~1cm from the goal — see policy_status.md "
        "2026-07-30), and STRUCTURALLY 0% for any checkpoint trained with "
        "terminate_on_success=False. Do not judge checkpoints by it until the env's "
        "termination/bonus design is reworked (arms_policy_finalisation.md, step 1).",
        "",
        "| Bucket | Episodes | Settled <2cm | Settled <3cm | Tail-settle (Ws) | Success rate (reach+hold, legacy) | Reach rate (no hold) | Mean reward | Mean dist (cm) | Median dist (cm) | p90 dist (cm) | Mean final dist (cm) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
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

    reward_breakdown_lines = [
        "",
        "## Reward component breakdown",
        "",
        "Per-step mean of each reward term (not per-episode sum) — added 2026-07-26 "
        "since this task previously had zero per-term visibility (unlike the walking "
        "task's automatic RewardManager logging), making it impossible to check e.g. "
        "whether joint_limit or torso_proximity penalties were disproportionately "
        "large without guessing. Works retroactively on any checkpoint, not just new "
        "training runs.",
        "",
        "| Bucket | Component | Mean value |",
        "|---|---|---|",
    ]

    for name in args_cli.buckets:
        stats, joint_ranges, reward_breakdown = _run_bucket(base_env, name, eval_root)
        if stats["episodes"] == 0:
            print(f"[Eval] Bucket '{name}': no completed episodes — increase --steps_per_bucket.")
            summary_lines.append(f"| {name} | 0 | — | — | — | — | — | — | — | — | — | — |")
            continue
        tail_cell = (
            f"{stats['tail_settle_rate']:.2%} (W={stats['tail_settle_window_s']:.1f}s"
            + (f", excl={stats['tail_settle_excluded']})" if stats["tail_settle_excluded"] else ")")
        )
        print(
            f"[Eval] Bucket '{name}': episodes={stats['episodes']} "
            f"settled_rate_2cm={stats['settled_rate_2cm']:.2%} "
            f"settled_rate_3cm={stats['settled_rate_3cm']:.2%} "
            f"tail_settle_rate={stats['tail_settle_rate']:.2%} "
            f"(window={stats['tail_settle_window_s']:.1f}s, excluded={stats['tail_settle_excluded']}) "
            f"success_rate(legacy)={stats['success_rate']:.2%} "
            f"reach_rate_no_hold={stats['reach_rate_no_hold']:.2%} "
            f"mean_reward={stats['mean_reward']:.3f} "
            f"mean_dist_cm={stats['mean_dist_cm']:.2f} "
            f"p50_dist_cm={stats['p50_dist_cm']:.2f} "
            f"p90_dist_cm={stats['p90_dist_cm']:.2f} "
            f"mean_final_dist_cm={stats['mean_final_dist_cm']:.2f}"
        )
        summary_lines.append(
            f"| {name} | {stats['episodes']} | {stats['settled_rate_2cm']:.2%} | {stats['settled_rate_3cm']:.2%} "
            f"| {tail_cell} "
            f"| {stats['success_rate']:.2%} | {stats['reach_rate_no_hold']:.2%} "
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
