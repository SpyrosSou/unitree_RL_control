# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single, comprehensive evaluation for a G1 29dof unified stand+walk checkpoint — one
command, one `summary.md`, everything we actually check for a walking checkpoint in one
place, mirroring how `eval_arm.py` works for arms. Replaces having to separately run
this script for command-tracking/drift, `check_real_displacement.py` for a raw
world-frame sanity check, and eval_walking.py --pin_disturbance_phase four times for the
per-phase disturbance fall-rate sweep — those were three separate Isaac Sim launches and
manual copy-pasted terminal output; this is one launch, one file on disk.

Rewritten 2026-07-27 to consolidate all three (previously: `eval_walking.py`'s own
command-bucket/drift sweep, `testing/general_testing/check_real_displacement.py`'s raw
`root_pos_w` sanity check, and repeated `--pin_disturbance_phase` runs for the 4-phase
arm-disturbance fall-rate check). All three reuse ONE `ManagerBasedRLEnv` instance
(rebuilding Isaac Sim's simulation context repeatedly within one process is unreliable —
see `_run_command_bucket`'s docstring) instead of three separate script invocations.

Sections run, in order:
1. Command-bucket sweep (tracking error, foot slip, fall rate, and — for the
   zero-lateral/zero-yaw "straight" buckets — heading/lateral drift).
2. Arm-disturbance phase sweep (fall rate under each of the 4 disturbance phases while
   standing still) — only if `--arm_disturbance` is set; skip with `--skip_phases`.
3. Raw world-frame displacement sanity check (does actual `root_pos_w` displacement over
   a fixed-speed rollout match what the commanded speed implies) — guards against a real
   past bug in this project where a reward-derived tracking-error metric was misread as
   achieved velocity (see git history 2026-07-23/24). Skip with `--skip_displacement`.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_walking.py \\
        --checkpoint logs/rsl_rl/walking/arm_disturbance/<run>/model_7999.pt \\
        --arm_disturbance \\
        --headless

    python validation/eval_walking.py \\
        --checkpoint logs/rsl_rl/walking/base/<run>/model_2998.pt \\
        --headless

Output:
    <checkpoint_dir>/walking_eval/summary.md — all three sections above, plus the raw
    per-episode CSVs per bucket/phase under their own subdirectories (reusing
    ``WalkingMetricsCsvWrapper``, so these numbers are directly comparable to a run's own
    ``walking_detailed.csv``).
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

# Command envelope: name -> (lin_vel_x, lin_vel_y, ang_vel_z). "straight" buckets (zero
# commanded lateral velocity and yaw rate) are the ones drift is meaningfully checked on.
#
# FIXED 2026-07-29: the turn/strafe/combo buckets previously commanded velocities the
# policy was NEVER trained on — training limit_ranges cap ang_vel_z at ±0.2 and
# lin_vel_y at ±0.3 (and ang_vel_z effectively trains at only ±0.1, since
# ang_vel_cmd_levels was never wired into CurriculumCfg — see mdp/curriculums.py), yet
# turn_left/right evaluated at ±0.6 (3-6x out of distribution) and strafe at ±0.4.
# The elevated turn_left fall rates recorded in policy_status.md were therefore
# stress-test numbers, not in-distribution performance. Default buckets now stay inside
# the trained envelope; the old values are kept as explicit "*_stress" buckets (opt-in
# via --buckets, excluded from the default list) so the OOD data point isn't lost.
# NOTE: turn_*/strafe_*/combo numbers from summaries generated before this date used
# the old (stress) commands — compare old summaries against the *_stress buckets, not
# the same-named default ones.
_BUCKETS = {
    "stand_still": (0.0, 0.0, 0.0),
    "forward_slow": (0.3, 0.0, 0.0),
    "forward_medium": (0.6, 0.0, 0.0),
    "forward_fast": (0.9, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "strafe_left": (0.0, 0.3, 0.0),
    "strafe_right": (0.0, -0.3, 0.0),
    "turn_left": (0.0, 0.0, 0.2),
    "turn_right": (0.0, 0.0, -0.2),
    "forward_turn_combo": (0.5, 0.0, 0.2),
    # Out-of-distribution stress variants (the pre-2026-07-29 bucket values):
    "strafe_left_stress": (0.0, 0.4, 0.0),
    "strafe_right_stress": (0.0, -0.4, 0.0),
    "turn_left_stress": (0.0, 0.0, 0.6),
    "turn_right_stress": (0.0, 0.0, -0.6),
    "forward_turn_combo_stress": (0.5, 0.0, 0.4),
}
_DEFAULT_BUCKETS = [name for name in _BUCKETS if not name.endswith("_stress")]
_STRAIGHT_BUCKETS = {"forward_slow", "forward_medium", "forward_fast", "backward"}
_PHASES = [0, 1, 2, 3]

parser = argparse.ArgumentParser(description="Comprehensive eval for a G1 29dof stand+walk checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint (.pt).")
parser.add_argument(
    "--arm_disturbance", action="store_true",
    help="Build the eval env from G1LocomotionArmDisturbanceEnvCfg_PLAY (disturbance forced on) "
    "instead of the plain G1LocomotionEnvCfg_PLAY — pass this for a checkpoint trained on "
    "G1-Locomotion-Velocity-ArmDisturbance-v0. Also gates whether the disturbance-phase sweep "
    "(section 2) runs at all — meaningless without the disturbance event in the first place.",
)
parser.add_argument(
    "--buckets", type=str, nargs="+", default=_DEFAULT_BUCKETS, choices=list(_BUCKETS.keys()),
    help="Which command buckets to evaluate in section 1. The *_stress buckets (the "
    "pre-2026-07-29 out-of-distribution turn/strafe commands) are opt-in, not default.",
)
parser.add_argument(
    "--phases", type=int, nargs="+", default=_PHASES, choices=_PHASES,
    help="Which arm-disturbance phases to evaluate in section 2 (only with --arm_disturbance).",
)
parser.add_argument("--skip_phases", action="store_true", help="Skip section 2 entirely.")
parser.add_argument("--skip_displacement", action="store_true", help="Skip section 3 entirely.")
parser.add_argument(
    "--displacement_speed", type=float, default=0.6,
    help="Commanded forward speed (m/s) for the section-3 raw displacement check.",
)
parser.add_argument(
    "--displacement_steps", type=int, default=500,
    help="Control steps for the section-3 raw displacement check (500 @ 50Hz = 10s).",
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs per bucket/phase.")
parser.add_argument(
    "--steps_per_bucket", type=int, default=1500,
    help="Control steps per bucket/phase (at 50 Hz; 1500 steps ~= 30 s, several episodes per env).",
)
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible rollouts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import csv
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionArmDisturbanceEnvCfg_PLAY,
    G1LocomotionEnvCfg_PLAY,
)
# Reuse the exact same per-episode metrics wrapper training runs use — keeps the eval's
# numbers directly comparable to what you'd see in a run's walking_detailed.csv.
from g1_locomotion.utils.eval_meta import write_eval_meta
from g1_locomotion.utils.metrics_wrappers import WalkingMetricsCsvWrapper


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x * var_y) ** 0.5


# 2026-07-28: a raw fall-rate percentage treats "fell at 3s" and "fell at 18s" as
# identical, but they're not the same failure in practice — a command held for the full
# episode duration is an eval artifact (see turn_left's dead ang_vel curriculum
# discussion), not something a real user is likely to sustain. Bin fall timing so "falls
# almost immediately" (a real, load-bearing problem) is visible separately from "falls
# only after sustaining an unrealistically long command" (lower priority).
_CONTROL_DT_S = 0.02  # sim.dt(0.005) * decimation(4), see g1_locomotion_env_cfg.py
_FALL_TIMING_BIN_EDGES_S = (3.0, 6.0, 12.0)  # bins: [0,3) [3,6) [6,12) [12,end)


def _fall_timing_bin_labels() -> list[str]:
    edges = _FALL_TIMING_BIN_EDGES_S
    labels = [f"<{edges[0]:.0f}s"]
    for lo, hi in zip(edges, edges[1:]):
        labels.append(f"{lo:.0f}-{hi:.0f}s")
    labels.append(f">{edges[-1]:.0f}s")
    return labels


def _summarize(csv_path: str, is_straight: bool, compute_drift: bool | None = None) -> dict:
    if not os.path.isfile(csv_path):
        return {"episodes": 0}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"episodes": 0}
    n = len(rows)
    fell = sum(int(r["fell"]) for r in rows)
    stats = {
        "episodes": n,
        "fall_rate": fell / n,
        "mean_lin_vel_track_err_m_s": sum(float(r["mean_lin_vel_track_err_m_s"]) for r in rows) / n,
        "mean_ang_vel_track_err_rad_s": sum(float(r["mean_ang_vel_track_err_rad_s"]) for r in rows) / n,
        "mean_foot_slip_speed_m_s": sum(float(r["mean_foot_slip_speed_m_s"]) for r in rows) / n,
    }
    if fell > 0:
        fall_times_s = [int(r["episode_steps"]) * _CONTROL_DT_S for r in rows if int(r["fell"])]
        edges = _FALL_TIMING_BIN_EDGES_S
        bin_counts = [0] * (len(edges) + 1)
        for t in fall_times_s:
            idx = next((i for i, e in enumerate(edges) if t < e), len(edges))
            bin_counts[idx] += 1
        stats["fall_count"] = fell
        stats["fall_timing_bin_counts"] = bin_counts
    # Knee-angle tracking (2026-07-27, see deferred_items_2026-07-21.md item 8): checks a
    # visually-observed hypothesis that near-extended knees correlate with the
    # asymmetric-step corrections that precede bad heading drift — computed for every
    # bucket (not just straight ones) since knee behavior itself isn't drift-specific,
    # only the correlation-with-drift check below is.
    mean_knee_angles = [float(r["mean_knee_angle_deg"]) for r in rows]
    min_knee_angles = [float(r["min_knee_angle_deg"]) for r in rows]
    stats["mean_knee_angle_deg"] = sum(mean_knee_angles) / n
    stats["mean_min_knee_angle_deg"] = sum(min_knee_angles) / n
    # 2026-07-28: real step count (foot lift + re-plant, air-time-threshold gated — see
    # WalkingMetricsCsvWrapper._update_stepping_metrics). Added because drift (net
    # displacement) and foot-slip (sliding while in contact) both miss a foot lifting and
    # re-planting nearby — visually a real "stationary step" but invisible to either of
    # those metrics if it doesn't net out to much displacement. Computed for every bucket,
    # same reasoning as knee angle above.
    step_counts = [int(r["step_count"]) for r in rows]
    max_air_times = [float(r["max_foot_air_time_s"]) for r in rows]
    stats["mean_step_count"] = sum(step_counts) / n
    stats["mean_max_foot_air_time_s"] = sum(max_air_times) / n
    # 2026-07-28: heading_drift_deg/lateral_drift_m are computed unconditionally by the
    # wrapper for every bucket (including zero-velocity ones — "expected position" trivially
    # reduces to start position when nothing is commanded, so the same math is exactly
    # "how far did it wander despite being told to do nothing"). Previously only surfaced
    # for `is_straight` buckets; widened so stand_still gets it too — this is the direct
    # quantification of standing-while-stationary drift, not just a walking-drift metric.
    if compute_drift is None:
        compute_drift = is_straight
    compute_drift = compute_drift or is_straight
    if compute_drift:
        heading_drifts = [float(r["heading_drift_deg"]) for r in rows]
        lateral_drifts = [float(r["lateral_drift_m"]) for r in rows]
        stats["mean_abs_heading_drift_deg"] = sum(abs(x) for x in heading_drifts) / n
        stats["max_abs_heading_drift_deg"] = max(abs(x) for x in heading_drifts)
        stats["mean_abs_lateral_drift_m"] = sum(abs(x) for x in lateral_drifts) / n
        stats["max_abs_lateral_drift_m"] = max(abs(x) for x in lateral_drifts)
        if is_straight:
            # Negative correlation = episodes that stay more extended (lower min knee
            # angle) tend to drift more — supports the hypothesis. Near zero = no
            # relationship found. Kept straight-buckets-only: this specific correlation is
            # about the walking-drift hypothesis, not the general standing-stillness check
            # compute_drift enables above.
            abs_heading = [abs(x) for x in heading_drifts]
            stats["knee_vs_heading_drift_corr"] = _pearson(min_knee_angles, abs_heading)
    return stats


def _build_base_env() -> ManagerBasedRLEnv:
    env_cfg_cls = G1LocomotionArmDisturbanceEnvCfg_PLAY if args_cli.arm_disturbance else G1LocomotionEnvCfg_PLAY
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # The _PLAY config disables observation noise (IMU/encoder Unoise) for clean
    # visualization — but that makes this eval meaningfully easier than real training
    # and deployment, and defeats the point of a sim2real robustness check. Restore the
    # exact training-time noise config (enable_corruption is just a group-level on/off
    # switch over the same, otherwise-unchanged Unoise terms).
    env_cfg.observations.policy.enable_corruption = True

    # Disable random pushes/external force disturbance for this eval — this measures the
    # policy's own tracking/gait behavior (and drift from it), not push recovery.
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None

    # Disable heading_command mode (if present) so ranges.ang_vel_z drives the yaw-rate
    # command directly instead of being overridden by an internal heading-tracking law
    # that chases an independently sampled target heading.
    base_velocity = env_cfg.commands.base_velocity
    if hasattr(base_velocity, "heading_command"):
        base_velocity.heading_command = False
    base_velocity.rel_standing_envs = 0.0
    # debug_vis=True (the class default) spawns a debug-arrow USD prim fetched from a
    # remote Nucleus/S3 asset path — with no local cache and a slow/unreachable network
    # path this hangs for up to a 300s timeout per env launch, even in --headless mode
    # (confirmed 2026-07-24 — see check_arm_disturbance_magnitude.py's identical fix).
    base_velocity.debug_vis = False

    return ManagerBasedRLEnv(cfg=env_cfg)


def _set_command(base_env: ManagerBasedRLEnv, vx: float, vy: float, wz: float):
    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (vx, vx)
    command_term.cfg.ranges.lin_vel_y = (vy, vy)
    command_term.cfg.ranges.ang_vel_z = (wz, wz)
    return command_term


def _run_command_bucket(base_env: ManagerBasedRLEnv, name: str, vx: float, vy: float, wz: float, out_dir: str) -> dict:
    """Run one fixed-command rollout on the shared, already-constructed base_env.

    Deliberately reuses one ManagerBasedRLEnv across every bucket/phase in this script
    instead of building a fresh one per section — repeatedly constructing/tearing down
    Isaac Sim's simulation context within a single process is unreliable. The velocity
    command term reads its cfg.ranges fresh on every resample, so mutating the live
    term's ranges and calling reset() is enough to switch commands — no rebuild needed.
    """
    print(f"\n[Eval] --- '{name}' (vx={vx}, vy={vy}, wz={wz}) ---")
    command_term = _set_command(base_env, vx, vy, wz)

    bucket_dir = os.path.join(out_dir, name)
    os.makedirs(bucket_dir, exist_ok=True)

    # Everything below touches base_env's live state tensors. From the second
    # bucket/phase onward those tensors get written to while already tagged as PyTorch
    # "inference tensors" (the first rollout ran inside inference_mode) — in-place writes
    # to an inference tensor are only legal from *inside* inference_mode, so
    # construction/reset must be in this block too, not just the stepping loop.
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        env = WalkingMetricsCsvWrapper(base_env, bucket_dir)
        device = base_env.device
        wrapped_env = RslRlVecEnvWrapper(env)

        agent_cfg = BasePPORunnerCfg()
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        # reset() forces every env to resample its command immediately, picking up the
        # ranges just set above rather than waiting for the natural resampling_time_range.
        obs, _ = wrapped_env.reset()
        # FOUND 2026-07-23 (g1_full_demo.py investigation): that same reset() also
        # re-rolls is_standing_env per env (probability rel_standing_envs), and
        # UniformVelocityCommand._update_command() force-zeroes vel_command_b every
        # step for any env still flagged — silently overriding the fixed (vx,vy,wz)
        # this bucket is supposed to be testing for whichever ~2% of envs get unlucky.
        command_term.is_standing_env[:] = False
        for _ in range(args_cli.steps_per_bucket):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

    # Close only this bucket's CSV file handles — not wrapped_env.close(), which would
    # cascade down and tear down base_env, breaking the next bucket/phase.
    env._csv.close()
    return _summarize(env.csv_path, is_straight=name in _STRAIGHT_BUCKETS, compute_drift=(name == "stand_still"))


def _run_displacement_check(base_env: ManagerBasedRLEnv) -> dict:
    """Raw root_pos_w displacement over a fixed-speed rollout — independent of the
    reward-derived tracking-error metric section 1 already reports. Guards against a
    real past bug (2026-07-23/24) where that metric was misread as achieved velocity;
    this reads world-frame position directly, no ambiguity possible."""
    print(f"\n[Eval] --- Displacement check (forward {args_cli.displacement_speed} m/s, "
          f"{args_cli.displacement_steps} steps) ---")
    command_term = _set_command(base_env, args_cli.displacement_speed, 0.0, 0.0)

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        wrapped_env = RslRlVecEnvWrapper(base_env)
        device = base_env.device
        agent_cfg = BasePPORunnerCfg()
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        obs, _ = wrapped_env.reset()
        command_term.is_standing_env[:] = False
        start_pos = base_env.scene["robot"].data.root_pos_w[:, :2].clone()

        for _ in range(args_cli.displacement_steps):
            command_term.is_standing_env[:] = False
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

        final_pos = base_env.scene["robot"].data.root_pos_w[:, :2]
        total_disp = (final_pos - start_pos).norm(dim=-1)

    expected_disp = args_cli.displacement_speed * (args_cli.displacement_steps / 50.0)
    return {
        "expected_disp_m": expected_disp,
        "mean_disp_m": total_disp.mean().item(),
        "min_disp_m": total_disp.min().item(),
        "max_disp_m": total_disp.max().item(),
        "frac_of_expected": total_disp.mean().item() / expected_disp if expected_disp > 0 else float("nan"),
    }


def main():
    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    eval_root = os.path.join(checkpoint_dir, "walking_eval")
    os.makedirs(eval_root, exist_ok=True)
    write_eval_meta(eval_root, args_cli, __file__)

    run_phases = args_cli.arm_disturbance and not args_cli.skip_phases

    print(f"[Eval] Checkpoint     : {args_cli.checkpoint}")
    print(f"[Eval] Arm disturbance: {args_cli.arm_disturbance}")
    print(f"[Eval] Buckets        : {args_cli.buckets}")
    print(f"[Eval] Phase sweep    : {args_cli.phases if run_phases else 'skipped'}")
    print(f"[Eval] Displacement   : {'skipped' if args_cli.skip_displacement else 'yes'}")
    print(f"[Eval] num_envs       : {args_cli.num_envs}   steps_per_bucket: {args_cli.steps_per_bucket}")

    print("[Eval] Building simulation (once, reused across every section)...")
    base_env = _build_base_env()

    summary_lines = [
        "# G1 29dof Walking Eval",
        "",
        f"Checkpoint: `{args_cli.checkpoint}`",
        f"Arm disturbance forced on: {args_cli.arm_disturbance}",
        "",
        "## 1. Command-bucket sweep",
        "",
        "| Bucket | vx, vy, wz | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) "
        "| Foot slip (m/s) | Mean\\|heading drift\\| (deg) | Mean\\|lateral drift\\| (m) "
        "| Mean step count | Mean max foot air time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    command_bucket_stats: dict[str, dict] = {}
    for name in args_cli.buckets:
        vx, vy, wz = _BUCKETS[name]
        stats = _run_command_bucket(base_env, name, vx, vy, wz, eval_root)
        command_bucket_stats[name] = stats
        if stats["episodes"] == 0:
            print(f"[Eval] Bucket '{name}': no completed episodes — increase --steps_per_bucket.")
            summary_lines.append(f"| {name} | {vx}, {vy}, {wz} | 0 | — | — | — | — | — | — | — | — |")
            continue

        drift_str = ""
        if "mean_abs_heading_drift_deg" in stats:
            drift_str = (
                f" mean_abs_heading_drift_deg={stats['mean_abs_heading_drift_deg']:.2f} "
                f"mean_abs_lateral_drift_m={stats['mean_abs_lateral_drift_m']:.3f}"
            )
        print(
            f"[Eval] Bucket '{name}': episodes={stats['episodes']} "
            f"fall_rate={stats['fall_rate']:.2%} "
            f"lin_track_err={stats['mean_lin_vel_track_err_m_s']:.3f} "
            f"ang_track_err={stats['mean_ang_vel_track_err_rad_s']:.3f} "
            f"foot_slip={stats['mean_foot_slip_speed_m_s']:.3f}" + drift_str +
            f" mean_step_count={stats['mean_step_count']:.2f} "
            f"mean_max_foot_air_time_s={stats['mean_max_foot_air_time_s']:.3f}"
        )

        heading_col = f"{stats['mean_abs_heading_drift_deg']:.2f}" if "mean_abs_heading_drift_deg" in stats else "—"
        lateral_col = f"{stats['mean_abs_lateral_drift_m']:.3f}" if "mean_abs_lateral_drift_m" in stats else "—"
        summary_lines.append(
            f"| {name} | {vx}, {vy}, {wz} | {stats['episodes']} | {stats['fall_rate']:.2%} "
            f"| {stats['mean_lin_vel_track_err_m_s']:.3f} | {stats['mean_ang_vel_track_err_rad_s']:.3f} "
            f"| {stats['mean_foot_slip_speed_m_s']:.3f} | {heading_col} | {lateral_col} "
            f"| {stats['mean_step_count']:.2f} | {stats['mean_max_foot_air_time_s']:.3f} |"
        )

    summary_lines += [
        "",
        "## 2. Arm-disturbance phase sweep (standing still, fall rate per phase)",
        "",
    ]
    if not run_phases:
        summary_lines.append("Skipped (`--arm_disturbance` not set, or `--skip_phases` passed).")
    else:
        summary_lines += [
            "| Phase | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) | Foot slip (m/s) |",
            "|---|---|---|---|---|---|",
        ]
        for phase in args_cli.phases:
            term_cfg = base_env.event_manager.get_term_cfg("arm_motion_disturbance")
            # Pins the phase for the whole rollout instead of letting it cycle over
            # elapsed eval time — see the old --pin_disturbance_phase flag's docstring
            # (now folded into this always-on sweep): sets phase_step_boundaries to N
            # copies of 0, which makes the phase-index lookup fall through immediately
            # to phase N regardless of step count. Mutating term_cfg.params in place
            # (not rebuilding the env) picks up live on this term's next "interval" call.
            term_cfg.params["phase_step_boundaries"] = tuple([0] * phase)
            term_cfg.params["phase_step_offset"] = 0

            name = f"disturbance_phase_{phase}"
            stats = _run_command_bucket(base_env, name, 0.0, 0.0, 0.0, eval_root)
            if stats["episodes"] == 0:
                print(f"[Eval] Phase {phase}: no completed episodes — increase --steps_per_bucket.")
                summary_lines.append(f"| {phase} | 0 | — | — | — | — |")
                continue
            print(
                f"[Eval] Phase {phase}: episodes={stats['episodes']} fall_rate={stats['fall_rate']:.2%} "
                f"lin_track_err={stats['mean_lin_vel_track_err_m_s']:.3f} "
                f"ang_track_err={stats['mean_ang_vel_track_err_rad_s']:.3f} "
                f"foot_slip={stats['mean_foot_slip_speed_m_s']:.3f}"
            )
            summary_lines.append(
                f"| {phase} | {stats['episodes']} | {stats['fall_rate']:.2%} "
                f"| {stats['mean_lin_vel_track_err_m_s']:.3f} | {stats['mean_ang_vel_track_err_rad_s']:.3f} "
                f"| {stats['mean_foot_slip_speed_m_s']:.3f} |"
            )

    summary_lines += [
        "",
        "## 3. Raw world-frame displacement check",
        "",
        "Reads `root_pos_w` directly — independent of section 1's reward-derived tracking "
        "error, guarding against misreading a tracking metric as achieved velocity "
        "(a real past bug in this project, 2026-07-23/24).",
        "",
    ]
    if args_cli.skip_displacement:
        summary_lines.append("Skipped (`--skip_displacement` passed).")
    else:
        d = _run_displacement_check(base_env)
        print(
            f"[Eval] Displacement: expected={d['expected_disp_m']:.2f}m "
            f"mean={d['mean_disp_m']:.3f}m ({d['frac_of_expected']:.1%} of expected) "
            f"min={d['min_disp_m']:.3f}m max={d['max_disp_m']:.3f}m"
        )
        summary_lines += [
            f"- Commanded {args_cli.displacement_speed} m/s for {args_cli.displacement_steps} steps "
            f"({args_cli.displacement_steps / 50:.1f}s @ 50Hz)",
            f"- Expected displacement if tracking perfectly: {d['expected_disp_m']:.2f}m",
            f"- Actual mean displacement across {args_cli.num_envs} envs: {d['mean_disp_m']:.3f}m "
            f"({d['frac_of_expected']:.1%} of expected)",
            f"- Range: {d['min_disp_m']:.3f}m - {d['max_disp_m']:.3f}m",
        ]

    summary_lines += [
        "",
        "## 4. Knee-angle tracking",
        "",
        "Checks a visually-observed hypothesis (round-2 drift checkpoint, 2026-07-27) "
        "that near-extended knees during straight-line walking correlate with the "
        "asymmetric-step corrections that precede bad heading drift — see "
        "`deferred_items_2026-07-21.md` item 8. Knee angle: 0 rad = fully extended, "
        "default pose is 0.3 rad (~17 deg), larger = more bent. `min_knee_angle_deg` is "
        "the most-extended moment reached per episode. Correlation is Pearson's r between "
        "each episode's `min_knee_angle_deg` and its `|heading_drift_deg|` — negative "
        "means episodes that stay more extended tend to drift more (supports the "
        "hypothesis); near zero means no relationship found in this data.",
        "",
        "| Bucket | Mean knee angle (deg) | Mean most-extended angle (deg) | Corr. vs \\|heading drift\\| |",
        "|---|---|---|---|",
    ]
    for name in args_cli.buckets:
        stats = command_bucket_stats.get(name, {})
        if stats.get("episodes", 0) == 0:
            summary_lines.append(f"| {name} | — | — | — |")
            continue
        corr = stats.get("knee_vs_heading_drift_corr")
        corr_col = f"{corr:.2f}" if corr is not None and corr == corr else "—"  # nan check
        summary_lines.append(
            f"| {name} | {stats['mean_knee_angle_deg']:.2f} | {stats['mean_min_knee_angle_deg']:.2f} "
            f"| {corr_col} |"
        )

    bin_labels = _fall_timing_bin_labels()
    summary_lines += [
        "",
        "## 5. Fall timing breakdown (buckets with any falls)",
        "",
        "A raw fall-rate percentage treats a fall at 3s and a fall at 18s as equally bad, "
        "but a command held for the full episode is an eval artifact, not something a "
        "real user is likely to sustain (e.g. turn_left/turn_right command wz continuously "
        "for the whole 20s — see run notes on the dead ang_vel curriculum). This bins WHEN "
        "falls happen so 'fails almost immediately' (a real problem) is visible separately "
        "from 'only fails after an unrealistically long sustained command' (lower priority).",
        "",
        "| Bucket | Total falls | " + " | ".join(bin_labels) + " |",
        "|---|---|" + "---|" * len(bin_labels),
    ]
    any_falls = False
    for name in args_cli.buckets:
        stats = command_bucket_stats.get(name, {})
        fall_count = stats.get("fall_count", 0)
        if not fall_count:
            continue
        any_falls = True
        counts = stats["fall_timing_bin_counts"]
        pct_cells = " | ".join(f"{c} ({c / fall_count:.0%})" for c in counts)
        summary_lines.append(f"| {name} | {fall_count} | {pct_cells} |")
    if not any_falls:
        summary_lines.append("| (no falls in any evaluated bucket) | — |" + " —|" * len(bin_labels))

    base_env.close()

    summary_path = os.path.join(eval_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n[Eval] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
