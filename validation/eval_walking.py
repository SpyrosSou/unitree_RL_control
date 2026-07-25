# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic, command-conditioned evaluation for a G1 29dof unified stand+walk
checkpoint — the "training-only" metrics eval (as opposed to
``validation/integration_validation/eval_full_demo.py``-style testing with a separate
arm policy layered on top, which doesn't exist for the 29dof arm yet anyway).

Rewritten 2026-07-21 for the 29dof pivot — replaces the 23dof-era version that used to
live at this exact path (preserved on the ``23_dof`` git branch, not carried forward
here) — see ``29dof_implementation_plan.md``. Structurally close to a straight port:
same command-bucket sweep, same drift-check methodology, same
``WalkingMetricsCsvWrapper`` reuse (so these numbers are directly comparable to a run's
own ``walking_detailed.csv`` — see that wrapper's docstring). Two real differences from
the 23dof version:

1. This project's 23dof phase had separate standing and walking checkpoints/tasks;
   the 29dof recipe is one unified policy (``rel_standing_envs`` covers near-zero-
   velocity "standing" as an edge case of the same command distribution) — so this one
   script covers what used to need two (``eval_standing.py`` + ``eval_walking.py``). The
   ``stand_still`` bucket already in ``_BUCKETS`` below exercises exactly that regime.
2. ``--arm_disturbance`` — if the checkpoint was trained on
   ``G1-Locomotion-Velocity-ArmDisturbance-v0`` (the arm-motion-disturbance
   curriculum, see Phase 2 of the plan), pass this to build the eval env from
   ``G1LocomotionArmDisturbanceEnvCfg_PLAY`` instead of the plain ``G1LocomotionEnvCfg_PLAY`` — forces
   the disturbance curriculum fully on (matching that _PLAY cfg's own phase-boundary
   override) instead of leaving arms scripted-static, so the command-tracking numbers
   reflect what the checkpoint actually has to cope with. Omit for a base-recipe
   checkpoint (``G1-Locomotion-Velocity-v0``) — arms just sit at default the whole
   time either way for that one.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_walking.py \\
        --checkpoint logs/rsl_rl/walking/base/<run>/model_2998.pt \\
        --headless

    python validation/eval_walking.py \\
        --checkpoint logs/rsl_rl/walking/arm_disturbance/<run>/model_2998.pt \\
        --arm_disturbance \\
        --headless

Output:
    Prints a per-bucket summary table (tracking error, foot slip, fall rate, and — for
    the straight-line buckets — heading/lateral drift) and writes the same summary, plus
    the raw per-episode rows (reusing ``WalkingMetricsCsvWrapper``), under
    <checkpoint_dir>/command_eval/.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

# Command envelope: name -> (lin_vel_x, lin_vel_y, ang_vel_z). "straight" buckets (zero
# commanded lateral velocity and yaw rate) are the ones drift is meaningfully checked on.
_BUCKETS = {
    "stand_still": (0.0, 0.0, 0.0),
    "forward_slow": (0.3, 0.0, 0.0),
    "forward_medium": (0.6, 0.0, 0.0),
    "forward_fast": (0.9, 0.0, 0.0),
    "backward": (-0.3, 0.0, 0.0),
    "strafe_left": (0.0, 0.4, 0.0),
    "strafe_right": (0.0, -0.4, 0.0),
    "turn_left": (0.0, 0.0, 0.6),
    "turn_right": (0.0, 0.0, -0.6),
    "forward_turn_combo": (0.5, 0.0, 0.4),
}
_STRAIGHT_BUCKETS = {"forward_slow", "forward_medium", "forward_fast", "backward"}

parser = argparse.ArgumentParser(description="Command-conditioned eval for a G1 29dof stand+walk checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained checkpoint (.pt).")
parser.add_argument(
    "--arm_disturbance", action="store_true",
    help="Build the eval env from G1LocomotionArmDisturbanceEnvCfg_PLAY (disturbance forced on) "
    "instead of the plain G1LocomotionEnvCfg_PLAY — pass this for a checkpoint trained on "
    "G1-Locomotion-Velocity-ArmDisturbance-v0.",
)
parser.add_argument(
    "--buckets", type=str, nargs="+", default=list(_BUCKETS.keys()), choices=list(_BUCKETS.keys()),
    help="Which command buckets to evaluate.",
)
parser.add_argument(
    "--pin_disturbance_phase", type=int, default=None, choices=[0, 1, 2, 3],
    help="Pin mdp.ArmMotionDisturbance at exactly this phase for the whole eval, instead of letting "
    "it cycle through phases over elapsed eval time (the default _PLAY phase_step_boundaries are "
    "tuned for a quick play.py look, not a clean per-phase measurement — a stand_still bucket run "
    "with this unset blends whatever mix of phases happened to occur during the eval window into "
    "one fall-rate number, dominated by whichever phase produced the most short/failing episodes). "
    "Works by setting phase_step_boundaries to N copies of 0, which makes the phase-index lookup "
    "fall through immediately to phase N regardless of step count — no new logic, just forces the "
    "existing one. Only meaningful with --arm_disturbance. Ignored otherwise.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs per bucket.")
parser.add_argument(
    "--steps_per_bucket", type=int, default=1500,
    help="Control steps to run per bucket (at 50 Hz; 1500 steps ~= 30 s, several episodes per env).",
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
# numbers directly comparable to what you'd see in a run's walking_detailed.csv. No
# 29dof-specific wrapper needed: WalkingMetricsCsvWrapper has no DOF-count assumptions
# (unlike StandingMetricsCsvWrapper, not used here — see 29dof_implementation_plan.md).
from g1_locomotion.utils.eval_meta import write_eval_meta
from g1_locomotion.utils.metrics_wrappers import WalkingMetricsCsvWrapper


def _summarize(csv_path: str, is_straight: bool) -> dict:
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
    if is_straight:
        heading_drifts = [float(r["heading_drift_deg"]) for r in rows]
        lateral_drifts = [float(r["lateral_drift_m"]) for r in rows]
        stats["mean_abs_heading_drift_deg"] = sum(abs(x) for x in heading_drifts) / n
        stats["max_abs_heading_drift_deg"] = max(abs(x) for x in heading_drifts)
        stats["mean_abs_lateral_drift_m"] = sum(abs(x) for x in lateral_drifts) / n
        stats["max_abs_lateral_drift_m"] = max(abs(x) for x in lateral_drifts)
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

    if args_cli.pin_disturbance_phase is not None and hasattr(env_cfg.events, "arm_motion_disturbance"):
        n = args_cli.pin_disturbance_phase
        env_cfg.events.arm_motion_disturbance.params["phase_step_boundaries"] = tuple([0] * n)
        env_cfg.events.arm_motion_disturbance.params["phase_step_offset"] = 0

    return ManagerBasedRLEnv(cfg=env_cfg)


def _run_bucket(base_env: ManagerBasedRLEnv, name: str, eval_root: str) -> dict:
    """Run one command bucket's rollout on the shared, already-constructed base_env.

    Deliberately reuses one ManagerBasedRLEnv across all buckets instead of building a
    fresh one per bucket — repeatedly constructing/tearing down Isaac Sim's simulation
    context within a single process is unreliable. The velocity command term reads its
    cfg.ranges fresh on every resample, so mutating the live term's ranges and calling
    reset() is enough to switch buckets — no rebuild needed.
    """
    vx, vy, wz = _BUCKETS[name]
    print(f"\n[Eval] --- Bucket '{name}' (vx={vx}, vy={vy}, wz={wz}) ---")

    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (vx, vx)
    command_term.cfg.ranges.lin_vel_y = (vy, vy)
    command_term.cfg.ranges.ang_vel_z = (wz, wz)

    bucket_dir = os.path.join(eval_root, name)
    os.makedirs(bucket_dir, exist_ok=True)

    # Everything below touches base_env's live state tensors. From the second bucket
    # onward those tensors get written to while already tagged as PyTorch "inference
    # tensors" (the first bucket's rollout ran inside inference_mode) — in-place writes
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
        # Small-scale here (num_envs is large, so only a handful of episodes per bucket
        # are affected, not the near-total corruption a 1-env demo saw) but still a
        # real, unnecessary contamination of buckets that are supposed to be a fixed,
        # pinned command for every env.
        command_term.is_standing_env[:] = False
        for _ in range(args_cli.steps_per_bucket):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

    # Close only this bucket's CSV file handles — not wrapped_env.close(), which would
    # cascade down and tear down base_env, breaking the next bucket.
    env._csv.close()
    return _summarize(env.csv_path, is_straight=name in _STRAIGHT_BUCKETS)


def main():
    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    eval_dir_name = "command_eval"
    if args_cli.pin_disturbance_phase is not None:
        # Namespaced separately per phase — otherwise repeated per-phase runs against the
        # same checkpoint would overwrite each other's summary.md/CSVs in "command_eval/".
        eval_dir_name = f"command_eval_phase{args_cli.pin_disturbance_phase}"
    eval_root = os.path.join(checkpoint_dir, eval_dir_name)
    os.makedirs(eval_root, exist_ok=True)
    write_eval_meta(eval_root, args_cli, __file__)

    print(f"[Eval] Checkpoint     : {args_cli.checkpoint}")
    print(f"[Eval] Arm disturbance: {args_cli.arm_disturbance}")
    print(f"[Eval] Buckets        : {args_cli.buckets}")
    print(f"[Eval] num_envs       : {args_cli.num_envs}   steps_per_bucket: {args_cli.steps_per_bucket}")

    print("[Eval] Building simulation (once, reused across all buckets)...")
    base_env = _build_base_env()

    summary_lines = [
        "# G1 29dof Locomotion Command-Conditioned Eval",
        "",
        f"Checkpoint: `{args_cli.checkpoint}`",
        f"Arm disturbance forced on: {args_cli.arm_disturbance}",
        "",
        "| Bucket | vx, vy, wz | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) "
        "| Foot slip (m/s) | Mean\\|heading drift\\| (deg) | Mean\\|lateral drift\\| (m) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for name in args_cli.buckets:
        stats = _run_bucket(base_env, name, eval_root)
        vx, vy, wz = _BUCKETS[name]
        if stats["episodes"] == 0:
            print(f"[Eval] Bucket '{name}': no completed episodes — increase --steps_per_bucket.")
            summary_lines.append(f"| {name} | {vx}, {vy}, {wz} | 0 | — | — | — | — | — | — |")
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
            f"foot_slip={stats['mean_foot_slip_speed_m_s']:.3f}" + drift_str
        )

        heading_col = f"{stats['mean_abs_heading_drift_deg']:.2f}" if "mean_abs_heading_drift_deg" in stats else "—"
        lateral_col = f"{stats['mean_abs_lateral_drift_m']:.3f}" if "mean_abs_lateral_drift_m" in stats else "—"
        summary_lines.append(
            f"| {name} | {vx}, {vy}, {wz} | {stats['episodes']} | {stats['fall_rate']:.2%} "
            f"| {stats['mean_lin_vel_track_err_m_s']:.3f} | {stats['mean_ang_vel_track_err_rad_s']:.3f} "
            f"| {stats['mean_foot_slip_speed_m_s']:.3f} | {heading_col} | {lateral_col} |"
        )

    base_env.close()

    summary_path = os.path.join(eval_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n[Eval] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
