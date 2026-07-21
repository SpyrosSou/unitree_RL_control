# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic, disturbance-conditioned evaluation for a standing checkpoint.

Runs fixed-seed rollouts of a trained standing policy at each arm-motion-disturbance
curriculum phase (0 = no motion .. however many phases the curriculum currently defines,
e.g. 5 = the 33 rad/s hardware-limit phase added 2026-07-07 — see
StandingArmTrajectoryDisturbance in mdp/events.py), independent of and decoupled from the
noisy training reward curve. This is the tool for answering "is this checkpoint actually
robust, and does it step when it needs to" rather than "did the reward go up." See
validation/README.md for what the numbers in the output mean and what "good" looks like.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_standing.py \\
        --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt \\
        --headless

    # Narrow to specific phases, more envs / longer rollout for tighter statistics
    python validation/eval_standing.py \\
        --checkpoint chosen_checkpoints/standing_latest.pt \\
        --phases 3 4 \\
        --num_envs 512 \\
        --steps_per_phase 3000 \\
        --headless

Output:
    Prints a per-phase summary table (fall rate, stepped-at-least-once rate, mean step
    count, mean/max tilt, mean peak foot air time, and fall rate split by whether a step
    was taken) and writes the same summary, plus the raw per-episode rows (reusing
    ``StandingMetricsCsvWrapper``), under <checkpoint_dir>/disturbance_eval/.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Disturbance-conditioned eval for a standing checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained standing checkpoint (.pt).")
parser.add_argument(
    "--phases", type=int, nargs="+", default=None,
    help="Which arm-motion-disturbance curriculum phases to evaluate. Defaults to all phases "
    "the curriculum currently defines.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs per phase.")
parser.add_argument(
    "--steps_per_phase", type=int, default=1500,
    help="Control steps to run per phase (at 50 Hz; 1500 steps ~= 30 s, several episodes per env).",
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
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import (
    G1LocomotionStandingFlatPPORunnerCfg,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionStandingFlatEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp.events import StandingArmTrajectoryDisturbance
# Reuse the exact same per-episode metrics wrapper training runs use — keeps the
# eval's numbers directly comparable to what you'd see in a run's standing_detailed.csv.
from g1_locomotion.utils.eval_meta import write_eval_meta
from g1_locomotion.utils.metrics_wrappers import StandingMetricsCsvWrapper

# Read straight from the curriculum class rather than keeping a separate hardcoded copy —
# this and the class definition silently drifting out of sync is exactly what almost
# happened when phase 5 was added (2026-07-07).
_PHASE_STEP_BOUNDARIES = StandingArmTrajectoryDisturbance._PHASE_STEP_BOUNDARIES


def _offset_for_phase(phase: int, common_step_counter: int) -> int:
    """A phase_step_offset that keeps common_step_counter inside `phase` for the whole
    eval rollout, given the env's *current* step count.

    The env is built once and reused across phases (see module docstring / main()), so
    common_step_counter keeps climbing across phases rather than starting at 0 each time
    — the offset has to account for whatever it already is, not assume a fresh env.
    """
    target = 0 if phase <= 0 else _PHASE_STEP_BOUNDARIES[phase - 1] + 1
    return target - common_step_counter


def _summarize(csv_path: str) -> dict:
    if not os.path.isfile(csv_path):
        return {"episodes": 0}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"episodes": 0}
    n = len(rows)
    fell = sum(int(r["fell"]) for r in rows)
    stepped_rows = [r for r in rows if int(r["step_count"]) > 0]
    no_step_rows = [r for r in rows if int(r["step_count"]) == 0]
    stepped = len(stepped_rows)
    stats = {
        "episodes": n,
        "fall_rate": fell / n,
        "stepped_at_least_once_rate": stepped / n,
        "mean_step_count": sum(int(r["step_count"]) for r in rows) / n,
        "mean_max_tilt_deg": sum(float(r["max_tilt_deg"]) for r in rows) / n,
        "max_max_tilt_deg": max(float(r["max_tilt_deg"]) for r in rows),
        "mean_max_foot_air_time_s": sum(float(r["max_foot_air_time_s"]) for r in rows) / n,
    }
    # Split fall rate by whether a corrective step was taken this episode. If stepping is
    # a genuinely learned recovery skill, stepped episodes should fall *less* often than
    # no-step episodes (stepping is the harder-disturbance response, deployed when leaning
    # alone wouldn't have been enough) — a stepped-episode fall rate *higher* than the
    # no-step rate is the red flag that stepping is a last-ditch, poorly-executed panic
    # response rather than a reliable skill. See README for the exact threshold to watch.
    stats["fall_rate_when_stepped"] = (sum(int(r["fell"]) for r in stepped_rows) / stepped) if stepped else None
    stats["fall_rate_when_no_step"] = (
        sum(int(r["fell"]) for r in no_step_rows) / len(no_step_rows) if no_step_rows else None
    )
    return stats


def _build_base_env() -> ManagerBasedRLEnv:
    env_cfg = G1LocomotionStandingFlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # The _PLAY config disables observation noise (IMU/encoder Unoise) for clean
    # visualization — but that makes this eval meaningfully easier than real training
    # and deployment, and defeats the point of a sim2real robustness check. Restore the
    # exact training-time noise config (enable_corruption is just a group-level on/off
    # switch over the same, otherwise-unchanged Unoise terms).
    env_cfg.observations.policy.enable_corruption = True

    return ManagerBasedRLEnv(cfg=env_cfg)


def _run_phase(base_env: ManagerBasedRLEnv, phase: int, eval_root: str) -> dict:
    """Run one phase's rollout on the shared, already-constructed base_env.

    Deliberately reuses one ManagerBasedRLEnv across all phases instead of building a
    fresh one per phase (the original design) — repeatedly constructing/tearing down
    Isaac Sim's simulation context within a single process is unreliable and was
    observed to hang indefinitely partway through a real eval run. Reconfiguring the
    live disturbance event's params and calling reset() is enough: the event term reads
    phase_step_boundaries/phase_step_offset fresh from its params on every call (see
    StandingArmTrajectoryDisturbance.__call__), so no rebuild is needed to change phase.
    """
    print(f"\n[Eval] --- Phase {phase} ---")

    disturbance = base_env.event_manager.get_term_cfg("standing_arm_motion_disturbance")
    disturbance.params["phase_step_boundaries"] = _PHASE_STEP_BOUNDARIES
    disturbance.params["phase_step_offset"] = _offset_for_phase(phase, base_env.common_step_counter)

    phase_dir = os.path.join(eval_root, f"phase_{phase}")
    os.makedirs(phase_dir, exist_ok=True)

    # Everything below touches base_env's live state tensors. From phase 1 onward those
    # tensors get written to while already tagged as PyTorch "inference tensors" (phase 0's
    # rollout ran inside inference_mode) — in-place writes to an inference tensor are only
    # legal from *inside* inference_mode, so construction/reset must be in this block too,
    # not just the stepping loop (outside it raises "Inplace update to inference tensor
    # outside InferenceMode is not allowed").
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        env = StandingMetricsCsvWrapper(base_env, phase_dir)
        device = base_env.device
        wrapped_env = RslRlVecEnvWrapper(env)

        agent_cfg = G1LocomotionStandingFlatPPORunnerCfg()
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        obs, _ = wrapped_env.reset()
        for _ in range(args_cli.steps_per_phase):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

    # Close only this phase's CSV file handles — not wrapped_env.close(), which would
    # cascade down and tear down base_env, breaking the next phase.
    env._csv.close()
    env._write_joint_diagnostics()
    return _summarize(env.csv_path)


def _fmt_pct(x) -> str:
    return f"{x:.2%}" if x is not None else "—"


def main():
    if args_cli.phases is None:
        args_cli.phases = list(range(len(_PHASE_STEP_BOUNDARIES) + 1))

    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    eval_root = os.path.join(checkpoint_dir, "disturbance_eval")
    os.makedirs(eval_root, exist_ok=True)
    write_eval_meta(eval_root, args_cli, __file__)

    print(f"[Eval] Checkpoint : {args_cli.checkpoint}")
    print(f"[Eval] Phases     : {args_cli.phases}")
    print(f"[Eval] num_envs   : {args_cli.num_envs}   steps_per_phase: {args_cli.steps_per_phase}")

    print("[Eval] Building simulation (once, reused across all phases)...")
    base_env = _build_base_env()

    summary_lines = [
        "# Standing Disturbance-Conditioned Eval",
        "",
        f"Checkpoint: `{args_cli.checkpoint}`",
        "",
        "| Phase | Episodes | Fall rate | Stepped-at-least-once rate | Mean step count "
        "| Fall rate (stepped) | Fall rate (no step) | Mean max tilt (deg) | Max tilt (deg) "
        "| Mean peak foot air time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for phase in args_cli.phases:
        stats = _run_phase(base_env, phase, eval_root)
        if stats["episodes"] == 0:
            print(f"[Eval] Phase {phase}: no completed episodes in the rollout window — increase --steps_per_phase.")
            summary_lines.append(f"| {phase} | 0 | — | — | — | — | — | — | — | — |")
            continue
        print(
            f"[Eval] Phase {phase}: episodes={stats['episodes']} "
            f"fall_rate={stats['fall_rate']:.2%} "
            f"stepped_at_least_once_rate={stats['stepped_at_least_once_rate']:.2%} "
            f"mean_step_count={stats['mean_step_count']:.2f} "
            f"fall_rate_when_stepped={_fmt_pct(stats['fall_rate_when_stepped'])} "
            f"fall_rate_when_no_step={_fmt_pct(stats['fall_rate_when_no_step'])} "
            f"mean_max_tilt_deg={stats['mean_max_tilt_deg']:.1f} "
            f"mean_peak_foot_air_time_s={stats['mean_max_foot_air_time_s']:.3f}"
        )
        summary_lines.append(
            f"| {phase} | {stats['episodes']} | {stats['fall_rate']:.2%} "
            f"| {stats['stepped_at_least_once_rate']:.2%} | {stats['mean_step_count']:.2f} "
            f"| {_fmt_pct(stats['fall_rate_when_stepped'])} | {_fmt_pct(stats['fall_rate_when_no_step'])} "
            f"| {stats['mean_max_tilt_deg']:.1f} | {stats['max_max_tilt_deg']:.1f} "
            f"| {stats['mean_max_foot_air_time_s']:.3f} |"
        )

    base_env.close()

    summary_path = os.path.join(eval_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n[Eval] Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
