# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fall-rate eval for a standing checkpoint against the *actual* disturbance it trained
on (mdp.StandingArmIKReachDisturbance — real per-arm differential IK reaching, see that
class's docstring), as opposed to validation/eval_standing.py (sweeps the *original*
scripted disturbance — a different motion pattern a G1-Locomotion-Standing-Flat-IKReach-v0
checkpoint was never trained against) or eval_full_demo.py's standing_arm_*_reach buckets
(drive the arm via the RL arm-IK policy in the *walking* env — StandingArmIKReachDisturbance
is never invoked there either). Neither of those exercises the thing a checkpoint from this
task actually learned; this script is the missing piece (2026-07-11 — see known_issues.md).

Uses G1-Locomotion-Standing-Flat-IKReach-Play-v0 (enable_step=ramp_full_step=0 override —
disturbance fully on from step 0, no training curriculum delay; there's no policy left to
protect from an abrupt disturbance when evaluating an already-finished checkpoint).

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_standing_ikreach.py \\
        --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_XXXX.pt \\
        --headless

Output: <checkpoint_dir>/ikreach_eval/<YYYY-MM-DD_HH-MM-SS>/
    (next to the checkpoint under logs/, matching eval_standing.py's disturbance_eval/,
    eval_walking.py's command_eval/ and eval_arm.py's arm_eval/ convention — this script
    used to write to validation/eval_standing_ikreach/<timestamp>/ instead, the odd one
    out, fixed 2026-07-15. Runs before that date live in the old location.)
    standing_detailed.csv — every episode, full StandingMetricsCsvWrapper column set.
    summary.csv           — one-row aggregate (fall_rate, mean_max_tilt_deg, etc.),
                             same mean/rate-based shape eval_full_demo.py's per-bucket
                             summaries use.
    run_meta.yaml         — which checkpoint/args/commit produced this run.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Fall-rate eval for a G1-Locomotion-Standing-Flat-IKReach-v0 checkpoint against its own disturbance."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained standing checkpoint (.pt).")
parser.add_argument(
    "--env_cfg", type=str, default="ikreach",
    choices=[
        "ikreach", "torso", "policyreach", "height", "symmetry", "consolidated",
        "consolidatedtorso", "intent", "noreach",
    ],
    help="Which *_PLAY env cfg to evaluate against — must match what the checkpoint was actually "
    "trained under (2026-07-13, added once a second and third training variant existed): "
    "'ikreach' = G1LocomotionStandingFlatIKReachEnvCfg_PLAY (analytic IK disturbance), "
    "'torso' = G1LocomotionStandingFlatIKReachTorsoEnvCfg_PLAY (same disturbance, re-tightened "
    "torso reward — reward doesn't affect a frozen policy's behavior, so this differs from "
    "'ikreach' in name only, but keeping them distinct avoids ambiguity about which checkpoint "
    "was evaluated against which cfg), 'policyreach' = G1LocomotionStandingFlatPolicyReachEnvCfg_PLAY "
    "(real arm-IK-policy-driven disturbance), 'height' = G1LocomotionStandingFlatIKReachHeightEnvCfg_PLAY "
    "(same analytic-IK disturbance as 'ikreach' plus a base_height_l2 reward), 'symmetry' = "
    "G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg_PLAY (height reward plus leg_symmetry_l2), "
    "'consolidated' = G1LocomotionStandingFlatConsolidatedEnvCfg_PLAY (policy-driven disturbance + "
    "height reward + deployment-matched arm gains — NOT reward-only: the gain match changes the "
    "physics the checkpoint trained under, so using the wrong cfg here matters even more than for "
    "the reward-only variants), 'consolidatedtorso' = G1LocomotionStandingFlatConsolidatedTorsoEnvCfg_PLAY "
    "(consolidated plus joint_deviation_torso at -0.05) — rewards don't affect inference, kept distinct for the same "
    "bookkeeping reason as 'torso'. Evaluating a checkpoint against the wrong one of these would "
    "silently test it against a different disturbance than it trained on, exactly the eval/train "
    "mismatch this whole script exists to avoid.",
)
parser.add_argument(
    "--freeze_arms", action="store_true",
    help="Hold the arms at their default pose for the whole eval (disturbance never enables) "
    "instead of the usual disturbance-on-from-step-0. Bisect tool for the native-vs-integration "
    "transfer collapse (2026-07-15, plan.md §3 insight 3): integration standing_still holds arms "
    "statically while every native eval has them actively reaching — this reproduces the static-arm "
    "condition inside the native env, isolating that one variable. Uses the same "
    "StandingArmBlendJointPositionAction hold mechanism either way (enable_step pushed past any "
    "reachable step), NOT a removal of the event term — removing it would leave the arm joints "
    "driven by the policy's raw (unshaped) arm output, a different condition entirely.",
)
parser.add_argument(
    "--no_push", action="store_true",
    help="Disable the push_robot interval event. Second bisect toggle: the native standing PLAY "
    "cfgs keep push_robot active (only base_external_force_torque is disabled) while the "
    "integration env disables both — this equalizes that difference toward the integration side.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs.")
parser.add_argument("--steps", type=int, default=3000, help="Control steps (50 Hz) — 3000 = 60s.")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible rollouts.")
parser.add_argument(
    "--output_root", type=str, default=None,
    help="Override the output root directory. Default: <checkpoint_dir>/ikreach_eval/ next to the "
    "checkpoint, matching the other validation/eval_*.py scripts' convention.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import csv
import datetime
import os

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import G1LocomotionStandingFlatPPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionStandingFlatConsolidatedEnvCfg_PLAY,
    G1LocomotionStandingFlatConsolidatedIntentEnvCfg_PLAY,
    G1LocomotionStandingFlatConsolidatedNoReachEnvCfg_PLAY,
    G1LocomotionStandingFlatConsolidatedTorsoEnvCfg_PLAY,
    G1LocomotionStandingFlatIKReachEnvCfg_PLAY,
    G1LocomotionStandingFlatIKReachHeightEnvCfg_PLAY,
    G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg_PLAY,
    G1LocomotionStandingFlatIKReachTorsoEnvCfg_PLAY,
    G1LocomotionStandingFlatPolicyReachEnvCfg_PLAY,
)
from g1_locomotion.utils.eval_meta import write_eval_meta
from g1_locomotion.utils.metrics_wrappers import StandingMetricsCsvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


_ENV_CFG_CLASSES = {
    "ikreach": G1LocomotionStandingFlatIKReachEnvCfg_PLAY,
    "torso": G1LocomotionStandingFlatIKReachTorsoEnvCfg_PLAY,
    "policyreach": G1LocomotionStandingFlatPolicyReachEnvCfg_PLAY,
    "height": G1LocomotionStandingFlatIKReachHeightEnvCfg_PLAY,
    "symmetry": G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg_PLAY,
    "consolidated": G1LocomotionStandingFlatConsolidatedEnvCfg_PLAY,
    "consolidatedtorso": G1LocomotionStandingFlatConsolidatedTorsoEnvCfg_PLAY,
    # "intent" adds the 10-D arm-intent observation term, so the env's obs space is 85-D —
    # matching only checkpoints trained on G1-Locomotion-Standing-Flat-Consolidated-Intent-v0.
    "intent": G1LocomotionStandingFlatConsolidatedIntentEnvCfg_PLAY,
    "noreach": G1LocomotionStandingFlatConsolidatedNoReachEnvCfg_PLAY,
}


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Next to the checkpoint under logs/, matching every other validation/eval_*.py
    # script's convention (see module docstring — this script used to be the odd one out,
    # writing to validation/eval_standing_ikreach/ instead; fixed 2026-07-15).
    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    output_root = args_cli.output_root or os.path.join(checkpoint_dir, "ikreach_eval")
    run_dir = os.path.join(output_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    write_eval_meta(run_dir, args_cli, __file__)

    print(f"[Eval] Checkpoint : {args_cli.checkpoint}")
    print(f"[Eval] num_envs={args_cli.num_envs}  steps={args_cli.steps}  output={run_dir}")

    env_cfg = _ENV_CFG_CLASSES[args_cli.env_cfg]()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Bisect toggles for the native-vs-integration transfer collapse — see the two flags'
    # help text. Applied after the PLAY cfg's own enable_step=0 override, so these win.
    if args_cli.freeze_arms:
        env_cfg.events.standing_arm_motion_disturbance.params["enable_step"] = 10**9
        env_cfg.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 10**9
        print("[Eval] --freeze_arms: arms held at default the whole eval (disturbance never enables).")
    if args_cli.no_push:
        env_cfg.events.push_robot = None
        print("[Eval] --no_push: push_robot event disabled.")

    print("[Eval] Building simulation...")
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    metrics_env = StandingMetricsCsvWrapper(base_env, run_dir, write_summary=False)
    wrapped_env = RslRlVecEnvWrapper(metrics_env)

    print("[Eval] Loading policy...")
    agent_cfg = G1LocomotionStandingFlatPPORunnerCfg()
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        obs, _ = wrapped_env.reset()
        for _ in range(args_cli.steps):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

    metrics_env._csv.close()
    detailed_path = metrics_env.csv_path
    base_env.close()

    with open(detailed_path) as f:
        rows = list(csv.DictReader(f))

    summary_path = os.path.join(run_dir, "summary.csv")
    fields = [
        "episodes", "fall_rate", "mean_episode_steps", "mean_max_tilt_deg",
        "mean_min_base_height_m", "mean_max_lin_speed_m_s", "mean_max_ang_speed_rad_s",
        "mean_step_count", "mean_max_foot_air_time_s", "mean_max_abs_torso_deg",
    ]
    if rows:
        fall_rate = sum(int(r["fell"]) for r in rows) / len(rows)
        # Torso column only exists in detailed CSVs written after 2026-07-15 (see
        # StandingMetricsCsvWrapper) — guard so re-summarizing an older CSV still works.
        mean_max_torso = _mean(rows, "max_abs_torso_deg")
        row = {
            "episodes": len(rows),
            "fall_rate": f"{fall_rate:.4f}",
            "mean_episode_steps": f"{_mean(rows, 'episode_steps'):.1f}",
            "mean_max_tilt_deg": f"{_mean(rows, 'max_tilt_deg'):.2f}",
            "mean_min_base_height_m": f"{_mean(rows, 'min_base_height_m'):.3f}",
            "mean_max_lin_speed_m_s": f"{_mean(rows, 'max_lin_speed_m_s'):.3f}",
            "mean_max_ang_speed_rad_s": f"{_mean(rows, 'max_ang_speed_rad_s'):.3f}",
            "mean_step_count": f"{_mean(rows, 'step_count'):.2f}",
            "mean_max_foot_air_time_s": f"{_mean(rows, 'max_foot_air_time_s'):.3f}",
            "mean_max_abs_torso_deg": f"{mean_max_torso:.2f}" if mean_max_torso is not None else "",
        }
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)
        print(
            f"\n[Eval] episodes={len(rows)} fall_rate={fall_rate:.2%} "
            f"mean_max_tilt_deg={row['mean_max_tilt_deg']} "
            f"mean_min_base_height_m={row['mean_min_base_height_m']}"
        )
    else:
        print("\n[Eval] WARNING: no episodes completed — nothing to summarize.")

    print(f"[Eval] Detailed: {detailed_path}")
    print(f"[Eval] Summary : {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
