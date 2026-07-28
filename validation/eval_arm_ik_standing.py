# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 3 (early pass) of `ik_arm_integration_plan.md`: headless, scripted in-sim
accuracy sweep for the IK-driven arm (standing, walking policy active at zero command),
combining the unified stand+walk policy with the tau_ff-compensated IK backend —
i.e. exactly `g1_full_demo.py --arm_backend ik`'s live control path, scripted instead of
interactive so it can sweep many goals unattended.

Built 2026-07-27 while the user was away, using their explicit one-time authorization to
run Isaac Sim autonomously for verification during that window (not a standing
permission — ask again next time). Directly follows up on the tau_ff fix (PD gravity
sag was stalling the rate-limiter itself — see ik_arm_integration_plan.md's Phase 2
status note) with the question that fix doesn't answer by itself: does it hold up
broadly across the goal box, not just the one point the user tested by hand, and does
real (PD + tau_ff + walking-policy-disturbance) tracking accuracy still land near
Phase 1's ~58%/2cm kinematic ceiling, or does the sim/dynamics layer cost more than that?

Per goal: solves+drives toward a fresh target (arm_ik.reset() first — landmine #9),
holds for `--hold_steps` (default matches what the demo showed converging by ~500
steps), then records: the REAL achieved wrist error, the PURE KINEMATIC error (solver's
own FK-recomputed answer for the same target, no dynamics involved — Phase 1's own
metric, for direct before/after-dynamics comparison), and whether the env's own
termination manager ever fired (fall) during the hold.

Usage:
    conda activate isaac_g1_ik
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_arm_ik_standing.py \\
        --loco_checkpoint chosen_checkpoints/walking_latest.pt \\
        --arm left --num_targets 30 --hold_steps 500 --headless

Output: printed coverage stats (2/5/10/15cm, matching Phase 1's ladder), a per-goal CSV,
and a summary written to validation/arm_ik_standing/<timestamp>/.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse
import importlib.util as _ilu
import os

ISAACLAB_PATH = os.path.expanduser("~/Elm/Code/IsaacLab")
_spec = _ilu.spec_from_file_location(
    "cli_args",
    os.path.join(ISAACLAB_PATH, "scripts/reinforcement_learning/rsl_rl/cli_args.py"),
)
cli_args_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cli_args_mod)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Headless in-sim IK accuracy sweep (standing, tau_ff-compensated).")
cli_args_mod.add_rsl_rl_args(parser)
parser.add_argument("--loco_checkpoint", type=str, default="chosen_checkpoints/walking_latest.pt",
                    help="Unified stand+walk checkpoint.")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right"], help="Which arm to sweep.")
parser.add_argument("--num_targets", type=int, default=30, help="Number of goal-box points to sample.")
parser.add_argument("--hold_steps", type=int, default=500, help="Control steps to hold each target before measuring.")
parser.add_argument("--settle_steps", type=int, default=100, help="Steps to let standing settle before the first target.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--apply_tauff", action="store_true", default=True,
                     help="Apply the solver's feedforward torque (the 2026-07-27 fix). Default on.")
parser.add_argument("--no_tauff", dest="apply_tauff", action="store_false",
                     help="Disable feedforward torque, to A/B against the fix (expect it to reproduce the stall).")
parser.add_argument(
    "--x_max", type=float, default=None,
    help="Override the goal box's x upper bound (default: None = full box, x in "
    "(0.20, 0.42)). Phase 1's dense sweep found the far half (x above the "
    "median, ~0.31) covers only 5-49%% within 2cm vs. 78-100%% for the near half "
    "-- pass --x_max 0.31 to sweep only the near (reliably-reachable) half.",
)
parser.add_argument(
    "--root_smoothing_alpha", type=float, default=0.05,
    help="EMA alpha for the root-orientation smoothing used in the pelvis-frame "
    "conversion (see _ROOT_SMOOTHING_ALPHA). Pass 1.0 to disable smoothing entirely "
    "(use the raw, unsmoothed orientation every step) -- for isolating whether the "
    "smoothing itself is helping or hurting, independent of the goal-box trim.",
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

import numpy as np
import torch

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.controllers.arm_ik import ARM_JOINT_NAMES, G1ArmIK
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_ARM_JOINTS,
    _RIGHT_EE_BODY,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import G1LocomotionEnvCfg_PLAY
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp import ArmDisturbanceBlendJointPositionAction
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

import math

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from g1_locomotion.utils.eval_meta import write_eval_meta

ARM_MAX_JOINT_DELTA_PER_STEP = 0.06  # matches g1_full_demo.py
_KINEMATIC_SANITY_M = 0.20  # 20cm; Phase 1's dense sweep never exceeded ~16cm
# Second, independent, always-on safety net (2026-07-27, matches g1_full_demo.py) — the
# once-per-target retry above only guards the FIRST solve after a target change; found
# a case where kinematic error was fine initially but drifted to ~221cm MID-hold
# (continuous live-warm-start re-solving wandering into a bad basin over many steps),
# correlating with a real fall. Checked every step (unlike the retry, doesn't reset
# solver state) — hold position and zero tau_ff rather than act on a solution this bad.
_KINEMATIC_HARD_REJECT_M = 0.30
# EMA smoothing alpha for the pelvis-frame conversion's root ORIENTATION (2026-07-27,
# matches g1_full_demo.py) -- default time constant ~0.4s at 50Hz. See
# g1_full_demo.py's ARM_IK_ROOT_SMOOTHING_ALPHA docstring for the derivation of why
# this matters, and --root_smoothing_alpha's own help text for a real, since-observed
# risk: if the torso doesn't just jitter but genuinely, slowly leans in reaction to the
# arm's own reach (a real, previously-documented phenomenon, not noise), smoothing
# lags behind that real trend rather than filtering fake noise -- actual value used is
# args_cli.root_smoothing_alpha (this module constant is not read directly, kept only
# as a documented reference/default).
_GAIN_ARM_ACTIVE = (40.0, 10.0)  # matches g1_full_demo.py / g1_arm_env.py training gain


def _load_loco_policy(ckpt: str, device: str):
    """Copied near-verbatim from g1_full_demo.py's _load_loco_policy (2026-07-22 fix:
    infer network shape/norm flags from the checkpoint itself, not agent_cfg) — reusing
    the exact proven loader rather than re-deriving it, to avoid a subtly-different new
    bug in code that already works. Uses the module-level cli_args_mod loaded at the
    top of this file (before argparse/AppLauncher), matching g1_full_demo.py's pattern."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args_mod.parse_rsl_rl_cfg("G1-Locomotion-Velocity-Play-v0", args_cli)
    state = torch.load(ckpt, map_location=device)["model_state_dict"]
    in_dim = state["actor.0.weight"].shape[1]
    critic_in_dim = state["critic.0.weight"].shape[1]
    num_actions = state[f"actor.{2 * len(agent_cfg.policy.actor_hidden_dims)}.weight"].shape[0]
    has_actor_norm = "actor_obs_normalizer._mean" in state
    has_critic_norm = "critic_obs_normalizer._mean" in state

    dummy_obs = TensorDict(
        {
            "policy": torch.zeros((1, in_dim), dtype=torch.float32, device=device),
            "critic": torch.zeros((1, critic_in_dim), dtype=torch.float32, device=device),
        },
        batch_size=[1], device=device,
    )
    actor_critic = ActorCritic(
        obs=dummy_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=num_actions,
        actor_obs_normalization=has_actor_norm,
        critic_obs_normalization=has_critic_norm,
        actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
        activation=agent_cfg.policy.activation,
        init_noise_std=agent_cfg.policy.init_noise_std,
    ).to(device)
    actor_critic.load_state_dict(state)
    actor_critic.eval()

    def policy(obs) -> torch.Tensor:
        obs_tensor = obs["policy"] if isinstance(obs, TensorDict) else obs
        td = TensorDict({"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]], device=device)
        with torch.inference_mode():
            return actor_critic.act_inference(td)

    return policy


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    assert list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS) == ARM_JOINT_NAMES, (
        "g1_arm_env.py joint lists and arm_ik.py's ARM_JOINT_NAMES have diverged."
    )

    env_cfg = G1LocomotionEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1_000_000
    env_cfg.curriculum = None
    env_cfg.observations.policy.enable_corruption = True  # matches g1_full_demo.py
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)  # we drive the command ourselves, every step
    if hasattr(env_cfg.actions, "JointPositionAction"):
        env_cfg.actions.JointPositionAction.class_type = ArmDisturbanceBlendJointPositionAction

    print("[EvalArmIK] Building env...")
    loco_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(loco_env)
    device = loco_env.device
    robot = loco_env.scene["robot"]
    ground_z_w = loco_env.scene.env_origins[0, 2].clone()

    all_arm_ids, _ = robot.find_joints(list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS))
    all_arm_joint_ids_robot = torch.tensor(all_arm_ids, dtype=torch.long, device=device)
    # Diagnostic-only (2026-07-27): arm_ik.py's reduced model LOCKS the waist at exactly
    # zero (see _LOCKED_JOINTS in arm_ik.py), but the loco policy's JointPositionAction
    # covers every joint ([".*"]) and only the arm portion is overridden by
    # ArmDisturbanceBlendJointPositionAction -- the waist is actively driven by the
    # walking/standing policy and is NOT guaranteed to sit at zero. A full-model FK check
    # (no sim needed) found ~0.6cm of wrist error per degree of waist deviation, on any
    # of the 3 waist axes, that the reduced-model solve has no way to see or correct for.
    # Logging real waist angles here to test whether this explains the achieved-vs-
    # kinematic gap (esp. targets that plateau at a persistent, non-decaying offset).
    waist_ids, _ = robot.find_joints(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"])
    waist_joint_ids_robot = torch.tensor(waist_ids, dtype=torch.long, device=device)

    l_ee_ids, _ = robot.find_bodies(_LEFT_EE_BODY)
    r_ee_ids, _ = robot.find_bodies(_RIGHT_EE_BODY)
    left_ee_body_id, right_ee_body_id = l_ee_ids[0], r_ee_ids[0]

    side = args_cli.arm
    other_side = "right" if side == "left" else "left"
    n_per_arm = len(_LEFT_ARM_JOINTS)
    side_slice = slice(0, n_per_arm) if side == "left" else slice(n_per_arm, 2 * n_per_arm)
    active_ee_body_id = left_ee_body_id if side == "left" else right_ee_body_id
    other_ee_body_id = right_ee_body_id if side == "left" else left_ee_body_id

    loco_policy = _load_loco_policy(args_cli.loco_checkpoint, device)
    arm_ik = G1ArmIK()

    zero_cmd = torch.zeros(1, 3, device=device)
    joint_cmd_this_step = [None]  # diagnostic: rate-limited joint-space command, active arm (7,)
    joint_actual_this_step = [None]  # diagnostic: actual joint pos at start of step, active arm (7,)
    solve_ok = [True]  # mutable box, updated by step_sim, read by the sweep loop
    # Set True once per new target (by the sweep loop, right after arm_ik.reset()) --
    # the retry check below only runs on the FIRST step_sim call for that target, not
    # every step. Checking every step (2026-07-27 finding) repeatedly wipes the
    # solver's warm-start continuity for any target whose primary solve occasionally
    # blips above threshold, making some targets measurably worse than not retrying.
    retry_pending = [False]
    did_retry = [False]  # set True iff the retry actually fired for the current target
                         # (distinct from solve_ok, which tracks IPOPT's own converged/
                         # fell-back-to-debug-iterate flag -- a solve can succeed by
                         # IPOPT's own definition and still be the thing we retried away)
    # EMA-smoothed root ORIENTATION ONLY (not position) for the pelvis-frame conversion
    # (2026-07-27, matches g1_full_demo.py's identical fix) -- found via the fixed-base
    # isolation test that walking's continuous weight-shifting was a major cause of bad
    # IK solves (kinematic mean 13.01cm walking vs. 3.16cm base-fixed, same 15 targets).
    # The raw, jittery root orientation was injecting real frame-to-frame perturbation
    # into the pelvis-relative conversion; smoothing it filters that out while still
    # tracking genuine relocation. Position deliberately stays raw/live (tested
    # smoothing it too -- made some targets measurably worse: it lags the "other arm,
    # hold in place" reference behind the arm's own live world position, injecting a
    # real mismatch; the derived leak mechanism only implicates orientation anyway).
    # None = not yet seeded (done on first use / reset per target).
    smoothed_root_quat = [None]

    def to_pelvis(world_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> np.ndarray:
        return quat_apply_inverse(root_quat, world_pos - root_pos).detach().cpu().numpy()

    def se3(p: np.ndarray) -> np.ndarray:
        tf = np.eye(4)
        tf[:3, 3] = p
        return tf

    _up = torch.tensor([0.0, 0.0, 1.0], device=device)

    def tilt_deg(root_quat: torch.Tensor) -> float:
        """Angle between the pelvis's local +z axis and world-up -- captures
        combined roll+pitch lean regardless of yaw, in one scalar."""
        world_z = quat_apply(root_quat, _up)
        return float(torch.rad2deg(torch.acos(world_z[2].clamp(-1.0, 1.0))).item())

    def step_sim(target_world: torch.Tensor | None):
        """One control step: loco policy drives locomotion (zero command throughout —
        standing), IK (if target_world given) drives the active arm, tau_ff applied on
        top per the 2026-07-27 fix (unless --no_tauff), rate-limited exactly like
        g1_full_demo.py's _compute_arm_targets_ik. Returns (achieved_err_m,
        kinematic_err_m, done) for the active arm when target_world is not None."""
        with torch.inference_mode():
            env_inner = env.unwrapped
            env_inner._arm_motion_joint_ids = all_arm_joint_ids_robot
            targets = robot.data.default_joint_pos[0:1, all_arm_joint_ids_robot].clone()

            achieved_err = kinematic_err = None
            if target_world is not None:
                root_pos = robot.data.root_pos_w[0]  # live, not smoothed -- see note above
                raw_root_quat = robot.data.root_quat_w[0]
                if smoothed_root_quat[0] is None:
                    smoothed_root_quat[0] = raw_root_quat.clone()
                else:
                    a = args_cli.root_smoothing_alpha
                    blended_quat = a * raw_root_quat + (1.0 - a) * smoothed_root_quat[0]
                    smoothed_root_quat[0] = blended_quat / blended_quat.norm()
                root_quat = smoothed_root_quat[0]
                target_pelvis = to_pelvis(target_world, root_pos, root_quat)
                other_pelvis = to_pelvis(robot.data.body_pos_w[0, other_ee_body_id], root_pos, root_quat)
                left_pelvis = target_pelvis if side == "left" else other_pelvis
                right_pelvis = other_pelvis if side == "left" else target_pelvis

                # 2026-07-28: REVERTED. Tried seeding solve_ik purely from its own
                # internal continuity (current_q=None) instead of the real, live robot
                # pose, on the theory that anchoring to a perpetually-lagging real pose
                # was destabilizing the redundant null-space choice for elbow/wrist_yaw.
                # Confirmed real (joint-space gaps for those two joints did shrink to
                # ~0 on several targets) but net harmful: without the real pose pulling
                # it back to what's actually achievable, the solver's own warm start
                # drifted away from reality over a long hold, producing WORSE kinematic
                # solves later in the sequence (one target's kinematic error went from
                # 0.5cm to 28.6cm) and one real fall that never happened before. Mean
                # real error got worse (25.76cm -> 39.65cm). Back to anchoring on the
                # real pose every step -- that anchoring is load-bearing, not the bug.
                current_q_full = robot.data.joint_pos[0, all_arm_joint_ids_robot].detach().cpu().numpy()
                sol_q, sol_tauff = arm_ik.solve_ik(se3(left_pelvis), se3(right_pelvis), current_q=current_q_full)
                solve_ok[0] = bool(arm_ik.last_solve_succeeded)
                kinematic_err = float(np.linalg.norm(arm_ik.fk_wrist(sol_q, side) - target_pelvis))

                # Hybrid warm-start fallback (2026-07-27, matches g1_full_demo.py's
                # _compute_arm_targets_ik) — checked ONCE per new target (retry_pending
                # set by the sweep loop right after arm_ik.reset()), not every step. A
                # kinematic error this far beyond anything Phase 1's dense sweep ever
                # produced (worst case ~16cm) means the solve itself is degenerate for
                # this call (bad warm start inherited from wherever the arm ended up
                # after the previous target), not a genuine hard-to-reach point.
                if retry_pending[0]:
                    retry_pending[0] = False
                    if kinematic_err > _KINEMATIC_SANITY_M:
                        did_retry[0] = True
                        default_q_full = robot.data.default_joint_pos[0, all_arm_joint_ids_robot].detach().cpu().numpy()
                        arm_ik.reset(default_q_full)
                        retry_q, retry_tauff = arm_ik.solve_ik(se3(left_pelvis), se3(right_pelvis), current_q=default_q_full)
                        solve_ok[0] = bool(arm_ik.last_solve_succeeded)
                        retry_err = float(np.linalg.norm(arm_ik.fk_wrist(retry_q, side) - target_pelvis))
                        sol_q, sol_tauff, kinematic_err = retry_q, retry_tauff, retry_err

                sol_q_t = torch.tensor(sol_q, dtype=torch.float32, device=device)
                sol_slice = sol_q_t[side_slice]
                current = robot.data.joint_pos[0, all_arm_joint_ids_robot[side_slice]]

                if kinematic_err > _KINEMATIC_HARD_REJECT_M:
                    # Hard safety net -- even after the once-per-target retry, this
                    # solution is too far off to act on. Hold position, zero tau_ff,
                    # for this step only (next step tries again fresh).
                    delta = torch.zeros_like(current)
                    targets[:, side_slice] = current
                    robot.set_joint_effort_target(
                        torch.zeros(1, all_arm_joint_ids_robot.numel(), device=device),
                        joint_ids=all_arm_joint_ids_robot,
                    )
                else:
                    delta = (sol_slice - current).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
                    targets[:, side_slice] = current + delta
                    if args_cli.apply_tauff:
                        tauff_t = torch.tensor(sol_tauff, dtype=torch.float32, device=device)
                        robot.set_joint_effort_target(tauff_t.unsqueeze(0), joint_ids=all_arm_joint_ids_robot)
                    else:
                        robot.set_joint_effort_target(
                            torch.zeros(1, all_arm_joint_ids_robot.numel(), device=device),
                            joint_ids=all_arm_joint_ids_robot,
                        )

                achieved_ee_w = robot.data.body_pos_w[0, active_ee_body_id]
                achieved_err = (target_world - achieved_ee_w).norm().item()
                # Diagnostic-only (2026-07-27): expose the joint-space commanded
                # reference (post rate-limit, this step) and the ACTUAL measured joint
                # position (pre-step, i.e. reflects tracking of the PREVIOUS step's
                # command) for the active arm -- never logged before now. Every
                # wrist-space hypothesis (smoothing, sway, waist, torque saturation,
                # solver null-space instability under realistic warm-starting) has been
                # directly tested and ruled out; this is the one remaining link
                # (commanded-vs-actual in JOINT space, across the step boundary) that
                # hasn't been observed from a real run.
                joint_cmd_this_step[0] = (current + delta).detach().cpu().numpy().copy()
                joint_actual_this_step[0] = current.detach().cpu().numpy().copy()

            env_inner._arm_motion_targets = targets
            robot.write_joint_stiffness_to_sim(_GAIN_ARM_ACTIVE[0], joint_ids=all_arm_joint_ids_robot)
            robot.write_joint_damping_to_sim(_GAIN_ARM_ACTIVE[1], joint_ids=all_arm_joint_ids_robot)

            loco_env.command_manager.get_term("base_velocity").vel_command_b[:] = zero_cmd
            loco_env.command_manager.get_term("base_velocity").is_standing_env[:] = False

            action = loco_policy(_last_obs[0])
            obs, _, dones, _ = env.step(action)
            _last_obs[0] = obs
            done = bool(torch.any(dones > 0).item())
        return achieved_err, kinematic_err, done, solve_ok[0], did_retry[0]

    with torch.inference_mode():
        obs, _ = env.reset()
    _last_obs = [obs]

    print(f"[EvalArmIK] Settling ({args_cli.settle_steps} steps, no arm target)...")
    any_fall = False
    # Diagnostic (2026-07-28): does the loco policy's OWN standing behavior -- zero
    # arm interaction at all -- already show large pelvis motion under this harness's
    # command regime (zero velocity, is_standing_env=False), or does sway only appear
    # once IK-driven arm reaching starts? Settles the question of whether a checkpoint
    # swap's worse eval numbers reflect the checkpoint's real standing behavior here,
    # or an arm-disturbance interaction specific to this eval's IK-driven reaching.
    settle_root_x, settle_root_y, settle_tilt = [], [], []
    for _ in range(args_cli.settle_steps):
        _, _, done, _, _ = step_sim(None)
        any_fall |= done
        rp = robot.data.root_pos_w[0]
        settle_root_x.append(rp[0].item())
        settle_root_y.append(rp[1].item())
        settle_tilt.append(tilt_deg(robot.data.root_quat_w[0]))
    settle_root_x, settle_root_y, settle_tilt = np.array(settle_root_x), np.array(settle_root_y), np.array(settle_tilt)
    settle_sway_cm = 100.0 * math.hypot(settle_root_x.max() - settle_root_x.min(),
                                        settle_root_y.max() - settle_root_y.min())
    print(f"[EvalArmIK] Pure-standing settle phase (NO arm target, NO arm interaction): "
          f"xy_sway={settle_sway_cm:.1f}cm  tilt_range={settle_tilt.max() - settle_tilt.min():.1f}deg  "
          f"fell={any_fall}")

    bounds = dict(_GOAL_BOUNDS[side])
    if args_cli.x_max is not None:
        bounds["x"] = (bounds["x"][0], args_cli.x_max)
        print(f"[EvalArmIK] Trimmed goal box: x in {bounds['x']} (was {_GOAL_BOUNDS[side]['x']})")
    rng = np.random.default_rng(args_cli.seed)
    targets_ground = np.stack([
        rng.uniform(*bounds["x"], args_cli.num_targets),
        rng.uniform(*bounds["y"], args_cli.num_targets),
        rng.uniform(*bounds["z"], args_cli.num_targets),
    ], axis=-1)

    rows = []
    trace_rows = []
    print(f"[EvalArmIK] Sweeping {args_cli.num_targets} targets, side={side}, "
          f"hold_steps={args_cli.hold_steps}, apply_tauff={args_cli.apply_tauff}")
    for i, tgt_ground in enumerate(targets_ground):
        # Ground-referenced target -> world (matches g1_full_demo.py's convention:
        # x/y rotated by current root yaw + root xy, z is ground-referenced absolute).
        #
        # 2026-07-28 FIX: this previously used the FULL root_quat (including whatever
        # real roll/pitch tilt the robot had at that instant), not yaw alone, despite
        # this comment always having said "yaw" -- an implementation gap, not a design
        # choice. Since `local`'s z-component is a large ABSOLUTE ground-referenced
        # height (~0.9-1.15m, not a small pelvis-relative offset), any nonzero roll/
        # pitch leaks that height into x/y through the rotation matrix's off-diagonal
        # terms (~sin(tilt) * height -- e.g. ~14-20cm at a modest 8-10 degree tilt).
        # Confirmed directly: two targets showing 23-30cm kinematic error live solved
        # to 1-2cm in a clean, static, zero-tilt frame with the exact same coordinates
        # -- the solver was never given a hard problem, it was given a CORRUPTED one.
        # yaw_quat() extracts just the facing-direction rotation, matching what
        # "ground-referenced" is actually supposed to mean (height from gravity,
        # x/y from facing direction) regardless of how much the robot happens to be
        # leaning at this exact instant.
        root_pos = robot.data.root_pos_w[0].clone()
        root_pos[2] = ground_z_w
        raw_root_quat_for_target = robot.data.root_quat_w[0]
        root_quat = yaw_quat(raw_root_quat_for_target)
        local = torch.tensor(tgt_ground, dtype=torch.float32, device=device)
        target_world = quat_apply(root_quat, local) + root_pos
        # Diagnostic-only (2026-07-28): measure the ACTUAL size of the leak this fix
        # removes, for THIS specific target, right now -- rather than assuming the
        # hypothetical-tilt estimate from earlier applies here. If this comes back
        # small, the leak wasn't the dominant driver for this target's kinematic error
        # and something else (ongoing per-step re-solving during the hold) needs to be
        # the focus instead of this fix.
        old_buggy_target_world = quat_apply(raw_root_quat_for_target, local) + root_pos
        leak_cm = (target_world - old_buggy_target_world).norm().item() * 100.0
        instant_tilt_deg = tilt_deg(raw_root_quat_for_target)
        print(f"  [target {i}] leak this fix removed: {leak_cm:.1f}cm  "
              f"(instantaneous tilt at target-creation: {instant_tilt_deg:.1f} deg)")

        did_retry[0] = False
        arm_ik.reset(robot.data.joint_pos[0, all_arm_joint_ids_robot].detach().cpu().numpy())
        retry_pending[0] = True
        smoothed_root_quat[0] = None  # re-seed from the live orientation for this new target

        achieved_err = kinematic_err = None
        fell = False
        any_solve_not_ok = False
        for step_i in range(args_cli.hold_steps):
            achieved_err, kinematic_err, done, ok, retried_flag = step_sim(target_world)
            any_solve_not_ok |= not ok
            # Diagnostic trace (2026-07-27): per-step record of pelvis motion vs.
            # achieved error, to test directly whether real error tracks the standing
            # policy's continuous weight-shift sway (a moving pelvis-relative target
            # throughout the WHOLE hold, not just at reset) rather than a one-off
            # frame bug or solver defect -- see ik_arm_integration_plan.md landmine
            # discussion. Cheap: no extra sim steps, just reading already-stepped state.
            rp = robot.data.root_pos_w[0]
            waist_q = robot.data.joint_pos[0, waist_joint_ids_robot]
            row = {
                "target_idx": i, "step": step_i,
                "achieved_err_cm": achieved_err * 100.0, "kinematic_err_cm": kinematic_err * 100.0,
                "root_x": rp[0].item(), "root_y": rp[1].item(), "root_z": rp[2].item(),
                "tilt_deg": tilt_deg(robot.data.root_quat_w[0]),
                "waist_yaw_deg": math.degrees(waist_q[0].item()),
                "waist_roll_deg": math.degrees(waist_q[1].item()),
                "waist_pitch_deg": math.degrees(waist_q[2].item()),
            }
            if joint_cmd_this_step[0] is not None:
                for j in range(7):
                    row[f"cmd_{j}"] = float(joint_cmd_this_step[0][j])
                    row[f"actual_{j}"] = float(joint_actual_this_step[0][j])
            trace_rows.append(row)
            if done:
                # IsaacLab auto-resets a terminated env inside step() itself — stop
                # immediately rather than keep commanding a now-stale target at a
                # freshly-reset robot for the rest of this hold (would silently corrupt
                # both this target's measurement and the next target's starting state).
                fell = True
                break
        any_fall |= fell
        retried_this_target = did_retry[0]

        rows.append({
            "target_x": tgt_ground[0], "target_y": tgt_ground[1], "target_z": tgt_ground[2],
            "achieved_err_cm": achieved_err * 100.0, "kinematic_err_cm": kinematic_err * 100.0,
            "fell": int(fell), "retried": int(retried_this_target), "ipopt_not_ok": int(any_solve_not_ok),
        })
        print(f"  [{i + 1}/{args_cli.num_targets}] target={tgt_ground.round(3)} "
              f"achieved={achieved_err * 100:.1f}cm kinematic={kinematic_err * 100:.1f}cm "
              f"fell={fell} retried={retried_this_target}")

        if fell:
            print("  [EvalArmIK] Fall detected — resetting env before continuing.")
            # Must be wrapped in inference_mode() too: internal buffers touched by
            # step_sim's own inference_mode block become "inference tensors" that
            # PyTorch refuses to update in-place from outside that context afterward
            # (real crash hit here on first run — RuntimeError: Inplace update to
            # inference tensor outside InferenceMode is not allowed).
            with torch.inference_mode():
                obs, _ = env.reset()
                _last_obs[0] = obs
            for _ in range(args_cli.settle_steps):
                step_sim(None)

    # NOTE: loco_env.close() deliberately NOT called here (or anywhere) before
    # reporting -- this repo has a documented, known issue (see eval_arm_long_hold.py-
    # adjacent scripts' own comments) where Isaac Sim/Kit teardown can hang after
    # close() in a standalone-scene setup. Confirmed here 2026-07-27: a run that
    # printed every target's result correctly still died silently (exit 0, no
    # traceback) immediately after the sweep loop, before ever reaching this summary --
    # consistent with close() hanging and something external eventually killing the
    # process. Fix: do ALL reporting first, then force-exit via os._exit(0) in
    # __main__ without a clean close() at all -- the process is ending regardless, so
    # skipping teardown is harmless.
    achieved = np.array([r["achieved_err_cm"] for r in rows])
    kinematic = np.array([r["kinematic_err_cm"] for r in rows])
    n = len(rows)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("validation", "arm_ik_standing", stamp)
    os.makedirs(out_dir, exist_ok=True)
    write_eval_meta(out_dir, args_cli, __file__)

    csv_path = os.path.join(out_dir, "goal_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    trace_path = os.path.join(out_dir, "step_trace.csv")
    with open(trace_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)

    trace_arr = {k: np.array([r[k] for r in trace_rows]) for k in trace_rows[0].keys()}
    print("\n[EvalArmIK] Per-target pelvis sway + waist deviation during the hold (diagnosing real-vs-kinematic gap):")
    print("  idx  achieved_cm  kinematic_cm  xy_sway_cm  tilt_range_deg  waist_end(y/r/p)deg  waist_pred_cm")
    # waist_pred_cm: a full-model FK check (2026-07-27, no sim needed) found ~0.6cm of
    # wrist error per degree of waist deviation on any single axis, roughly linear and
    # roughly additive across axes -- this reduced-model IK locks the waist at zero and
    # cannot see or correct for whatever the walking policy actually does with it.
    _CM_PER_DEG_WAIST = 0.6
    for i in range(n):
        m = trace_arr["target_idx"] == i
        xy_sway_cm = 100.0 * math.hypot(
            trace_arr["root_x"][m].max() - trace_arr["root_x"][m].min(),
            trace_arr["root_y"][m].max() - trace_arr["root_y"][m].min(),
        )
        tilt_range_deg = trace_arr["tilt_deg"][m].max() - trace_arr["tilt_deg"][m].min()
        wy_end = trace_arr["waist_yaw_deg"][m][-1]
        wr_end = trace_arr["waist_roll_deg"][m][-1]
        wp_end = trace_arr["waist_pitch_deg"][m][-1]
        waist_pred_cm = _CM_PER_DEG_WAIST * (abs(wy_end) + abs(wr_end) + abs(wp_end))
        print(f"  {i:3d}  {rows[i]['achieved_err_cm']:10.1f}  {rows[i]['kinematic_err_cm']:11.1f}  "
              f"{xy_sway_cm:9.1f}  {tilt_range_deg:14.1f}  "
              f"{wy_end:5.1f}/{wr_end:5.1f}/{wp_end:5.1f}       {waist_pred_cm:12.1f}")

    # Joint-space commanded-vs-actual tracking gap (2026-07-27) -- the one link never
    # directly observed before: is the ACTUAL simulated joint position tracking the
    # RATE-LIMITED command we send it, or is there a persistent joint-space gap that
    # wrist-space error alone couldn't distinguish from a frame/kinematics bug? Compares
    # step k's command to step k+1's actual (the physics update in between is what
    # should be closing that gap).
    print("\n[EvalArmIK] Joint-space commanded-vs-actual tracking gap (final-step, active arm, degrees):")
    print("  idx  achieved_cm  " + "  ".join(f"j{j}_gap_deg" for j in range(7)))
    for i in range(n):
        m = trace_arr["target_idx"] == i
        cmds = np.stack([trace_arr[f"cmd_{j}"][m] for j in range(7)], axis=-1)
        actuals = np.stack([trace_arr[f"actual_{j}"][m] for j in range(7)], axis=-1)
        # shift: command at step k should be reflected in actual at step k+1
        gap_deg = np.degrees(cmds[:-1] - actuals[1:])
        final_gap = gap_deg[-1]
        print(f"  {i:3d}  {rows[i]['achieved_err_cm']:10.1f}  " +
              "  ".join(f"{g:9.2f}" for g in final_gap))

    n_retried = sum(r["retried"] for r in rows)
    n_fell = sum(r["fell"] for r in rows)
    print(f"\n[EvalArmIK] ===== SUMMARY (side={side}, apply_tauff={args_cli.apply_tauff}, n={n}) =====")
    print(f"any fall during sweep: {any_fall}  (targets that fell: {n_fell}/{n})")
    print(f"targets whose initial solve was bad enough to trigger the once-per-target retry (>{_KINEMATIC_SANITY_M * 100:.0f}cm): {n_retried}/{n}")
    for tol in (2.0, 5.0, 10.0, 15.0):
        print(f"  within {tol:4.0f}cm  real={100 * (achieved <= tol).mean():5.1f}%  "
              f"kinematic={100 * (kinematic <= tol).mean():5.1f}%")
    print(f"  real error:      mean={achieved.mean():.2f}cm  median={np.median(achieved):.2f}cm  max={achieved.max():.2f}cm")
    print(f"  kinematic error: mean={kinematic.mean():.2f}cm  median={np.median(kinematic):.2f}cm  max={kinematic.max():.2f}cm")
    print(f"[EvalArmIK] Per-goal CSV: {csv_path}")


if __name__ == "__main__":
    main()
    # simulation_app.close() deliberately SKIPPED (2026-07-27): confirmed by direct
    # observation (user's own run) that all reporting -- every target's line, the
    # SUMMARY block, the CSV, run_meta.yaml -- completes and is written to disk
    # correctly, and the hang is specifically in Kit's own teardown afterward (the
    # process sat idle post-summary until manually interrupted). Since main() has
    # already returned and every side effect that matters is on disk, there is nothing
    # left for a clean close() to protect -- os._exit(0) terminates immediately
    # instead of risking another indefinite hang.
    os._exit(0)
