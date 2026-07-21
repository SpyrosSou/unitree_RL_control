# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, vectorized integration eval — runs the same combined standing + walking +
arm setup as testing/general_testing/g1_full_demo.py (same env, same action-override
math, same observation-frame fix), scripted instead of interactive, across many
parallel envs, and logs metrics to CSV.

Deliberately built on the *walking* env (G1LocomotionFlatEnvCfg_PLAY), exactly like
g1_full_demo.py, not the standing env — the standing task swaps in a custom action term
(StandingArmBlendJointPositionAction) that unconditionally overrides arm joints with a
scripted disturbance target, which fights any attempt to inject a different signal from
outside. The walking env uses a plain JointPositionAction, so overriding the arm
columns of the action vector (same technique g1_full_demo.py already uses) works
directly, no extra plumbing needed.

Bucket list, each a single continuous condition (no mode transitions within a bucket —
each one picks one policy and holds the command steady for its whole duration):

- standing_still  — zero command, arms held at default. Fall rate + tilt with no input.
- walking_straight — constant forward command, arms held at default. Fall rate + drift.
- standing_arm_{left,right}_reach[_edge] — zero command, one arm continuously cycling
  through goals while the other is held at default. Arm success rate + steps-to-success,
  plus fall rate/tilt as a free byproduct (answers "does reaching destabilize
  standing"). Four of these by default (left/right x in-range/edge — see below);
  --skip_edge_bucket drops it to two (left/right, in-range only).

  Goal sampling per bucket:
    in-range — uniform within _GOAL_BOUNDS[side], the exact box g1_arm_env.py trains
      on (fraction=1.0, no curriculum) and the same box g1_full_demo.py calls "the
      trained range" and warns about leaving (_parse_and_confirm_target). This was
      already the only range in use before --skip_edge_bucket/edge existed; nothing
      changed here.
    edge — "just outside, close to the trained range": exactly one axis per goal is
      pushed past its bound by a random amount in [0, --edge_margin_m], the other two
      axes stay in-range. A single-axis violation, not simultaneous extrapolation on
      all three, so it's genuinely adjacent to the box rather than a different region
      — meant to approximate what happens when someone nudges a manually-typed
      g1_full_demo.py target just past the printed reachable range on one coordinate.

  Right-arm reaching has no separately-trained checkpoint (see
  testing/arm_testing/checkpoints.yaml — "right: checkpoint: ~"). It runs the
  left-trained policy on a mirrored observation and mirrors the resulting action back,
  the same technique g1_full_demo.py's mirror-testing mode (Y/U keys) and
  mdp/symmetry.py use — not an approximation invented for this script.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/integration_validation/eval_full_demo.py \\
        --standing_checkpoint chosen_checkpoints/standing_latest.pt \\
        --walking_checkpoint chosen_checkpoints/walking_latest.pt \\
        --arm_checkpoint chosen_checkpoints/arm_left_latest.pt \\
        --headless

Output (mirrors logs/'s timestamped-folder convention):
    validation/integration_validation/<YYYY-MM-DD_HH-MM-SS>/
        detailed.csv          — every episode/attempt row from every bucket, tagged by
                                 bucket, columns not relevant to a given bucket left blank.
                                 Cross-bucket, reduced column set — for scripts/pandas, not
                                 for eyeballing any one test's full diagnostics.
        <bucket>/<name>_detailed.csv — full per-episode/attempt rows for exactly one test,
                                 every diagnostic column that test's metrics wrapper tracks.
        <bucket>/<name>_summary.csv  — that same test's one-row aggregate: means (mean_*)
                                 and rates (fall_rate, success_rate_*). Distance/rotation
                                 metrics that can be signed either way per episode (heading
                                 drift, lateral drift) are averaged by |value| — a plain
                                 signed mean would hide real drift behind episodes that
                                 happened to cancel out in opposite directions. Every
                                 *_detailed.csv gets exactly one matching *_summary.csv,
                                 each shaped for what that test actually measures rather
                                 than forced into one shared row/column set.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Headless integration eval: standing + walking + arm, combined (mirrors g1_full_demo.py)."
)
parser.add_argument(
    "--standing_checkpoint", type=str, required=True, help="Path to a trained standing checkpoint (.pt)."
)
parser.add_argument(
    "--walking_checkpoint", type=str, required=True, help="Path to a trained walking checkpoint (.pt)."
)
parser.add_argument(
    "--arm_checkpoint", type=str, required=True,
    help="Path to a trained left-arm-IK checkpoint (.pt). Also drives right-arm reaching, via mirroring "
    "(see module docstring) — there is no separate right-arm checkpoint yet.",
)
parser.add_argument(
    "--arm_hidden_dims", type=int, nargs="+", default=None,
    help="Override arm actor/critic hidden dims (e.g. 512 256 128 for a wide-net checkpoint). "
    "Default: baseline [256, 128, 64].",
)
parser.add_argument(
    "--arm_driver", type=str, default="policy", choices=["policy", "ik", "event"],
    help="What drives the actively-reaching arm in the standing_arm_*_reach buckets (2026-07-16). "
    "'policy' = the trained RL arm checkpoint (the historical behavior). 'ik' = the same "
    "isaaclab DifferentialIKController + clamped-delta scheme StandingArmIKReachDisturbance "
    "already uses inside standing's training — i.e. for an IK-trained standing checkpoint "
    "(height_reward etc.), deployment then matches training *exactly*. Also the "
    "industry-standard humanoid architecture (RL legs + IK arms at hardware gains), removes "
    "the RL arm's near-goal jitter/branch problems by construction, and drives the right arm "
    "natively instead of via the mirror transform. 'event' (DIAGNOSTIC, 2026-07-16) = attach "
    "standing training's OWN StandingArmIKReachDisturbance event term to this env and let it "
    "drive the arms exactly as it does in training/the native eval — no reimplementation in "
    "the loop at all. Completes the native-vs-integration 2x2: if falls persist even under the "
    "literal training disturbance code, the base-env difference is the culprit; if they "
    "vanish, this script's own arm-driving path is. In event mode the four side/edge reach "
    "buckets collapse into one 'standing_arm_event_reach' bucket (the term picks sides "
    "per-episode itself, goals are its own, so there's no per-attempt arm CSV — the standing "
    "fall rate is the measurement), and the walking bucket runs with arms pinned at default "
    "(the term's buffer overrides them) — ignore its arm realism in this mode.",
)
parser.add_argument(
    "--active_arm_gain", type=float, nargs=2, default=[200.0, 20.0], metavar=("KP", "KD"),
    help="PD gain written to the actively-reaching arm's joints in the reach buckets "
    "(default 200 20 = the RL arm checkpoint's training gain, the historical behavior). "
    "Pass '60 1.5' for the hardware-realistic gain (unitree_sdk2_python's own arm examples) "
    "— which is also exactly the gain every standing checkpoint's training drives its "
    "disturbance arm at, so combined with --arm_driver ik the standing policy faces "
    "literally its own training-time disturbance.",
)
parser.add_argument(
    "--torso_clip_deg", type=float, default=None,
    help="Match the standing checkpoint's OWN torso_joint action clip, in +/-degrees (2026-07-20). "
    "This env is deliberately built on the walking cfg (G1LocomotionFlatEnvCfg_PLAY, see "
    "_build_env_cfg's docstring), which has no relationship to the standing lineage's own cfg "
    "chain — so a clip set on a standing training config (e.g. "
    "G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg's +/-30) is NEVER "
    "carried into this eval unless passed here explicitly. Omitting this for a torso-clip-trained "
    "checkpoint reproduces the exact pre-fix unclipped exploit (torso pinned at its ~150deg hard "
    "limit, cascading into falls) — not because the checkpoint regressed, but because it's being "
    "evaluated under an action space it was never trained within, the same category of "
    "train/deploy mismatch this whole project has repeatedly hunted down elsewhere. Pass the "
    "SAME value the checkpoint's own training config used (30 for TorsoClip, 0 for TorsoLock); "
    "omit entirely (default None = no clip applied) for any checkpoint trained WITHOUT a torso "
    "clip (GainMatch, height_reward, everything before 2026-07-20 night).",
)
parser.add_argument("--num_envs", type=int, default=32, help="Parallel envs per bucket.")
parser.add_argument(
    "--steps_standing_still", type=int, default=3000, help="Control steps (50 Hz) for the standing_still bucket."
)
parser.add_argument(
    "--steps_walking_straight", type=int, default=3000, help="Control steps (50 Hz) for the walking_straight bucket."
)
parser.add_argument(
    "--steps_arm_reach", type=int, default=3000,
    help="Control steps (50 Hz) for each standing_arm_*_reach[_edge] bucket — applied per bucket, so total "
    "arm-reach runtime scales with how many of those buckets run (see --skip_edge_bucket).",
)
parser.add_argument(
    "--max_steps_per_goal", type=int, default=750,
    help="Control steps before an arm-reach attempt counts as a timeout (750 @ 50Hz = 15s).",
)
parser.add_argument(
    "--edge_margin_m", type=float, default=0.05,
    help="Max distance (m) a single axis is pushed past _GOAL_BOUNDS in the *_edge buckets. See module docstring.",
)
parser.add_argument(
    "--skip_edge_bucket", action="store_true",
    help="Only run the in-range standing_arm_{left,right}_reach buckets, skip the *_edge (near-out-of-range) ones.",
)
parser.add_argument(
    "--forward_speed", type=float, default=0.5, help="Commanded forward speed for walking_straight (m/s)."
)
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible rollouts.")
parser.add_argument(
    "--output_root", type=str, default=None,
    help="Override the output root directory. Default: validation/integration_validation/ next to this script.",
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
import math
import os
import subprocess

import yaml

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import G1ArmIKLeftPPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_ARM_JOINTS,
    _RIGHT_EE_BODY,
)
from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import mirror_arm_actions, mirror_arm_obs
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import G1LocomotionFlatPPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import G1LocomotionFlatEnvCfg_PLAY
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp import (
    StandingArmBlendJointPositionAction,
    StandingArmIKReachDisturbance,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp.observations import standing_arm_motion_targets
from g1_locomotion.utils.metrics_wrappers import StandingMetricsCsvWrapper, WalkingMetricsCsvWrapper, _DualCsvWriter
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Same constants g1_full_demo.py uses for this exact action conversion — see
# velocity_env_cfg.py's ActionsCfg (scale=0.5, use_default_offset=True), shared by
# every G1Flat-derived task.
ARM_ACTION_SCALE = 0.5
# Matches G1ArmIKEnvCfg.max_action_delta_per_step (was 0.05 here — a stale mismatch, not
# a deliberate choice; g1_arm_env.py's actual training value is 0.06).
ARM_MAX_JOINT_DELTA_PER_STEP = 0.06
# Matches G1ArmIKEnvCfg.action_filter_alpha. Training never hands the raw policy output
# straight to the actuator: g1_arm_env.py's _pre_physics_step does
# filtered = alpha*raw + (1-alpha)*filtered_prev, THEN scales/clamps that — an EMA low-
# pass meant to emulate actuator lag and avoid abrupt jumps (see its own comment). This
# deployment path was applying the raw, unfiltered policy output directly instead — found
# 2026-07-09 while chasing why standing_arm_reach's fall rate was so much worse here than
# casual manual g1_full_demo.py testing suggested: this script's continuous back-to-back
# goal-cycling (a new goal every few seconds, non-stop) stresses an unfiltered-vs-trained
# mismatch far harder than a few manually-typed targets with pauses ever would.
ARM_ACTION_FILTER_ALPHA = 0.25
GOAL_THRESHOLD_M = 0.02  # matches G1ArmIKEnvCfg.goal_threshold (reverted 2026-07-12 from a 3cm change made without asking)

_ARM_JOINTS = {"left": _LEFT_ARM_JOINTS, "right": _RIGHT_ARM_JOINTS}
_EE_BODY = {"left": _LEFT_EE_BODY, "right": _RIGHT_EE_BODY}


# ---------------------------------------------------------------------------
# Env / policy construction — mirrors g1_full_demo.py's G1FullDemo.__init__
# ---------------------------------------------------------------------------

def _build_env_cfg() -> G1LocomotionFlatEnvCfg_PLAY:
    cfg = G1LocomotionFlatEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    # Restore training-time observation noise — the _PLAY config disables it for clean
    # visualization, same workaround every eval_*.py script in this repo applies.
    cfg.observations.policy.enable_corruption = True

    # Use the same action term standing's own training env uses (2026-07-12, replaces
    # the previous "zero out action[:, arm_cols] before .step()" approach in every bucket
    # below). That approach corrupted the "last_action" observation term: ActionManager
    # stores whatever raw tensor is passed to .step() verbatim (action_manager.py:385-386)
    # and that's exactly what mdp.last_action reads back next step — it is NOT derived
    # from what StandingArmBlendJointPositionAction.apply_actions() actually sends to the
    # simulated joints. So masking the action tensor before stepping meant the standing
    # policy's own arm-column output was being fed back to it every step as a hard-pinned
    # 0.0 (or an IK delta) it never produced and never saw during training — a genuine
    # observation mismatch, not just a simulated-target one. StandingArmBlendJointPositionAction
    # decouples this correctly: it only overrides what's *applied* to the arm joints
    # (reading env._standing_arm_motion_targets/_standing_arm_motion_joint_ids, set by each
    # bucket below), while the policy's raw output — whatever it naturally is — still flows
    # through untouched into action_manager's stored action and therefore into next step's
    # last_action observation, exactly matching training. Falls through to identical
    # behavior to the previous stock JointPositionAction when those two env attributes
    # aren't set (e.g. for walking_straight, which never touches them).
    cfg.actions.joint_pos.class_type = StandingArmBlendJointPositionAction

    # Carry over the standing checkpoint's OWN torso_joint action clip, if any — see
    # --torso_clip_deg's help text for why this doesn't happen automatically (this env is
    # built from the walking cfg lineage, unrelated to the standing cfg chain a clip lives
    # on). Must match whatever the checkpoint was actually trained with, or this eval
    # silently tests a different action space than the one that produced it.
    if args_cli.torso_clip_deg is not None:
        cfg.actions.joint_pos.clip = {
            "torso_joint": (-math.radians(args_cli.torso_clip_deg), math.radians(args_cli.torso_clip_deg))
        }

    # No arm-gain override here any more (2026-07-12) — no single value serves every
    # policy's own training. Each bucket runner sets the correct gain live via
    # _set_arm_gains() right before it steps (see that function). Leaves this cfg at the
    # stock Isaac Lab default (40/10) at spawn time, immediately overwritten before any
    # bucket actually runs.

    # --arm_driver event (diagnostic, 2026-07-16): attach standing training's own
    # disturbance term to this (walking) env so the reach condition is produced by the
    # literal training code, not any reimplementation in this script — see the flag's
    # help text for the 2x2 this completes. Constructed enabled; _set_event_disturbance
    # toggles it per bucket (only the event-reach bucket wants it active).
    if args_cli.arm_driver == "event":
        cfg.events.standing_arm_motion_disturbance = EventTerm(
            func=StandingArmIKReachDisturbance,
            mode="interval",
            interval_range_s=(cfg.sim.dt * cfg.decimation, cfg.sim.dt * cfg.decimation),
            params={"asset_cfg": SceneEntityCfg("robot"), "enable_step": 0, "ramp_full_step": 0},
            is_global_time=False,
        )
    return cfg


def _set_event_disturbance(base_env, enabled: bool):
    """--arm_driver event only: toggle the attached training-disturbance term between
    buckets by pushing enable_step out of reach (the same gating knob the term's own
    curriculum uses). No-op in the other driver modes (the term doesn't exist there)."""
    if args_cli.arm_driver != "event":
        return
    term = base_env.event_manager.get_term_cfg("standing_arm_motion_disturbance").func
    step_val = 0 if enabled else 10**12
    term.enable_step = step_val
    term.ramp_full_step = step_val


# Kp/Kd each bucket actually trained under — see _set_arm_gains(). Not one shared value:
# 2026-07-12 confirmed arm-IK reach precision and standing's compliant-arm balance
# strategy need genuinely different gains (arm-IK regressed 86%->30% success when
# retrained under standing's softer value), and walking was never touched at all so it
# should see its own stock training gain, not either policy's.
_STANDING_ARM_GAIN = (60.0, 1.5)  # matches g1_locomotion_env_cfg.py's standing training gain
_ARM_IK_GAIN = (200.0, 20.0)  # matches g1_arm_env.py's arm-IK training gain
_WALKING_ARM_GAIN = (40.0, 10.0)  # stock Isaac Lab default — walking never overrides "arms"


def _set_arm_gains(base_env, stiffness: float, damping: float, device, joint_names=None):
    """Live-update arm joints' PD gains via the PhysX view directly — the actuator
    cfg's stiffness/damping are fixed at spawn time, but write_joint_stiffness_to_sim/
    write_joint_damping_to_sim let this reused-across-buckets simulation switch gains
    between buckets instead of being stuck with whatever _build_env_cfg() set once.

    joint_names: which joints to write (default: all arm joints, both sides). Passing one
    side's joints lets the reach buckets split gains per arm — see _run_standing_arm_reach."""
    robot = base_env.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(
        joint_names if joint_names is not None else _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS
    )
    arm_joint_ids_t = torch.tensor(arm_joint_ids, dtype=torch.long, device=device)
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=arm_joint_ids_t)
    robot.write_joint_damping_to_sim(damping, joint_ids=arm_joint_ids_t)


def _load_loco_policy(base_env, agent_cfg, checkpoint: str, device):
    """Load a locomotion (standing/walking) policy with its input width inferred from the
    checkpoint itself, not from the env (2026-07-15, replaces the OnPolicyRunner path).

    Why: the arm-intent standing task (G1-Locomotion-Standing-Flat-Consolidated-Intent-v0)
    appends a 10-D arm-targets block to the END of the policy observation (75 -> 85), but
    this eval's env is the plain walking cfg (75-D obs) shared by BOTH policies —
    OnPolicyRunner sizes the network from the env, so it could only ever load 75-D
    checkpoints. Building the ActorCritic directly (same pattern _load_arm_policy has
    always used) from the checkpoint's own first-layer width handles every combination:
    old 75-D standing checkpoints, new 85-D intent ones, and walking (always 75-D) —
    no flags needed.

    The returned policy closure auto-extends: if the checkpoint expects more dims than the
    obs it's handed, it appends mdp.standing_arm_motion_targets(base_env) — the exact
    function the intent task trains with, reading the same env buffer every bucket in this
    script already maintains — so train and deploy compute the extra block identically."""
    state = torch.load(checkpoint, map_location=device)["model_state_dict"]
    in_dim = state["actor.0.weight"].shape[1]
    num_actions = state[f"actor.{2 * len(agent_cfg.policy.actor_hidden_dims)}.weight"].shape[0]

    # The locomotion runner cfgs never define obs_groups (dataclasses.MISSING — found
    # 2026-07-15 when this crashed the first overnight integration eval of the new loader);
    # only the arm task's cfg sets it explicitly. Fall back to the standard single-group
    # mapping, which is exactly what that cfg uses and what these policies trained under.
    obs_groups = agent_cfg.obs_groups
    if not isinstance(obs_groups, dict):
        obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    dummy_obs = TensorDict(
        {"policy": torch.zeros((1, in_dim), dtype=torch.float32, device=device)},
        batch_size=[1], device=device,
    )
    actor_critic = ActorCritic(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=num_actions,
        actor_obs_normalization=agent_cfg.policy.actor_obs_normalization,
        critic_obs_normalization=agent_cfg.policy.critic_obs_normalization,
        actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
        activation=agent_cfg.policy.activation,
        init_noise_std=agent_cfg.policy.init_noise_std,
    ).to(device)
    actor_critic.load_state_dict(state)
    actor_critic.eval()

    def policy(obs) -> torch.Tensor:
        # RslRlVecEnvWrapper.reset()/step() hand back a TensorDict ({"policy": tensor},
        # batch dims only) — a TensorDict's .shape is its BATCH size and torch.cat on a
        # mixed [TensorDict, Tensor] list crashes, so normalize to the policy-group tensor
        # first (found 2026-07-15, first real run of this loader).
        obs_tensor = obs["policy"] if isinstance(obs, TensorDict) else obs
        if obs_tensor.shape[-1] < in_dim:
            obs_tensor = torch.cat([obs_tensor, standing_arm_motion_targets(base_env)], dim=-1)
        td = TensorDict({"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]], device=device)
        with torch.inference_mode():
            return actor_critic.act_inference(td)

    return policy


def _load_arm_policy(device):
    obs_dim, action_dim = 28, 5
    agent_cfg = G1ArmIKLeftPPORunnerCfg()
    if args_cli.arm_hidden_dims is not None:
        agent_cfg.policy.actor_hidden_dims = list(args_cli.arm_hidden_dims)
        agent_cfg.policy.critic_hidden_dims = list(args_cli.arm_hidden_dims)

    arm_obs = TensorDict(
        {"policy": torch.zeros((1, obs_dim), dtype=torch.float32, device=device)},
        batch_size=[1], device=device,
    )
    actor_critic = ActorCritic(
        obs=arm_obs,
        obs_groups=agent_cfg.obs_groups,
        num_actions=action_dim,
        actor_obs_normalization=agent_cfg.policy.actor_obs_normalization,
        critic_obs_normalization=agent_cfg.policy.critic_obs_normalization,
        actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
        activation=agent_cfg.policy.activation,
        init_noise_std=agent_cfg.policy.init_noise_std,
    ).to(device)
    checkpoint = torch.load(args_cli.arm_checkpoint, map_location=device)
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    actor_critic.eval()

    def policy(obs: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=device)
        with torch.inference_mode():
            return actor_critic.act_inference(td)

    return policy


_GOAL_PRECISION_M = 0.001  # round sampled goals to mm — see _round_goal's docstring


def _round_goal(goals: torch.Tensor) -> torch.Tensor:
    """Round to the nearest mm. Raw torch.rand()-derived coordinates carry ~7 significant
    figures (e.g. 0.2034718...) — false precision nobody asked for and the real arm can't
    hit anyway; logging/reporting on numbers like that (goal_x_m/y_m/z_m in arm_detailed.csv,
    printed targets) is noise a human then has to mentally round themselves. mm is already
    tighter than the 2cm success threshold, so this doesn't change what's reachable or
    change success/fail outcomes — it just stops manufacturing sub-mm precision that was
    never meaningful."""
    return torch.round(goals / _GOAL_PRECISION_M) * _GOAL_PRECISION_M


def _sample_goals(side: str, num: int, device, edge: bool = False, edge_margin_m: float = 0.05) -> torch.Tensor:
    """Uniform sample within _GOAL_BOUNDS[side], robot-local frame — same formula as
    G1ArmIKEnv._sample_goal_positions (fraction=1.0, no curriculum). This *is* "the
    recommended/trained range" g1_full_demo.py warns about leaving.

    edge=True instead samples "just outside, close to the trained range" — see module
    docstring for the exact scheme (one axis pushed past its bound by up to
    edge_margin_m, the other two stay in-range)."""
    bounds = _GOAL_BOUNDS[side]
    goals = torch.zeros((num, 3), device=device)
    for i, key in enumerate(("x", "y", "z")):
        lo, hi = bounds[key]
        goals[:, i] = lo + torch.rand(num, device=device) * (hi - lo)

    if not edge or num == 0:
        return _round_goal(goals)

    axis = torch.randint(0, 3, (num,), device=device)
    push_high = torch.rand(num, device=device) < 0.5
    margin = torch.rand(num, device=device) * edge_margin_m
    for i, key in enumerate(("x", "y", "z")):
        lo, hi = bounds[key]
        mask = axis == i
        if not bool(mask.any().item()):
            continue
        goals[mask, i] = torch.where(push_high[mask], hi + margin[mask], lo - margin[mask])
    return _round_goal(goals)


def _build_arm_obs(
    robot, arm_joint_ids: torch.Tensor, ee_body_id: int, goal_world: torch.Tensor
) -> torch.Tensor:
    """Same 28-D layout g1_full_demo.py's _build_arm_obs builds, including the
    observation-frame fix (2026-07-09): error must be in the robot's *current* body
    frame, not raw world — g1_arm_env.py trains with a physically fixed, never-rotating
    root, so world frame is body frame throughout training; this env's robot can
    actually turn (reset_base randomizes yaw over the full +/-180 degrees), so a raw
    world-frame error stops meaning "forward/lateral relative to the robot" the moment
    it isn't at ~0 yaw. Works for either arm — it's just whichever joint_ids/ee_body_id
    are passed in."""
    base_lin_vel = robot.data.root_lin_vel_b
    base_ang_vel = robot.data.root_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b

    joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    joint_vel = robot.data.joint_vel[:, arm_joint_ids]
    ee_pos = robot.data.body_pos_w[:, ee_body_id, :]
    error = quat_apply_inverse(robot.data.root_quat_w, goal_world - ee_pos)

    return torch.cat(
        [base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal_world, error], dim=-1
    )


class _IKArmDriver:
    """Analytic-IK arm driver for the reach buckets (--arm_driver ik) — a lean, per-bucket
    copy of the exact frame/Jacobian/clamp math StandingArmIKReachDisturbance._step_side
    uses inside standing's training (see that class — validated there over multiple
    training runs), just fed by the eval tracker's goals instead of its own cycling.
    Gravity droop at soft gains self-corrects: the delta is recomputed from the *measured*
    joint position every step, so persistent lag keeps advancing the commanded target
    until the end-effector converges — an implicit integral action."""

    def __init__(self, base_env, robot, joint_ids: torch.Tensor, ee_body_id: int, device):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
        self._ik = DifferentialIKController(cfg=ik_cfg, num_envs=base_env.num_envs, device=device)
        self._base_env = base_env
        self._robot = robot
        self._joint_ids = joint_ids
        self._ee_body_id = ee_body_id
        # Jacobian indexing convention for floating vs fixed base — same as
        # StandingArmIKReachDisturbance.__init__ (this env's robot is floating-base).
        if robot.is_fixed_base:
            self._jacobi_body_idx = ee_body_id - 1
            self._jacobi_joint_ids = joint_ids
        else:
            self._jacobi_body_idx = ee_body_id
            self._jacobi_joint_ids = joint_ids + 6

    def compute_targets(self, goal_local: torch.Tensor) -> torch.Tensor:
        """Absolute joint targets (num_envs, 5) for this step. goal_local is the tracker's
        root-frame goal (x/y root-relative, z ground-referenced — converted to root-frame z
        from the robot's current height, same as training)."""
        from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

        robot = self._robot
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_id]
        ee_quat_w = robot.data.body_quat_w[:, self._ee_body_id]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
        )
        root_height_above_ground = robot.data.root_pos_w[:, 2] - self._base_env.scene.env_origins[:, 2]
        target_b = goal_local.clone()
        target_b[:, 2] = target_b[:, 2] - root_height_above_ground

        self._ik.set_command(target_b, ee_quat=ee_quat_b)
        jacobian_w = robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])

        joint_pos = robot.data.joint_pos[:, self._joint_ids]
        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)
        delta = (joint_pos_des - joint_pos).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
        limits = robot.data.soft_joint_pos_limits[:, self._joint_ids]
        return (joint_pos + delta).clamp(limits[:, :, 0], limits[:, :, 1])


def _compute_arm_delta(
    side: str, arm_policy, robot, arm_joint_ids: torch.Tensor, ee_body_id: int, goal_world: torch.Tensor
) -> torch.Tensor:
    """5-D joint-delta action for `side`'s arm, in that arm's own (unmirrored) frame.

    Left just runs the left-trained policy directly. Right has no checkpoint of its own
    (testing/arm_testing/checkpoints.yaml: "right: checkpoint: ~") — mirror the real
    right-arm observation into the left-arm convention, run the left-trained policy on
    it, then mirror the resulting action back. Same technique, same sign/index maps, as
    g1_full_demo.py's _mirror_action_override / mdp/symmetry.py."""
    obs = _build_arm_obs(robot, arm_joint_ids, ee_body_id, goal_world)
    if side == "left":
        return arm_policy(obs)
    mirrored_obs = mirror_arm_obs(obs)
    delta_left_frame = arm_policy(mirrored_obs)
    return mirror_arm_actions(delta_left_frame)


# ---------------------------------------------------------------------------
# Arm-attempt tracker (standing_arm_*_reach buckets) — cycles goals independent of
# the walking env's own episode boundary (fall/timeout), same structure as
# g1_full_demo.py's target-setting flow, vectorized.
# ---------------------------------------------------------------------------

class _ArmAttemptTracker:
    _SUMMARY_FIELDS = [
        "episode_index", "env_id", "env_step", "attempt_steps", "steps_to_reach", "outcome", "success",
        "min_dist_to_goal_cm",
    ]
    _DETAILED_FIELDS = _SUMMARY_FIELDS + ["goal_x_m", "goal_y_m", "goal_z_m"]

    def __init__(
        self,
        base_env: ManagerBasedRLEnv,
        robot,
        arm_joint_ids: torch.Tensor,
        ee_body_id: int,
        log_dir: str,
        device,
        sample_fn,
    ):
        self.base_env = base_env
        self.robot = robot
        self.arm_joint_ids = arm_joint_ids
        self.ee_body_id = ee_body_id
        self.device = device
        self.n = base_env.num_envs
        self._sample_fn = sample_fn

        self._csv = _DualCsvWriter(log_dir, "arm", self._DETAILED_FIELDS, self._SUMMARY_FIELDS, write_summary=False)
        self.csv_path = self._csv.detailed_path

        self._episode_index = torch.full((self.n,), self._csv.next_episode_index(), dtype=torch.long, device=device)
        self.goal_local = torch.zeros((self.n, 3), device=device)
        self.attempt_steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.min_dist = torch.full((self.n,), float("inf"), device=device)
        # Control step (within the attempt) at which min_dist first crossed the success
        # threshold; -1 = not reached. Needed since the dwell fix below decouples "when it
        # reached" from "when the attempt ends" (always the 15s timeout now).
        self.reach_step = torch.full((self.n,), -1, dtype=torch.long, device=device)
        # EMA action-filter state (see ARM_ACTION_FILTER_ALPHA) — one 5-D filtered-action
        # vector per env, matching g1_arm_env.py's own self.filtered_actions. Reset to 0
        # per-env at the start of each new attempt, same as training resets it per-env on
        # every episode reset (g1_arm_env.py: self.filtered_actions[env_ids_tensor] = 0.0)
        # — a fresh attempt shouldn't start pre-biased by whatever the previous attempt's
        # filter had converged to.
        self.filtered_action = torch.zeros((self.n, 5), device=device)

    def goal_world(self) -> torch.Tensor:
        """Recomputed every call from the robot's *current* pose — deliberately NOT a
        pose snapshotted once at attempt-start (that was the previous behavior, and it
        was a bug: found 2026-07-09 while investigating why standing_arm_reach's fall
        rate was catastrophic — a goal fixed in world space forces the arm to compensate
        for base tilt/drift to keep reaching it, and g1_arm_env.py trains with
        fix_root_link=True, i.e. the policy has *zero* training experience compensating
        for base motion. As the base tilted further trying to recover from the arm's own
        reaching motion, the fixed-world goal effectively receded, the policy pushed the
        arm further to chase it (out-of-distribution behavior, not a considered
        compensation), which added more destabilizing torque, which tilted the base
        further — a runaway feedback loop ending in a fall.

        goal_local is instead treated as a fixed offset from the torso: recomputing
        goal_world from the *current* root pose every step reproduces what the
        fixed-root training setup gave the policy for free — a goal that only moves
        because the robot pursuing it does, never because the base wandered out from
        under a stationary target. z still pins to ground level (env_origins), not the
        root link's own elevation — the root sits ~0.75-0.8m off the ground, and goal z
        is "height above ground" (matching _GOAL_BOUNDS / g1_arm_env.py's own
        convention), not "height above the root"."""
        pos = self.robot.data.root_pos_w.clone()
        pos[:, 2] = self.base_env.scene.env_origins[:, 2]
        return quat_apply(self.robot.data.root_quat_w, self.goal_local) + pos

    def filter_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        self.filtered_action = (
            ARM_ACTION_FILTER_ALPHA * raw_action + (1.0 - ARM_ACTION_FILTER_ALPHA) * self.filtered_action
        )
        return self.filtered_action

    def start_new_attempts(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.goal_local[env_ids] = self._sample_fn(env_ids.numel(), self.device)
        self.filtered_action[env_ids] = 0.0
        self.attempt_steps[env_ids] = 0
        self.min_dist[env_ids] = float("inf")
        self.reach_step[env_ids] = -1

        # No per-goal arm reset any more (2026-07-14): this used to teleport the arm to its
        # default pose via write_joint_state_to_sim on every new goal — an instantaneous,
        # nonphysical CoM/momentum jump the standing policy never experiences in training
        # (nothing in mdp/events.py's disturbance classes ever writes joint *state*; goals
        # resample and the arm moves continuously from wherever it is) and that can't happen
        # on real hardware either. Starting each attempt from the arm's current pose is
        # exactly what standing trained against, so the fall metric stays faithful; the arm
        # policy reaching goal-to-goal (instead of always from default, as its own training
        # does) is the same condition StandingArmPolicyReachDisturbance exposes standing to.
        # Envs that just *fell* still restart from default implicitly — the env reset that
        # follows a termination resets joint states itself, which is a legitimate episode
        # boundary, not a mid-episode teleport.

    def step(self, standing_reset_this_step: torch.Tensor):
        ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        dist = torch.linalg.vector_norm(self.goal_world() - ee_pos, dim=-1)
        self.min_dist = torch.minimum(self.min_dist, dist.detach())
        self.attempt_steps += 1

        success = self.min_dist < GOAL_THRESHOLD_M
        newly_reached = success & (self.reach_step < 0)
        self.reach_step = torch.where(newly_reached, self.attempt_steps, self.reach_step)

        # Dwell until timeout (2026-07-16): an attempt now concludes ONLY at
        # max_steps_per_goal, never the instant the hand first crosses the success
        # threshold. The old behavior resampled a fresh goal immediately on first
        # crossing — presenting the standing policy with relentless back-to-back reach
        # motion that training explicitly does NOT contain: the disturbance's own
        # 2026-07-12 dwell fix holds every goal for the full 15s precisely because
        # instant re-cycling was unrealistic, and this tracker was never updated to
        # match. Found 2026-07-16 via the controlled height_reward comparison — same
        # checkpoint, same IK driver, same 60/1.5 gains: 1.2% falls native vs ~75% here,
        # a gap only a code-level train/deploy mismatch can explain. Reach *timing* is
        # preserved in steps_to_reach; success still means "got within threshold at any
        # point during the attempt".
        timeout = self.attempt_steps >= args_cli.max_steps_per_goal
        finished_naturally = timeout & ~standing_reset_this_step

        self._flush(finished_naturally, success)
        if bool(standing_reset_this_step.any().item()):
            self._flush(standing_reset_this_step, success & standing_reset_this_step)

    def _flush(self, mask: torch.Tensor, success: torch.Tensor):
        env_ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        env_step = int(self.base_env.common_step_counter)
        for env_id in env_ids.tolist():
            steps = max(int(self.attempt_steps[env_id].item()), 1)
            was_success = bool(success[env_id].item())
            reach_step = int(self.reach_step[env_id].item())
            timed_out = self.attempt_steps[env_id].item() >= args_cli.max_steps_per_goal
            # A reached-then-interrupted attempt keeps outcome "success" — the old
            # (pre-dwell) tracker would have concluded it as a success at first crossing,
            # so this preserves success-rate comparability; the fall itself is still
            # counted in the standing CSV.
            outcome = "success" if was_success else ("timeout" if timed_out else "interrupted")
            goal_local = self.goal_local[env_id].tolist()
            row = {
                "episode_index": int(self._episode_index[env_id].item()),
                "env_id": env_id,
                "env_step": env_step,
                "attempt_steps": steps,
                "steps_to_reach": reach_step if reach_step >= 0 else "",
                "outcome": outcome,
                "success": int(was_success),
                "min_dist_to_goal_cm": float(self.min_dist[env_id].item()) * 100.0,
                "goal_x_m": goal_local[0], "goal_y_m": goal_local[1], "goal_z_m": goal_local[2],
            }
            self._csv.write_row(row)
            self._episode_index[env_id] += 1
        self.start_new_attempts(env_ids)

    def close(self):
        self._csv.close()


# ---------------------------------------------------------------------------
# Bucket runners
# ---------------------------------------------------------------------------

def _force_command(base_env, vx: float, vy: float, wz: float):
    """Pin the base_velocity command term to a fixed value, same technique
    eval_walking.py uses — setting cfg.ranges to a single point means every
    resample (on the command manager's own schedule) still lands on the same
    value, so both the policy's observation *and* anything reading
    env.command_manager directly (e.g. WalkingMetricsCsvWrapper's drift calc)
    agree. Overwriting the observation tensor alone (the first draft of this
    script did this) only fixes what the policy sees — the wrapper reads the
    command manager independently, so its drift numbers would silently reflect
    whatever the command manager happened to resample on its own, not this
    bucket's intended fixed command."""
    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (vx, vx)
    command_term.cfg.ranges.lin_vel_y = (vy, vy)
    command_term.cfg.ranges.ang_vel_z = (wz, wz)


def _run_standing_still(base_env, bucket_dir, standing_policy, device):
    print("\n[Eval] --- Bucket 'standing_still' ---")
    os.makedirs(bucket_dir, exist_ok=True)
    _force_command(base_env, 0.0, 0.0, 0.0)
    _set_arm_gains(base_env, *_STANDING_ARM_GAIN, device)
    _set_event_disturbance(base_env, False)
    torch.manual_seed(args_cli.seed)

    # Hold the arms at default via the *simulated target only*, the same mechanism
    # standing's own training uses (StandingArmBlendJointPositionAction, now installed as
    # this env's action term in _build_env_cfg — see the comment there). Previously this
    # zeroed action[:, arm_action_indices] before calling .step(), which also corrupted the
    # "last_action" observation term the policy reads back next step (ActionManager stores
    # whatever raw tensor is passed to .step() verbatim, independent of what an action term
    # actually simulates) — the policy was being shown a hard-pinned 0.0 for its own arm
    # output every step, a value it never produced or saw in training. The buffer approach
    # below leaves the policy's raw output (whatever it naturally is) untouched in the
    # action tensor, so last_action stays representative of training, while the arm joints
    # still only ever get driven to their default pose.
    robot = base_env.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(_LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS)
    arm_joint_ids_t = torch.tensor(arm_joint_ids, dtype=torch.long, device=device)
    base_env._standing_arm_motion_joint_ids = arm_joint_ids_t
    base_env._standing_arm_motion_targets = robot.data.default_joint_pos[:, arm_joint_ids_t].clone()

    with torch.inference_mode():
        metrics_env = StandingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)
        obs, _ = wrapped_env.reset()
        # Reset can respawn envs with slightly different default joint pos buffers than at
        # construction time — re-snapshot after reset so the held target is accurate.
        base_env._standing_arm_motion_targets = robot.data.default_joint_pos[:, arm_joint_ids_t].clone()
        for _ in range(args_cli.steps_standing_still):
            action = standing_policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
    metrics_env._csv.close()
    metrics_env._write_joint_diagnostics()
    print(f"[Eval] standing_still done — {metrics_env.csv_path}")
    return metrics_env.csv_path


def _run_walking_straight(base_env, bucket_dir, walking_policy, device):
    print("\n[Eval] --- Bucket 'walking_straight' ---")
    os.makedirs(bucket_dir, exist_ok=True)
    _force_command(base_env, args_cli.forward_speed, 0.0, 0.0)
    _set_arm_gains(base_env, *_WALKING_ARM_GAIN, device)
    _set_event_disturbance(base_env, False)
    # Buckets share one base_env and run sequentially — standing_still (which runs right
    # before this one) sets env._standing_arm_motion_targets to hold arms at default via
    # StandingArmBlendJointPositionAction (see _build_env_cfg). Clear it here or that stale
    # buffer would keep overriding this bucket's arm joints too, instead of letting the
    # walking policy's own raw output through (walking never masked arm actions, even
    # before this action-term change).
    base_env._standing_arm_motion_targets = None
    base_env._standing_arm_motion_joint_ids = None
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        metrics_env = WalkingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)
        obs, _ = wrapped_env.reset()
        for _ in range(args_cli.steps_walking_straight):
            action = walking_policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
    metrics_env._csv.close()
    print(f"[Eval] walking_straight done — {metrics_env.csv_path}")
    return metrics_env.csv_path


def _run_standing_arm_event_reach(base_env, bucket_dir, standing_policy, device):
    """--arm_driver event: the reach condition produced by standing training's OWN
    disturbance term (attached in _build_env_cfg), zero arm-driving code in this loop.
    The term samples its own goals and sides per episode, so there's no per-attempt arm
    CSV — the standing fall rate is the measurement (vs. the same checkpoint's native
    eval under the identical term). Both arms at standing's training gain, exactly as in
    training (the term itself never touches gains unless match_deployment_arm_gains)."""
    print("\n[Eval] --- Bucket 'standing_arm_event_reach' (training's own disturbance term) ---")
    os.makedirs(bucket_dir, exist_ok=True)
    _force_command(base_env, 0.0, 0.0, 0.0)
    _set_arm_gains(base_env, *_STANDING_ARM_GAIN, device)
    _set_event_disturbance(base_env, True)
    torch.manual_seed(args_cli.seed)

    # Reach-accuracy logging (2026-07-16): the disturbance term tracks its distance to
    # every goal internally (_min_dist, per side) but nothing anywhere ever logged it —
    # neither in training nor in the native eval — so "does the IK-driven arm actually
    # reach its goals" has never been measured. Snapshot the term's per-side state each
    # step; a drop in _attempt_steps marks a goal ending (timeout resample or episode
    # reset), at which point the previous snapshot holds that goal's final min distance.
    term = base_env.event_manager.get_term_cfg("standing_arm_motion_disturbance").func
    sides = ("left", "right")
    goal_rows: list[dict] = []

    def _snapshot():
        return (
            {s: term._attempt_steps[s].clone() for s in sides},
            {s: term._min_dist[s].clone() for s in sides},
            {s: term._active[s].clone() for s in sides},
        )

    with torch.inference_mode():
        metrics_env = StandingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)
        obs, _ = wrapped_env.reset()
        prev_steps, prev_min, prev_active = _snapshot()
        for _ in range(args_cli.steps_arm_reach):
            action = standing_policy(obs)
            obs, _, dones, _ = wrapped_env.step(action)
            for s in sides:
                ended = term._attempt_steps[s] < prev_steps[s]
                if bool(ended.any().item()):
                    for env_id in ended.nonzero(as_tuple=False).squeeze(-1).tolist():
                        if bool(prev_active[s][env_id].item()):
                            goal_rows.append({
                                "env_id": env_id,
                                "side": s,
                                "min_dist_cm": float(prev_min[s][env_id].item()) * 100.0,
                                "goal_steps": int(prev_steps[s][env_id].item()),
                            })
            prev_steps, prev_min, prev_active = _snapshot()
    metrics_env._csv.close()
    metrics_env._write_joint_diagnostics()
    _set_event_disturbance(base_env, False)

    goals_csv = os.path.join(bucket_dir, "event_arm_goals.csv")
    with open(goals_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env_id", "side", "min_dist_cm", "goal_steps"])
        writer.writeheader()
        writer.writerows(goal_rows)
    for s in sides:
        dists = sorted(r["min_dist_cm"] for r in goal_rows if r["side"] == s)
        if not dists:
            print(f"[Eval] event reach accuracy ({s}): no completed goals")
            continue
        mean = sum(dists) / len(dists)
        p50 = dists[len(dists) // 2]
        p90 = dists[int(len(dists) * 0.9)]
        under3 = sum(1 for d in dists if d < 3.0) / len(dists)
        print(
            f"[Eval] event reach accuracy ({s}): goals={len(dists)} mean={mean:.1f}cm "
            f"p50={p50:.1f}cm p90={p90:.1f}cm  <3cm: {under3:.1%}"
        )
    print(f"[Eval] standing_arm_event_reach done — {metrics_env.csv_path} / {goals_csv}")
    return metrics_env.csv_path, None


def _run_standing_arm_reach(base_env, bucket_dir, standing_policy, arm_policy, device, side: str, edge: bool):
    bucket_name = os.path.basename(bucket_dir)
    print(f"\n[Eval] --- Bucket '{bucket_name}' (side={side}, edge={edge}) ---")
    os.makedirs(bucket_dir, exist_ok=True)

    robot = base_env.scene["robot"]
    other_side = "right" if side == "left" else "left"
    arm_joint_ids_robot, _ = robot.find_joints(_ARM_JOINTS[side])
    arm_joint_ids_robot = torch.tensor(arm_joint_ids_robot, dtype=torch.long, device=device)
    other_joint_ids_robot, _ = robot.find_joints(_ARM_JOINTS[other_side])
    other_joint_ids_robot = torch.tensor(other_joint_ids_robot, dtype=torch.long, device=device)
    ee_body_id, _ = robot.find_bodies(_EE_BODY[side])
    ee_body_id = ee_body_id[0]

    # Both arms' *simulated* targets are driven through env._standing_arm_motion_targets
    # (StandingArmBlendJointPositionAction, installed in _build_env_cfg) instead of by
    # masking the action tensor — see that function's comment. The active arm's slice is
    # overwritten every step below with the IK-derived target; the other arm's slice is
    # set once to default and never touched again. The policy's own raw output — for both
    # arms — still flows through into action_manager's stored action untouched, so
    # last_action stays representative of training.
    combined_joint_ids = torch.cat([arm_joint_ids_robot, other_joint_ids_robot])
    base_env._standing_arm_motion_joint_ids = combined_joint_ids
    n_active = arm_joint_ids_robot.numel()

    def sample_fn(num, dev):
        return _sample_goals(side, num, dev, edge=edge, edge_margin_m=args_cli.edge_margin_m)

    _force_command(base_env, 0.0, 0.0, 0.0)
    # Per-arm gain split (2026-07-14), matching g1_full_demo.py's _update_arm_gains
    # convention instead of the previous "both arms at 200/20": only the actively IK-driven
    # arm gets the arm checkpoint's own training gain; the held arm keeps standing's trained
    # gain. Hardware note (verified against ~/Elm/Code/unitree_sdk2_python): Unitree's own
    # g1_arm5/g1_arm7 SDK examples command the arms at kp=60/kd=1.5 — i.e. _STANDING_ARM_GAIN
    # *is* the hardware-realistic arm gain, and _ARM_IK_GAIN (200/20) is a sim-only crutch
    # the current arm checkpoint requires (its reach success regressed 86%->30% when
    # retrained at 60/1.5). Getting the arm policy to work at ~60/1.5 is a real sim2real
    # prerequisite, tracked separately — until then, deployment matches each policy's own
    # training gain, split per arm.
    # Active-arm gain now CLI-controlled (--active_arm_gain, default = the RL arm
    # checkpoint's 200/20); held arm keeps standing's trained gain either way.
    _set_arm_gains(base_env, *args_cli.active_arm_gain, device, joint_names=list(_ARM_JOINTS[side]))
    _set_arm_gains(base_env, *_STANDING_ARM_GAIN, device, joint_names=list(_ARM_JOINTS[other_side]))
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        metrics_env = StandingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)

        tracker = _ArmAttemptTracker(base_env, robot, arm_joint_ids_robot, ee_body_id, bucket_dir, device, sample_fn)
        ik_driver = (
            _IKArmDriver(base_env, robot, arm_joint_ids_robot, ee_body_id, device)
            if args_cli.arm_driver == "ik" else None
        )

        obs, _ = wrapped_env.reset()
        tracker.start_new_attempts(torch.arange(base_env.num_envs, device=device))
        base_env._standing_arm_motion_targets = torch.cat(
            [
                robot.data.default_joint_pos[:, arm_joint_ids_robot],
                robot.data.default_joint_pos[:, other_joint_ids_robot],
            ],
            dim=1,
        )

        for _ in range(args_cli.steps_arm_reach):
            action = standing_policy(obs)

            if ik_driver is not None:
                # Analytic IK: absolute targets straight from the tracker's root-frame
                # goal — no EMA filter/action scale (those replicate the RL arm's own
                # training conventions; the IK path replicates training's disturbance
                # conventions instead, which have their own clamp inside the driver).
                new_targets = ik_driver.compute_targets(tracker.goal_local)
            else:
                raw_delta = _compute_arm_delta(
                    side, arm_policy, robot, arm_joint_ids_robot, ee_body_id, tracker.goal_world()
                )
                delta = tracker.filter_action(raw_delta)
                current = robot.data.joint_pos[:, arm_joint_ids_robot]
                limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids_robot]
                step_delta = (delta * ARM_ACTION_SCALE).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
                new_targets = (current + step_delta).clamp(limits[:, :, 0], limits[:, :, 1])
            base_env._standing_arm_motion_targets[:, :n_active] = new_targets

            obs, _, dones, _ = wrapped_env.step(action)
            tracker.step(torch.as_tensor(dones, device=device, dtype=torch.bool))

    metrics_env._csv.close()
    metrics_env._write_joint_diagnostics()
    tracker.close()
    print(f"[Eval] {bucket_name} done — {metrics_env.csv_path} / {tracker.csv_path}")
    return metrics_env.csv_path, tracker.csv_path


# ---------------------------------------------------------------------------
# Merge per-bucket CSVs into one detailed.csv at the run root
# ---------------------------------------------------------------------------

_UNIFIED_FIELDS = [
    "bucket", "episode_index", "env_id", "env_step", "steps", "outcome",
    "fell", "max_tilt_deg", "heading_drift_deg", "lateral_drift_m",
    "success", "min_dist_to_goal_cm",
]


def _read_rows(path: str) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _merge_detailed(run_dir: str, standing_still_csv, walking_csv, arm_reach_results: list[tuple[str, str, str]]):
    """arm_reach_results: list of (bucket_name, standing_csv, attempt_csv), one per
    standing_arm_*_reach[_edge] bucket that ran."""
    out_path = os.path.join(run_dir, "detailed.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_UNIFIED_FIELDS)
        writer.writeheader()

        for r in _read_rows(standing_still_csv):
            writer.writerow({
                "bucket": "standing_still", "episode_index": r["episode_index"], "env_id": r["env_id"],
                "env_step": r["env_step"], "steps": r["episode_steps"], "outcome": r["outcome"],
                "fell": r["fell"], "max_tilt_deg": r["max_tilt_deg"],
            })
        for r in _read_rows(walking_csv):
            writer.writerow({
                "bucket": "walking_straight", "episode_index": r["episode_index"], "env_id": r["env_id"],
                "env_step": r["env_step"], "steps": r["episode_steps"], "outcome": r["outcome"],
                "fell": r["fell"], "max_tilt_deg": r["max_tilt_deg"],
                "heading_drift_deg": r["heading_drift_deg"], "lateral_drift_m": r["lateral_drift_m"],
            })
        for bucket_name, standing_csv, attempt_csv in arm_reach_results:
            for r in _read_rows(standing_csv):
                writer.writerow({
                    "bucket": f"{bucket_name}_stability", "episode_index": r["episode_index"],
                    "env_id": r["env_id"], "env_step": r["env_step"], "steps": r["episode_steps"],
                    "outcome": r["outcome"], "fell": r["fell"], "max_tilt_deg": r["max_tilt_deg"],
                })
            for r in _read_rows(attempt_csv):
                writer.writerow({
                    "bucket": bucket_name, "episode_index": r["episode_index"], "env_id": r["env_id"],
                    "env_step": r["env_step"], "steps": r["attempt_steps"], "outcome": r["outcome"],
                    "success": r["success"], "min_dist_to_goal_cm": r["min_dist_to_goal_cm"],
                })
    return out_path


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def _mean_abs(rows: list[dict], key: str) -> float | None:
    """Same as _mean, but averages |value|. Required for any signed distance/rotation
    metric (heading drift, lateral drift): those are equally likely to land positive or
    negative episode-to-episode, so a plain mean can land near zero even when every
    single episode drifted substantially — averaging the magnitude is what actually
    answers "how much does it drift," which a signed mean does not."""
    vals = [abs(float(r[key])) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def _summary_path_for(detailed_csv_path: str) -> str:
    assert detailed_csv_path.endswith("_detailed.csv")
    return detailed_csv_path[: -len("_detailed.csv")] + "_summary.csv"


def _write_bucket_summary(detailed_csv_path: str, fields: list[str], row: dict) -> str:
    out_path = _summary_path_for(detailed_csv_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    return out_path


_STANDING_STABILITY_FIELDS = [
    "episodes", "fall_rate", "mean_episode_steps", "mean_max_tilt_deg",
    "mean_min_base_height_m", "mean_max_lin_speed_m_s", "mean_max_ang_speed_rad_s",
    "mean_step_count", "mean_max_foot_air_time_s", "mean_max_abs_torso_deg",
]


def _standing_stability_summary_row(rows: list[dict]) -> dict | None:
    """Shared by every bucket's stability side (standing_still, walking's own tilt aside,
    and each standing_arm_*_reach[_edge]'s stability rows) — all come from
    StandingMetricsCsvWrapper's detailed CSV with identical columns, so using the same
    field set lets them be diffed side by side (that comparison is the whole point of
    the standing_arm_*_reach buckets: "does reaching destabilize standing, and does it
    matter which arm / whether the goal is in-range")."""
    if not rows:
        return None
    fall_rate = sum(int(r["fell"]) for r in rows) / len(rows)
    # Torso rotation column only exists in CSVs written after 2026-07-15 (see
    # StandingMetricsCsvWrapper) — guard so re-summarizing an older detailed CSV still works.
    mean_max_torso = _mean(rows, "max_abs_torso_deg")
    return {
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


_WALKING_FIELDS = [
    "episodes", "fall_rate", "mean_episode_steps", "mean_max_tilt_deg",
    "mean_lin_vel_track_err_m_s", "mean_ang_vel_track_err_rad_s", "mean_foot_slip_speed_m_s",
    "mean_abs_heading_drift_deg", "mean_abs_lateral_drift_m",
]


def _walking_summary_row(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    fall_rate = sum(int(r["fell"]) for r in rows) / len(rows)
    return {
        "episodes": len(rows),
        "fall_rate": f"{fall_rate:.4f}",
        "mean_episode_steps": f"{_mean(rows, 'episode_steps'):.1f}",
        "mean_max_tilt_deg": f"{_mean(rows, 'max_tilt_deg'):.2f}",
        "mean_lin_vel_track_err_m_s": f"{_mean(rows, 'mean_lin_vel_track_err_m_s'):.4f}",
        "mean_ang_vel_track_err_rad_s": f"{_mean(rows, 'mean_ang_vel_track_err_rad_s'):.4f}",
        "mean_foot_slip_speed_m_s": f"{_mean(rows, 'mean_foot_slip_speed_m_s'):.4f}",
        # abs-mean, not mean — see _mean_abs docstring. A policy that drifts left half
        # the time and right the other half would otherwise report ~0 drift here.
        "mean_abs_heading_drift_deg": f"{_mean_abs(rows, 'heading_drift_deg'):.2f}",
        "mean_abs_lateral_drift_m": f"{_mean_abs(rows, 'lateral_drift_m'):.3f}",
    }


_ARM_ATTEMPT_FIELDS = [
    "attempts", "success_rate_overall", "success_rate_concluded", "concluded_attempts",
    "interrupted_rate", "timeout_rate", "mean_steps_to_success", "mean_min_dist_to_goal_cm",
]


def _arm_attempt_summary_row(attempt_rows: list[dict]) -> dict | None:
    """"interrupted" = the standing policy fell mid-attempt, cutting the reach off before
    it could succeed or time out on its own terms — not a failure of the arm policy in the
    same sense as a timeout. Two different questions need two different denominators:
    "concluded" (excludes interrupted) answers "when the arm actually got to finish the
    attempt, how often did it succeed" — but interrupted attempts skew towards the
    harder/slower ones (falls take time to arrive at), so this rate alone is optimistic
    about the combined system. "overall" (every attempt, interrupted counted as failure)
    answers the real end-to-end question: "starting a reach while standing, how often does
    it actually complete successfully." Report both."""
    if not attempt_rows:
        return None
    concluded = [r for r in attempt_rows if r["outcome"] != "interrupted"]
    overall_success_rate = sum(int(r["success"]) for r in attempt_rows) / len(attempt_rows)
    row = {
        "attempts": len(attempt_rows),
        "success_rate_overall": f"{overall_success_rate:.4f}",
        "interrupted_rate": f"{sum(1 for r in attempt_rows if r['outcome'] == 'interrupted') / len(attempt_rows):.4f}",
        "timeout_rate": f"{sum(1 for r in attempt_rows if r['outcome'] == 'timeout') / len(attempt_rows):.4f}",
        "mean_min_dist_to_goal_cm": f"{_mean(attempt_rows, 'min_dist_to_goal_cm'):.2f}",
    }
    if concluded:
        concluded_success_rate = sum(int(r["success"]) for r in concluded) / len(concluded)
        row["success_rate_concluded"] = f"{concluded_success_rate:.4f}"
        row["concluded_attempts"] = len(concluded)
        succ = [r for r in concluded if int(r["success"]) == 1]
        if succ:
            # steps_to_reach = first threshold crossing (dwell-era CSVs, 2026-07-16+);
            # older CSVs concluded attempts at first crossing, so attempt_steps meant the
            # same thing there — fall back to it when the new column is absent/empty.
            steps_to_success = _mean(succ, "steps_to_reach")
            if steps_to_success is None:
                steps_to_success = _mean(succ, "attempt_steps")
            row["mean_steps_to_success"] = f"{steps_to_success:.1f}"
    return row


def _write_bucket_summaries(
    standing_still_csv, walking_csv, arm_reach_results: list[tuple[str, str, str]]
) -> list[str]:
    """One summary CSV per test, written next to its own *_detailed.csv (not one merged
    file) — each test's summary needs a different shape (fall rate + tilt vs. drift vs.
    reach success rate), and keeping them separate means every test reliably gets a
    summary instead of silently sharing a slot with another bucket in the same file.

    arm_reach_results: list of (bucket_name, standing_csv, attempt_csv), one per
    standing_arm_*_reach[_edge] bucket that ran."""
    written = []

    rows = _read_rows(standing_still_csv)
    row = _standing_stability_summary_row(rows)
    if row is not None:
        path = _write_bucket_summary(standing_still_csv, _STANDING_STABILITY_FIELDS, row)
        written.append(path)
        print(f"[Eval] standing_still: episodes={row['episodes']} fall_rate={float(row['fall_rate']):.2%} "
              f"mean_max_tilt_deg={row['mean_max_tilt_deg']}  -> {path}")

    rows = _read_rows(walking_csv)
    row = _walking_summary_row(rows)
    if row is not None:
        path = _write_bucket_summary(walking_csv, _WALKING_FIELDS, row)
        written.append(path)
        print(f"[Eval] walking_straight: episodes={row['episodes']} fall_rate={float(row['fall_rate']):.2%} "
              f"mean_abs_heading_drift_deg={row['mean_abs_heading_drift_deg']} "
              f"mean_abs_lateral_drift_m={row['mean_abs_lateral_drift_m']}  -> {path}")

    for bucket_name, standing_csv, attempt_csv in arm_reach_results:
        stab_rows = _read_rows(standing_csv)
        stab_row = _standing_stability_summary_row(stab_rows)
        if stab_row is not None:
            path = _write_bucket_summary(standing_csv, _STANDING_STABILITY_FIELDS, stab_row)
            written.append(path)
            print(f"[Eval] {bucket_name}_stability: episodes={stab_row['episodes']} "
                  f"fall_rate={float(stab_row['fall_rate']):.2%} "
                  f"mean_max_tilt_deg={stab_row['mean_max_tilt_deg']}  -> {path}")

        attempt_rows = _read_rows(attempt_csv)
        attempt_row = _arm_attempt_summary_row(attempt_rows)
        if attempt_row is not None:
            path = _write_bucket_summary(attempt_csv, _ARM_ATTEMPT_FIELDS, attempt_row)
            written.append(path)
            concluded_str = (
                f"{float(attempt_row['success_rate_concluded']):.2%}"
                if "success_rate_concluded" in attempt_row else "n/a"
            )
            print(
                f"[Eval] {bucket_name}: attempts={attempt_row['attempts']} "
                f"success_rate_overall={float(attempt_row['success_rate_overall']):.2%} "
                f"success_rate_concluded={concluded_str}  -> {path}"
            )

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_run_meta(run_dir: str):
    """run_meta.yaml — makes every output directory self-describing (2026-07-14: until
    now, mapping a `<timestamp>/` folder back to which checkpoints produced it required
    digging through whichever overnight log happened to invoke this script — the run
    folders themselves recorded nothing)."""
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except OSError:
        git_commit = None

    meta = {
        "script": os.path.abspath(__file__),
        "git_commit": git_commit,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoints": {
            "standing": args_cli.standing_checkpoint,
            "walking": args_cli.walking_checkpoint,
            "arm": args_cli.arm_checkpoint,
        },
        "arm_hidden_dims": args_cli.arm_hidden_dims,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "steps": {
            "standing_still": args_cli.steps_standing_still,
            "walking_straight": args_cli.steps_walking_straight,
            "arm_reach_per_bucket": args_cli.steps_arm_reach,
        },
        "max_steps_per_goal": args_cli.max_steps_per_goal,
        "edge_margin_m": args_cli.edge_margin_m,
        "skip_edge_bucket": args_cli.skip_edge_bucket,
        "forward_speed": args_cli.forward_speed,
        "arm_driver": args_cli.arm_driver,
        "arm_gains": {
            "standing_held_arm": list(_STANDING_ARM_GAIN),
            "active_arm": list(args_cli.active_arm_gain),
            "walking": list(_WALKING_ARM_GAIN),
        },
        "per_goal_arm_teleport": False,  # removed 2026-07-14 — see _ArmAttemptTracker.start_new_attempts
        "goal_dwell_until_timeout": True,  # 2026-07-16 — see _ArmAttemptTracker.step; pre-fix runs resampled on first success
    }
    meta_path = os.path.join(run_dir, "run_meta.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[Eval] Run metadata: {meta_path}")


def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = args_cli.output_root or os.path.join(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(output_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    _write_run_meta(run_dir)

    print(f"[Eval] Standing checkpoint : {args_cli.standing_checkpoint}")
    print(f"[Eval] Walking checkpoint  : {args_cli.walking_checkpoint}")
    print(f"[Eval] Arm checkpoint      : {args_cli.arm_checkpoint}  (drives both arms — right via mirroring)")
    print(f"[Eval] num_envs={args_cli.num_envs}  output={run_dir}")

    print("[Eval] Building simulation (once, reused across all buckets)...")
    env_cfg = _build_env_cfg()
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    print("[Eval] Loading policies...")
    agent_cfg = G1LocomotionFlatPPORunnerCfg()
    standing_policy = _load_loco_policy(base_env, agent_cfg, args_cli.standing_checkpoint, device)
    walking_policy = _load_loco_policy(base_env, agent_cfg, args_cli.walking_checkpoint, device)
    arm_policy = _load_arm_policy(device)

    standing_still_csv = _run_standing_still(
        base_env, os.path.join(run_dir, "standing_still"), standing_policy, device
    )
    walking_csv = _run_walking_straight(
        base_env, os.path.join(run_dir, "walking_straight"), walking_policy, device
    )

    if args_cli.arm_driver == "event":
        bucket_name = "standing_arm_event_reach"
        standing_csv, attempt_csv = _run_standing_arm_event_reach(
            base_env, os.path.join(run_dir, bucket_name), standing_policy, device
        )
        arm_reach_results = [(bucket_name, standing_csv, attempt_csv)]
    else:
        arm_reach_specs = [(side, False) for side in ("left", "right")]
        if not args_cli.skip_edge_bucket:
            arm_reach_specs += [(side, True) for side in ("left", "right")]

        arm_reach_results = []
        for side, edge in arm_reach_specs:
            bucket_name = f"standing_arm_{side}_reach" + ("_edge" if edge else "")
            standing_csv, attempt_csv = _run_standing_arm_reach(
                base_env, os.path.join(run_dir, bucket_name), standing_policy, arm_policy, device, side, edge
            )
            arm_reach_results.append((bucket_name, standing_csv, attempt_csv))

    base_env.close()

    detailed_path = _merge_detailed(run_dir, standing_still_csv, walking_csv, arm_reach_results)
    summary_paths = _write_bucket_summaries(standing_still_csv, walking_csv, arm_reach_results)

    print(f"\n[Eval] Detailed (merged): {detailed_path}")
    print("[Eval] Per-test summaries:")
    for path in summary_paths:
        print(f"[Eval]   {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
