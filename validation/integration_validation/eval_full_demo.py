# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, vectorized integration eval — runs the combined unified stand+walk policy
plus the 7-DOF arm policy together, scripted instead of interactive, across many
parallel envs, and logs metrics to CSV.

Rewritten 2026-07-21 for the 29dof pivot — replaces the 23dof-era version that used to
live at this exact path (preserved on the `23_dof` git branch) — see
`29dof_implementation_plan.md` Phase 4. Same overall methodology and bucket/CSV
structure as before (this is deliberately NOT a from-scratch redesign — the
interrupted/timeout/success semantics, the goal-dwell-until-timeout tracker, the
per-bucket stability + reach-accuracy summary split, the mirror-driven right arm, all
carry forward essentially unchanged), but with real, load-bearing architecture changes:

1. **One unified `loco_policy`**, not separate standing/walking checkpoints — the 29dof
   recipe covers both regimes with one policy (near-zero commanded velocity IS
   "standing"). This removes the old per-mode arm-gain-by-active-policy logic and the
   whole locomotion-side mode-switch concept entirely — a real simplification, not
   something ported around.
2. **7-DOF arm** (was 5-DOF) — joint lists/EE body names/observation width updated
   throughout (see `g1_arm_env.py`'s module docstring for the DOF breakdown).
3. **`--arm_driver ik`/`event` modes REMOVED** (were present in the 23dof-era version).
   Both depended on `StandingArmIKReachDisturbance` (a real analytic-IK training-
   disturbance mechanism) — which has NOT been ported to the new
   `g1_locomotion/mdp/events.py` yet (blocked on Phase 2's IK-reach variant, see the
   plan doc; only the scripted `ArmMotionDisturbance` curriculum exists so far). Faking
   support for those modes would silently test nothing meaningful, so they're gone
   rather than stubbed — re-add once that disturbance class exists.
4. **New bucket: `walk_stop_reach`** (2026-07-21, user request) — command forward
   velocity, then command a stop, and start an arm reach *immediately* on the stop
   command with no settling delay. This is deliberately reproducing a known historical
   failure mode (documented in the 23dof-era `g1_full_demo.py`'s own
   `--wait_arm_rest` flag, which existed specifically to work around momentum-carryover
   instability at a stand/walk mode boundary) rather than avoiding it — the point of
   this bucket is to measure how bad it is on the new unified policy before deciding
   whether a similar mitigation is needed here too.
5. **`ArmDisturbanceBlendJointPositionAction` reused as-is**, not rebuilt — it already
   overrides simulated arm targets from `env._arm_motion_targets`/`_arm_motion_joint_ids`
   generically (doesn't care whether the training curriculum or this eval script is the
   one writing those buffers each step), same pattern the 23dof-era
   `StandingArmBlendJointPositionAction` served for both training AND
   `g1_full_demo.py`/this script.

Bucket list, each a single continuous condition (no mode transitions within a bucket
except `walk_stop_reach`, whose entire point IS a transition):

- standing_still  — zero command, arms held at default. Fall rate + tilt with no input.
- walking_straight — constant forward command, arms held at default. Fall rate + drift.
- walk_stop_reach — forward command, then an abrupt stop + immediate arm reach, no
  settling delay. Fall rate specifically around that transition is the measurement.
- arm_{left,right}_reach[_edge] — zero command, one arm continuously cycling through
  goals while the other is held at default. Arm success rate + steps-to-success, plus
  fall rate/tilt as a free byproduct (answers "does reaching destabilize the policy").
  Four of these by default (left/right x in-range/edge); --skip_edge_bucket drops it to
  two (left/right, in-range only).

  Goal sampling per bucket:
    in-range — uniform within _GOAL_BOUNDS[side], the exact box g1_arm_env.py trains on.
    edge — "just outside, close to the trained range": exactly one axis per goal is
      pushed past its bound by a random amount in [0, --edge_margin_m], the other two
      axes stay in-range.

  Right-arm reaching has no separately-trained checkpoint. It runs the left-trained
  policy on a mirrored observation and mirrors the resulting action back (see
  mdp/symmetry.py) — not an approximation invented for this script.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/integration_validation/eval_full_demo.py \\
        --loco_checkpoint chosen_checkpoints/walking_latest.pt \\
        --arm_checkpoint chosen_checkpoints/arm_left_latest.pt \\
        --headless

Output (mirrors logs/'s timestamped-folder convention):
    validation/integration_validation/<YYYY-MM-DD_HH-MM-SS>/
        detailed.csv          — every episode/attempt row from every bucket, tagged by
                                 bucket, columns not relevant to a given bucket left blank.
        <bucket>/<name>_detailed.csv — full per-episode/attempt rows for exactly one test.
        <bucket>/<name>_summary.csv  — that same test's one-row aggregate.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Headless integration eval: unified stand+walk policy + arm policy, combined."
)
parser.add_argument(
    "--loco_checkpoint", type=str, required=True,
    help="Path to a trained unified stand+walk checkpoint (.pt) — either the base recipe "
    "or the arm-disturbance-trained variant.",
)
parser.add_argument(
    "--arm_checkpoint", type=str, required=True,
    help="Path to a trained left-arm checkpoint (.pt). Also drives right-arm reaching, via "
    "mirroring (see module docstring) — there is no separate right-arm checkpoint.",
)
parser.add_argument(
    "--arm_hidden_dims", type=int, nargs="+", default=None,
    help="Override arm actor/critic hidden dims. Default: None (use G1ArmLeftPPORunnerCfg's "
    "own default, [512, 256, 128] — wide-net is the default network for this task, not an "
    "opt-in variant). Only pass this if evaluating a checkpoint with a different network size.",
)
parser.add_argument(
    "--active_arm_gain", type=float, nargs=2, default=[40.0, 10.0], metavar=("KP", "KD"),
    help="PD gain written to the actively-reaching arm's joints in the reach buckets "
    "(default 40 10 = the arm checkpoint's own training gain, matching g1_arm_env.py's "
    "2026-07-22 real-hardware-gain fix — same value as _LOCO_ARM_GAIN below, since that "
    "fix intentionally made the arm task train at the real hardware/held gain rather than "
    "a softer experimental one). "
    "The held (non-reaching) arm and the loco policy's own arm joints during "
    "standing_still/walking_straight/walk_stop_reach always use the loco env's own stock "
    "gain (see _LOCO_ARM_GAIN) — never overridden.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Parallel envs per bucket.")
parser.add_argument(
    "--steps_standing_still", type=int, default=3000, help="Control steps (50 Hz) for the standing_still bucket."
)
parser.add_argument(
    "--steps_walking_straight", type=int, default=3000, help="Control steps (50 Hz) for the walking_straight bucket."
)
parser.add_argument(
    "--steps_walk_before_stop", type=int, default=250,
    help="Control steps (50 Hz) of forward walking before the walk_stop_reach bucket commands "
    "a stop + starts the arm reach (250 @ 50Hz = 5s — enough to reach steady-state gait).",
)
parser.add_argument(
    "--steps_reach_after_stop", type=int, default=1500,
    help="Control steps (50 Hz) of arm reaching (starting immediately at the stop command, no "
    "settling delay) for the walk_stop_reach bucket.",
)
parser.add_argument(
    "--steps_arm_reach", type=int, default=3000,
    help="Control steps (50 Hz) for each arm_{left,right}_reach[_edge] bucket — applied per "
    "bucket, so total arm-reach runtime scales with how many of those buckets run.",
)
parser.add_argument(
    "--max_steps_per_goal", type=int, default=750,
    help="Control steps before an arm-reach attempt counts as a timeout (750 @ 50Hz = 15s).",
)
parser.add_argument(
    "--edge_margin_m", type=float, default=0.05,
    help="Max distance (m) a single axis is pushed past _GOAL_BOUNDS in the *_edge buckets.",
)
parser.add_argument(
    "--skip_edge_bucket", action="store_true",
    help="Only run the in-range arm_{left,right}_reach buckets, skip the *_edge ones.",
)
parser.add_argument(
    "--forward_speed", type=float, default=0.5, help="Commanded forward speed for walking buckets (m/s)."
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
import os
import subprocess

import yaml

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import G1ArmLeftPPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_ARM_JOINTS,
    _RIGHT_EE_BODY,
)
from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import mirror_arm_actions, mirror_arm_obs
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import G1LocomotionEnvCfg_PLAY
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp import ArmDisturbanceBlendJointPositionAction
from g1_locomotion.utils.metrics_wrappers import StandingMetricsCsvWrapper, WalkingMetricsCsvWrapper, _DualCsvWriter
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Same constants g1_full_demo.py uses for this exact action conversion — see
# g1_locomotion_env_cfg.py's ActionsCfg (scale=0.25, use_default_offset=True) and
# g1_arm_env.py's own action_scale/max_action_delta_per_step/action_filter_alpha.
ARM_ACTION_SCALE = 0.5
ARM_MAX_JOINT_DELTA_PER_STEP = 0.06
ARM_ACTION_FILTER_ALPHA = 0.25
GOAL_THRESHOLD_M = 0.02  # matches G1ArmEnvCfg.goal_threshold
# FIXED 2026-07-23 (same bug/fix as G1ArmEnvCfg.goal_hold_steps in g1_arm_env.py):
# this tracker's own success flag was "distance ever dipped under GOAL_THRESHOLD_M,
# even for one instant" (a running minimum), so a hand swinging past the goal
# mid-oscillation counted the same as a genuine, settled reach — confirmed wrong by
# direct visual test of a "successful" logged coordinate. Requires GOAL_HOLD_STEPS_LOCO
# CONSECUTIVE in-threshold steps before counting as reached. This tracker runs inside
# the LOCOMOTION env's control loop (50Hz: sim.dt=0.005, decimation=4), not the arm
# task's own 30Hz, so 25 steps here matches g1_arm_env.py's 15 steps @ 30Hz in real
# time (both ~0.5s).
GOAL_HOLD_STEPS_LOCO = 25

_ARM_JOINTS = {"left": _LEFT_ARM_JOINTS, "right": _RIGHT_ARM_JOINTS}
_EE_BODY = {"left": _LEFT_EE_BODY, "right": _RIGHT_EE_BODY}

# Arm gain when NOT actively being driven by the arm policy — the loco env's own stock
# gain for arm joints (g1_locomotion.assets.robots.unitree.UNITREE_G1_29DOF_CFG's
# N5020-16/W4010-25 groups: 40 stiffness / 10 damping, uniform across all 7 arm joints).
# Must match that asset config, or this eval silently tests a different held-arm gain
# than the loco policy actually trained under (same class of train/deploy mismatch
# lessons_learned.md warns about generally).
_LOCO_ARM_GAIN = (40.0, 10.0)
# Active-reach gain — matches g1_arm_env.py's own training gain, now 40/10 as of that
# file's 2026-07-22 real-hardware-gain fix (was 200/20 before that; this constant and
# --active_arm_gain's default were stale/unfixed until 2026-07-23 — see that flag's own
# help text). CLI-overridable via --active_arm_gain.


# ---------------------------------------------------------------------------
# Env / policy construction
# ---------------------------------------------------------------------------

def _build_env_cfg() -> G1LocomotionEnvCfg_PLAY:
    cfg = G1LocomotionEnvCfg_PLAY()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    # Restore training-time observation noise — the _PLAY config disables it for clean
    # visualization, same workaround every eval_*.py script in this repo applies.
    cfg.observations.policy.enable_corruption = True

    # ArmDisturbanceBlendJointPositionAction overrides only the *simulated* arm target
    # (from env._arm_motion_targets/_arm_motion_joint_ids, written live by each bucket
    # below), leaving the policy's own raw output flowing into action_manager's stored
    # action — and therefore into next step's last_action observation — untouched. This
    # is the exact same action term the training curriculum uses (mdp/events.py's
    # ArmMotionDisturbance); reused here unchanged, not rebuilt, since it's already
    # generic about who writes the override buffers.
    if hasattr(cfg.actions, "JointPositionAction"):
        cfg.actions.JointPositionAction.class_type = ArmDisturbanceBlendJointPositionAction

    return cfg


def _set_arm_gains(base_env, stiffness: float, damping: float, device, joint_names=None):
    """Live-update arm joints' PD gains via the PhysX view directly — the actuator
    cfg's stiffness/damping are fixed at spawn time; write_joint_stiffness_to_sim/
    write_joint_damping_to_sim let this reused-across-buckets simulation switch gains
    between buckets instead of being stuck with whatever _build_env_cfg() set once."""
    robot = base_env.scene["robot"]
    arm_joint_ids, _ = robot.find_joints(
        joint_names if joint_names is not None else _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS
    )
    arm_joint_ids_t = torch.tensor(arm_joint_ids, dtype=torch.long, device=device)
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=arm_joint_ids_t)
    robot.write_joint_damping_to_sim(damping, joint_ids=arm_joint_ids_t)


def _load_loco_policy(base_env, agent_cfg, checkpoint: str, device):
    """Load the unified stand+walk policy. Network shape (including the critic's) and
    obs-normalization flags are all inferred directly from the checkpoint's own
    state_dict, not trusted from agent_cfg — 2026-07-22 bug fix (found via
    g1_full_demo.py hitting the identical code path first). This task's CriticCfg is
    genuinely wider than PolicyCfg (495 vs 480 columns: it adds a privileged
    base_lin_vel term the actor never sees — see g1_locomotion_env_cfg.py), so the old
    fallback ``obs_groups = {"policy": [...], "critic": ["policy"]}`` silently built a
    critic sized for the actor's (narrower) input, causing a load_state_dict shape
    mismatch on critic.0.weight. agent_cfg.policy.actor_obs_normalization/
    critic_obs_normalization also didn't reliably match what the checkpoint was
    actually trained with (BasePPORunnerCfg never sets them explicitly, unlike the arm
    task's PPO cfg) — checking for the normalizer buffers' presence in the checkpoint
    directly sidesteps needing to know why that mismatch happens."""
    state = torch.load(checkpoint, map_location=device)["model_state_dict"]
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


def _load_arm_policy(device):
    obs_dim, action_dim = 32, 7
    agent_cfg = G1ArmLeftPPORunnerCfg()
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


_GOAL_PRECISION_M = 0.001  # round sampled goals to mm


def _round_goal(goals: torch.Tensor) -> torch.Tensor:
    """Round to the nearest mm — false precision beyond what the real arm can hit or
    training used anyway; doesn't change what's reachable or success/fail outcomes."""
    return torch.round(goals / _GOAL_PRECISION_M) * _GOAL_PRECISION_M


def _sample_goals(side: str, num: int, device, edge: bool = False, edge_margin_m: float = 0.05) -> torch.Tensor:
    """Uniform sample within _GOAL_BOUNDS[side], robot-local frame — same formula as
    G1ArmEnv._sample_goal_positions. edge=True samples just outside the trained range
    (one axis pushed past its bound by up to edge_margin_m, other two stay in-range)."""
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
    """Same 32-D layout g1_arm_env.py trains on (9-D base-state prefix + 7 joint_pos +
    7 joint_vel + 3 ee_pos + 3 goal + 3 error), including the observation-frame fix:
    error must be in the robot's *current* body frame, not raw world — g1_arm_env.py
    trains with a physically fixed, never-rotating root, so world frame is body frame
    throughout training; this env's robot can actually turn, so a raw world-frame error
    stops meaning "forward/lateral relative to the robot" the moment it isn't at ~0 yaw."""
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


def _compute_arm_delta(
    side: str, arm_policy, robot, arm_joint_ids: torch.Tensor, ee_body_id: int, goal_world: torch.Tensor
) -> torch.Tensor:
    """7-D joint-delta action for `side`'s arm, in that arm's own (unmirrored) frame.

    Left runs the left-trained policy directly. Right has no checkpoint of its own —
    mirror the real right-arm observation into the left-arm convention, run the
    left-trained policy on it, then mirror the resulting action back."""
    obs = _build_arm_obs(robot, arm_joint_ids, ee_body_id, goal_world)
    if side == "left":
        return arm_policy(obs)
    mirrored_obs = mirror_arm_obs(obs)
    delta_left_frame = arm_policy(mirrored_obs)
    return mirror_arm_actions(delta_left_frame)


# ---------------------------------------------------------------------------
# Arm-attempt tracker (arm_{left,right}_reach buckets) — cycles goals independent of
# the env's own episode boundary (fall/timeout).
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
        self._n_joints = len(arm_joint_ids)

        self._csv = _DualCsvWriter(log_dir, "arm", self._DETAILED_FIELDS, self._SUMMARY_FIELDS, write_summary=False)
        self.csv_path = self._csv.detailed_path

        self._episode_index = torch.full((self.n,), self._csv.next_episode_index(), dtype=torch.long, device=device)
        self.goal_local = torch.zeros((self.n, 3), device=device)
        self.attempt_steps = torch.zeros(self.n, dtype=torch.long, device=device)
        self.min_dist = torch.full((self.n,), float("inf"), device=device)
        self.reach_step = torch.full((self.n,), -1, dtype=torch.long, device=device)
        # Consecutive-steps-within-threshold counter, and a sticky "genuinely settled
        # at some point this attempt" flag — see GOAL_HOLD_STEPS_LOCO's own comment.
        self._hold_counter = torch.zeros(self.n, dtype=torch.long, device=device)
        self._ever_settled = torch.zeros(self.n, dtype=torch.bool, device=device)
        # EMA action-filter state, one per env — matches g1_arm_env.py's own
        # self.filtered_actions. Reset per-env at the start of each new attempt.
        self.filtered_action = torch.zeros((self.n, self._n_joints), device=device)

    def goal_world(self) -> torch.Tensor:
        """Recomputed every call from the robot's *current* pose — goal_local is a
        fixed offset from the torso, not a world-fixed point (see the 23dof-era
        version's identical fix for why: a world-fixed goal forces the arm to
        compensate for base tilt/drift it was never trained to handle, and that
        compensation itself destabilizes the base further — a runaway feedback loop)."""
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
        self._hold_counter[env_ids] = 0
        self._ever_settled[env_ids] = False

    def step(self, standing_reset_this_step: torch.Tensor):
        ee_pos = self.robot.data.body_pos_w[:, self.ee_body_id, :]
        dist = torch.linalg.vector_norm(self.goal_world() - ee_pos, dim=-1)
        self.min_dist = torch.minimum(self.min_dist, dist.detach())
        self.attempt_steps += 1

        # Consecutive-steps-within-threshold, gated at GOAL_HOLD_STEPS_LOCO — a single
        # instantaneous crossing (the old bug) no longer counts. _ever_settled is sticky
        # for the rest of the attempt once the hold requirement is first satisfied, so a
        # later drift away from the goal doesn't retroactively un-succeed a genuine reach.
        within_threshold = dist < GOAL_THRESHOLD_M
        self._hold_counter = torch.where(
            within_threshold, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        newly_settled = self._hold_counter >= GOAL_HOLD_STEPS_LOCO
        self._ever_settled = self._ever_settled | newly_settled
        success = self._ever_settled
        newly_reached = newly_settled & (self.reach_step < 0)
        self.reach_step = torch.where(newly_reached, self.attempt_steps, self.reach_step)

        # Dwell until timeout — an attempt concludes ONLY at max_steps_per_goal, never
        # the instant the hand first crosses the success threshold (matches training's
        # own disturbance dwell behavior). Reach *timing* is preserved in
        # steps_to_reach; success now means "held within threshold for
        # GOAL_HOLD_STEPS_LOCO consecutive steps at some point," not just a touch.
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
    """Pin the base_velocity command term to a fixed value.

    Writes BOTH `cfg.ranges` (so every future resample — including the one every
    `reset()` triggers — still lands on this value, and anything reading the ranges
    directly agrees) AND the command term's own live buffer, `vel_command_b`, directly
    (2026-07-21, real bug fix: the walk_stop_reach bucket calls this mid-rollout, with
    no reset() in between the walk and stop phases — the whole point is testing a
    command change WITHOUT a reset boundary — so relying on cfg.ranges alone would
    silently keep the previous (forward-walking) command live until the base env's own
    ~10s resampling_time_range next happened to fire, defeating the bucket entirely.
    Every other call site here already runs right before a reset(), where this was
    already effectively true — writing it directly here too is a pure correctness fix,
    not a change to any of those buckets' behavior; see mdp/commands/velocity_command.py's
    `UniformVelocityCommand.vel_command_b` for the same buffer g1_full_demo.py's live
    keyboard control writes for the identical reason (history_length=5 observation
    stacking means the flat obs tensor can't be hand-patched directly anymore either)."""
    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (vx, vx)
    command_term.cfg.ranges.lin_vel_y = (vy, vy)
    command_term.cfg.ranges.ang_vel_z = (wz, wz)
    command_term.vel_command_b[:, 0] = vx
    command_term.vel_command_b[:, 1] = vy
    command_term.vel_command_b[:, 2] = wz
    # FOUND 2026-07-23 (g1_full_demo.py investigation, same root cause): UniformVelocity
    # Command._update_command() runs every step and force-zeroes vel_command_b for any
    # env flagged is_standing_env (a per-env coin flip, probability rel_standing_envs,
    # rolled at reset) — silently undoing the write above for whichever envs got
    # unlucky. Force it off for every env this call covers.
    command_term.is_standing_env[:] = False


def _hold_arms_at_default(base_env, robot, device):
    """Set env._arm_motion_targets/_arm_motion_joint_ids to hold both arms at default
    via ArmDisturbanceBlendJointPositionAction's simulated-target override, leaving the
    policy's own raw arm-column output untouched in the action tensor (and therefore in
    next step's last_action observation) — same rationale as every other use of this
    mechanism in this repo."""
    arm_joint_ids, _ = robot.find_joints(_LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS)
    arm_joint_ids_t = torch.tensor(arm_joint_ids, dtype=torch.long, device=device)
    base_env._arm_motion_joint_ids = arm_joint_ids_t
    base_env._arm_motion_targets = robot.data.default_joint_pos[:, arm_joint_ids_t].clone()
    return arm_joint_ids_t


def _run_standing_still(base_env, bucket_dir, loco_policy, device):
    print("\n[Eval] --- Bucket 'standing_still' ---")
    os.makedirs(bucket_dir, exist_ok=True)
    _force_command(base_env, 0.0, 0.0, 0.0)
    _set_arm_gains(base_env, *_LOCO_ARM_GAIN, device)
    torch.manual_seed(args_cli.seed)

    robot = base_env.scene["robot"]
    _hold_arms_at_default(base_env, robot, device)

    with torch.inference_mode():
        metrics_env = StandingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)
        obs, _ = wrapped_env.reset()
        # Reset can respawn envs with slightly different default joint pos buffers —
        # re-snapshot after reset so the held target is accurate.
        arm_joint_ids_t = _hold_arms_at_default(base_env, robot, device)
        for _ in range(args_cli.steps_standing_still):
            action = loco_policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
    metrics_env._csv.close()
    metrics_env._write_joint_diagnostics()
    print(f"[Eval] standing_still done — {metrics_env.csv_path}")
    return metrics_env.csv_path


def _run_walking_straight(base_env, bucket_dir, loco_policy, device):
    print("\n[Eval] --- Bucket 'walking_straight' ---")
    os.makedirs(bucket_dir, exist_ok=True)
    _force_command(base_env, args_cli.forward_speed, 0.0, 0.0)
    _set_arm_gains(base_env, *_LOCO_ARM_GAIN, device)
    # FIXED 2026-07-23 (user request): this used to set _arm_motion_targets/
    # _arm_motion_joint_ids to None, letting the loco policy's own RAW arm-column
    # output drive the robot directly. Under ArmDisturbanceBlendJointPositionAction
    # (which this checkpoint trained under), raw arm-column actions are never applied
    # to physics during training — only observed, never actuated — so this was testing
    # "arms while walking" against never-trained, unshaped output. That capability is
    # explicitly deferred (see 29dof_implementation_plan.md's deferred-items list),
    # not something any current checkpoint has practiced, so this bucket was
    # guaranteed to fail regardless of checkpoint quality — not a meaningful signal.
    # Hold arms at default instead, matching what training's own standing-only gate
    # already does for any env commanded to walk, and matching eval_walking.py's
    # forward_slow/forward_medium buckets (which show 0% fall on this exact checkpoint).
    robot = base_env.scene["robot"]
    _hold_arms_at_default(base_env, robot, device)
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        metrics_env = WalkingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)
        obs, _ = wrapped_env.reset()
        # Reset can respawn envs with slightly different default joint pos buffers —
        # re-snapshot after reset so the held target is accurate (same reasoning as
        # _run_standing_still's own second _hold_arms_at_default call).
        _hold_arms_at_default(base_env, robot, device)
        for _ in range(args_cli.steps_walking_straight):
            action = loco_policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
    metrics_env._csv.close()
    print(f"[Eval] walking_straight done — {metrics_env.csv_path}")
    return metrics_env.csv_path


def _run_walk_stop_reach(base_env, bucket_dir, loco_policy, arm_policy, device):
    """New bucket (2026-07-21, user request): walk, then stop, then immediately start
    an arm reach — no settling delay between the stop command and the reach starting.
    Deliberately reproduces a known historical failure mode (momentum carryover right
    at a stand/walk boundary, the exact thing the 23dof-era demo's --wait_arm_rest flag
    existed to work around) rather than avoiding it — the fall rate specifically in the
    steps right after the stop command is the measurement. Left arm only (this bucket
    is about the walk/reach transition, not arm-side coverage — arm_left_reach/
    arm_right_reach already cover that)."""
    print("\n[Eval] --- Bucket 'walk_stop_reach' ---")
    os.makedirs(bucket_dir, exist_ok=True)

    robot = base_env.scene["robot"]
    side = "left"
    arm_joint_ids_robot, _ = robot.find_joints(_ARM_JOINTS[side])
    arm_joint_ids_robot = torch.tensor(arm_joint_ids_robot, dtype=torch.long, device=device)
    other_joint_ids_robot, _ = robot.find_joints(_ARM_JOINTS["right"])
    other_joint_ids_robot = torch.tensor(other_joint_ids_robot, dtype=torch.long, device=device)
    ee_body_id, _ = robot.find_bodies(_EE_BODY[side])
    ee_body_id = ee_body_id[0]
    combined_joint_ids = torch.cat([arm_joint_ids_robot, other_joint_ids_robot])
    n_active = arm_joint_ids_robot.numel()

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        metrics_env = StandingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)

        # Phase 1: walk. Arms held at default via the override, loco arm gain.
        _force_command(base_env, args_cli.forward_speed, 0.0, 0.0)
        _set_arm_gains(base_env, *_LOCO_ARM_GAIN, device)
        base_env._arm_motion_joint_ids = combined_joint_ids
        obs, _ = wrapped_env.reset()
        base_env._arm_motion_targets = robot.data.default_joint_pos[:, combined_joint_ids].clone()
        for _ in range(args_cli.steps_walk_before_stop):
            action = loco_policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

        # Phase 2: stop + reach, starting on the SAME step the stop command is issued —
        # no wait for velocity to settle. Active arm switches to its own training gain
        # immediately too (matching how a real deployment would switch gain the instant
        # a reach is commanded, not after some grace period).
        _force_command(base_env, 0.0, 0.0, 0.0)
        _set_arm_gains(base_env, *args_cli.active_arm_gain, device, joint_names=list(_ARM_JOINTS[side]))
        _set_arm_gains(base_env, *_LOCO_ARM_GAIN, device, joint_names=list(_ARM_JOINTS["right"]))

        tracker = _ArmAttemptTracker(base_env, robot, arm_joint_ids_robot, ee_body_id, bucket_dir, device,
                                      lambda num, dev: _sample_goals(side, num, dev))
        tracker.start_new_attempts(torch.arange(base_env.num_envs, device=device))

        for _ in range(args_cli.steps_reach_after_stop):
            action = loco_policy(obs)
            raw_delta = _compute_arm_delta(side, arm_policy, robot, arm_joint_ids_robot, ee_body_id, tracker.goal_world())
            delta = tracker.filter_action(raw_delta)
            current = robot.data.joint_pos[:, arm_joint_ids_robot]
            limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids_robot]
            step_delta = (delta * ARM_ACTION_SCALE).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
            new_targets = (current + step_delta).clamp(limits[:, :, 0], limits[:, :, 1])
            base_env._arm_motion_targets[:, :n_active] = new_targets

            obs, _, dones, _ = wrapped_env.step(action)
            tracker.step(torch.as_tensor(dones, device=device, dtype=torch.bool))

    metrics_env._csv.close()
    metrics_env._write_joint_diagnostics()
    tracker.close()
    print(f"[Eval] walk_stop_reach done — {metrics_env.csv_path} / {tracker.csv_path}")
    return metrics_env.csv_path, tracker.csv_path


def _run_arm_reach(base_env, bucket_dir, loco_policy, arm_policy, device, side: str, edge: bool):
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

    combined_joint_ids = torch.cat([arm_joint_ids_robot, other_joint_ids_robot])
    base_env._arm_motion_joint_ids = combined_joint_ids
    n_active = arm_joint_ids_robot.numel()

    def sample_fn(num, dev):
        return _sample_goals(side, num, dev, edge=edge, edge_margin_m=args_cli.edge_margin_m)

    _force_command(base_env, 0.0, 0.0, 0.0)
    # Per-arm gain split: only the actively-reaching arm gets the arm checkpoint's own
    # training gain; the held arm and everything else keeps the loco policy's own gain.
    _set_arm_gains(base_env, *args_cli.active_arm_gain, device, joint_names=list(_ARM_JOINTS[side]))
    _set_arm_gains(base_env, *_LOCO_ARM_GAIN, device, joint_names=list(_ARM_JOINTS[other_side]))
    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        metrics_env = StandingMetricsCsvWrapper(base_env, bucket_dir, write_summary=False)
        wrapped_env = RslRlVecEnvWrapper(metrics_env)

        tracker = _ArmAttemptTracker(base_env, robot, arm_joint_ids_robot, ee_body_id, bucket_dir, device, sample_fn)

        obs, _ = wrapped_env.reset()
        tracker.start_new_attempts(torch.arange(base_env.num_envs, device=device))
        base_env._arm_motion_targets = torch.cat(
            [
                robot.data.default_joint_pos[:, arm_joint_ids_robot],
                robot.data.default_joint_pos[:, other_joint_ids_robot],
            ],
            dim=1,
        )

        for _ in range(args_cli.steps_arm_reach):
            action = loco_policy(obs)

            raw_delta = _compute_arm_delta(
                side, arm_policy, robot, arm_joint_ids_robot, ee_body_id, tracker.goal_world()
            )
            delta = tracker.filter_action(raw_delta)
            current = robot.data.joint_pos[:, arm_joint_ids_robot]
            limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids_robot]
            step_delta = (delta * ARM_ACTION_SCALE).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
            new_targets = (current + step_delta).clamp(limits[:, :, 0], limits[:, :, 1])
            base_env._arm_motion_targets[:, :n_active] = new_targets

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


def _merge_detailed(run_dir: str, standing_still_csv, walking_csv, walk_stop_reach, arm_reach_results):
    """arm_reach_results / walk_stop_reach: list of (bucket_name, standing_csv, attempt_csv)."""
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
        for bucket_list in (walk_stop_reach, arm_reach_results):
            for bucket_name, standing_csv, attempt_csv in bucket_list:
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
    """Averages |value| — required for any signed distance/rotation metric (heading
    drift, lateral drift): a plain mean can land near zero even when every episode
    drifted substantially, if it happened to drift both ways about equally often."""
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
    """Shared by every zero-command-ish bucket's stability side — all come from
    StandingMetricsCsvWrapper's detailed CSV with identical columns, so using the same
    field set lets them be diffed side by side (the whole point of the arm-reach
    buckets: "does reaching destabilize the policy, and does it matter which arm /
    whether the goal is in-range")."""
    if not rows:
        return None
    fall_rate = sum(int(r["fell"]) for r in rows) / len(rows)
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
        "mean_abs_heading_drift_deg": f"{_mean_abs(rows, 'heading_drift_deg'):.2f}",
        "mean_abs_lateral_drift_m": f"{_mean_abs(rows, 'lateral_drift_m'):.3f}",
    }


_ARM_ATTEMPT_FIELDS = [
    "attempts", "success_rate_overall", "success_rate_concluded", "concluded_attempts",
    "interrupted_rate", "timeout_rate", "mean_steps_to_success", "mean_min_dist_to_goal_cm",
]


def _arm_attempt_summary_row(attempt_rows: list[dict]) -> dict | None:
    """"interrupted" = the policy fell mid-attempt, cutting the reach off before it
    could succeed or time out on its own terms. "concluded" (excludes interrupted)
    answers "when the arm actually got to finish the attempt, how often did it
    succeed" — but interrupted attempts skew toward the harder/slower ones (falls take
    time to arrive at), so this rate alone is optimistic about the combined system.
    "overall" (every attempt, interrupted counted as failure) answers the real
    end-to-end question. Report both."""
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
            steps_to_success = _mean(succ, "steps_to_reach")
            if steps_to_success is None:
                steps_to_success = _mean(succ, "attempt_steps")
            row["mean_steps_to_success"] = f"{steps_to_success:.1f}"
    return row


def _write_bucket_summaries(standing_still_csv, walking_csv, walk_stop_reach, arm_reach_results) -> list[str]:
    """One summary CSV per test, written next to its own *_detailed.csv."""
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

    for bucket_name, standing_csv, attempt_csv in walk_stop_reach + arm_reach_results:
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
    """run_meta.yaml — makes every output directory self-describing."""
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
            "loco": args_cli.loco_checkpoint,
            "arm": args_cli.arm_checkpoint,
        },
        "arm_hidden_dims": args_cli.arm_hidden_dims,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "steps": {
            "standing_still": args_cli.steps_standing_still,
            "walking_straight": args_cli.steps_walking_straight,
            "walk_before_stop": args_cli.steps_walk_before_stop,
            "reach_after_stop": args_cli.steps_reach_after_stop,
            "arm_reach_per_bucket": args_cli.steps_arm_reach,
        },
        "max_steps_per_goal": args_cli.max_steps_per_goal,
        "edge_margin_m": args_cli.edge_margin_m,
        "skip_edge_bucket": args_cli.skip_edge_bucket,
        "forward_speed": args_cli.forward_speed,
        "arm_gains": {
            "loco_held_arm": list(_LOCO_ARM_GAIN),
            "active_arm": list(args_cli.active_arm_gain),
        },
        "goal_dwell_until_timeout": True,
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

    print(f"[Eval] Loco checkpoint : {args_cli.loco_checkpoint}")
    print(f"[Eval] Arm checkpoint  : {args_cli.arm_checkpoint}  (drives both arms — right via mirroring)")
    print(f"[Eval] num_envs={args_cli.num_envs}  output={run_dir}")

    print("[Eval] Building simulation (once, reused across all buckets)...")
    env_cfg = _build_env_cfg()
    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    device = base_env.device

    print("[Eval] Loading policies...")
    agent_cfg = BasePPORunnerCfg()
    loco_policy = _load_loco_policy(base_env, agent_cfg, args_cli.loco_checkpoint, device)
    arm_policy = _load_arm_policy(device)

    standing_still_csv = _run_standing_still(
        base_env, os.path.join(run_dir, "standing_still"), loco_policy, device
    )
    walking_csv = _run_walking_straight(
        base_env, os.path.join(run_dir, "walking_straight"), loco_policy, device
    )

    walk_stop_standing_csv, walk_stop_attempt_csv = _run_walk_stop_reach(
        base_env, os.path.join(run_dir, "walk_stop_reach"), loco_policy, arm_policy, device
    )
    walk_stop_reach = [("walk_stop_reach", walk_stop_standing_csv, walk_stop_attempt_csv)]

    arm_reach_specs = [(side, False) for side in ("left", "right")]
    if not args_cli.skip_edge_bucket:
        arm_reach_specs += [(side, True) for side in ("left", "right")]

    arm_reach_results = []
    for side, edge in arm_reach_specs:
        bucket_name = f"arm_{side}_reach" + ("_edge" if edge else "")
        standing_csv, attempt_csv = _run_arm_reach(
            base_env, os.path.join(run_dir, bucket_name), loco_policy, arm_policy, device, side, edge
        )
        arm_reach_results.append((bucket_name, standing_csv, attempt_csv))

    base_env.close()

    detailed_path = _merge_detailed(run_dir, standing_still_csv, walking_csv, walk_stop_reach, arm_reach_results)
    summary_paths = _write_bucket_summaries(standing_still_csv, walking_csv, walk_stop_reach, arm_reach_results)

    print(f"\n[Eval] Detailed (merged): {detailed_path}")
    print("[Eval] Per-test summaries:")
    for path in summary_paths:
        print(f"[Eval]   {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
