# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof arm policy via RL — DirectRLEnv.

Deliberately not called "IK" anywhere (2026-07-21, user request): the policy learns to
move the palm(s) to randomly sampled 3-D goal positions using delta joint-position
commands on the 7-DOF arm(s) — pure RL joint-space control, no inverse-kinematics solve
anywhere in this file or in the deployed policy. "IK" is reserved for the actual
analytic-IK-based training disturbance generator used elsewhere (the not-yet-ported
``StandingArmIKReachDisturbance``-equivalent in ``g1_locomotion/mdp/events.py`` —
see ``29dof_implementation_plan.md``), which is a genuinely different mechanism and must
not be confused with this policy.

Adapted 2026-07-21 from the 23dof-era ``g1_locomotion/tasks/manager_based/g1_arm/
g1_arm_env.py`` (5-DOF arm, no wrist) — see ``29dof_implementation_plan.md``'s Phase 3.
Structural DOF change, not just "add 2 joints": the 29dof model's elbow is a single
``elbow_joint`` (not the old ``elbow_pitch``/``elbow_roll`` pair), and 3 wrist joints
(roll/pitch/yaw) are added — net 5 -> 7 DOF per arm.

Goal stays position-only (3-D x/y/z), per explicit user decision 2026-07-21 — the wrist
exists for orientation control, which is an intentional, planned follow-up once this
position-only version is validated, not something to build now. That means the 2 extra
wrist DOF (beyond wrist_roll, which was arguably already needed to reach some
orientations even in 5-DOF-equivalent terms) are additional redundant DOF for a
position-only goal — expect the manipulator-redundancy issue documented in
``lessons_learned.md``/``phase_logs/phase_2.md`` (a policy landing on one of several
valid joint configurations for the same goal, inconsistently) to be at least as present
here as it was at 5-DOF, likely more so.

Single arm (arm="left" or "right"):
    Observation (32-D): base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) |
                         joint_pos(7) | joint_vel(7) | ee_pos(3) | goal(3) | error(3)
    Action      ( 7-D): delta joint targets for the active arm

Both arms (arm="both"):
    Observation (64-D): [left 32-D] + [right 32-D] (base-state block duplicated per arm
                         so each arm's block stays independently mirror-transformable)
    Action      (14-D): [left 7-D]  + [right 7-D]
    Episode terminates when BOTH palms reach their respective goals.

Design choices carried forward directly from the 23dof phase's hard-won findings
(``lessons_learned.md``), not re-litigated from scratch:
- Wide network ([512,256,128], not [256,128,64]) as the DEFAULT here, not an opt-in
  experiment — it reproducibly improved precision/tail behavior twice in the 23dof phase
  (once at the ~55% baseline, once at the ~85% reachability-fixed baseline), so there's
  no reason to re-discover that from a narrower default.
- Left-right symmetry-augmented PPO training (see ``mdp/symmetry.py``) from the start,
  not added reactively after observing an asymmetric policy.
- Observation noise + startup domain randomization (actuator gains, joint friction/
  armature) on from the start, matching G1FlatEnvCfg-family magnitudes.
- A null-space regularization term (uniform per-joint weight here, NOT the 23dof
  version's elbow_pitch-specific 3x weighting — that joint doesn't exist in this form
  anymore, and there's no data yet on which of the 7 joints plays the analogous role in
  a redundancy branch-selection problem here; revisit once training data shows it).
- A torso-proximity penalty (end-effector vs. torso_link) plus enabled_self_collisions,
  same two-layer defense as before.

Explicitly NOT carried forward without re-verification:
- ``_GOAL_BOUNDS`` below reuses the 23dof-era's reachability-validated numeric box as a
  STARTING HYPOTHESIS ONLY (position reach is governed mostly by upper-arm/forearm
  length, which the wrist addition shouldn't change) — but per
  ``29dof_implementation_plan.md`` Phase 3.2, this MUST be re-validated via
  ``validation/check_arm_reachability.py`` before trusting it for a real
  training run. Do not skip that step because the box "should" still be reachable.
- Arm actuator gains (200/20) are the 23dof-era's own RL-experiment-derived value (real
  SDK gains starved the arm of gravity-compensation torque it needs in this pure-PD sim
  — see that finding's own note below), carried forward as a starting point, not
  re-verified for the 7-DOF arm's different mass distribution.
- Per-joint asymmetric joint-limit margins (the old elbow_pitch-specific 1%/5% split) —
  replaced with a uniform 5% margin for all 7 joints until real data says otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul
from isaaclab.utils.noise import UniformNoiseCfg, uniform_noise

from g1_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_CFG

# ---------------------------------------------------------------------------
# Domain randomization events
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Startup-only domain randomization for the arm task.

    Only the arm actuator group is randomized (payload/hardware variance on the joints
    actually being controlled) — legs/torso/waist are held rigid (see A5-equivalent in
    __post_init__) and don't need it. No friction/mass/CoM randomization on bodies: this
    task has no contact interactions at all (pure point-goal reaching).
    """

    arm_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
            ),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    arm_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"]
            ),
            "friction_distribution_params": (0.0, 0.03),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "abs",
        },
    )


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------


@configclass
class G1ArmEnvCfg(DirectRLEnvCfg):
    """Base configuration for the G1 29dof arm-reaching task."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    episode_length_s: float = 10.0
    decimation: int = 2  # 30 Hz control

    # Observation / action / state dims
    # 9-D base-state prefix (base_lin_vel, base_ang_vel, projected_gravity) + 23-D per arm
    # (joint_pos(7) + joint_vel(7) + ee_pos(3) + goal(3) + error(3)) = 32-D per arm.
    observation_space: int = 32
    action_space: int = 7
    state_space: int = 0  # no separate critic state; must be set (None crashes serialization)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0)

    # Domain randomization
    events: EventCfg = EventCfg()

    # Which arm(s) to train: "left", "right", or "both"
    arm: str = "left"

    # Reward scales
    position_reward_scale: float = 10.0
    # REVERTED 2026-07-24 back to 0.0 (the original) — briefly set to 3.0 alongside an
    # entropy_coef bump (see rsl_rl_ppo_cfg.py) as a combined guess to fix a plateaued
    # training run. That combination caused Policy/mean_noise_std to explode (1.0 -> 57
    # over 2000 iterations) and Train/mean_reward to get steadily worse — a real
    # regression, not an improvement, and bundling both changes together means it's
    # unclear which one (or their interaction) caused it. Superseded by
    # G1ArmLeftAblationExpScaleEnvCfg below, which tests this exact value (3.0) in
    # isolation, entropy_coef left at baseline 0.0.
    position_reward_exp_scale: float = 0.0
    position_reward_exp_sigma: float = 0.05  # 5 cm
    action_smoothness_scale: float = 0.01
    joint_limit_penalty_scale: float = 1.0
    goal_reached_bonus: float = 50.0
    goal_threshold: float = 0.02  # 2 cm
    # FIXED 2026-07-23 (user request): previously "success" meant the hand's distance to
    # goal dropped under goal_threshold on ANY single step — the episode terminated that
    # exact instant, so a hand swinging past the target mid-oscillation counted exactly
    # the same as one that actually converged and settled there. Confirmed by direct
    # visual test: a checkpoint's "successful" reach (per this old definition) visibly
    # orbited the target without ever stopping. goal_hold_steps requires the goal to
    # stay reached for this many CONSECUTIVE steps before termination/success fires —
    # 15 steps @ 30 Hz (this task's control rate) = 0.5s, long enough to rule out a
    # fly-through, short enough not to make training impractically slower. This also
    # fixes goal_reached_bonus's own incentive: previously it could only ever apply for
    # at most one step before the episode ended, so it never actually rewarded holding;
    # now it accumulates for every step of a genuine hold.
    goal_hold_steps: int = 15
    action_scale: float = 0.5
    action_filter_alpha: float = 0.25
    max_action_delta_per_step: float = 0.06

    # Fraction of the (real hardware) joint range each reset randomizes the arm's
    # starting position within, centered on the hardware range's own midpoint.
    reset_range_fraction: float = 0.15

    # Torso-proximity penalty (soft defense against the end-effector entering
    # torso_link; complements enabled_self_collisions below).
    torso_proximity_margin_m: float = 0.12
    torso_proximity_penalty_scale: float = 2.0

    # Root-pose wobble — see _apply_root_wobble. Bounded, slow, per-env-randomized
    # oscillation so the arm policy is actually exposed to (and must observe/adapt to) a
    # moving reference frame, instead of a perfectly static one it's never seen a
    # disturbed version of. Root itself stays physically fixed (fix_root_link=True) —
    # this only changes what base_ang_vel/projected_gravity *report*, not anything
    # physical. Values carried forward from the 23dof-era task's own final, widened
    # amplitudes (calibrated against measured integration-eval tilt, not re-derived here).
    enable_root_wobble: bool = True
    root_wobble_enable_step: int = 30_000
    root_wobble_max_roll_rad: float = 0.35  # ~20 deg
    root_wobble_max_pitch_rad: float = 0.35  # ~20 deg
    root_wobble_max_yaw_rate_rad_s: float = 0.5
    root_wobble_max_lin_vel_mps: float = 0.3
    root_wobble_freq_hz_range: tuple[float, float] = (0.1, 0.3)

    # Observation noise magnitudes — matches G1FlatEnvCfg-family magnitudes (same values
    # this repo's 23dof-era arm task used, and the same ones Unitree's own 29dof
    # locomotion recipe uses for the equivalent terms — see
    # 29dof_implementation_plan.md's sim2real notes).
    base_lin_vel_noise: float = 0.1
    base_ang_vel_noise: float = 0.2
    projected_gravity_noise: float = 0.05
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 1.5

    # Null-space regularization — a soft tiebreaker among the multiple joint
    # configurations that reach the same (x, y, z) goal (manipulator redundancy — 7 DOF
    # for a 3-DOF position-only goal, even more redundant than the old 5-DOF version).
    # Uniform weight across all 7 joints (NOT the 23dof-era's elbow_pitch-specific 3x —
    # that joint no longer exists in this form; revisit with per-joint weighting once
    # training data identifies which joint(s), if any, play the analogous
    # branch-selection role here).
    null_space_penalty_scale: float = 0.05

    # Goal workspace x-range override — None = use _GOAL_BOUNDS as-is. Kept for parity
    # with the 23dof-era task's stress-testing pattern; unused by default.
    goal_bounds_x_override: tuple[float, float] | None = None

    # ADDED 2026-07-24: exclude wrist_pitch/wrist_yaw from the RL-controlled joint set
    # (held rigidly at default instead — see G1ArmEnv.__init__/_apply_action). This
    # task's own module docstring already flags these two as "additional redundant DOF
    # for a position-only goal" beyond wrist_roll (which was arguably already needed at
    # 5-DOF-equivalent terms) — a real, anticipated manipulator-redundancy contributor,
    # not a new theory. The 23dof-era 5-DOF arm (no wrist at all) reached ~55% baseline
    # success before any reward/entropy tuning; our own 7-DOF task has been plateauing
    # around ~28%. Locking these 2 is a structural simplification orthogonal to the
    # reward-shape/entropy experiments (both of which turned out risky or inconclusive
    # — see g1_arm_env.py git history 2026-07-24), not a replacement for fixing those,
    # but untested and not contraindicated by anything found so far.
    lock_wrist_pitch_yaw: bool = False

    # Robot — G1 29dof asset, fixed base (this task doesn't need locomotion-grade
    # ground-contact fidelity; only arm joints are RL-actuated, everything else held
    # rigid by its own default PD gains, see __post_init__).
    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        super().__post_init__()
        # Root is NOT rigidly welded via a real PhysX fixed joint (fix_root_link=True
        # does exactly that) — the arm policy needs a base it can actually observe
        # moving (see root wobble above), and a real fixed joint can't be reprogrammed
        # at runtime. Instead the root is fully self-managed every control step
        # (_apply_root_wobble writes synthetic observation values only — see that
        # method's docstring for why an earlier, genuinely-free-root version of this
        # task in the 23dof phase turned into an unintended, un-winnable balance task
        # and was reverted).
        self.robot.spawn.articulation_props.fix_root_link = True

        # Self-collision — UnitreeUsdFileCfg already defaults this True, kept explicit
        # here for clarity/defensiveness (same "arm entering torso" structural fix the
        # 23dof-era task relied on) — NOT YET VISUALLY VERIFIED for this asset/task via
        # play.py, same caveat every untested config change in this repo carries.
        self.robot.spawn.articulation_props.enabled_self_collisions = True

        # This task never needs ground-contact resolution (nothing touches the ground)
        # — cut solver cost vs. the locomotion-tuned default.
        self.robot.spawn.articulation_props.solver_position_iteration_count = 4
        self.robot.spawn.articulation_props.solver_velocity_iteration_count = 1

        # Arm actuator gain — FIXED 2026-07-22 (user request) from 200/20 (the 23dof-era
        # task's own RL-experiment-derived value, carried forward unverified — its own
        # note: real SDK gains "starved the arm of gravity-compensation torque" at a pure
        # PD term with no feedforward) to 40/10, the REAL hardware gain — same value
        # assets/robots/unitree.py's N5020-16/W4010-25 groups use for these exact joints
        # (already reconciled against deploy.yaml line-by-line for the locomotion task,
        # see that file's own module docstring). This override was never covered by that
        # reconciliation — it's a separate code path the arm task's own robot spawn
        # replaces the shared config with, not something unitree.py's fix could reach.
        # If training struggles to hold precision at this softer gain, that's real signal
        # about what the actual hardware will need too (a plain PD drive has no gravity
        # feedforward regardless of sim vs. real), not a training inconvenience to dodge
        # by training against an easier gain than deployment will ever provide.
        self.robot.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            stiffness=40.0,
            damping=10.0,
        )
        # Legs/waist/feet stay at UNITREE_G1_29DOF_CFG's own stock gains (see
        # g1_locomotion.assets.robots.unitree) — already reasonably strong for a
        # "stay put" role, not overridden here.

        # Contact sensors not needed for this task.
        self.robot.spawn.activate_contact_sensors = False


@configclass
class G1ArmLeftEnvCfg(G1ArmEnvCfg):
    arm: str = "left"
    observation_space: int = 32
    action_space: int = 7


@configclass
class G1ArmLeftLockedWristEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-24: wrist_pitch/wrist_yaw excluded from RL control, held rigidly at
    default instead — see G1ArmEnvCfg.lock_wrist_pitch_yaw's own docstring. 5
    controlled joints (shoulder x3, elbow, wrist_roll), not 7 — observation/action
    dims shrink accordingly: 32 -> 28 (joint_pos/joint_vel each drop from 7 to 5
    elements), 7 -> 5."""

    lock_wrist_pitch_yaw: bool = True
    observation_space: int = 28
    action_space: int = 5


@configclass
class G1ArmLeftAblationExpScaleEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-24 ablation: position_reward_exp_scale=3.0 in isolation (entropy_coef
    stays at baseline 0.0 — see G1ArmLeftAblationEntropyCoefPPORunnerCfg for that one
    tested alone). See G1ArmEnvCfg's own field comment for the full rationale."""

    position_reward_exp_scale: float = 3.0


@configclass
class G1ArmRightEnvCfg(G1ArmEnvCfg):
    arm: str = "right"
    observation_space: int = 32
    action_space: int = 7


@configclass
class G1ArmBothEnvCfg(G1ArmEnvCfg):
    """Both arms trained simultaneously (64-D obs, 14-D actions)."""
    arm: str = "both"
    observation_space: int = 64
    action_space: int = 14


@configclass
class G1ArmLeftEnvCfg_PLAY(G1ArmLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


@configclass
class G1ArmRightEnvCfg_PLAY(G1ArmRightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


@configclass
class G1ArmBothEnvCfg_PLAY(G1ArmBothEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

# Joint names for each arm (must match the G1 29dof articulation) — 7 per side:
# shoulder pitch/roll/yaw, elbow (single joint, not split pitch/roll like the 23dof/
# 5-DOF model), wrist roll/pitch/yaw.
_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# End-effector body: the link immediately after wrist_yaw — this asset has no dexterous
# hand/finger bodies (confirmed: UNITREE_G1_29DOF_CFG's actuator dict has no "hands"
# group, unlike isaaclab_assets's G1_29DOF_CFG), and unitree_rl_lab's own mimic tracking
# tasks use exactly this body as their end-effector reference for the same asset
# (tasks/mimic/robots/g1_29dof/*/tracking_env_cfg.py).
_LEFT_EE_BODY = "left_wrist_yaw_link"
_RIGHT_EE_BODY = "right_wrist_yaw_link"
_TORSO_BODY = "torso_link"

# Goal workspace bounds in the robot's local frame [m] (env origin added later to
# convert to world frame).
#
# STARTING HYPOTHESIS ONLY, reused numerically from the 23dof-era task's own
# reachability-validated box (after two rounds of fixes there — see that file's history
# / lessons_learned.md / phase_logs/phase_2.md — ~47% -> ~65%+ coverage at 2cm
# tolerance). Reasoning for reuse: position reach is governed mostly by upper-arm +
# forearm link length, which the wrist addition shouldn't change — the wrist joints sit
# near the end of the chain and mostly affect orientation, not how far the chain can
# reach. But this is a hypothesis, not a verified fact for THIS asset/kinematic chain —
# per 29dof_implementation_plan.md Phase 3.2, run
# validation/check_arm_reachability.py against it before trusting it for a real
# training run. Do not skip that step because it "should" still be fine.
_GOAL_BOUNDS = {
    "left":  {"x": (0.20, 0.42), "y": (0.08, 0.40), "z": (0.9, 1.15)},
    "right": {"x": (0.20, 0.42), "y": (-0.40, -0.08), "z": (0.9, 1.15)},
}


class G1ArmEnv(DirectRLEnv):
    """G1 29dof arm-reaching environment using direct RL (``DirectRLEnv``).

    Supports arm="left", "right", or "both". When arm="both" the observation and action
    spaces are doubled and both palms must reach their goals for the episode to
    terminate early.
    """

    cfg: G1ArmEnvCfg

    def __init__(self, cfg: G1ArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------
        # Post-init: build arm-group list (safe after sim is running)
        # ------------------------------------------------------------------
        def _bounds_for(side: str) -> dict:
            b = dict(_GOAL_BOUNDS[side])
            if cfg.goal_bounds_x_override is not None:
                b["x"] = cfg.goal_bounds_x_override
            return b

        # lock_wrist_pitch_yaw: filter these 2 names out of the RL-controlled set before
        # find_joints() — inline here, NOT a change to the module-level _LEFT_ARM_JOINTS/
        # _RIGHT_ARM_JOINTS constants themselves, since g1_full_demo.py/eval_full_demo.py/
        # mdp/events.py all import and rely on those being the full 7-joint lists.
        _WRIST_LOCK_SUFFIXES = ("wrist_pitch_joint", "wrist_yaw_joint")

        def _controlled(names: list[str]) -> list[str]:
            if not cfg.lock_wrist_pitch_yaw:
                return names
            return [n for n in names if not n.endswith(_WRIST_LOCK_SUFFIXES)]

        def _locked(names: list[str]) -> list[str]:
            if not cfg.lock_wrist_pitch_yaw:
                return []
            return [n for n in names if n.endswith(_WRIST_LOCK_SUFFIXES)]

        self._arm_groups: list[dict] = []
        locked_names: list[str] = []
        if cfg.arm in ("left", "both"):
            ids, _ = self.robot.find_joints(_controlled(_LEFT_ARM_JOINTS))
            ee, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            self._arm_groups.append({
                "joint_tensor": torch.tensor(ids, dtype=torch.long, device=self.device),
                "ee_idx": ee[0],
                "bounds": _bounds_for("left"),
            })
            locked_names += _locked(_LEFT_ARM_JOINTS)
        if cfg.arm in ("right", "both"):
            ids, _ = self.robot.find_joints(_controlled(_RIGHT_ARM_JOINTS))
            ee, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            self._arm_groups.append({
                "joint_tensor": torch.tensor(ids, dtype=torch.long, device=self.device),
                "ee_idx": ee[0],
                "bounds": _bounds_for("right"),
            })
            locked_names += _locked(_RIGHT_ARM_JOINTS)

        # Locked wrist joints (if any) get explicitly held at default every step in
        # _apply_action — not left to chance/implicit PD-target behavior. Holding, not
        # removing: they stay physically present and gain-controlled, just never given
        # a policy-driven target.
        if locked_names:
            lids, _ = self.robot.find_joints(locked_names)
            self._locked_wrist_joint_indices = torch.tensor(lids, dtype=torch.long, device=self.device)
            self._locked_wrist_default_pos = self.robot.data.default_joint_pos[
                :, self._locked_wrist_joint_indices
            ].clone()
        else:
            self._locked_wrist_joint_indices = None
            self._locked_wrist_default_pos = None

        self.n_arms = len(self._arm_groups)

        # Combined joint tensor (7 for single arm, 14 for both) — used for reset
        self.arm_joint_indices_tensor = torch.cat(
            [g["joint_tensor"] for g in self._arm_groups]
        )
        self.arm_joint_indices = self.arm_joint_indices_tensor.tolist()

        # Uniform per-joint null-space weight — see cfg.null_space_penalty_scale's
        # docstring for why this is uniform (not per-joint-weighted like the 23dof-era
        # elbow_pitch-specific 3x) until real training data identifies an analogous
        # branch-selection joint here.
        for arm in self._arm_groups:
            arm["null_space_weight"] = torch.ones(len(arm["joint_tensor"]), device=self.device)

        # Real hardware joint range (soft_joint_pos_limits) — the action-target clamp
        # and the joint-limit-avoidance reward both use this directly.
        self._arm_hw_limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_indices_tensor].clone()

        # Uniform 5% joint-limit-penalty margin for every joint/bound — see
        # cfg docstring: the 23dof-era per-joint asymmetric margin was tuned for that
        # model's specific elbow_pitch range asymmetry, which doesn't map cleanly onto
        # this model's single elbow_joint. Revisit with real per-joint data once
        # available (same *_deg_at_min_dist eval columns this repo already has).
        self._joint_limit_margin_fraction = torch.full(
            (len(self.arm_joint_indices), 2), 0.05, device=self.device
        )

        # Torso body index, for the proximity penalty.
        torso_ids, _ = self.robot.find_bodies(_TORSO_BODY)
        self._torso_body_idx = torso_ids[0]

        # Runtime buffers
        self.goal_positions = torch.zeros((self.num_envs, self.n_arms, 3), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.filtered_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Consecutive-steps-within-threshold counter — see goal_hold_steps' own comment.
        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Root-pose wobble state — synthetic *observation* signal only, root itself
        # stays physically fixed (see __post_init__ for why).
        self._default_root_quat = self.robot.data.default_root_state[:, 3:7].clone()
        self._wobble_roll_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_pitch_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_roll_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_pitch_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_roll_phase = torch.zeros(self.num_envs, device=self.device)
        self._wobble_pitch_phase = torch.zeros(self.num_envs, device=self.device)
        self._wobble_yaw_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_yaw_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_yaw_phase = torch.zeros(self.num_envs, device=self.device)
        self._wobble_lin_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_lin_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_lin_phase = torch.zeros(self.num_envs, device=self.device)
        self._wobble_t = torch.zeros(self.num_envs, device=self.device)
        self._synthetic_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._synthetic_projected_gravity = self.robot.data.projected_gravity_b.clone()
        self._synthetic_lin_vel = self.robot.data.root_lin_vel_b.clone()

        # Goal visualisation markers
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/GoalMarkers"
        marker_cfg.markers["sphere"].radius = 0.03
        self.goal_markers = VisualizationMarkers(marker_cfg)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

    # ------------------------------------------------------------------
    # Root-pose wobble
    # ------------------------------------------------------------------

    def _apply_root_wobble(self):
        """Compute a synthetic bounded roll/pitch/yaw-rate/lin-vel wobble for the
        *observation* only. See cfg.enable_root_wobble's docstring — the root itself
        stays physically fixed, this does not move anything."""
        if not self.cfg.enable_root_wobble:
            return
        if self.common_step_counter < self.cfg.root_wobble_enable_step:
            self._synthetic_ang_vel[:] = 0.0
            self._synthetic_projected_gravity[:] = self.robot.data.projected_gravity_b
            self._synthetic_lin_vel[:] = 0.0
            return

        self._wobble_t += self.step_dt
        roll = self._wobble_roll_amp * torch.sin(
            2.0 * torch.pi * self._wobble_roll_freq * self._wobble_t + self._wobble_roll_phase
        )
        pitch = self._wobble_pitch_amp * torch.sin(
            2.0 * torch.pi * self._wobble_pitch_freq * self._wobble_t + self._wobble_pitch_phase
        )
        roll_rate = self._wobble_roll_amp * self._wobble_roll_freq * 2.0 * torch.pi * torch.cos(
            2.0 * torch.pi * self._wobble_roll_freq * self._wobble_t + self._wobble_roll_phase
        )
        pitch_rate = self._wobble_pitch_amp * self._wobble_pitch_freq * 2.0 * torch.pi * torch.cos(
            2.0 * torch.pi * self._wobble_pitch_freq * self._wobble_t + self._wobble_pitch_phase
        )
        yaw_rate = self._wobble_yaw_amp * torch.sin(
            2.0 * torch.pi * self._wobble_yaw_freq * self._wobble_t + self._wobble_yaw_phase
        )

        zero = torch.zeros_like(roll)
        wobble_quat = quat_from_euler_xyz(roll, pitch, zero)
        fake_quat = quat_mul(self._default_root_quat, wobble_quat)

        gravity_dir_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        self._synthetic_projected_gravity[:] = quat_apply_inverse(fake_quat, gravity_dir_w)
        self._synthetic_ang_vel[:, 0] = roll_rate
        self._synthetic_ang_vel[:, 1] = pitch_rate
        self._synthetic_ang_vel[:, 2] = yaw_rate

        lin_phase = 2.0 * torch.pi * self._wobble_lin_freq * self._wobble_t + self._wobble_lin_phase
        self._synthetic_lin_vel[:, 0] = self._wobble_lin_amp * torch.cos(lin_phase)
        self._synthetic_lin_vel[:, 1] = self._wobble_lin_amp * torch.sin(lin_phase)
        self._synthetic_lin_vel[:, 2] = 0.0

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        self._apply_root_wobble()

        alpha = float(self.cfg.action_filter_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        self.filtered_actions = alpha * self.actions + (1.0 - alpha) * self.filtered_actions

        current = self.robot.data.joint_pos[:, self.arm_joint_indices_tensor]
        delta = self.filtered_actions * self.cfg.action_scale
        max_delta = float(self.cfg.max_action_delta_per_step)
        if max_delta > 0.0:
            delta = delta.clamp(min=-max_delta, max=max_delta)

        targets = current + delta
        targets = targets.clamp(self._arm_hw_limits[:, 0], self._arm_hw_limits[:, 1])
        self.robot.set_joint_position_target(targets, joint_ids=self.arm_joint_indices_tensor)
        self.previous_actions[:] = self.filtered_actions

        # cfg.lock_wrist_pitch_yaw: hold these explicitly at default every step, not
        # left to implicit PD-target carryover — guarantees no drift regardless of
        # what IsaacLab does with an untouched joint's target across a reset.
        if self._locked_wrist_joint_indices is not None:
            self.robot.set_joint_position_target(
                self._locked_wrist_default_pos, joint_ids=self._locked_wrist_joint_indices
            )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        base_lin_vel = uniform_noise(
            self._synthetic_lin_vel,
            UniformNoiseCfg(n_min=-self.cfg.base_lin_vel_noise, n_max=self.cfg.base_lin_vel_noise),
        )
        base_ang_vel = uniform_noise(
            self._synthetic_ang_vel,
            UniformNoiseCfg(n_min=-self.cfg.base_ang_vel_noise, n_max=self.cfg.base_ang_vel_noise),
        )
        projected_gravity = uniform_noise(
            self._synthetic_projected_gravity,
            UniformNoiseCfg(n_min=-self.cfg.projected_gravity_noise, n_max=self.cfg.projected_gravity_noise),
        )

        parts = []
        for i, arm in enumerate(self._arm_groups):
            jt = arm["joint_tensor"]
            joint_pos = uniform_noise(
                self.robot.data.joint_pos[:, jt],
                UniformNoiseCfg(n_min=-self.cfg.joint_pos_noise, n_max=self.cfg.joint_pos_noise),
            )  # (N, 7)
            joint_vel = uniform_noise(
                self.robot.data.joint_vel[:, jt],
                UniformNoiseCfg(n_min=-self.cfg.joint_vel_noise, n_max=self.cfg.joint_vel_noise),
            )  # (N, 7)
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]  # (N, 3)
            goal = self.goal_positions[:, i, :]                        # (N, 3)
            error = goal - ee_pos                                       # (N, 3)
            parts.extend([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error])
        return {"policy": torch.cat(parts, dim=-1)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total = torch.zeros(self.num_envs, device=self.device)
        all_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        torso_pos = self.robot.data.body_pos_w[:, self._torso_body_idx, :]  # (N, 3)
        margin = self.cfg.torso_proximity_margin_m

        for i, arm in enumerate(self._arm_groups):
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]  # (N, 3)
            dist = torch.norm(self.goal_positions[:, i, :] - ee_pos, dim=-1)
            total += -dist * self.cfg.position_reward_scale
            total += torch.exp(-dist / self.cfg.position_reward_exp_sigma) * self.cfg.position_reward_exp_scale
            reached = dist < self.cfg.goal_threshold
            total += reached.float() * self.cfg.goal_reached_bonus
            all_reached &= reached

            torso_dist = torch.norm(ee_pos - torso_pos, dim=-1)
            total += -torch.clamp(margin - torso_dist, min=0.0) * self.cfg.torso_proximity_penalty_scale

            jt = arm["joint_tensor"]
            ref_pose = self.robot.data.default_joint_pos[:, jt]
            pose_deviation = torch.norm(
                (self.robot.data.joint_pos[:, jt] - ref_pose) * arm["null_space_weight"], dim=-1
            )
            total += -pose_deviation * self.cfg.null_space_penalty_scale

        total += -torch.norm(self.previous_actions, dim=-1) * self.cfg.action_smoothness_scale

        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices_tensor]
        limits = self._arm_hw_limits
        span = limits[:, 1] - limits[:, 0]
        margin_lo = self._joint_limit_margin_fraction[:, 0] * span
        margin_hi = self._joint_limit_margin_fraction[:, 1] * span
        at_limit = (
            (joint_pos < limits[:, 0] + margin_lo) | (joint_pos > limits[:, 1] - margin_hi)
        ).float().sum(-1)
        total += -at_limit * self.cfg.joint_limit_penalty_scale

        # Success requires holding all_reached for goal_hold_steps CONSECUTIVE steps, not
        # just touching it once — see goal_hold_steps' own comment for the full rationale.
        self._hold_counter = torch.where(
            all_reached, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        self.successes = self._hold_counter >= self.cfg.goal_hold_steps
        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.successes.clone()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device) \
            if not isinstance(env_ids, torch.Tensor) else env_ids

        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(joint_pos)

        for i, idx in enumerate(self.arm_joint_indices):
            lo = self._arm_hw_limits[i, 0]
            hi = self._arm_hw_limits[i, 1]
            centre = (lo + hi) * 0.5
            half = (hi - lo) * self.cfg.reset_range_fraction
            joint_pos[:, idx] = centre + (torch.rand(num_resets, device=self.device) * 2 - 1) * half

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)

        # Re-sample this episode's wobble profile.
        self._wobble_roll_amp[env_ids_tensor] = (
            torch.rand(num_resets, device=self.device) * self.cfg.root_wobble_max_roll_rad
        )
        self._wobble_pitch_amp[env_ids_tensor] = (
            torch.rand(num_resets, device=self.device) * self.cfg.root_wobble_max_pitch_rad
        )
        freq_lo, freq_hi = self.cfg.root_wobble_freq_hz_range
        freq_span = freq_hi - freq_lo
        self._wobble_roll_freq[env_ids_tensor] = freq_lo + torch.rand(num_resets, device=self.device) * freq_span
        self._wobble_pitch_freq[env_ids_tensor] = freq_lo + torch.rand(num_resets, device=self.device) * freq_span
        self._wobble_roll_phase[env_ids_tensor] = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        self._wobble_pitch_phase[env_ids_tensor] = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        self._wobble_yaw_amp[env_ids_tensor] = (
            torch.rand(num_resets, device=self.device) * self.cfg.root_wobble_max_yaw_rate_rad_s
        )
        self._wobble_yaw_freq[env_ids_tensor] = freq_lo + torch.rand(num_resets, device=self.device) * freq_span
        self._wobble_yaw_phase[env_ids_tensor] = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        self._wobble_lin_amp[env_ids_tensor] = (
            torch.rand(num_resets, device=self.device) * self.cfg.root_wobble_max_lin_vel_mps
        )
        self._wobble_lin_freq[env_ids_tensor] = freq_lo + torch.rand(num_resets, device=self.device) * freq_span
        self._wobble_lin_phase[env_ids_tensor] = torch.rand(num_resets, device=self.device) * 2.0 * torch.pi
        self._wobble_t[env_ids_tensor] = 0.0

        # Sample goals in robot-local frame, shift to world frame via env origins.
        for i, arm in enumerate(self._arm_groups):
            self.goal_positions[env_ids_tensor, i, :] = (
                self._sample_goal_positions(num_resets, arm["bounds"])
                + self.scene.env_origins[env_ids_tensor]
            )

        self.previous_actions[env_ids_tensor] = 0.0
        self.filtered_actions[env_ids_tensor] = 0.0
        self.successes[env_ids_tensor] = False
        self._hold_counter[env_ids_tensor] = 0
        self.episode_length_buf[env_ids_tensor] = 0

        self._update_goal_markers(env_ids_tensor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_position(self, arm_index: int = 0) -> torch.Tensor:
        """End-effector world position for one arm group (N, 3)."""
        return self.robot.data.body_pos_w[:, self._arm_groups[arm_index]["ee_idx"], :]

    def _sample_goal_positions(self, num_goals: int, bounds: dict) -> torch.Tensor:
        """Sample goals in the robot's local frame for one arm."""
        goals = torch.zeros((num_goals, 3), device=self.device)
        for i, key in enumerate(("x", "y", "z")):
            lo, hi = bounds[key]
            centre = (lo + hi) * 0.5
            half_span = (hi - lo) * 0.5
            goals[:, i] = centre + (torch.rand(num_goals, device=self.device) * 2 - 1) * half_span
        return goals

    def _update_goal_markers(self, env_ids: torch.Tensor):
        self.goal_markers.visualize(self.goal_positions[env_ids].reshape(-1, 3))
