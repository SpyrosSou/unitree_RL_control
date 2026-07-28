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
    # 9-D base-state prefix (base_lin_vel, base_ang_vel, projected_gravity) + 30-D per arm
    # (joint_pos(7) + joint_vel(7) + ee_pos(3) + goal(3) + error(3) + action_fb(7)) =
    # 39-D per arm. action_fb ADDED 2026-07-26 — see _get_observations' own comment.
    observation_space: int = 39
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

    # 2026-07-26: goal-distance curriculum. When enabled, _sample_goal_positions shrinks
    # each bounds box around its own center to goal_curriculum_start_frac of its full
    # half-span at common_step_counter=0, linearly expanding to 100% by
    # goal_curriculum_end_step (30_000 steps matches root_wobble_enable_step's existing
    # step-based-schedule convention, ~1250 iterations at num_steps_per_env=24). Motivated
    # by 2026-07-26 diagnostics: holding is solved (96%+ conversion from ever-within-2cm
    # to actual success across every checkpoint tested this session), but reaching itself
    # plateaus at ~30-40% ever-within-2cm and the plateau is genuine — frac_envs_reached/
    # position_dist go flat by ~step 500 through 2000 iterations for best_combined, not
    # still climbing — with no goal-space clustering (success/failure goal-position
    # distributions nearly identical on x/y/z). A uniform capability shortfall across the
    # whole box is the textbook case for easy-to-hard curriculum, not more iterations at
    # fixed difficulty. Off by default — every existing task is unaffected.
    goal_curriculum_enabled: bool = False
    goal_curriculum_start_frac: float = 0.15
    goal_curriculum_end_step: int = 30_000

    # 2026-07-26: lets an eval-only cfg (G1ArmLeftEnvCfg_LongHold200) disable the
    # early-termination-on-success behavior so an episode keeps running well past the
    # goal_hold_steps threshold, to check whether the hold is actually sustained rather
    # than just momentarily satisfied. True (stock behavior) everywhere else.
    terminate_on_success: bool = True
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

    # 2026-07-28: distance-adaptive rate limit — off by default, every existing task
    # unaffected. Motivated by a real visual observation (g1_full_demo.py) that the arm
    # moves noticeably slowly throughout an entire reach, plausibly running out of the
    # episode's time budget while still closing in on a distant goal rather than ever
    # actually converging — consistent with `policy_status.md`'s own finding that only
    # 1.4% of failed eval episodes ever got within the 2cm success radius (i.e. most
    # failures aren't "reached but didn't hold," they're "never got close"), though
    # that alone doesn't prove a speed problem specifically (see the new
    # dist_to_goal_cm_at_t*/final_dist_to_goal_cm eval columns for a more direct check).
    # This is NOT the same thing as the already-tried-and-rejected uniform rate-limit
    # relaxation (action_filter_alpha 0.25->0.6, max_action_delta_per_step 0.06->0.15 —
    # see G1ArmLeftEnvCfg_RateLimit, final eval 21-24% vs baseline's 27-30%, i.e. worse):
    # that sped up movement everywhere, including right at the goal, which plausibly
    # hurt final settling precision. This instead keeps max_action_delta_per_step (the
    # existing, precision-tuned value) unchanged near the goal, and only raises the cap
    # while still far away — trying to get faster traversal without touching the
    # behavior that produces precise settling.
    use_adaptive_rate_limit: bool = False
    max_action_delta_per_step_far: float = 0.18
    adaptive_rate_limit_near_dist_m: float = 0.05
    adaptive_rate_limit_far_dist_m: float = 0.20

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

    # ADDED 2026-07-26: settle/braking penalty — see _get_rewards' own comment for the
    # full rationale (comparative reward-breakdown data confirmed this is a
    # holding-stability problem, not a reaching problem: every 40/10-gain variant
    # touches goal_threshold at least as often per step as the 99.98%-success 200/20
    # reference, yet succeeds an order of magnitude less often). settle_proximity_m is
    # deliberately looser than goal_threshold (2cm) so the penalty can encourage braking
    # INTO the success zone, not just once already inside it. Weight is a first,
    # unvalidated estimate — comparable order of magnitude to null_space_penalty_scale,
    # not yet ablated in isolation.
    settle_proximity_m: float = 0.05
    settle_velocity_penalty_scale: float = 0.05

    # ADDED 2026-07-26: made configurable so a subclass can test a different gain
    # cleanly (see G1ArmLeftEnvCfg_Gain60Kd1p5 below) — was hardcoded 40.0/10.0 inline
    # in __post_init__ before. Default (40/10) unchanged — see __post_init__'s own long
    # comment for why that value was chosen and its provenance. 2026-07-26: found that
    # Unitree's own dedicated arm-control SDK examples (g1_arm7_sdk_dds_example.py,
    # g1_arm5_sdk_dds_example.py — the actual gesture/reaching interface, not the
    # locomotion/whole-body one) use kp=60, kd=1.5 for all arm joints, not 40/10 — the
    # 40/10 value's damping component in particular traces to one historical commit in
    # unitree_rl_lab (a research repo), not the deployment SDK. Worth testing directly.
    arm_actuator_stiffness: float = 40.0
    arm_actuator_damping: float = 10.0

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

    # ADDED 2026-07-26: append the action-filter's current internal state (self.
    # filtered_actions) to the observation — see _get_observations' own comment for the
    # full POMDP-gap rationale. Default True for all NEW training. False only exists so
    # pre-2026-07-26 checkpoints (32/28/64-D obs, e.g. the 200/20-gain reference and the
    # 4-way ablation sweep's baseline/privileged_critic/log_std runs) can still be
    # loaded and evaluated — the observation_space field must ALSO be set back to its
    # pre-change value on any cfg used this way (see G1ArmLeftEnvCfg_Legacy32 below),
    # since this flag alone does not resize the tensor the network expects.
    include_action_feedback: bool = True

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
        # FIXED 2026-07-25: joint_names_expr used a bare ".*_" prefix, which matches
        # BOTH left_*/right_* joints regardless of self.arm — so a "left"-only task
        # was also softening the RIGHT arm's gain from its stock value to 40/10, even
        # though the right arm is never given a policy-driven target here (only
        # G1ArmEnv.arm_joint_indices_tensor, built from self.arm, gets
        # set_joint_position_target calls). An uncontrolled limb with a softened gain
        # and no active driving just settles once toward a new gravity/dynamics
        # equilibrium and holds there — confirmed via direct visual test (--arm left)
        # showing the right elbow "move once and lock," which this explains without
        # any action-routing bug (reward/obs code only ever reads self._arm_groups,
        # scoped correctly to self.arm the whole time). Scope the regex prefix(es) to
        # the side(s) this task actually controls; the other side (if any) keeps
        # whatever gain UNITREE_G1_29DOF_CFG's stock config already gives it.
        side_prefixes = {"left": ["left"], "right": ["right"], "both": ["left", "right"]}[self.arm]
        self.robot.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[
                f"{side}_{suffix}"
                for side in side_prefixes
                for suffix in (
                    "shoulder_pitch_joint",
                    "shoulder_roll_joint",
                    "shoulder_yaw_joint",
                    "elbow_joint",
                    "wrist_roll_joint",
                    "wrist_pitch_joint",
                    "wrist_yaw_joint",
                )
            ],
            stiffness=self.arm_actuator_stiffness,
            damping=self.arm_actuator_damping,
        )
        # Legs/waist/feet (and the uncontrolled arm, if any) stay at
        # UNITREE_G1_29DOF_CFG's own stock gains (see g1_locomotion.assets.robots.unitree)
        # — already reasonably strong for a "stay put" role, not overridden here.

        # Contact sensors not needed for this task.
        self.robot.spawn.activate_contact_sensors = False


@configclass
class G1ArmLeftEnvCfg(G1ArmEnvCfg):
    arm: str = "left"
    observation_space: int = 39
    action_space: int = 7


@configclass
class G1ArmLeftEnvCfg_Gain60Kd1p5(G1ArmLeftEnvCfg):
    """2026-07-26: tests kp=60, kd=1.5 — the gain Unitree's own dedicated arm-control
    SDK examples actually use (g1_arm7_sdk_dds_example.py, g1_arm5_sdk_dds_example.py),
    as opposed to the 40/10 this task inherited from a locomotion-context historical
    commit in unitree_rl_lab. Motivated by: (1) direct SDK evidence this is the real
    value for arm-control specifically, not just a doubt, and (2) real-world
    observation that the physical robot holds arm gestures/positions reliably, which
    is hard to reconcile with 40/10 if that's genuinely what governs arm-holding on
    hardware. Includes the settle-reward fix automatically (baked into the shared base
    reward code, not gated behind a flag) — this is not a gain-only isolated test."""

    arm_actuator_stiffness: float = 60.0
    arm_actuator_damping: float = 1.5


@configclass
class G1ArmLeftEnvCfg_Gain15Kd1(G1ArmLeftEnvCfg):
    """2026-07-26: tests kp=15, kd=1 — matches unitreerobotics/unitree_rl_mjlab's actual
    RL-policy deploy config (deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml),
    where arm joints run at kp=14.3-16.8, kd=0.9-1.1. This supersedes Gain60Kd1p5 as the
    better real-hardware candidate: 60/1.5 was the *scripted-motion* SDK example's gain
    (g1_arm7/g1_arm5_sdk_dds_example.py), which caused ~86-90% mean_joints_at_limit in eval
    (underdamped oscillation, high kp with low kd). A deployed RL policy is expected to
    supply its own corrective effort each step rather than lean on a stiff PD controller,
    so real RL deployments run low kp *and* low kd together, not just low kd. Note the
    kp/kd ratio here (~15) is close to the working 200/20 reference's ratio (10) — both
    are much closer to each other than to 40/10's ratio (4), which may be the more
    relevant similarity. Includes the settle-reward fix automatically (baked into the
    shared base reward code, not gated behind a flag) — not a gain-only isolated test."""

    arm_actuator_stiffness: float = 15.0
    arm_actuator_damping: float = 1.0


@configclass
class G1ArmLeftEnvCfg_GoalCurriculum(G1ArmLeftEnvCfg):
    """2026-07-26: goal-distance curriculum — see goal_curriculum_enabled's own field
    comment on G1ArmEnvCfg for the full motivation. Includes the settle-reward fix
    automatically (baked into shared base reward code, not gated behind a flag)."""

    goal_curriculum_enabled: bool = True


@configclass
class G1ArmLeftEnvCfg_Legacy32(G1ArmLeftEnvCfg):
    """For evaluating checkpoints trained before 2026-07-26's action_fb observation
    addition (32-D obs) — e.g. the 200/20-gain reference and the 4-way ablation sweep's
    baseline/privileged_critic/log_std runs, all of which predate it. Not for new
    training — include_action_feedback should stay True there so future checkpoints get
    the (likely helpful) action feedback by default. episode_length_s=20.0 matches
    G1ArmLeftEnvCfg_PLAY (used for every other eval path) for a fair comparison."""

    include_action_feedback: bool = False
    observation_space: int = 32

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


@configclass
class G1ArmLeftEnvCfg_LongHold200(G1ArmLeftEnvCfg_Legacy32):
    """2026-07-26: long-hold verification for the 200/20-gain reference checkpoint.
    goal_hold_steps=15 (0.5s at 30Hz) only proves the arm can touch-and-briefly-hold a
    target long enough to trigger early termination; it says nothing about whether
    that hold is *sustained*. Disables early termination on success and extends
    episode_length_s to 45s so a fixed goal can be observed for much longer than the
    minimum hold window. Eval-only — not for training (see validation/
    eval_arm_long_hold.py)."""

    terminate_on_success: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0


@configclass
class G1ArmLeftLockedWristEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-24: wrist_pitch/wrist_yaw excluded from RL control, held rigidly at
    default instead — see G1ArmEnvCfg.lock_wrist_pitch_yaw's own docstring. 5
    controlled joints (shoulder x3, elbow, wrist_roll), not 7 — observation/action
    dims shrink accordingly: 39 -> 33 (joint_pos/joint_vel/action_fb each drop from 7
    to 5 elements), 7 -> 5."""

    lock_wrist_pitch_yaw: bool = True
    observation_space: int = 33
    action_space: int = 5


@configclass
class G1ArmLeftAblationExpScaleEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-24 ablation: position_reward_exp_scale=3.0 in isolation (entropy_coef
    stays at baseline 0.0 — see G1ArmLeftAblationEntropyCoefPPORunnerCfg for that one
    tested alone). See G1ArmEnvCfg's own field comment for the full rationale."""

    position_reward_exp_scale: float = 3.0


@configclass
class G1ArmLeftAblationRateLimitEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-25 ablation: relax the action-filtering/rate-limiting that gates how fast
    the commanded joint target can move, in isolation from anything RL-hyperparameter-
    related (entropy_coef, exp_scale — both already ruled out, see this task's own
    conversation history / policy_status.md).

    Motivation: at the real hardware gain (40/10, no gravity feedforward — confirmed via
    unitree_rl_lab/deploy/include/FSM/State_RLBase.h, tau()=0 on real hardware too),
    Train/mean_episode_length sits at ~285-297 (out of a 300-step max) across EVERY 40/10
    variant tried so far (baseline, exp_scale=3.0, entropy_coef=0.003/0.01) — meaning
    episodes almost never trigger the early-success termination (hold goal 15 consecutive
    steps), regardless of any RL-side tuning. The working 200/20-gain checkpoint converges
    to ~60 steps. Three different RL-hyperparameter interventions, zero effect on this
    number — strong evidence the bottleneck isn't exploration or reward shape.

    action_filter_alpha (0.25, heavy EMA smoothing of each new action against the
    previous one) and max_action_delta_per_step (0.06 rad, hard per-control-step cap) are
    both unchanged since the very first commit (confirmed via git log) — not a confound
    with the 200/20-vs-40/10 comparison. Both gate the SAME mechanism (how fast the
    effective commanded target can move) — testing them together is one coherent
    hypothesis ("rate-limiting prevents settling"), not an uncontrolled multi-variable
    bundle like the earlier entropy_coef+exp_scale mistake. At the softer 40/10 gain the
    joint's own physical response is already slower than at 200/20; layering heavy
    smoothing and a tight delta cap on top may mean the effective target never actually
    catches up and holds still long enough to satisfy goal_hold_steps, independent of
    what the policy has learned. Relaxed here (not removed — still bounded/safe):
    action_filter_alpha 0.25 -> 0.6 (much less lag), max_action_delta_per_step 0.06 -> 0.15
    rad (2.5x higher cap). If Train/mean_episode_length starts dropping meaningfully below
    ~290 within a short run, this is the mechanism; if it doesn't, this is ruled out too."""

    action_filter_alpha: float = 0.6
    max_action_delta_per_step: float = 0.15


@configclass
class G1ArmLeftAdaptiveRateEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-28: distance-adaptive rate limit — see G1ArmEnvCfg's
    use_adaptive_rate_limit field docstring for the full mechanism and how this differs
    from the already-tried-and-rejected G1ArmLeftAblationRateLimitEnvCfg (that one raised
    the rate limit uniformly, everywhere, including right at the goal, and ended up
    worse: 21-24% vs baseline's 27-30% final eval, despite looking better early —
    policy_status.md flags this as "the one case where the early-signal heuristic was
    misleading," so don't over-trust an early read here either).

    Motivated by a direct visual observation (g1_full_demo.py, 2026-07-28) that the arm
    moves noticeably slowly throughout an entire reach, plausibly running out of the
    episode's time budget while still closing in on a distant goal — separately
    consistent with (not proof of, see the module's own eval-column additions for a
    direct check) `policy_status.md`'s finding that only 1.4% of failed eval episodes
    ever got within the 2cm success radius, and that failure is uniform across the whole
    goal box (not clustered at hard-to-reach regions) — a "never got close, regardless of
    where the goal was" pattern is at least dimensionally consistent with "ran out of
    time before arriving," though a slow policy and a poorly-converged one would look
    similar from aggregate stats alone; the new final_dist_to_goal_cm/
    dist_to_goal_cm_at_t* eval columns are what actually distinguishes them.

    max_action_delta_per_step_far/adaptive_rate_limit_near_dist_m/
    adaptive_rate_limit_far_dist_m are all first-guess starting points, not tuned
    values (same spirit as every other new lever in this project) — chosen so behavior
    within 5cm of the goal exactly matches the existing, precision-tuned baseline
    (unchanged max_action_delta_per_step=0.06), while beyond 20cm it can move up to 3x
    faster (0.18 rad/step, comparable in magnitude to the rejected uniform ablation's
    0.15, but only far from the goal instead of everywhere)."""

    use_adaptive_rate_limit: bool = True


@configclass
class G1ArmLeftAblationLowVelNoiseEnvCfg(G1ArmLeftEnvCfg):
    """2026-07-25 ablation: joint_vel_noise reduced from 1.5 rad/s to 0.1 rad/s, in
    isolation (no other change — same gain, same reward, symmetry intact).

    1.5 rad/s was never derived for this task — it's borrowed wholesale from
    G1FlatEnvCfg-family/Unitree's own 29dof LOCOMOTION recipe (see the field's own
    comment), where leg joints swing at several rad/s during a stride, so that much
    noise is proportionally reasonable there. Arm-reaching's useful velocity signal
    during fine settling (the exact regime goal_hold_steps=15 needs to succeed in) is on
    the order of 0.05-0.3 rad/s — a factor of 5-30x smaller than the noise band itself,
    meaning the policy may not be able to reliably perceive "am I still moving, should I
    decelerate" through that much noise. 0.1 rad/s is a first, defensible estimate (still
    real, nonzero sensor noise, just no longer larger than the signal it's supposed to
    accompany) — not derived from an actual encoder spec, revisit if this direction pans
    out. This has never been touched; every previous arm experiment used 1.5 rad/s
    unchanged, including the 200/20-gain reference that got 99.98% (so this alone isn't
    sufficient to explain that gap — but it may be a real, compounding contributor at the
    much-more-precision-sensitive 40/10 gain)."""

    joint_vel_noise: float = 0.1


@configclass
class G1ArmRightEnvCfg(G1ArmEnvCfg):
    arm: str = "right"
    observation_space: int = 39
    action_space: int = 7


@configclass
class G1ArmBothEnvCfg(G1ArmEnvCfg):
    """Both arms trained simultaneously (78-D obs, 14-D actions)."""
    arm: str = "both"
    observation_space: int = 78
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

        # Per-arm slice into the flat action/filtered_actions vector (which is ordered
        # the same way arm_joint_indices_tensor is — concatenated in _arm_groups order)
        # — used by _get_observations to feed each arm's own action-feedback slice.
        _offset = 0
        for arm in self._arm_groups:
            n = len(arm["joint_tensor"])
            arm["action_slice"] = slice(_offset, _offset + n)
            _offset += n

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
        base_max_delta = float(self.cfg.max_action_delta_per_step)

        if self.cfg.use_adaptive_rate_limit and base_max_delta > 0.0:
            # Per-arm, per-env rate limit: base_max_delta (the existing, precision-tuned
            # value) near the goal, ramping up to max_action_delta_per_step_far while
            # still far away — see cfg field docstring for the full rationale. Computed
            # per arm group since each arm has its own end-effector/goal and its own
            # slice of the flat delta tensor (self._arm_groups' "action_slice").
            far_max_delta = float(self.cfg.max_action_delta_per_step_far)
            near_d = float(self.cfg.adaptive_rate_limit_near_dist_m)
            far_d = float(self.cfg.adaptive_rate_limit_far_dist_m)
            span = max(far_d - near_d, 1e-6)
            for i, arm in enumerate(self._arm_groups):
                ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]
                dist = torch.norm(self.goal_positions[:, i, :] - ee_pos, dim=-1)  # (N,)
                frac = ((dist - near_d) / span).clamp(0.0, 1.0)
                per_env_max_delta = (base_max_delta + (far_max_delta - base_max_delta) * frac).unsqueeze(-1)
                sl = arm["action_slice"]
                delta[:, sl] = torch.max(torch.min(delta[:, sl], per_env_max_delta), -per_env_max_delta)
        elif base_max_delta > 0.0:
            delta = delta.clamp(min=-base_max_delta, max=base_max_delta)

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

        # ADDED 2026-07-25: clean (noise-free) counterparts for a privileged/asymmetric
        # critic — see G1ArmLeftAblationPrivilegedCriticPPORunnerCfg's own docstring.
        # Always computed (cheap) but only USED when a runner cfg's obs_groups actually
        # points "critic" at this "critic" key instead of the default "policy" key — the
        # default behavior (obs_groups critic -> ["policy"]) is completely unaffected by
        # this addition.
        clean_base_lin_vel = self._synthetic_lin_vel
        clean_base_ang_vel = self._synthetic_ang_vel
        clean_projected_gravity = self._synthetic_projected_gravity

        parts = []
        critic_parts = []
        for i, arm in enumerate(self._arm_groups):
            jt = arm["joint_tensor"]
            joint_pos_raw = self.robot.data.joint_pos[:, jt]
            joint_vel_raw = self.robot.data.joint_vel[:, jt]
            joint_pos = uniform_noise(
                joint_pos_raw,
                UniformNoiseCfg(n_min=-self.cfg.joint_pos_noise, n_max=self.cfg.joint_pos_noise),
            )  # (N, 7)
            joint_vel = uniform_noise(
                joint_vel_raw,
                UniformNoiseCfg(n_min=-self.cfg.joint_vel_noise, n_max=self.cfg.joint_vel_noise),
            )  # (N, 7)
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]  # (N, 3)
            goal = self.goal_positions[:, i, :]                        # (N, 3)
            error = goal - ee_pos                                       # (N, 3)
            # ADDED 2026-07-26: the policy previously had NO way to observe its own
            # action-filter's internal state (self.filtered_actions — an EMA of past
            # actions, action_filter_alpha=0.25, i.e. ~75% memory of history each step).
            # _apply_action's actual commanded delta depends on this hidden state, not
            # just the current raw action, but a memoryless feedforward policy (no
            # recurrence) was never given it as an observation — a real POMDP-vs-MDP
            # gap: the policy couldn't know how heavily a new action would actually be
            # blended in, making it harder to plan precise, well-calibrated corrections
            # during exactly the fine-settling phase goal_hold_steps needs. No noise
            # added — this is the policy's own exact internal state, not a sensor
            # reading, nothing to simulate imprecision on.
            parts.extend([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error])
            critic_parts.extend([
                clean_base_lin_vel, clean_base_ang_vel, clean_projected_gravity,
                joint_pos_raw, joint_vel_raw, ee_pos, goal, error,
            ])
            if self.cfg.include_action_feedback:
                action_fb = self.filtered_actions[:, arm["action_slice"]]  # (N, 7)
                parts.append(action_fb)
                critic_parts.append(action_fb)
        return {"policy": torch.cat(parts, dim=-1), "critic": torch.cat(critic_parts, dim=-1)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total = torch.zeros(self.num_envs, device=self.device)
        all_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # ADDED 2026-07-26: per-component reward breakdown — this task (DirectRLEnv) had
        # ZERO per-term visibility before this, unlike the walking task (ManagerBasedRLEnv,
        # which logs every RewTerm to tensorboard automatically). Every piece of reasoning
        # this session about whether e.g. joint_limit_penalty or torso_proximity_penalty
        # might be disproportionately large was pure guesswork — there was no data to
        # check it against. Logged via self.extras["log"] (the key rsl_rl's own
        # OnPolicyRunner reads — see on_policy_runner.py's rollout loop), same
        # "Episode_Reward/<name>" naming convention the walking task's RewardManager
        # already uses, so tensorboard shows both tasks' reward breakdowns consistently.
        position_dist_term = torch.zeros(self.num_envs, device=self.device)
        position_exp_term = torch.zeros(self.num_envs, device=self.device)
        goal_bonus_term = torch.zeros(self.num_envs, device=self.device)
        torso_proximity_term = torch.zeros(self.num_envs, device=self.device)
        null_space_term = torch.zeros(self.num_envs, device=self.device)
        settle_term = torch.zeros(self.num_envs, device=self.device)
        mean_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        torso_pos = self.robot.data.body_pos_w[:, self._torso_body_idx, :]  # (N, 3)
        margin = self.cfg.torso_proximity_margin_m

        for i, arm in enumerate(self._arm_groups):
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]  # (N, 3)
            dist = torch.norm(self.goal_positions[:, i, :] - ee_pos, dim=-1)
            mean_dist_to_goal += dist
            position_dist_term += -dist * self.cfg.position_reward_scale
            position_exp_term += torch.exp(-dist / self.cfg.position_reward_exp_sigma) * self.cfg.position_reward_exp_scale
            reached = dist < self.cfg.goal_threshold
            goal_bonus_term += reached.float() * self.cfg.goal_reached_bonus
            all_reached &= reached

            torso_dist = torch.norm(ee_pos - torso_pos, dim=-1)
            torso_proximity_term += -torch.clamp(margin - torso_dist, min=0.0) * self.cfg.torso_proximity_penalty_scale

            jt = arm["joint_tensor"]
            ref_pose = self.robot.data.default_joint_pos[:, jt]
            pose_deviation = torch.norm(
                (self.robot.data.joint_pos[:, jt] - ref_pose) * arm["null_space_weight"], dim=-1
            )
            null_space_term += -pose_deviation * self.cfg.null_space_penalty_scale

            # ADDED 2026-07-26: settle/braking penalty. Real data (comparative
            # Episode_Reward breakdown across the 200/20-gain reference and every
            # 40/10-gain variant tried) confirmed this isn't a reaching problem — every
            # 40/10 variant touches goal_threshold at least as often per step as the
            # 99.98%-success reference, some MORE often, yet succeeds (15 CONSECUTIVE
            # steps held) an order of magnitude less. It reaches the zone fine and
            # can't stabilize there. Nothing in the reward before this rewarded actually
            # slowing down near the goal — position_dist/position_exp_bonus reward being
            # close, not being STILL while close. Penalize joint velocity, but ONLY once
            # within settle_proximity_m of the goal (a looser radius than goal_threshold
            # itself, so it can brake INTO the success zone rather than only once
            # already inside it) — gated, not applied everywhere, so it can't discourage
            # the initial approach motion.
            joint_vel_norm = torch.norm(self.robot.data.joint_vel[:, jt], dim=-1)
            within_settle_zone = (dist < self.cfg.settle_proximity_m).float()
            settle_term += -joint_vel_norm * within_settle_zone * self.cfg.settle_velocity_penalty_scale

        mean_dist_to_goal /= self.n_arms

        action_smoothness_term = -torch.norm(self.previous_actions, dim=-1) * self.cfg.action_smoothness_scale

        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices_tensor]
        limits = self._arm_hw_limits
        span = limits[:, 1] - limits[:, 0]
        margin_lo = self._joint_limit_margin_fraction[:, 0] * span
        margin_hi = self._joint_limit_margin_fraction[:, 1] * span
        at_limit_mask = (joint_pos < limits[:, 0] + margin_lo) | (joint_pos > limits[:, 1] - margin_hi)
        at_limit = at_limit_mask.float().sum(-1)
        joint_limit_term = -at_limit * self.cfg.joint_limit_penalty_scale

        total = (
            position_dist_term + position_exp_term + goal_bonus_term + torso_proximity_term
            + null_space_term + action_smoothness_term + joint_limit_term + settle_term
        )

        self.extras["log"] = {
            "Episode_Reward/position_dist": position_dist_term.mean(),
            "Episode_Reward/position_exp_bonus": position_exp_term.mean(),
            "Episode_Reward/goal_reached_bonus": goal_bonus_term.mean(),
            "Episode_Reward/torso_proximity": torso_proximity_term.mean(),
            "Episode_Reward/null_space": null_space_term.mean(),
            "Episode_Reward/action_smoothness": action_smoothness_term.mean(),
            "Episode_Reward/joint_limit": joint_limit_term.mean(),
            "Episode_Reward/settle": settle_term.mean(),
            # Raw (non-reward-scaled) diagnostics — interpretable units, not just reward contribution.
            "Metrics/mean_dist_to_goal_cm": mean_dist_to_goal.mean() * 100.0,
            "Metrics/mean_joints_at_limit": at_limit.mean(),
            "Metrics/frac_envs_reached": all_reached.float().mean(),
            "Curriculum/goal_bounds_frac": torch.tensor(
                getattr(self, "_last_goal_curriculum_frac", 1.0), device=self.device
            ),
        }

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
        if self.cfg.terminate_on_success:
            terminated = self.successes.clone()
        else:
            terminated = torch.zeros_like(self.successes)
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
        if self.cfg.goal_curriculum_enabled:
            end_step = max(1, self.cfg.goal_curriculum_end_step)
            progress = min(1.0, max(0.0, self.common_step_counter / end_step))
            frac = self.cfg.goal_curriculum_start_frac + (1.0 - self.cfg.goal_curriculum_start_frac) * progress
        else:
            frac = 1.0
        self._last_goal_curriculum_frac = frac

        goals = torch.zeros((num_goals, 3), device=self.device)
        for i, key in enumerate(("x", "y", "z")):
            lo, hi = bounds[key]
            centre = (lo + hi) * 0.5
            half_span = (hi - lo) * 0.5 * frac
            goals[:, i] = centre + (torch.rand(num_goals, device=self.device) * 2 - 1) * half_span
        return goals

    def _update_goal_markers(self, env_ids: torch.Tensor):
        self.goal_markers.visualize(self.goal_positions[env_ids].reshape(-1, 3))
