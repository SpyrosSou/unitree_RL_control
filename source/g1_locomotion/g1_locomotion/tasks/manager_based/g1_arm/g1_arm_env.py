# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 arm IK via RL — DirectRLEnv.

The policy learns to move the palm(s) to randomly sampled 3-D goal positions
using delta joint-position commands on the 5-DOF arm(s).

Single arm  (arm="left" or "right"):
    Observation (28-D): base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) |
                         joint_pos(5) | joint_vel(5) | ee_pos(3) | goal(3) | error(3)
    Action      ( 5-D): delta joint targets for the active arm

Both arms   (arm="both"):
    Observation (56-D): [left 28-D] + [right 28-D] (base-state block duplicated per arm
                         so each arm's block stays independently mirror-transformable)
    Action      (10-D): [left 5-D]  + [right 5-D]
    Episode terminates when BOTH palms reach their respective goals.

Phase 2 additions (2026-07-07, see known_issues.md / training_regimes.md):
    - Base-orientation/velocity observation terms (above) — the arm policy was
      previously blind to the base entirely.
    - A small scripted wobble fed into the base-observation terms during training (``_apply_root_wobble``),
      on a curriculum (off for the first ``root_wobble_enable_step`` env-steps, on after) so
      the policy learns core reaching on an easy/static base first, then generalizes to a
      seemingly moving/tilting one. The root itself stays physically fixed
      (``fix_root_link=True``) — this only changes what ``base_ang_vel``/``projected_gravity``
      *report*, not anything physical; an earlier version actually moved the root and
      caused the whole robot to fall over under random actions (no active balance
      controller to hold a free base up) — see known_issues.md.
    - Sim2real: observation noise (Unoise, matching G1FlatEnvCfg's magnitudes) and startup
      domain randomization (actuator gains, joint friction/armature) on the arm joints —
      previously this task had none at all.
    - A soft distance penalty keeping the end-effector away from ``torso_link`` (the
      "arm entering torso" interpenetration issue) — see the ``enabled_self_collisions``
      note in ``__post_init__`` for the complementary structural fix.

Fixed 2026-07-08 (see known_issues.md): ``joint_vel`` was silently sliced to the first 3
of 5 arm joints (``jt[:3]``) since before Phase 2 — the policy has never been able to see
elbow_pitch/elbow_roll velocity at all. Found while diagnosing the ~55% success plateau: a
difficulty-bucketed breakdown of eval failures showed a roughly *uniform* ~45% failure rate
across easy-to-hard goals (not concentrated at the hard tail, which would point at a
control-authority/reachability limit instead), and failures missed by a real margin
(median 8.5cm at timeout, not a near-miss) — a pattern more consistent with a broad control-
quality gap than a reachability one. Now observes all 5 joint velocities, bumping per-arm
observation 26-D -> 28-D (52 -> 56 for "both"). Deliberately tested alone, before any of the
3 overnight-sweep experiments (entropy/wide-net/reward-shaping) get layered on top — those
were all small, inconsistent-between-buckets effects on top of an already-mediocre baseline,
whereas this is a genuine bug fix.
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

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

# ---------------------------------------------------------------------------
# Domain randomization events (Phase 2 — see class docstring)
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Startup-only domain randomization for the arm task.

    Only the arm actuator group is randomized (payload/hardware variance on the joints
    actually being controlled) — legs/torso/fingers are held rigid (see A5 in
    __post_init__) and don't need it. Friction/mass/CoM randomization (used by
    walking/standing via G1FlatEnvCfg) isn't included here: this task has no contact
    interactions at all (pure point-goal reaching), so those wouldn't have any effect —
    only including randomization that actually reaches something the policy experiences.
    """

    arm_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    arm_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"]),
            "friction_distribution_params": (0.0, 0.03),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "abs",
        },
    )


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------


@configclass
class G1ArmIKEnvCfg(DirectRLEnvCfg):
    """Base configuration for the G1 arm IK reaching task."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    episode_length_s: float = 10.0
    decimation: int = 2  # 30 Hz control

    # Observation / action / state dims
    # 9-D base-state prefix (base_lin_vel, base_ang_vel, projected_gravity) + 19-D per arm
    # (joint_pos(5) + joint_vel(5) + ee_pos(3) + goal(3) + error(3)).
    observation_space: int = 28
    action_space: int = 5
    state_space: int = 0  # no separate critic state; must be set (None crashes serialization)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0)

    # Domain randomization (Phase 2 — see EventCfg docstring)
    events: EventCfg = EventCfg()

    # Which arm(s) to train: "left", "right", or "both"
    arm: str = "left"

    # Reward scales
    position_reward_scale: float = 10.0
    # Exponential proximity bonus, OFF by default — this is the confirmed-good baseline
    # (0.0 = off). A first attempt (scale=5.0, sigma=5cm) bundled with an entropy_coef
    # change and caused a clear regression (35.5% -> 10.9% success within one run);
    # reverted, see known_issues.md. A second, isolated attempt is now its own opt-in
    # variant, G1ArmIKLeftRewardShapeEnvCfg below (scale=1.5, sigma=2cm) — trained and
    # compared against this baseline via a separate gym task
    # (G1-Arm-IK-Left-RewardShape-v0), not by changing the default here, so the baseline
    # stays available for comparison.
    position_reward_exp_scale: float = 0.0
    position_reward_exp_sigma: float = 0.05  # 5 cm
    action_smoothness_scale: float = 0.01
    joint_limit_penalty_scale: float = 1.0
    goal_reached_bonus: float = 50.0
    goal_threshold: float = 0.02  # 2 cm
    action_scale: float = 0.5
    action_filter_alpha: float = 0.25
    max_action_delta_per_step: float = 0.06

    # Fraction of the (real hardware) joint range each reset randomizes the arm's
    # starting position within, centered on the hardware range's own midpoint — e.g. 0.15
    # means the reset spread covers 30% of the total range. Deliberately does NOT affect
    # what's reachable via actions (self._arm_hw_limits, the full hardware range, used by
    # _apply_action/_get_rewards) — only where an episode starts. Reduced from the
    # original 0.3 (Phase 2, 2026-07-07) to make typical starting poses less extreme,
    # after a *different* attempt at this (restricting the action range itself) made a
    # large fraction of _GOAL_BOUNDS unreachable — see known_issues.md.
    reset_range_fraction: float = 0.15

    # Torso-proximity penalty (Phase 2, item B2 — soft defense against the end-effector
    # entering torso_link; see the enabled_self_collisions note below for the structural
    # complement to this).
    torso_proximity_margin_m: float = 0.12
    torso_proximity_penalty_scale: float = 2.0

    # Root-pose wobble (Phase 2, item A2) — see _apply_root_wobble. Bounded, slow,
    # per-env-randomized oscillation so the arm policy is actually exposed to (and must
    # observe/adapt to) a moving reference frame, instead of a perfectly static one it's
    # never seen a disturbed version of.
    #
    # Amplitudes widened 2026-07-09 — the original roll/pitch values (0.10 rad, ~5.7°)
    # were matched to validation/eval_standing.py's summary, i.e. the *standing-alone*
    # task with no arm reaching happening. That was the wrong reference: once arm
    # reaching runs on top of standing (validation/integration_validation/eval_full_demo.py),
    # actual measured mean_max_tilt_deg during standing_arm_left_reach ran 54-64° — the
    # policy had only ever been asked to cope with ~1/10th of the tilt it actually meets
    # in the field, which was borne out concretely: a standalone diagnostic against the
    # actual trained checkpoint (no Isaac Sim — reconstructed the actor MLP straight from
    # the checkpoint's state_dict) found the network's response to base-tilt features
    # diverges further from mirror-symmetric the larger the tilt gets, i.e. behavior in
    # that regime is close to undefined/untrained rather than a considered response. New
    # values below aim to cover "starting to struggle but still plausibly recoverable"
    # (roughly 3.5x the old cap), not the 60°+ range, which is already-falling territory
    # no arm behavior is going to fix regardless of how well-trained it is. First-pass
    # estimate from the integration eval's numbers, not a rigorously tuned value — revisit
    # after seeing how a policy trained under this range actually performs.
    #
    # Two axes added at the same time, previously always exactly zero even after the
    # curriculum turned on: yaw rate and base linear velocity. The base_lin_vel comment
    # used to read "faking a linear velocity... risked a misleading training signal for
    # no real benefit" — reasonable when nothing else here modeled *why* the base would
    # be moving, but the integration eval now shows real, non-negligible base_lin_vel
    # during actual standing+reach failures, so leaving the arm policy with zero training
    # exposure to it is itself the misleading signal (an always-static-base assumption
    # deployment doesn't honor). Driven by the same smooth per-env-randomized sinusoid
    # style as roll/pitch, not white noise, for the same reason the original comment
    # cared about a "clean" signal — see _apply_root_wobble.
    #
    # Curriculum, not constant-on: phase 0 (env-steps < root_wobble_enable_step) has no
    # wobble at all, so the policy learns the core reaching skill on an easy/static base
    # first; phase 1 turns it on. Same reasoning and mechanism (env.common_step_counter
    # threshold) as StandingArmTrajectoryDisturbance's curriculum in
    # g1_locomotion/mdp/events.py, just two phases instead of five — this disturbance is
    # far milder, doesn't need more. Default boundary is 25% of the default 5000-iteration
    # budget (num_steps_per_env=24 -> 120,000 env-steps total, so 30,000 here).
    #
    # Visually confirmed via scripts/zero_agent.py (2026-07-07) for the original
    # (narrower) roll/pitch-only version: smooth per-env wobble, no jitter/explosions,
    # self-collision behaved, some minor coupled leg movement (expected — the root is
    # kinematically re-snapped every step, so the hip joints feel a brief reaction each
    # time before the legs' strong PD hold corrects it back). The widened range and the
    # two new axes below have NOT been visually re-verified the same way — do that via
    # scripts/zero_agent.py before trusting this for a real training run, same standing
    # instruction as any other untested curriculum change in this repo.
    enable_root_wobble: bool = True
    root_wobble_enable_step: int = 30_000
    root_wobble_max_roll_rad: float = 0.35  # ~20 deg, was 0.10 (~5.7 deg)
    root_wobble_max_pitch_rad: float = 0.35  # ~20 deg, was 0.10 (~5.7 deg)
    root_wobble_max_yaw_rate_rad_s: float = 0.5  # new axis, was always exactly 0
    root_wobble_max_lin_vel_mps: float = 0.3  # new axis, was always the real (~0) reading
    root_wobble_freq_hz_range: tuple[float, float] = (0.1, 0.3)

    # Observation noise magnitudes — matches the same terms/values G1FlatEnvCfg already
    # uses for walking/standing (this task previously had zero observation noise at all).
    base_lin_vel_noise: float = 0.1
    base_ang_vel_noise: float = 0.2
    projected_gravity_noise: float = 0.05
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 1.5

    # Null-space regularization (2026-07-08, isolated retrain #2 on top of the joint_vel
    # fix — see known_issues.md). This is a 5-DOF arm reaching a 3-DOF (position-only)
    # goal: for most goals there's a whole family of joint configurations that all reach
    # the same point (manipulator redundancy), and a standard PPO policy's unimodal
    # Gaussian action distribution is a poor structural fit for a target with multiple
    # equally-valid solutions — it can waver between them instead of consistently
    # committing to one. This term breaks the tie: a small, constant penalty for the
    # arm's *joint angles* drifting far from a fixed reference pose (default_joint_pos,
    # the asset's own rest pose — same value reset() and the interactive test scripts
    # already use as "home"), biasing the policy toward one consistent solution per goal
    # instead of leaving the redundant DOF free to wander. Deliberately weak relative to
    # position_reward_scale so reaching the goal always dominates when the goal is far
    # from the reference pose — this is a tiebreaker among valid solutions, not a leash
    # pulling the arm back to rest. Does NOT touch end-effector orientation at all (the
    # goal stays position-only, x/y/z) — this is a joint-space preference, unrelated to
    # what direction the palm points.
    null_space_penalty_scale: float = 0.05

    # Goal-difficulty curriculum — one of 3 isolated overnight experiments (2026-07-07)
    # targeting the confirmed ~55% success plateau (see known_issues.md). OFF by default
    # (this is the baseline); turned on only by G1ArmIKLeftGoalCurriculumEnvCfg below.
    # Idea: start episodes with goals confined to a shrunk region around the workspace
    # centre (easier reaches), then linearly widen to the full _GOAL_BOUNDS box as
    # training progresses — same env-step-gated curriculum mechanism as the root wobble
    # above, applied to goal sampling instead. Rationale: if part of the plateau is the
    # policy failing to ever nail the hardest (edge-of-workspace) goals early on and
    # settling into a local optimum that ignores them, learning core reaching on easy
    # goals first (curriculum learning) may leave more capacity/exploration for the hard
    # ones once they're introduced, vs. facing the full difficulty distribution from step 0.
    enable_goal_curriculum: bool = False
    goal_curriculum_start_fraction: float = 0.4  # 40% of full box extent at step 0
    goal_curriculum_full_step: int = 60_000  # linearly reaches 100% by here (50% of budget)

    # Goal-box x-range override (2026-07-08, elbow-extension stress test — see
    # known_issues.md). None = use _GOAL_BOUNDS as-is (default). When set, replaces just
    # the x-range for every active arm, leaving y/z untouched — used to restrict training
    # to the outer "far face" of the box (the region needing near-full elbow extension,
    # per the joint-config correlation check), instead of the full volume. x rather than
    # y/z specifically because x (forward reach) was already established as the dominant
    # lever for reach difficulty (see the reachability check's per-axis octant data).
    goal_bounds_x_override: tuple[float, float] | None = None

    # Robot — G1 asset with prim_path set for multi-env cloning.
    # Uses G1_MINIMAL_CFG (fewer collision meshes than G1_CFG) since this task doesn't
    # need locomotion-grade collision fidelity on the legs/feet — same joint/articulation
    # structure, cheaper to simulate. Arm actuator stiffness/damping are overridden in
    # __post_init__ for responsive position tracking (defaults are tuned for torque control).
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        super().__post_init__()
        # Root is NOT rigidly welded (Phase 2 change, was fix_root_link=True): the arm
        # policy needs a base it can actually observe moving (see root wobble above), and
        # a real PhysX fixed joint can't be reprogrammed at runtime. Instead the root is
        # fully self-managed every control step (_apply_root_wobble writes its pose/
        # velocity directly, kinematically) — it is NOT free-falling/balancing, since we
        # overwrite it every step before physics has a chance to meaningfully integrate
        # gravity on it. This is NOT a balance task: only the pelvis root pose is
        # scripted; legs/waist/fingers are held rigid below (A5), only arm joints are
        # RL-actuated.
        #
        # REVERTED 2026-07-07 after an actual play test: fix_root_link=False (a genuinely
        # free root) turned this into an unintended balance task. With no active balance
        # controller (legs are just PD-held to a fixed pose, no hip/ankle/stepping
        # strategy), the robot was only standing on passive leg stiffness + ground
        # contact — thin enough margin that random arm actions (real momentum, unlike
        # zero actions) tipped it over forward every time. See known_issues.md. Root
        # stays fixed; the wobble is now a synthetic *observation* signal only (see
        # _apply_root_wobble) — the policy still sees varying tilt/rotation-rate values
        # it must be robust to, without any physical motion or fall risk.
        self.robot.spawn.articulation_props.fix_root_link = True

        # Self-collision (Phase 2, item B1 — the structural fix for "arm entering
        # torso"). Every G1 asset variant in isaaclab_assets ships with
        # enabled_self_collisions=False by default. Enabling it here is the real,
        # physical guarantee (vs. B2's soft penalty below, which is a nudge). Unlike
        # deformable bodies, rigid articulations have no self_collision_filter_distance
        # safeguard against rest-pose collision-mesh overlap — NOT YET VISUALLY VERIFIED
        # for stability or profiled for step-time cost. Check via play.py (small
        # num_envs) before trusting this in a real training run, same as the root wobble
        # above.
        self.robot.spawn.articulation_props.enabled_self_collisions = True

        # This task never needs ground-contact resolution (nothing touches the ground) —
        # the inherited iteration counts (8/4) are tuned for locomotion contact fidelity
        # we don't need here. Note: legs/torso/fingers are still full simulated DOFs (not
        # removed) since there's no lighter "arms-only" asset available; this only cuts
        # solver cost, not DOF count. Revisit with a stripped asset if profiling ever
        # shows this isn't enough.
        self.robot.spawn.articulation_props.solver_position_iteration_count = 4
        self.robot.spawn.articulation_props.solver_velocity_iteration_count = 1

        # Reverted 2026-07-12 to 200/20 after a same-day attempt to switch to Unitree's
        # real arm5-SDK gains (Kp=60, Kd=1.5 — unitree_sdk2_python's
        # example/g1/high_level/g1_arm5_sdk_dds_example.py) measurably regressed reach
        # quality when retrained under it: success rate 86% -> 30% in this task's own
        # native eval, nothing to do with deployment mismatch. Root cause, confirmed by
        # reading isaaclab's ImplicitActuator.compute() directly: it's pure PD
        # (stiffness*pos_error + damping*vel_error), no gravity-compensation feedforward
        # anywhere in the pipeline. The real SDK's low gains are almost certainly meant to
        # ride on top of a separate gravity-comp term in the real controller (that
        # example's own weight/weight_rate blend-in over its first 3s is exactly that
        # pattern) — applied as the *only* torque source here, the arm is simply too weak
        # to hold itself against gravity, let alone track a target precisely. Not worth
        # chasing sim-to-real gain fidelity for a sim-only pipeline anyway; 200/20 is what
        # this task's own checkpoints actually perform well under.
        self.robot.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ],
            stiffness=200.0,
            damping=20.0,
        )
        # Legs/waist ("legs" actuator group, which also covers torso_joint) and feet
        # ("feet" group) are left at G1_MINIMAL_CFG's own stock gains (150-200 stiffness
        # for legs+torso, 20 for ankles) — deliberately kept, not silently inherited: they
        # were already reasonably strong for a "stay put" role, verified here rather than
        # assumed (finding #6 / A5).
        #
        # Fingers previously had NO actuator group at all: the "arms" override above
        # replaces that dict key wholesale, and G1_MINIMAL_CFG's stock "arms" group
        # covers shoulder/elbow *and* every finger joint — so overriding "arms" to only
        # shoulder/elbow silently dropped finger actuation entirely (they'd have gone
        # fully limp under gravity). Real bug, found while doing A5; fixed here with a
        # dedicated modest PD hold since fingers aren't otherwise used in this task.
        self.robot.actuators["fingers"] = ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint",
                ".*_zero_joint", ".*_one_joint", ".*_two_joint",
            ],
            stiffness=20.0,
            damping=2.0,
            armature=0.001,
        )
        # Contact sensors not needed for this task
        self.robot.spawn.activate_contact_sensors = False


@configclass
class G1ArmIKLeftEnvCfg(G1ArmIKEnvCfg):
    arm: str = "left"
    observation_space: int = 28
    action_space: int = 5


@configclass
class G1ArmIKRightEnvCfg(G1ArmIKEnvCfg):
    arm: str = "right"
    observation_space: int = 28
    action_space: int = 5


@configclass
class G1ArmIKBothEnvCfg(G1ArmIKEnvCfg):
    """Both arms trained simultaneously (56-D obs, 10-D actions)."""
    arm: str = "both"
    observation_space: int = 56
    action_space: int = 10


@configclass
class G1ArmIKLeftRewardShapeEnvCfg(G1ArmIKLeftEnvCfg):
    """Experiment 1/4 (2026-07-07 overnight sweep): exponential proximity bonus, isolated.

    Only this one term differs from G1ArmIKLeftEnvCfg's baseline — see the field's
    docstring in G1ArmIKEnvCfg for the reasoning and how this differs from the first
    (bundled, regressed) attempt. entropy_coef and everything else stays at baseline
    (see G1ArmIKLeftRewardShapePPORunnerCfg in agents/rsl_rl_ppo_cfg.py).
    """
    position_reward_exp_scale: float = 1.5
    position_reward_exp_sigma: float = 0.02


@configclass
class G1ArmIKLeftGoalCurriculumEnvCfg(G1ArmIKLeftEnvCfg):
    """Experiment 2/4 (2026-07-07 overnight sweep): goal-difficulty curriculum, isolated.

    Only enable_goal_curriculum differs from baseline — see its docstring in
    G1ArmIKEnvCfg. Reward/network/entropy all stay at baseline.
    """
    enable_goal_curriculum: bool = True


@configclass
class G1ArmIKLeftStressRegionEnvCfg(G1ArmIKLeftEnvCfg):
    """1b (2026-07-08): isolated diagnostic — train exclusively on the elbow-extension
    stress region (see known_issues.md), to test whether focused training there actually
    improves precision, before investing in a proper curriculum (1a) if it does.

    Restricts x to the outer ~30% of the box (0.35-0.42 vs the full 0.20-0.42), y/z
    untouched — traces the whole "far face" of the box (all the corners on that face and
    everything between them), not one isolated corner, so the region has real variety
    rather than being a single repeated point.
    """
    goal_bounds_x_override: tuple[float, float] = (0.35, 0.42)


@configclass
class G1ArmIKLeftEnvCfg_PLAY(G1ArmIKLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


@configclass
class G1ArmIKRightEnvCfg_PLAY(G1ArmIKRightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


@configclass
class G1ArmIKBothEnvCfg_PLAY(G1ArmIKBothEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

# Joint names for each arm (must match the G1 USD articulation)
_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
]
_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]
_LEFT_EE_BODY = "left_palm_link"
_RIGHT_EE_BODY = "right_palm_link"
_TORSO_BODY = "torso_link"

# Goal workspace bounds in the robot's local frame [m]
# (env origin is added later to convert to world frame)
#
# Reshaped 2026-07-08 (see known_issues.md) after validation/check_arm_reachability.py
# showed only ~47% of the previous box (x:0.1-0.5, y:0.05-0.45, z:0.9-1.2) was within 2cm
# of anything kinematically reachable — a hard ceiling on success rate no amount of
# training could cross. Two distinct problems, fixed on opposite edges of the box:
#   - Far corner (max x, max y, max z simultaneously) was genuinely unreachable — needs
#     max forward + max lateral + max height reach all at once, beyond arm length. Pulled
#     all three upper bounds in.
#   - Near corner (min x, min y, min z) wasn't unreachable, but sits ~3cm from
#     torso_link's collision geometry (derived from the real G1 URDF) — well inside the
#     12cm safety margin torso_proximity_penalty_scale already enforces, so the policy
#     was being asked to violate its own anti-collision incentive to succeed there.
#     Pushed the x lower bound out to clear the margin.
#
# Refined again 2026-07-08 after training on the first reshape (74.7%/76.8% success, up
# from ~55-59% — confirms reachability was the dominant factor) and re-running the
# reachability check against the *new* bounds: coverage improved a lot (2cm tolerance:
# 47.1% -> 65.3%, median nearest-reachable distance 2.65cm -> 0.76cm) but the low-x/low-y
# octant was still clearly the worst (mean 4.1-4.6cm, vs <1cm for high-x/low-y) — the
# first x push (0.10->0.15) helped but didn't fully clear the torso margin. Per-axis
# octant data showed x is the dominant lever (holding y low, moving x from low->high
# alone dropped that corner's mean distance from ~4.6cm to ~0.9cm) with y a secondary
# contributor (~4.6cm -> ~1.8cm) — pushed x further, y a smaller amount.
_GOAL_BOUNDS = {
    "left":  {"x": (0.20, 0.42), "y": (0.08, 0.40), "z": (0.9, 1.15)},
    "right": {"x": (0.20, 0.42), "y": (-0.40, -0.08), "z": (0.9, 1.15)},
}

# REVERTED 2026-07-07 — see known_issues.md. A first attempt at this used hand-picked
# absolute joint-angle bounds per joint, reasoned from real hardware limits + the goal
# workspace geometry but never verified against actual forward kinematics (no Isaac Sim
# access to check). It made a large fraction of _GOAL_BOUNDS unreachable — a 1500-
# iteration test run plateaued at 0% success and a median 17.7cm from goal (vs. a 2cm
# threshold), with a wide, bimodal-looking distance distribution (some goals solved
# well, most stuck at a geometric floor) that's the signature of an unreachable-goal
# problem, not a "needs more training" one. Picking new absolute bounds by hand again
# risks the same mistake. Fixed differently below: the action-target clamp goes back to
# the real hardware range (restores exactly the reachability the task had before any of
# this — reaching converged fine under it previously), and only *reset* randomization
# gets a proportionally narrower spread (see RESET_RANGE_FRACTION in G1ArmIKEnvCfg) —
# safe regardless of exact value, since it only affects the starting pose, never what's
# reachable via the policy's own actions.


class G1ArmIKEnv(DirectRLEnv):
    """G1 arm reaching environment using direct RL (``DirectRLEnv``).

    Supports arm="left", "right", or "both".  When arm="both" the observation
    and action spaces are doubled and both palms must reach their goals for the
    episode to terminate early.
    """

    cfg: G1ArmIKEnvCfg

    def __init__(self, cfg: G1ArmIKEnvCfg, render_mode: str | None = None, **kwargs):
        # Nothing arm-specific needed before super().__init__ now
        super().__init__(cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------
        # Post-init: build arm-group list (safe after sim is running)
        # ------------------------------------------------------------------
        # Each entry: {joint_tensor, ee_idx, bounds}
        def _bounds_for(side: str) -> dict:
            b = dict(_GOAL_BOUNDS[side])
            if cfg.goal_bounds_x_override is not None:
                b["x"] = cfg.goal_bounds_x_override
            return b

        self._arm_groups: list[dict] = []
        if cfg.arm in ("left", "both"):
            ids, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            ee, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            self._arm_groups.append({
                "joint_tensor": torch.tensor(ids, dtype=torch.long, device=self.device),
                "ee_idx": ee[0],
                "bounds": _bounds_for("left"),
            })
        if cfg.arm in ("right", "both"):
            ids, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            ee, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            self._arm_groups.append({
                "joint_tensor": torch.tensor(ids, dtype=torch.long, device=self.device),
                "ee_idx": ee[0],
                "bounds": _bounds_for("right"),
            })

        self.n_arms = len(self._arm_groups)

        # Combined joint tensor (5 for single arm, 10 for both) — used for reset
        self.arm_joint_indices_tensor = torch.cat(
            [g["joint_tensor"] for g in self._arm_groups]
        )
        self.arm_joint_indices = self.arm_joint_indices_tensor.tolist()

        # Per-joint null-space weight (2026-07-08, joint-config correlation check round
        # 2 — see known_issues.md). Goal-position logging showed the ~22% "near-limit
        # elbow" failure mode isn't tied to any goal-box region at all — it's a free,
        # roughly goal-independent choice between two redundant solution branches for
        # the *same* goals (one with elbow_pitch bent ~42deg, ~91% success; one with it
        # pinned near -0.1deg, ~59% success). The plain (unweighted) null-space penalty
        # doesn't discriminate between the branches well: total deviation-from-default
        # across all 5 joints is actually *similar* for both (~79deg bad vs ~82deg
        # good), because shoulder_pitch is far from default in both branches, diluting
        # the one joint that actually differs a lot (elbow_pitch: ~50deg off in the bad
        # branch vs ~8deg off in the good one). Weighting elbow_pitch's contribution 3x
        # separates the branches clearly (~162deg bad vs ~85deg good) — a much sharper,
        # more targeted signal than just turning up the scale uniformly, which would
        # pressure all 5 joints toward default without specifically discouraging the
        # branch that's actually the problem.
        joint_names_all = self.robot.data.joint_names
        for arm in self._arm_groups:
            weights = [3.0 if "elbow_pitch" in joint_names_all[idx] else 1.0 for idx in arm["joint_tensor"].tolist()]
            arm["null_space_weight"] = torch.tensor(weights, device=self.device)

        # Real hardware joint range (soft_joint_pos_limits) — the action-target clamp and
        # the joint-limit-avoidance reward both use this directly (see _apply_action /
        # _get_rewards), restoring exactly the reachability this task had before the
        # reverted tighter-range attempt above. Cached once here since it's read every
        # step/reset.
        self._arm_hw_limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_indices_tensor].clone()

        # Per-joint, per-bound joint-limit-penalty margin fraction (2026-07-08, joint-
        # config correlation check — see known_issues.md). Found via the new
        # *_deg_at_min_dist eval columns: elbow_pitch's range (-2.5deg, 185.5deg) is
        # heavily asymmetric — 0deg is roughly a straight, fully-extended arm, and the
        # joint barely goes past straight (-2.5deg) but folds a lot (up to 185.5deg).
        # Full extension is a normal, necessary pose for far reaches, not a dangerous
        # edge case — but the flat 5%-of-range margin applied to every joint uniformly
        # doesn't know that, and was penalizing exactly the pose needed for precise far
        # reaches. Deliberately asymmetric, not just a smaller flat margin for the whole
        # joint: 100% of the near-limit failures found were at the lower bound (-2.5deg),
        # 0% at the upper (185.5deg), so only the lower-bound margin is reduced — the
        # upper bound (and both bounds on the other 4 joints, which showed zero
        # near-limit involvement in either success or failure episodes) keep the
        # original, more conservative 5%. Shape (n_joints, 2), columns are [lower, upper].
        joint_names = self.robot.data.joint_names
        margin_lo, margin_hi = [], []
        for idx in self.arm_joint_indices:
            name = joint_names[idx]
            is_elbow_pitch = "elbow_pitch" in name
            margin_lo.append(0.01 if is_elbow_pitch else 0.05)
            margin_hi.append(0.05)
        self._joint_limit_margin_fraction = torch.tensor(
            list(zip(margin_lo, margin_hi)), device=self.device
        )

        # Torso body index, for the proximity penalty (B2)
        torso_ids, _ = self.robot.find_bodies(_TORSO_BODY)
        self._torso_body_idx = torso_ids[0]

        # Runtime buffers
        # goal_positions: (num_envs, n_arms, 3)
        self.goal_positions = torch.zeros((self.num_envs, self.n_arms, 3), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.filtered_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Root-pose wobble state (A2) — synthetic *observation* signal only, root itself
        # stays physically fixed (see __post_init__ for why). default_root_quat is the
        # reference orientation the fake wobble rotates on top of when computing what
        # projected_gravity/base_ang_vel *would* read if the base were actually tilting.
        self._default_root_quat = self.robot.data.default_root_state[:, 3:7].clone()
        self._wobble_roll_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_pitch_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_roll_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_pitch_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_roll_phase = torch.zeros(self.num_envs, device=self.device)
        self._wobble_pitch_phase = torch.zeros(self.num_envs, device=self.device)
        # Yaw-rate axis — single sinusoid (yaw itself isn't observed, only its rate, so
        # there's no analogous "yaw angle" to track the way roll/pitch track a fake quat).
        self._wobble_yaw_amp = torch.zeros(self.num_envs, device=self.device)
        self._wobble_yaw_freq = torch.zeros(self.num_envs, device=self.device)
        self._wobble_yaw_phase = torch.zeros(self.num_envs, device=self.device)
        # Linear-velocity axis — one shared amp/freq/phase driving a circular drift
        # (lin_vel_x = amp*cos, lin_vel_y = amp*sin(phase-shifted)) rather than two fully
        # independent oscillators, so the direction smoothly rotates instead of two
        # unrelated sinusoids happening to overlap; z left at 0 (no vertical bobbing
        # modeled, root height isn't part of this observation anyway).
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

        # Ground plane (global — not cloned per env)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Clone environments and filter inter-env collisions
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

    # ------------------------------------------------------------------
    # Root-pose wobble (Phase 2, item A2)
    # ------------------------------------------------------------------

    def _apply_root_wobble(self):
        """Compute a synthetic bounded roll/pitch/yaw-rate/lin-vel wobble for the
        *observation* only.

        The root itself stays physically fixed (fix_root_link=True) — this does not move
        anything. It only fills self._synthetic_ang_vel / self._synthetic_projected_gravity
        / self._synthetic_lin_vel with what those observation terms *would* read if the
        base were actually tilting and drifting, so the policy still has to learn to reach
        despite a varying base-state input.

        REVERTED from an earlier version that used real (fix_root_link=False) kinematic
        root pose writes: with no active balance controller, a free root left the whole
        robot standing on nothing but passive leg stiffness, and random arm actions
        reliably tipped it over forward — see known_issues.md. This synthetic-only
        version carries zero fall risk since nothing physically moves, at the cost of not
        capturing how a real tilting/drifting base would also change the arm's own
        gravity-compensation dynamics — judged an acceptable trade (see
        root_wobble_max_roll_rad's docstring for how the amplitude/axis choices below
        were set, and why the original "a few degrees, roll/pitch only" scope changed).
        """
        if not self.cfg.enable_root_wobble:
            return
        if self.common_step_counter < self.cfg.root_wobble_enable_step:
            # Curriculum phase 0: report the real (untilted, undrifting) sensor values.
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
        self._synthetic_ang_vel[:, 0] = roll_rate  # small-angle approx: body-frame roll/pitch rate
        self._synthetic_ang_vel[:, 1] = pitch_rate
        self._synthetic_ang_vel[:, 2] = yaw_rate

        # Circular drift: one amp/freq/phase pair drives both x and y (y phase-shifted by
        # 90°) so the drift direction smoothly rotates over time instead of two unrelated
        # sinusoids happening to overlap. z left at 0 — no vertical bobbing modeled.
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

        # Filter and clamp action deltas to emulate actuator lag and avoid abrupt jumps.
        # Works for single arm (5 joints) and both arms (10 joints) identically.
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

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Base-state prefix (Phase 2, A1) — same per-arm block, duplicated for "both" so
        # each 26-D arm block stays independently mirror-transformable (see A4 plan).
        # base_lin_vel/base_ang_vel/projected_gravity all read the synthetic wobble
        # buffers (see _apply_root_wobble), not the real (always-static) sensor values —
        # the root is physically fixed, this is what makes it *look* tilted and drifting
        # to the policy without actually moving anything. base_lin_vel used to be left as
        # the real (always ~0) reading — see root_wobble_max_lin_vel_mps's docstring for
        # why that changed.
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

        # Build per-arm block, then concatenate (→ 28 or 56 total)
        parts = []
        for i, arm in enumerate(self._arm_groups):
            jt = arm["joint_tensor"]
            joint_pos = uniform_noise(
                self.robot.data.joint_pos[:, jt],
                UniformNoiseCfg(n_min=-self.cfg.joint_pos_noise, n_max=self.cfg.joint_pos_noise),
            )  # (N, 5)
            joint_vel = uniform_noise(
                self.robot.data.joint_vel[:, jt],
                UniformNoiseCfg(n_min=-self.cfg.joint_vel_noise, n_max=self.cfg.joint_vel_noise),
            )  # (N, 5) — was jt[:3] (elbow_pitch/elbow_roll vel invisible), see module docstring
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
            # Exponential proximity bonus — steep, localized gradient near the goal that
            # the linear term above doesn't provide on its own (see cfg field docstring).
            total += torch.exp(-dist / self.cfg.position_reward_exp_sigma) * self.cfg.position_reward_exp_scale
            reached = dist < self.cfg.goal_threshold
            total += reached.float() * self.cfg.goal_reached_bonus
            all_reached &= reached

            # Soft penalty (B2): discourage the end-effector getting too close to the
            # torso — the actual structural fix is enabled_self_collisions=True in
            # __post_init__ (B1), this is a cheap, safe complement/fallback.
            torso_dist = torch.norm(ee_pos - torso_pos, dim=-1)
            total += -torch.clamp(margin - torso_dist, min=0.0) * self.cfg.torso_proximity_penalty_scale

            # Null-space regularization — see cfg field docstring. A soft tiebreaker
            # among the multiple joint configurations that reach the same (x, y, z)
            # goal, not a constraint on the goal itself. Per-joint weighted (see
            # arm["null_space_weight"] in __init__) — elbow_pitch counts 3x so this
            # actually discriminates between the two solution branches found in the
            # joint-config correlation check, instead of being diluted by shoulder_pitch
            # (which is far from default in both branches and doesn't distinguish them).
            jt = arm["joint_tensor"]
            ref_pose = self.robot.data.default_joint_pos[:, jt]
            pose_deviation = torch.norm(
                (self.robot.data.joint_pos[:, jt] - ref_pose) * arm["null_space_weight"], dim=-1
            )
            total += -pose_deviation * self.cfg.null_space_penalty_scale

        # Smoothness and joint-limit penalties (shared across all arm joints)
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

        # Episode terminates when ALL arms have reached their goals
        self.successes = all_reached
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

        # Start from default standing pose, randomise only the arm joints
        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(joint_pos)

        # Randomize around the hardware range's own center, same formula this task used
        # before Phase 2 (known to converge fine) — only the fraction shrank (0.3 -> see
        # cfg.reset_range_fraction), to make typical starting poses less extreme without
        # touching what's reachable via actions (self._arm_hw_limits, unchanged, full
        # hardware range — see known_issues.md for why a *tighter* range there broke
        # reachability instead).
        for i, idx in enumerate(self.arm_joint_indices):
            lo = self._arm_hw_limits[i, 0]
            hi = self._arm_hw_limits[i, 1]
            centre = (lo + hi) * 0.5
            half = (hi - lo) * self.cfg.reset_range_fraction
            joint_pos[:, idx] = centre + (torch.rand(num_resets, device=self.device) * 2 - 1) * half

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)

        # Root stays physically fixed (fix_root_link=True) — no explicit root pose/
        # velocity reset needed, the fixed joint handles that. Only the *synthetic*
        # wobble profile (used purely for the observation, see _apply_root_wobble) needs
        # re-sampling each episode.

        # Re-sample this episode's wobble profile (amplitude drawn from [0, max] so some
        # envs get little/no wobble and others get the full range — variety, not a
        # constant worst case every episode).
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

        # Sample goals in robot-local frame, shift to world frame via env origins
        goal_fraction = self._current_goal_curriculum_fraction()
        for i, arm in enumerate(self._arm_groups):
            self.goal_positions[env_ids_tensor, i, :] = (
                self._sample_goal_positions(num_resets, arm["bounds"], goal_fraction)
                + self.scene.env_origins[env_ids_tensor]
            )

        # Reset buffers
        self.previous_actions[env_ids_tensor] = 0.0
        self.filtered_actions[env_ids_tensor] = 0.0
        self.successes[env_ids_tensor] = False
        self.episode_length_buf[env_ids_tensor] = 0

        self._update_goal_markers(env_ids_tensor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_position(self, arm_index: int = 0) -> torch.Tensor:
        """End-effector world position for one arm group (N, 3)."""
        return self.robot.data.body_pos_w[:, self._arm_groups[arm_index]["ee_idx"], :]

    def _current_goal_curriculum_fraction(self) -> float:
        """Fraction (0-1] of the full goal-box extent to sample from right now.

        See enable_goal_curriculum's docstring in G1ArmIKEnvCfg. Off (returns 1.0, full
        range) unless the cfg flag is set; linearly ramps start_fraction -> 1.0 by
        goal_curriculum_full_step env-steps, then stays at 1.0.
        """
        if not self.cfg.enable_goal_curriculum:
            return 1.0
        full_step = self.cfg.goal_curriculum_full_step
        start = self.cfg.goal_curriculum_start_fraction
        if full_step <= 0:
            return 1.0
        progress = min(1.0, self.common_step_counter / full_step)
        return start + (1.0 - start) * progress

    def _sample_goal_positions(self, num_goals: int, bounds: dict, fraction: float = 1.0) -> torch.Tensor:
        """Sample reachable goals in the robot's local frame for one arm.

        fraction scales each axis's sampling range around the box's own centre (1.0 =
        full bounds, <1.0 = a smaller box centred the same place) — see
        enable_goal_curriculum / _current_goal_curriculum_fraction.
        """
        goals = torch.zeros((num_goals, 3), device=self.device)
        for i, key in enumerate(("x", "y", "z")):
            lo, hi = bounds[key]
            centre = (lo + hi) * 0.5
            half_span = (hi - lo) * 0.5 * fraction
            goals[:, i] = centre + (torch.rand(num_goals, device=self.device) * 2 - 1) * half_span
        return goals

    def _update_goal_markers(self, env_ids: torch.Tensor):
        # Flatten (n_reset_envs, n_arms, 3) → (n_reset_envs * n_arms, 3) for the marker visualizer
        self.goal_markers.visualize(self.goal_positions[env_ids].reshape(-1, 3))
