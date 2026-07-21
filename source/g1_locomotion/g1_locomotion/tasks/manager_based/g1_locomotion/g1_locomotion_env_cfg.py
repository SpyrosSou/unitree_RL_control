# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomotion environment configurations.

These classes inherit from Isaac Lab's pre-built G1 velocity-tracking environments.
Override any field here to tune or extend the base configs.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp

# For mdp.base_height_l2 — a stock Isaac Lab reward, not one of this project's custom
# terms, so it isn't in the project's own `mdp` package above. Aliased to avoid shadowing
# that import.
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp

# symmetry.py is deliberately not part of the mdp package's star-import (see that file —
# rsl_rl_ppo_cfg.py already imports compute_symmetric_states from it directly the same way).
from .mdp.symmetry import leg_symmetry_l2

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)


@configclass
class G1LocomotionFlatEnvCfg(G1FlatEnvCfg):
    """G1 flat-terrain locomotion (customisable).

    Inherits the full flat-ground G1 setup from Isaac Lab.
    Add per-project overrides below, e.g.:
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
    """

    def __post_init__(self):
        super().__post_init__()

        # Self-collision (Phase 2, item B1 in the roadmap plan — the "arm entering
        # torso" fix). G1_MINIMAL_CFG ships with enabled_self_collisions=False; this is
        # a shared-asset property so it's enabled here too, not just for the arm task,
        # but it's NOT YET VISUALLY VERIFIED for stability or profiled for step-time
        # cost — check via play.py (small num_envs) before the next real training run,
        # same caveat as g1_arm_env.py's identical override.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        base_velocity = getattr(self.commands, "base_velocity", None)
        if base_velocity is not None:
            # Add frequent command changes and a small near-zero/zero slice.
            if hasattr(base_velocity, "resampling_time_range"):
                base_velocity.resampling_time_range = (0.9, 2.6)
            if hasattr(base_velocity, "rel_standing_envs"):
                base_velocity.rel_standing_envs = 0.10
            if hasattr(base_velocity, "ranges"):
                ranges = base_velocity.ranges
                if hasattr(ranges, "lin_vel_x"):
                    ranges.lin_vel_x = (-0.5, 1.0)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-0.6, 0.6)
                if hasattr(ranges, "ang_vel_z"):
                    ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1LocomotionFlatTransitionEnvCfg(G1FlatEnvCfg):
    """Flat locomotion config with transition-heavy command sampling.

    Designed to expose frequent stop/start, direction reversal, and short pulses.
    """

    def __post_init__(self):
        super().__post_init__()

        # See G1LocomotionFlatEnvCfg's __post_init__ for the rationale/caveat.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        base_velocity = getattr(self.commands, "base_velocity", None)
        if base_velocity is not None:
            if hasattr(base_velocity, "rel_standing_envs"):
                base_velocity.rel_standing_envs = 0.30
            if hasattr(base_velocity, "resampling_time_range"):
                base_velocity.resampling_time_range = (0.5, 1.6)
            if hasattr(base_velocity, "ranges"):
                ranges = base_velocity.ranges
                if hasattr(ranges, "lin_vel_x"):
                    ranges.lin_vel_x = (-1.0, 1.0)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-0.9, 0.9)
                if hasattr(ranges, "ang_vel_z"):
                    ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1LocomotionFlatEnvCfg_PLAY(G1LocomotionFlatEnvCfg):
    """Play / evaluation variant of the flat env (fewer envs, no randomisation).

    REVERTED FROM A BUG 2026-07-07: this used to inherit from Isaac Lab's stock
    G1FlatEnvCfg_PLAY directly instead of from G1LocomotionFlatEnvCfg above — meaning
    every project-level customization on walking (command ranges, and critically
    enabled_self_collisions=True) silently never applied to this PLAY variant, only to
    real training. g1_full_demo.py and any other PLAY-based interactive testing were
    therefore always running under stock Isaac Lab settings, self-collision included —
    this is why "arm colliding with hips during walking" was still visible in the full
    demo despite the self-collision fix already being in place for training. Fixed by
    inheriting from the customized class instead, and re-applying the same PLAY-specific
    tweaks (fewer envs, no observation corruption, no random pushes) the stock
    G1FlatEnvCfg_PLAY makes — same pattern G1LocomotionFlatTransitionEnvCfg_PLAY and
    G1LocomotionStandingFlatEnvCfg_PLAY already correctly use below.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1LocomotionFlatTransitionEnvCfg_PLAY(G1LocomotionFlatTransitionEnvCfg):
    """Play variant for transition-heavy walking policy."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1LocomotionRoughEnvCfg(G1RoughEnvCfg):
    """G1 rough-terrain locomotion (customisable).

    Inherits the full rough-terrain G1 setup from Isaac Lab.
    """

    pass


@configclass
class G1LocomotionRoughEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play / evaluation variant of the rough env (fewer envs, no randomisation)."""

    pass


@configclass
class G1LocomotionStandingFlatEnvCfg(G1FlatEnvCfg):
    """Flat terrain standing-only policy configuration.

    Commands are mostly zero with tiny pulses so the policy can learn
    minimal corrective stepping instead of only in-place balancing.
    """

    def __post_init__(self):
        super().__post_init__()

        # See G1LocomotionFlatEnvCfg's __post_init__ for the rationale/caveat. Especially
        # relevant here since standing's arm-motion disturbance curriculum is exactly the
        # mechanism that swings an arm close to the torso.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        # Kept at 60/1.5 (2026-07-12) — unlike the same-day attempt to use this value for
        # the arm-IK task (reverted there, see g1_arm_env.py — regressed reach accuracy
        # badly), this softer gain measurably *helped* standing: fall rate 0% either way,
        # but mean tilt in eval_standing_ikreach.py dropped from 26.3deg (old 25/8) to
        # 5.8deg (this value) — the best standing result across every gain tried. Arm-IK
        # reaching needs a stiff gain for precision; standing's compliant/shock-absorbing
        # arm response benefits from a soft one — no single value serves both, which is
        # why deployment (eval_full_demo.py) now applies gains per-bucket instead of one
        # global override, matching whichever policy's own training each bucket tests.
        if "arms" in self.scene.robot.actuators:
            self.scene.robot.actuators["arms"].stiffness = 60.0
            self.scene.robot.actuators["arms"].damping = 1.5

        # Keep action dimension unchanged, but override arm-joint targets from disturbance curriculum.
        if hasattr(self.actions, "joint_pos"):
            self.actions.joint_pos.class_type = mdp.StandingArmBlendJointPositionAction

        # Arm-motion curriculum event: starts with no motion, then small/slow, then larger/faster,
        # and finally asymmetric motions with occasional sudden reversals.
        self.events.standing_arm_motion_disturbance = EventTerm(
            func=mdp.StandingArmTrajectoryDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"asset_cfg": SceneEntityCfg("robot")},
            is_global_time=False,
        )

        # Ensure reset keeps the arm-disturbance trajectory anchored to default posture.
        self.events.standing_arm_motion_reset = EventTerm(
            func=mdp.reset_arm_targets_to_default,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_elbow_roll_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "right_elbow_roll_joint",
                    ],
                )
            },
        )

        base_velocity = getattr(self.commands, "base_velocity", None)
        if base_velocity is not None:
            if hasattr(base_velocity, "heading_command"):
                base_velocity.heading_command = False
            if hasattr(base_velocity, "rel_standing_envs"):
                # Keep standing dominant while reserving a small slice for
                # micro-corrections around zero velocity.
                base_velocity.rel_standing_envs = 0.92
            if hasattr(base_velocity, "resampling_time_range"):
                base_velocity.resampling_time_range = (0.8, 2.0)
            if hasattr(base_velocity, "ranges"):
                ranges = base_velocity.ranges
                if hasattr(ranges, "lin_vel_x"):
                    ranges.lin_vel_x = (-0.06, 0.06)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-0.04, 0.04)
                if hasattr(ranges, "ang_vel_z"):
                    ranges.ang_vel_z = (-0.10, 0.10)

        rewards = getattr(self, "rewards", None)
        if rewards is not None:
            if hasattr(rewards, "lin_vel_z_l2"):
                rewards.lin_vel_z_l2.weight = -2.2
            if hasattr(rewards, "ang_vel_xy_l2"):
                rewards.ang_vel_xy_l2.weight = -0.12
            if hasattr(rewards, "action_rate_l2"):
                rewards.action_rate_l2.weight = -0.008
            if hasattr(rewards, "dof_acc_l2"):
                rewards.dof_acc_l2.weight = -1.5e-7
            if hasattr(rewards, "feet_air_time"):
                rewards.feet_air_time.weight = 0.0
            if hasattr(rewards, "joint_deviation_torso"):
                # Relaxed from the inherited -0.1 (experimental, phase 1): the torso needs to be
                # free to act as a balance-compensation DOF when the arm-motion disturbance above
                # shifts the CoG, instead of being penalized for moving at all. Re-tighten this
                # first if standing looks too wobbly through the torso.
                rewards.joint_deviation_torso.weight = 0.0


@configclass
class G1LocomotionStandingFlatEnvCfg_PLAY(G1LocomotionStandingFlatEnvCfg):
    """Play variant for the standing flat terrain policy."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None

        # In play mode, begin arm-motion disturbances quickly for visualization.
        if hasattr(self.events, "standing_arm_motion_disturbance"):
            self.events.standing_arm_motion_disturbance.params["phase_step_boundaries"] = (20, 80, 180, 320, 500)
            # Start play close to phase-3 so stronger motions appear quickly. To preview
            # the new (untested) phase 5 instead, pass a phase_step_offset >= 320 - <first
            # env-step this instance reaches> (or just let a long-enough play session
            # advance into it naturally, per the boundaries above).
            self.events.standing_arm_motion_disturbance.params["phase_step_offset"] = 140


@configclass
class G1LocomotionStandingFlatIKReachEnvCfg(G1LocomotionStandingFlatEnvCfg):
    """Standing-flat config using real per-arm differential IK to drive the arm-reach
    disturbance during training, instead of StandingArmTrajectoryDisturbance's scripted
    joint-space random walk — see mdp.StandingArmIKReachDisturbance's docstring for the
    full rationale (short version: a real x/y/z reach target, solved with actual IK, is
    symmetric between arms by construction and matches what deployment actually tests,
    unlike an arbitrary scripted joint-angle trajectory).

    Isolated variant (new gym id, not a change to G1LocomotionStandingFlatEnvCfg itself,
    2026-07-10): the existing standing checkpoint was trained entirely under the scripted
    disturbance, and swapping the disturbance mechanism out from under an in-progress run
    mid-curriculum risks an unintended distribution shift. This is for a deliberate,
    separate experiment — either a fresh run or an explicit fine-tune from an existing
    standing checkpoint via --resume/--load_run, your choice.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.standing_arm_motion_disturbance = EventTerm(
            func=mdp.StandingArmIKReachDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"asset_cfg": SceneEntityCfg("robot")},
            is_global_time=False,
        )
        # standing_arm_motion_reset (reset_arm_targets_to_default) is inherited unchanged
        # — it only resets the shared _standing_arm_motion_targets buffer to default pose,
        # independent of which disturbance term is actually generating it.


@configclass
class G1LocomotionStandingFlatIKReachHeightEnvCfg(G1LocomotionStandingFlatIKReachEnvCfg):
    """Same as G1LocomotionStandingFlatIKReachEnvCfg (analytic-IK disturbance, dwell-until-
    timeout goal cycling — the arm holds a reached target instead of moving on instantly)
    with exactly one addition: a direct base-height reward.

    2026-07-13: found while chasing why standing had started resolving into a deep,
    stable squat (~0.4-0.5m root height vs. the ~0.75m it spawns at) as its general
    strategy under disturbance. Checked every reward term touching hip/knee joints —
    joint_deviation_hip only covers hip_yaw/hip_roll (not hip_pitch, the squat-depth
    joint); hip/knee otherwise only appear in dof_acc_l2/dof_torques_l2, which penalize
    jerkiness and effort, not pose. Nothing anywhere in this reward stack penalizes
    settling into a slow, smooth, low-torque squat — that's inherited correctly from
    Isaac Lab's walking-oriented reward set (a gait needs hip-pitch/knee flexion, so
    excluding them from a deviation penalty there is right), but standing never added
    anything back to cover the gap, so a lower CoM was a genuinely free way to buy
    stability once the disturbance got harder. mdp.base_height_l2 (stock Isaac Lab,
    isaaclab/envs/mdp/rewards.py) is the direct fix — an L2 penalty on root height
    against a target, rather than another indirect joint-deviation proxy. target_height
    matches G1_MINIMAL_CFG's own spawn height (0.75m, isaaclab_assets/robots/unitree.py).
    Weight is a starting point (-10.0), not a tuned value — check standing's own reward
    curves after training and adjust if it's either not discouraging the squat enough or
    fighting the disturbance-recovery crouch too hard to let it absorb anything at all.

    Isolated as its own gym id, built directly on G1LocomotionStandingFlatIKReachEnvCfg
    (not on either of last night's two experiments — joint_deviation_torso re-tightening
    measurably didn't help and is dropped; the arm-policy-driven disturbance solved
    standing_still but wrecked standing_arm_left_reach, most likely because the deep
    squat it also produced puts the torso at a height/geometry the arm-IK policy — fixed-
    root-trained at nominal height — was never built to reach relative to. Fixing the
    squat directly here, on the baseline that was already reasonably balanced, is a
    cleaner single-variable step than compounding it on top of that result).
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2 = RewTerm(
            func=locomotion_mdp.base_height_l2,
            weight=-10.0,
            params={"target_height": 0.75},
        )


@configclass
class G1LocomotionStandingFlatIKReachHeightEnvCfg_PLAY(G1LocomotionStandingFlatIKReachHeightEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentEnvCfg(G1LocomotionStandingFlatIKReachHeightEnvCfg):
    """Step 2 of the 2026-07-20 IK-fix recovery (definitive_next_steps.md): height_reward's
    own recipe (analytic-IK disturbance + base_height_l2, this class's parent) plus the two
    add-ons already proven independently on the Consolidated (policy-driven-arm) lineage —
    the arm-intent observation and the no_reach_prob idle slice — carried over to the
    analytic-IK lineage instead, now that `StandingArmIKReachDisturbance`'s joint-ordering
    bug is fixed (see events.py's `_col_idx` comment and the doc's 2026-07-20 section).

    Why re-derive from IKReachHeight rather than just re-running Consolidated/
    ConsolidatedIntent: every checkpoint on disk, height_reward included, was trained
    against the SAME broken disturbance class (`StandingArmIKReachDisturbance` is also
    what `StandingArmPolicyReachDisturbance`, i.e. the Consolidated lineage, subclasses
    and calls `super()._step_side()` /overrides `_step_side` from) — the column-index bug
    lived in code both lineages exercise. height_reward is the strongest known baseline
    (0% falls in the 2026-07-16 sweep) and was never combined with either add-on; this is
    the natural next single retrain rather than re-deriving from a weaker/differently-
    confounded line.

    Two adds, matching G1LocomotionStandingFlatConsolidatedIntentEnvCfg's own reasoning
    exactly (see that class's docstring for the full mechanism of each):
    - `no_reach_prob=0.15` — 15% idle episodes, closes the static-arm distribution hole.
    - the arm-intent observation (`mdp.standing_arm_motion_targets`, 10-D, appended to
      the policy obs, 75 -> 85) — lets standing anticipate per-target instead of settling
      on one fixed bracing posture; produced the best posture ever measured (~8° torso)
      on the Consolidated line, and directly serves the user's no-rotation requirement.

    Both are now trained against an IK disturbance that actually reaches its goals
    (p50 miss 0.0cm, ~80-90% within 3-5cm, verified via check_ik_accuracy.py) instead of
    the ~40-54cm-off version every other checkpoint saw — expect this to be a materially
    harder disturbance to stay upright against than anything in the 2026-07-16 sweep, per
    rule 3 in the doc (2 seeds, no single-run conclusions).
    """

    def __post_init__(self):
        super().__post_init__()

        # No noise: the commanded targets are internally known, not a sensor reading —
        # same as ConsolidatedIntentEnvCfg's identical line.
        self.observations.policy.arm_motion_targets = ObsTerm(func=mdp.standing_arm_motion_targets)

        # 15% idle episodes — see class docstring.
        self.events.standing_arm_motion_disturbance.params["no_reach_prob"] = 0.15


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentEnvCfg_PLAY(G1LocomotionStandingFlatIKReachHeightIntentEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchEnvCfg(G1LocomotionStandingFlatIKReachHeightIntentEnvCfg):
    """IKReachHeightIntent (parent) plus exactly one addition: `match_deployment_arm_gains`
    on the arm-reach disturbance term.

    2026-07-20, same day as the parent's own first training run: that run (checkpoint
    `2026-07-20_11-32-18_ikreach_height_intent`, kept on disk as a documented data point,
    NOT deleted) passed its gate (0% falls, static-arm hole closed) and held up reasonably
    under its own training disturbance (event-mode integration: 55% falls) — but collapsed
    to 99-100% falls specifically in `--arm_driver policy` integration buckets, the actual
    deployment condition. Root cause, confirmed by cross-referencing which buckets change
    which variable: `eval_full_demo.py`'s policy-mode reach buckets
    (`_run_standing_arm_reach`) set the actively-reaching arm to `--active_arm_gain`
    (default 200/20, matching the arm-IK policy's own training gain) while the held arm
    and everything else stays at 60/1.5 — event mode never touches gains at all (always
    60/1.5, `_run_standing_arm_event_reach`), which is exactly why event mode looked fine
    while policy mode collapsed. The parent config never enabled
    `match_deployment_arm_gains`, so it never once trained against a 200/20-stiff arm's
    reaction torques before meeting one for the first time at eval — the identical failure
    mode the project's 2026-07-14 notes already diagnosed and that motivated adding this
    exact flag to `StandingArmIKReachDisturbance` in the first place (previously exercised
    only by the separate Consolidated/policy-driven-disturbance lineage, never carried
    over to the analytic-IK lineage until now).

    Deliberately the ONLY change from the parent (single-variable discipline) — do NOT
    also add a torso-specific lever in the same run (e.g. a new posture term): the parent
    config already carries the one proven-safe torso lever (the arm-intent observation),
    torso-deviation reward penalties are a confirmed dead end (tried twice, see the
    experiment verdicts table — do not retry a third time), and this run already gives a
    free read on whether intent+height+now-correct-gains resolves the torso lift/tilt the
    user flagged 2026-07-20 (`mean_max_abs_torso_deg`, plus a visual check) — no need to
    add anything new to get that signal, and adding one now would confound whether it was
    the gain fix or the new lever that changed the outcome.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.standing_arm_motion_disturbance.params["match_deployment_arm_gains"] = True


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchEnvCfg_PLAY(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchEnvCfg
):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchEnvCfg
):
    """GainMatch (parent, fixes the arm-gain-mismatch fall collapse) plus exactly one more
    change: a hard clip on `torso_joint`'s commanded action range, +/-30 degrees.

    2026-07-20, found via the new per-joint diagnostics (metrics_wrappers.py's
    joint_diagnostics.csv) plus a live g1_full_demo.py check the user did by eye:
    `torso_joint` sits at 60-70deg even completely at rest (freeze_arms gate, 0% falls),
    climbing to ~120deg under active reaching — well past what any legitimate standing/
    reaching posture needs, and visually confirmed as "clearly wrong," not just a number.
    Root cause (traced via max_action_abs, which peaked at ~45 in that same checkpoint —
    wildly outside a normal PPO actor's range): the stock G1FlatEnvCfg's ActionsCfg.joint_pos
    (`JointPositionActionCfg(joint_names=[".*"], scale=0.5)`) sets NO clip at all on any
    joint, so nothing stops the policy from requesting an arbitrarily large target for any
    one DOF — torso_joint just happened to be the one PPO found and exploited as a free,
    unpenalized (no reward term prices in torso deviation at all in this recipe) way to
    sit somewhere. It isn't reward-tunable the way the two prior torso experiments assumed
    (torso_retighten/consolidated_torso, both dead ends — see the experiment verdicts
    table): those added a deviation PENALTY, which the policy must learn to respect and
    which fights any legitimate corrective use of the joint; this is a hard, deterministic
    ACTION-SPACE constraint instead, applied only to this one joint, every other joint
    (including the rest of `.*`) completely untouched.

    `clip` on `JointPositionActionCfg` is applied to the FINAL commanded target (i.e.
    already scale+offset'd, in real joint-angle radians) — see
    isaaclab/envs/mdp/actions/joint_actions.py's `process_actions`: `processed = raw*scale
    + offset`, THEN clamped to `clip`. So bounding torso_joint directly to +/-30deg here
    is exact, not an indirect scale/raw-action guess. +/-30deg (not a full lock/exclusion
    from the action space) is a deliberate choice: `torso_joint` is a waist YAW joint,
    real recovery-authority for counteracting reach-induced angular momentum — removing it
    entirely risks quietly hurting disturbance recovery in a way that wouldn't show up
    until later. +/-30deg keeps meaningful authority while making the observed exploit
    (60-120deg) physically impossible.

    MUST be retrained, not just clipped post-hoc on an already-trained checkpoint: this
    changes what the environment lets the policy do, i.e. the dynamics it's optimized
    against, not a filter bolted onto a frozen network — deploying the existing GainMatch
    checkpoint under a newly-clipped action space it never trained within would be exactly
    the kind of train/deploy mismatch this whole day has been about eliminating (see the
    2026-07-20 gain-mismatch section above).

    Sibling experiment to G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoLockEnvCfg
    (+/-30deg here vs. a full 0-width lock there) — 2026-07-20, explicit user call: run ONE
    seed of EACH clip width rather than two seeds of this one, to directly answer whether
    the "keep some recovery authority" concern above is real before spending a second seed
    confirming just this width. A single seed of each is a real, if smaller, signal than
    zero seeds of the comparison; revisit with a second seed per width once one direction
    looks like the right one.
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.clip = {"torso_joint": (-math.radians(30.0), math.radians(30.0))}


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg_PLAY(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg
):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipOrientHipEnvCfg(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg
):
    """TorsoClip (parent, +/-30deg hard action-space cap on torso_joint) plus strengthening
    two reward terms that were already active this whole time but never retuned for
    standing: `flat_orientation_l2` (tilt penalty, inherited at -1.0) and
    `joint_deviation_hip` (hip_yaw/hip_roll deviation penalty, inherited at -0.1) — both
    come unchanged from G1RoughEnvCfg, tuned for a WALKING gait that legitimately needs
    some sway and hip motion, never revisited for standing specifically.

    2026-07-21: after retraining Clip/Lock, gate-eval (idle, arms frozen, 0% falls in
    every one of these checkpoints) tilt got WORSE once torso was capped, not better —
    GainMatch 8.42deg -> Clip 37.19deg -> Lock 25.80deg — even though torso itself is now
    correctly bounded (60.3deg -> 32.2deg -> 0.7deg). Cross-checked against
    joint_diagnostics.csv's frac_of_soft_limit_used on hip_roll (native eval): GainMatch
    left/right 10.9%/27.1% -> Clip 60.7%/10.9% -> Lock 55.2%/30.9% — a dramatic jump on
    whichever hip is doing the compensating. Live g1_full_demo.py check (user, by eye)
    confirmed both Clip and Lock stand with legs spread far wider than GainMatch and a
    visually obvious, unacceptable amount of tilt.

    Conclusion: GainMatch's comparatively normal-looking idle pose was never actually the
    product of these two reward terms working — it was torso rotation acting as a free,
    unpenalized escape hatch that happened to look tolerable. Capping that hatch (correctly
    — the 60-120deg twist was real and had to go, see TorsoClip's own docstring) didn't
    remove the underlying balance-compensation need, it just forced it onto the next-
    cheapest lever: hip abduction. -1.0/-0.1 apparently aren't strong enough to price that
    trade-off correctly once the cheaper option is gone — they were only ever "good enough"
    while masked by the free DOF.

    -3.0 / -0.5 below are starting points, not tuned values (same spirit as
    base_height_l2's own -10.0 starting point) — check standing's reward curves after
    training and adjust if either term isn't discouraging the compensation enough or is
    fighting legitimate disturbance-recovery motion too hard to let it absorb anything.
    Deliberately built on TorsoClip, not TorsoLock: Lock is dropped as a direction as of
    today (numerically worse on every axis in the policy-mode integration eval AND
    visually worse live — see the 2026-07-21 write-up in definitive_next_steps.md) —
    keeping SOME torso authority while ALSO pricing in tilt/hip-deviation properly is the
    live hypothesis being tested here, not a fallback to reconsider Lock later.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.joint_deviation_hip.weight = -0.5


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipOrientHipEnvCfg_PLAY(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipOrientHipEnvCfg
):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipBadOrientationEnvCfg(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg
):
    """TorsoClip (parent, +/-30deg hard action-space cap on torso_joint) plus one addition:
    a hard TERMINATION on excessive whole-body tilt, `isaaclab.envs.mdp.bad_orientation`
    (checks the angle between the projected-gravity vector and the z-axis, i.e. whole-body
    tilt from vertical — the SAME quantity this project's own eval scripts already log as
    `mean_max_tilt_deg`, just applied live during training instead of only measured after).

    2026-07-21: found while comparing our reward stack against Unitree's own G1 velocity-
    task recipe (~/Elm/Code/unitree_rl_lab, source/unitree_rl_lab/unitree_rl_lab/tasks/
    locomotion/robots/g1/29dof/velocity_env_cfg.py — real, hardware-deployed). Their
    TerminationsCfg has `bad_orientation = DoneTerm(func=mdp.bad_orientation,
    params={"limit_angle": 0.8})` in addition to a time_out and a height-floor
    termination; ours (inherited from G1FlatEnvCfg/G1RoughEnvCfg) only ever terminates on
    time_out and base_contact (actually touching the ground) — orientation itself has
    never been anything but a soft reward cost here, at any point in this project's
    history.

    Why this is a structurally different fix from every other lever pulled so far
    (torso_retighten, consolidated_torso, leg_symmetry_l2, the torso action clip, tonight's
    OrientHip reward-reweight sibling): every one of those prices in ONE specific joint or
    joint group. PPO can always trade a soft per-joint cost against the disturbance-
    recovery benefit of using that joint, and when one joint gets constrained (torso,
    2026-07-20) the exact same trade just re-forms around whichever joint is now cheapest
    (hip_roll, discovered the same day). A hard termination on the OUTCOME (excessive
    tilt) instead of the cause (which joint produced it) can't be dodged that way — it
    doesn't matter which DOF the policy uses to tilt too far, crossing the line always
    costs the same (the rest of the episode's reward), so there's no "cheaper joint" to
    discover and move the exploit to next.

    limit_angle=0.8 (~45.8deg) is Unitree's own literal value, not re-derived — used as a
    starting point specifically BECAUSE it's a real number from a working, hardware-
    deployed system, not an arbitrary guess. Flagged risk going in: our own checkpoints
    already show mean_max_tilt_deg at or above this threshold under active reach recovery
    (TorsoClip's own native eval: 48.94deg mean-of-episode-max, meaning a real fraction of
    individual peaks already exceed 45.8deg) — so this will very likely terminate a
    meaningful share of episodes early, especially before the policy has adapted to it.
    That pressure is the intended mechanism (it should push the emergent policy toward
    keeping tilt down across the board, not just on this one checkpoint's current
    strategy), but it's also the thing to check first if training looks unstable or
    collapses: if termination fires so often/early that the policy can't discover
    successful recovery before running out of episode, loosen limit_angle (try 60-70deg)
    or consider ramping it in via curriculum rather than applying it from step 0.

    Deliberately built on TorsoClip directly, NOT stacked on top of the same day's
    TorsoClip-OrientHip sibling (reward reweight): single-variable discipline — this tests
    "does a hard tilt termination alone fix the wide-stance/tilt regression" as an
    independent data point from "does reweighting flat_orientation_l2/joint_deviation_hip
    alone fix it." Only combine the two once each has its own result.
    """

    def __post_init__(self):
        super().__post_init__()

        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8},
        )


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipBadOrientationEnvCfg_PLAY(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipBadOrientationEnvCfg
):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoLockEnvCfg(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchEnvCfg
):
    """Same as G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg (see
    that class's docstring for the full root-cause writeup — identical here) except the
    clip is a full 0-width lock instead of +/-30deg: `torso_joint`'s commanded target is
    pinned to exactly its default offset every step, functionally equivalent to excluding
    the joint from the action space entirely (the "legs" actuator's own 200/5 PD gain
    holds it there — the same "held rigid" pattern g1_arm_env.py already uses for its own
    non-RL-actuated joints, just reached via a zero-width clip instead of a smaller
    action_space, so it's the same mechanism/class as the +/-30deg sibling rather than a
    structurally different one).

    2026-07-20, explicit user call: run this alongside the +/-30deg version (one seed
    each) rather than only the softer clip, specifically to test whether keeping SOME
    torso authority for reach-disturbance recovery actually matters, or whether it was
    never doing real work and can just be removed outright. Real risk flagged going in:
    `torso_joint` is a waist YAW joint with plausible legitimate use for counteracting
    reach-induced angular momentum — a full lock forces the standing policy to solve
    disturbance recovery entirely through legs/ankles/hips instead. Watch the ACTIVE-reach
    fall-rate buckets specifically (native + policy-mode integration) for any regression
    vs. the +/-30deg sibling and vs. GainMatch's own numbers — that's the sentinel for
    "locking it fully cost real recovery capability," not just whether torso itself reads
    near 0deg (it will, by construction).
    """

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.clip = {"torso_joint": (0.0, 0.0)}


@configclass
class G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoLockEnvCfg_PLAY(
    G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoLockEnvCfg
):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg(G1LocomotionStandingFlatIKReachHeightEnvCfg):
    """Same as G1LocomotionStandingFlatIKReachHeightEnvCfg (analytic-IK disturbance,
    dwell-until-timeout goal cycling, base_height_l2 against the squat) plus one addition:
    mdp.symmetry.leg_symmetry_l2, penalizing left-right leg asymmetry directly.

    2026-07-14: added after visually confirming (g1_full_demo.py) that even the
    height-reward checkpoint settles into a persistent, lopsided rest pose — tilted,
    uneven legs — with zero arm disturbance active, so no legitimate disturbance response
    could be causing it. See leg_symmetry_l2's own docstring for the full mechanism and
    why it's scoped to legs only (never fights a legitimate one-armed reach — arms aren't
    in its joint set at all — and can't touch torso, which has no left/right pair to
    compare against and stays governed by joint_deviation_torso instead).

    Isolated as its own gym id, single-variable step on top of the height-reward baseline
    (not stacked with a torso change too — see leg_symmetry_l2's docstring: torso is
    deliberately left alone here in case fixing leg asymmetry resolves the torso rotation
    as a side effect, without needing a second, confounding lever added at the same time).
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.leg_symmetry_l2 = RewTerm(
            func=leg_symmetry_l2,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_hip_pitch_joint",
                        "left_hip_roll_joint",
                        "left_hip_yaw_joint",
                        "left_knee_joint",
                        "left_ankle_pitch_joint",
                        "left_ankle_roll_joint",
                    ],
                )
            },
        )


@configclass
class G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg_PLAY(G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatConsolidatedEnvCfg(G1LocomotionStandingFlatIKReachHeightEnvCfg):
    """Phase 1 of the integration-recovery plan (see plan.md §4), 2026-07-14: ONE config
    combining the three levers that were each validated individually but never together —
    and whose combination is the actual deployment condition the integration eval tests.

    1. StandingArmPolicyReachDisturbance — the real frozen arm-IK policy as the training
       disturbance, jitter included (validated alone 2026-07-13: fixed standing_still 0%
       and right-reach 9% falls, but produced a deep squat that broke reach geometry).
    2. base_height_l2 (inherited from G1LocomotionStandingFlatIKReachHeightEnvCfg) — the
       squat fix (validated alone 2026-07-13/14: height restored to 0.71-0.73m everywhere,
       reach precision way up, but fall rate alone unchanged).
    3. match_deployment_arm_gains — the one lever never tried anywhere: the actively
       reaching arm trains at 200/20 (what deployment actually runs it at, and what the
       arm checkpoint was trained under) instead of standing's 60/1.5. Standing has
       literally never felt a 200/20-stiff arm before this config, and the 2026-07-14
       fall-timing diagnostics (falls = slow lean build-up over 2.6-7.6s of sustained
       reaching, not goal-switch shocks) point at exactly the larger-than-trained
       reaction torques a stiff arm transmits.

    Deliberately bundles all three (breaking the usual single-variable rule) because each
    variable has already been tested alone — what has never existed is a standing policy
    trained under the full deployment condition. If this config's integration eval still
    fails, the next step is architecture (plan.md §5), not more reward tuning here.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.standing_arm_motion_disturbance = EventTerm(
            func=mdp.StandingArmPolicyReachDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "arm_checkpoint": "chosen_checkpoints/arm_left_latest.pt",
                "match_deployment_arm_gains": True,
            },
            is_global_time=False,
        )


@configclass
class G1LocomotionStandingFlatConsolidatedEnvCfg_PLAY(G1LocomotionStandingFlatConsolidatedEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatConsolidatedIntentEnvCfg(G1LocomotionStandingFlatConsolidatedEnvCfg):
    """Consolidated (see that class's docstring) + exactly one change: the arm-intent
    observation (mdp.standing_arm_motion_targets — the arm's currently-commanded joint
    targets as offsets from default, 10-D, appended at the END of the policy observation).

    2026-07-15, plan.md §4.5 option 2: the consolidated checkpoint balances *reactively* —
    it only ever feels the arm's motion after the fact, so its best strategy was one fixed
    bracing posture (a persistently rotated torso) that costs most of the inner reach
    workspace, and each policy-driven-trained run solves only one arm. Deployed humanoid
    stacks put the upper-body command in the locomotion policy's observation for exactly
    this reason: seeing where the arm is told to go next allows *per-target* anticipation
    instead of one posture that must serve every reach. The buffer this term exposes
    (env._standing_arm_motion_targets) already exists identically in training (the
    disturbance classes write it) and in both deployment scripts (they write it too), so
    train and deploy see the same signal by construction.

    Observation dim grows 75 -> 85; the term appends at the end, so deployment feeds old
    75-D policies by simply not appending (both deployment scripts auto-detect the
    checkpoint's input width — see their _load_loco_policy). Standing's PPO runner cfg has
    no symmetry augmentation (confirmed in agents/rsl_rl_ppo_cfg.py — only walking uses
    it), so no mirror-transform changes are needed for the extra block.

    Sibling experiment to G1LocomotionStandingFlatConsolidatedTorsoEnvCfg (posture *penalty*
    vs posture *information*) — same baseline. The torso branch was run first and failed
    catastrophically (2026-07-15: rotation persisted at ~48.5 deg natively AND the
    checkpoint joined the collapse group — 100% integration falls even standing still),
    so this branch is now the main line.

    Second deliberate addition (2026-07-15, after the torso branch's collapse):
    no_reach_prob=0.15 — 15% of episodes hold both arms at default with no reaching. The
    --freeze_arms bisect proved "standing with motionless arms" is a lethal
    training-distribution hole for checkpoints that never practice it (100% falls within
    ~2.5s, natively, for symmetry/consolidated_torso-class checkpoints). Strict
    single-variable discipline would omit this, but each training is a single seed anyway
    and the failure mode is proven + cheap to cover — insurance beats attribution here.
    It also composes naturally with the intent term: idle episodes give the policy an
    explicit all-zeros intent signal to associate with "just stand".
    """

    def __post_init__(self):
        super().__post_init__()

        # No noise: the commanded targets are internally known, not a sensor reading.
        self.observations.policy.arm_motion_targets = ObsTerm(func=mdp.standing_arm_motion_targets)

        # 15% idle episodes — see class docstring.
        self.events.standing_arm_motion_disturbance.params["no_reach_prob"] = 0.15


@configclass
class G1LocomotionStandingFlatConsolidatedIntentEnvCfg_PLAY(G1LocomotionStandingFlatConsolidatedIntentEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatConsolidatedNoReachEnvCfg(G1LocomotionStandingFlatConsolidatedEnvCfg):
    """Consolidated + exactly one change: no_reach_prob=0.15 (15% of episodes hold both
    arms at default, no reaching — see StandingArmIKReachDisturbance.no_reach_prob for the
    static-arm distribution hole this closes).

    2026-07-15: the CONTROL for the consolidated_intent run, which deliberately bundles
    two changes (the arm-intent observation AND this same idle slice — see its docstring
    for why insurance beat attribution there). This config isolates the idle slice alone:
    whatever the intent run shows, comparing it against this run separates "the policy
    learned from seeing the arm's intent" from "the policy just stopped being fragile
    about motionless arms." Also the natural fallback baseline if the intent term itself
    misbehaves.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.standing_arm_motion_disturbance.params["no_reach_prob"] = 0.15


@configclass
class G1LocomotionStandingFlatConsolidatedNoReachEnvCfg_PLAY(G1LocomotionStandingFlatConsolidatedNoReachEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatConsolidatedTorsoEnvCfg(G1LocomotionStandingFlatConsolidatedEnvCfg):
    """Consolidated (see that class's docstring) + exactly one change: joint_deviation_torso
    partially re-tightened, 0.0 -> -0.05.

    2026-07-15: the consolidated checkpoint fixed left-reach falls (78% -> 2.1%) but did it
    by holding a *persistently rotated torso* from the moment it stands (visually confirmed
    in g1_full_demo by the user) — a bracing posture that displaces the shoulder relative
    to the pelvis-frame goal box and swallowed the near-inner reach workspace: goals the
    height_reward checkpoint reaches at 1.8cm / 100% success (43-attempt sample, near-inner
    corner x<0.27, y<0.18) timed out at 15-20cm under consolidated (concluded success
    collapsed to ~16%). This config asks for the same fall robustness *without* the
    permanent rotation, by making a sustained torso offset cost something while leaving it
    cheap enough to still use transiently for genuine recoveries.

    Note the history: the same -0.05 was tried before (G1LocomotionStandingFlatIKReachTorsoEnvCfg,
    2026-07-13) and made integration *worse* — but that was on the pre-gain-match baseline,
    where the policy plausibly needed every posture trick available just to survive at all.
    With the gain match now carrying the load (2.1% falls with the torso free), re-testing
    whether the rotation is actually load-bearing or just an unpenalized habit is a clean
    single-variable question on the new baseline. Watch fall rate AND the new
    max_abs_torso_deg / mean_abs_torso_deg columns (StandingMetricsCsvWrapper, added
    2026-07-15 for exactly this experiment) plus arm success together: success = falls stay
    ~2%, torso rotation drops, reach success recovers toward the height_reward run's ~96%
    concluded. If falls come back instead, the rotation was load-bearing and the fix is
    target-conditioned posture (arm-intent observation, plan.md §4.5 option 2), not a
    posture penalty.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.joint_deviation_torso.weight = -0.05


@configclass
class G1LocomotionStandingFlatConsolidatedTorsoEnvCfg_PLAY(G1LocomotionStandingFlatConsolidatedTorsoEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachEnvCfg_PLAY(G1LocomotionStandingFlatIKReachEnvCfg):
    """Eval variant for an *already-trained* IK-reach checkpoint (validation/eval_standing_ikreach.py) —
    disturbance fully on from step 0 (enable_step=ramp_full_step=0, see
    mdp.StandingArmIKReachDisturbance._ramp_fraction) instead of training's own 30k-step
    curriculum delay, since there's no policy left to protect from an abrupt disturbance
    here — the whole point is measuring how the finished policy handles it. Observation
    noise restored (the inherited _PLAY convention disables it for clean visualization;
    this needs training-representative noise for a fair eval, same reasoning
    eval_full_demo.py's _build_env_cfg already documents for the same fix)."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatIKReachTorsoEnvCfg(G1LocomotionStandingFlatIKReachEnvCfg):
    """Same as G1LocomotionStandingFlatIKReachEnvCfg (analytic-IK disturbance, dwell-until-
    timeout goal cycling) with exactly one additional change: joint_deviation_torso's
    weight, relaxed to 0.0 in G1LocomotionStandingFlatEnvCfg to let the torso act as a
    free balance-compensation DOF (see that class's own comment: "re-tighten this first
    if standing looks too wobbly through the torso"). 2026-07-13: partially re-tightened,
    not reverted to the original -0.1 — the torso still needs *some* freedom to
    compensate, this just discourages committing to an extreme, sustained lean now that
    goals are held for up to 15s instead of cycling continuously (see the dwell-until-
    timeout fix in mdp.StandingArmIKReachDisturbance). Isolated as its own gym id so this
    is a single-variable experiment against the already-validated dwell-phase-fix
    baseline, not bundled with the separate arm-policy-driven-disturbance experiment
    (G1LocomotionStandingFlatPolicyReachEnvCfg) — same reasoning, different lever, run
    independently so results are attributable.
    """

    def __post_init__(self):
        super().__post_init__()

        self.rewards.joint_deviation_torso.weight = -0.05


@configclass
class G1LocomotionStandingFlatIKReachTorsoEnvCfg_PLAY(G1LocomotionStandingFlatIKReachTorsoEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0


@configclass
class G1LocomotionStandingFlatPolicyReachEnvCfg(G1LocomotionStandingFlatIKReachEnvCfg):
    """Same as G1LocomotionStandingFlatIKReachEnvCfg (dwell-until-timeout goal cycling,
    same curriculum ramp) but drives the arm-reach disturbance with the actual trained
    arm-IK policy instead of analytic IK — see mdp.StandingArmPolicyReachDisturbance's
    docstring for the full rationale (short version: a checkpoint trained against clean
    analytic IK scored 0.13% fall rate in its own native eval under this exact disturbance
    but 73%+ when deployed against the real policy's actual — visibly jittery near a
    reached goal — behavior; every other difference between those two environments has
    been ruled out, so this closes the one that's left).

    Isolated as its own gym id, not a change to G1LocomotionStandingFlatIKReachEnvCfg
    itself, same reasoning that class documents for its own isolation from the original
    scripted-disturbance config: this is a deliberate, separate experiment against the
    already-validated dwell-phase-fix baseline, run independently from the torso-reward
    experiment (G1LocomotionStandingFlatIKReachTorsoEnvCfg) so each change's effect is
    attributable on its own.
    """

    def __post_init__(self):
        super().__post_init__()

        self.events.standing_arm_motion_disturbance = EventTerm(
            func=mdp.StandingArmPolicyReachDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "arm_checkpoint": "chosen_checkpoints/arm_left_latest.pt",
            },
            is_global_time=False,
        )


@configclass
class G1LocomotionStandingFlatPolicyReachEnvCfg_PLAY(G1LocomotionStandingFlatPolicyReachEnvCfg):
    """Eval variant, same pattern as G1LocomotionStandingFlatIKReachEnvCfg_PLAY."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.enable_corruption = True
        self.events.standing_arm_motion_disturbance.params["enable_step"] = 0
        self.events.standing_arm_motion_disturbance.params["ramp_full_step"] = 0
