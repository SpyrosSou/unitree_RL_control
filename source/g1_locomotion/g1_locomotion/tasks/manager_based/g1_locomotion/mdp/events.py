# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event terms for G1 locomotion tasks."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


# Matches velocity_env_cfg.py's ActionsCfg (scale=0.5) and every deployment script's own
# ARM_ACTION_SCALE constant (eval_full_demo.py, g1_full_demo.py) — the arm-IK policy's raw
# output is a joint-space delta in this same convention, so StandingArmPolicyReachDisturbance
# has to convert it identically or it isn't actually replaying what deployment does.
_ARM_ACTION_SCALE = 0.5

_ARM_JOINT_NAMES = [
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
]


def reset_arm_targets_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset arm motion trajectory buffers back to default for selected envs."""

    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device)

    joint_ids, _ = asset.find_joints(_ARM_JOINT_NAMES)
    joint_ids_t = torch.tensor(joint_ids, dtype=torch.long, device=asset.device)

    if hasattr(env, "_standing_arm_motion_targets"):
        env._standing_arm_motion_targets[env_ids] = asset.data.default_joint_pos[env_ids][:, joint_ids_t].clone()


class StandingArmTrajectoryDisturbance(ManagerTermBase):
    """Inject smooth random arm trajectories during standing-policy training.

    The curriculum is phased by global env step count:
    - Phase 0: no arm motion (standing baseline warm-up)
    - Phase 1: mild motions (0.03 rad/step ≈ 1.5 rad/s @ 50 Hz)
    - Phase 2: moderate motions (0.05 rad/step ≈ 2.5 rad/s @ 50 Hz)
    - Phase 3: stronger motions (0.10 rad/step ≈ 5.0 rad/s @ 50 Hz)
    - Phase 4: stress-test bursts (0.25 rad/step ≈ 12.5 rad/s @ 50 Hz)
    - Phase 5: hardware-limit stress-test (0.66 rad/step ≈ 33 rad/s @ 50 Hz) — the agreed
      ceiling from phase_1.md, safely under the real G1 arm joint speed limit (~37 rad/s).
      Added 2026-07-07, purely additive (phases 0-4 unchanged) — not yet trained or
      validated, see known_issues.md.
    """

    # 5 boundaries -> 6 phases (0-5).
    # Spend longer at lower speeds where the real platform normally operates,
    # then introduce progressively harder disturbances.
    #
    # Phase 5's amplitude/asymmetry/reversal continue the same per-phase increment as
    # phases 1-4 rather than scaling proportionally with delta (delta roughly doubles-to-
    # 2.6x's each phase, but amplitude growth is much shallower — 0.24, 0.45, 0.70, 0.90 —
    # consistent with staying within a comfortable, real shoulder/elbow range of motion
    # rather than tracking speed 1:1). This has NOT been visually verified via play.py —
    # do that before trusting it for a real training run, same as any other untested
    # curriculum change in this repo (see phase_0.md's own known-limitation note).
    _PHASE_STEP_BOUNDARIES = (12000, 30000, 50000, 80000, 120000)
    _NO_MOTION_PROB    = (1.00, 0.00, 0.00, 0.00, 0.00, 0.00)
    _MAX_DELTA_RAD     = (0.00, 0.030, 0.050, 0.100, 0.250, 0.660)
    _MAX_AMPLITUDE_RAD = (0.00, 0.24,  0.45,  0.70,  0.90,  1.10)
    _ASYMMETRY_PROB    = (0.00, 0.08,  0.20,  0.45,  0.75,  0.85)
    _REVERSAL_PROB     = (0.00, 0.00,  0.02,  0.06,  0.15,  0.22)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        joint_ids, joint_names = self._asset.find_joints(_ARM_JOINT_NAMES)
        if len(joint_ids) == 0:
            raise RuntimeError("StandingArmTrajectoryDisturbance could not find arm joints.")

        self._arm_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self._arm_joint_names = list(joint_names)

        self._targets = self._asset.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._default = self._targets.clone()
        self._env._standing_arm_motion_targets = self._targets
        self._env._standing_arm_motion_joint_ids = self._arm_joint_ids

        # Build left-right joint index pairs and mirror signs for symmetric motions.
        # Roll and yaw components are mirrored with opposite sign under left-right symmetry.
        name_to_col = {name: idx for idx, name in enumerate(self._arm_joint_names)}
        pair_left_cols = []
        pair_right_cols = []
        pair_signs = []
        for left_name in self._arm_joint_names:
            if not left_name.startswith("left_"):
                continue
            right_name = left_name.replace("left_", "right_", 1)
            if right_name not in name_to_col:
                continue
            pair_left_cols.append(name_to_col[left_name])
            pair_right_cols.append(name_to_col[right_name])
            is_mirrored_axis = ("_roll_" in left_name) or ("_yaw_" in left_name)
            pair_signs.append(-1.0 if is_mirrored_axis else 1.0)

        if len(pair_left_cols) > 0:
            self._pair_left = torch.tensor(pair_left_cols, dtype=torch.long, device=self.device)
            self._pair_right = torch.tensor(pair_right_cols, dtype=torch.long, device=self.device)
            self._pair_sign = torch.tensor(pair_signs, dtype=torch.float32, device=self.device).view(1, -1)
        else:
            self._pair_left = None
            self._pair_right = None
            self._pair_sign = None

        # Per-joint scaling for realism: shoulder joints get wider excursions,
        # elbow joints remain comparatively constrained.
        scale = []
        for name in self._arm_joint_names:
            if "shoulder_pitch" in name:
                scale.append(1.00)
            elif "shoulder_roll" in name or "shoulder_yaw" in name:
                scale.append(0.95)
            elif "elbow_pitch" in name:
                scale.append(0.75)
            elif "elbow_roll" in name:
                scale.append(0.65)
            else:
                scale.append(1.00)
        self._joint_motion_scale = torch.tensor(scale, dtype=torch.float32, device=self.device).view(1, -1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._targets[env_ids] = self._default[env_ids]
        self._env._standing_arm_motion_targets = self._targets

    def _phase_index(self, common_step_counter: int) -> int:
        for i, boundary in enumerate(self._phase_step_boundaries):
            if common_step_counter < boundary:
                return i
        return len(self._phase_step_boundaries)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        phase_step_boundaries: tuple[int, ...] | None = None,
        phase_step_offset: int = 0,
    ) -> None:
        del asset_cfg

        if phase_step_boundaries is None:
            self._phase_step_boundaries = self._PHASE_STEP_BOUNDARIES
        else:
            self._phase_step_boundaries = tuple(int(v) for v in phase_step_boundaries)

        self._phase_step_offset = int(phase_step_offset)

        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)

        if len(env_ids) == 0:
            return

        phase = self._phase_index(env.common_step_counter + self._phase_step_offset)
        current = self._targets[env_ids]
        defaults = self._default[env_ids]

        # Phase 0: pure baseline standing behavior (no arm motion disturbance).
        if phase == 0:
            updated = 0.9 * current + 0.1 * defaults
        else:
            delta_mag = self._MAX_DELTA_RAD[phase]
            amp = self._MAX_AMPLITUDE_RAD[phase]

            delta = (2.0 * torch.rand_like(current) - 1.0) * delta_mag
            delta = delta * self._joint_motion_scale

            if phase >= 3:
                reversal_mask = torch.rand_like(delta) < self._REVERSAL_PROB[phase]
                delta = torch.where(reversal_mask, -2.0 * delta, delta)

            if self._pair_left is not None:
                symmetric_roll = torch.rand((len(env_ids), 1), device=self.device) > self._ASYMMETRY_PROB[phase]
                symmetric_mask = symmetric_roll.expand(-1, len(self._pair_left))
                mirrored_right = delta[:, self._pair_left] * self._pair_sign
                delta[:, self._pair_right] = torch.where(symmetric_mask, mirrored_right, delta[:, self._pair_right])

            no_motion_mask = torch.rand((len(env_ids), 1), device=self.device) < self._NO_MOTION_PROB[phase]
            updated = current + delta
            amp_vec = amp * self._joint_motion_scale
            updated = torch.clamp(updated, defaults - amp_vec, defaults + amp_vec)
            relaxed = 0.9 * current + 0.1 * defaults
            updated = torch.where(no_motion_mask, relaxed, updated)

        self._targets[env_ids] = updated
        self._env._standing_arm_motion_targets = self._targets


class StandingArmIKReachDisturbance(ManagerTermBase):
    """Differential-IK-driven reach disturbance for standing training.

    Replaces StandingArmTrajectoryDisturbance's scripted joint-space random walk with a
    real x/y/z reach target per arm, solved every control step with
    isaaclab.controllers.DifferentialIKController (damped-least-squares Jacobian IK) —
    the same "goal in _GOAL_BOUNDS" abstraction g1_arm_env.py trains on and
    g1_full_demo.py/eval_full_demo.py test against, instead of an arbitrary joint-angle
    trajectory that has no particular relationship to reaching anything.

    Deliberately real numerical IK, not the trained arm-IK RL policy (2026-07-10): this
    decouples standing's robustness training from whatever quality/asymmetry the RL arm
    policy currently has — the mirror-based right-arm path is measurably imperfect
    (~8% average deviation from true equivariance against the actual trained checkpoint,
    concentrated on roll/yaw) and got *worse*, not better, under a wider network. Baking
    that specific policy's specific flaws into what standing has to learn to tolerate
    would be a confound, and would go stale the next time the arm policy is retrained.
    Real IK is exact and symmetric between arms by construction — standing learns to
    counteract reaching motion in general, a property that survives future arm-policy
    changes entirely.

    Goal x/y are expressed directly in the robot's current root frame (forward/lateral
    relative to wherever it's currently facing) — since the IK solve itself operates in
    root frame (ee pose and Jacobian both rotated into it, see _jacobian_b), this makes
    the goal automatically track the robot's current heading for free, without the
    explicit world-frame re-anchoring eval_full_demo.py/g1_full_demo.py needed (their
    observations are built in world frame; IK's native frame already is the root frame).
    Goal z stays ground-referenced (matching _GOAL_BOUNDS' own convention: height above
    the floor, not above the root) and is converted to root-frame z fresh every step from
    the robot's *current* height, so a target "1.0m off the floor" stays at 1.0m off the
    floor even as the robot crouches/rises during recovery, rather than drifting with the
    torso the way a naive root-relative z would.

    Curriculum-gated (env-steps < enable_step -> arms held at default, no reaching), same
    on/off mechanism g1_arm_env.py's own root wobble uses. Once enabled, each env is
    randomly assigned a reach mode at every episode reset: left-only, right-only (most of
    the probability mass between them — matches how this is actually tested one arm at a
    time in g1_full_demo.py/eval_full_demo.py), or both arms simultaneously (a harder
    stress case, deliberately the minority — see both_arms_prob).

    Ramped, not on/off, added 2026-07-11: the first version of this class went straight
    from "no reaching" to full speed + full _GOAL_BOUNDS range the instant enable_step
    was crossed, for every env at once, with no relaxation between attempts — a much more
    abrupt, relentless disturbance than StandingArmTrajectoryDisturbance's own proven
    5-phase ramp (spread over 120,000 env-steps) ever asks the policy to handle. Fine-
    tuning a healthy checkpoint against that abrupt version for ~4000 iterations
    collapsed it (fine at eval_standing.py's phase 0-2, 38-81% fall rate at phase 3-5,
    versus a healthy baseline beforehand) — this ramp (start_fraction -> 1.0, linear from
    enable_step to ramp_full_step, applied to *both* reach speed and goal-box range) is
    the fix, using the exact fraction formula g1_arm_env.py's own
    _current_goal_curriculum_fraction already validates, not a new one. ramp_full_step
    defaults to 120,000 to match the original disturbance's own full-ramp point — a
    proven timescale, not a guess.
    """

    # Deployment gain for an actively IK/policy-driven arm — matches g1_arm_env.py's
    # training gain and the _ARM_IK_GAIN both deployment scripts apply to whichever arm
    # is actively reaching (eval_full_demo.py's reach buckets, g1_full_demo.py's
    # _update_arm_gains). Only used when match_deployment_arm_gains is enabled (see
    # __init__): standing otherwise trains its whole arm group at the env cfg's own
    # 60/1.5 — a gain that has never matched what deployment actually runs the reaching
    # arm at, and (2026-07-14, verified against unitree_sdk2_python) the 60/1.5 value is
    # the hardware-realistic one while 200/20 is a sim-only property of the current arm
    # checkpoint. Until the arm policy works at soft gains, standing must train against
    # the stiff arm it will actually face.
    _DEPLOY_ACTIVE_ARM_GAIN = (200.0, 20.0)

    enable_step: int = 30_000
    ramp_full_step: int = 120_000
    # 0.15, not g1_arm_env.py's 0.4 — the 2026-07-11 training run's TensorBoard curve
    # (Train/mean_reward, Episode_Termination/base_contact) showed a sharp shock exactly
    # at enable_step even with the ramp in place: reward -2.4 and 96% fall rate the very
    # first logged iteration after the disturbance switched on at 0.4 intensity, before
    # recovering over the next few hundred iterations. 0.4 is apparently still a big first
    # step from "nothing" for this task; starting lower gives the policy an easier first
    # foothold before the ramp carries it up to full strength by ramp_full_step regardless.
    start_fraction: float = 0.15
    both_arms_prob: float = 0.2
    # Fraction of episodes with NO reaching at all — both arms held at their default pose
    # for the whole episode (2026-07-15). Closes a proven, lethal training-distribution
    # hole: the --freeze_arms bisect showed checkpoints trained with the disturbance
    # always active post-enable fall 100% (within ~2.5s) when the arms are simply held
    # still at default — the exact condition of integration's standing_still bucket and
    # of any real deployment moment with no arm command. "Arms quietly at default" was
    # something these policies had literally never practiced after the curriculum
    # enabled. Default 0.0 so existing configs/checkpoints keep their historical
    # behavior; opt in per config via params={"no_reach_prob": ...}.
    no_reach_prob: float = 0.0
    goal_threshold_m: float = 0.03  # matches eval_full_demo.py's GOAL_THRESHOLD_M
    max_steps_per_goal: int = 750  # 15s @ 50Hz, matches eval_full_demo.py
    max_joint_delta_per_step: float = 0.06  # matches G1ArmIKEnvCfg.max_action_delta_per_step
    ik_method: str = "dls"

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Per-instance overrides for the three ramp knobs, same mechanism
        # StandingArmTrajectoryDisturbance's phase_step_boundaries/phase_step_offset
        # params already use — lets an eval/play config force the disturbance fully on
        # from step 0 (enable_step=0, ramp_full_step=0) to test an already-trained
        # checkpoint, without touching the class defaults training itself uses.
        self.enable_step = cfg.params.get("enable_step", self.enable_step)
        self.ramp_full_step = cfg.params.get("ramp_full_step", self.ramp_full_step)
        self.start_fraction = cfg.params.get("start_fraction", self.start_fraction)
        self.no_reach_prob = cfg.params.get("no_reach_prob", self.no_reach_prob)

        # Imported here, not at module level, so this file stays importable even if the
        # g1_arm package ever isn't (matches how eval_full_demo.py already reuses these
        # exact constants across task packages — g1_arm has no reverse dependency on
        # g1_locomotion, so this is a one-way edge, not a cycle).
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
            _GOAL_BOUNDS,
            _LEFT_ARM_JOINTS,
            _LEFT_EE_BODY,
            _RIGHT_ARM_JOINTS,
            _RIGHT_EE_BODY,
        )

        self._goal_bounds = _GOAL_BOUNDS

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        self._sides = ("left", "right")
        self._arm_joint_names = {"left": _LEFT_ARM_JOINTS, "right": _RIGHT_ARM_JOINTS}
        self._ee_body_name = {"left": _LEFT_EE_BODY, "right": _RIGHT_EE_BODY}

        # NOTE (2026-07-20): find_joints returns ids in ASSET order (preserve_order
        # defaults to False), and the G1 articulation orders joints breadth-first, which
        # INTERLEAVES left/right (L_shoulder_pitch, R_shoulder_pitch, L_shoulder_roll, ...)
        # — NOT _ARM_JOINT_NAMES's left-block-then-right-block layout. A previous version
        # assumed block layout here (per-side column slices 0:5 / 5:10 into _targets),
        # silently scrambling every IK write across both arms: e.g. in a right-only reach
        # the right shoulder pitch/roll were pinned at default while the left elbow
        # received the right arm's commands. This was THE dominant reach-accuracy bug
        # (FK-verified via validation/check_ik_accuracy.py: p50 miss 14cm left / 30cm
        # right with perfect tracking, ~0% within 3cm on the right). The buffer's column
        # convention (asset order, matching these ids) is kept — every consumer
        # (actions.py's blend term, observations.py's intent term, the deployment
        # scripts) pairs columns to _standing_arm_motion_joint_ids dynamically — and the
        # per-side column mapping below is now derived from the ids themselves
        # (_col_idx), never assumed.
        all_joint_ids, _ = self._asset.find_joints(_ARM_JOINT_NAMES)
        # Keep our own reference — __call__ re-asserts BOTH env attributes every step
        # (2026-07-16): external code can null them between uses (eval_full_demo.py's
        # walking bucket does exactly that, deliberately), and the blend action term
        # falls through to stock behavior unless both are set — which silently hands the
        # arm joints to the policy's unshaped raw output instead of this term's targets.
        self._all_joint_ids = torch.tensor(all_joint_ids, dtype=torch.long, device=self.device)
        self._env._standing_arm_motion_joint_ids = self._all_joint_ids
        self._targets = self._asset.data.default_joint_pos[:, self._all_joint_ids].clone()
        self._default = self._targets.clone()
        self._env._standing_arm_motion_targets = self._targets

        self._joint_ids: dict[str, torch.Tensor] = {}
        self._jacobi_joint_ids: dict[str, torch.Tensor] = {}
        self._ee_body_idx: dict[str, int] = {}
        self._jacobi_body_idx: dict[str, int] = {}
        self._col_idx: dict[str, torch.Tensor] = {}
        self._ik: dict[str, DifferentialIKController] = {}

        # Column of each asset joint id within _all_joint_ids — the ordering-robust
        # replacement for the old per-side block slices (see the interleaving note above).
        id_to_col = {int(j): c for c, j in enumerate(self._all_joint_ids.tolist())}

        is_fixed_base = self._asset.is_fixed_base
        ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method=self.ik_method)
        for side in self._sides:
            joint_ids, _ = self._asset.find_joints(self._arm_joint_names[side])
            self._joint_ids[side] = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
            self._col_idx[side] = torch.tensor(
                [id_to_col[int(j)] for j in joint_ids], dtype=torch.long, device=self.device
            )

            body_ids, _ = self._asset.find_bodies(self._ee_body_name[side])
            self._ee_body_idx[side] = body_ids[0]
            if is_fixed_base:
                self._jacobi_body_idx[side] = body_ids[0] - 1
                self._jacobi_joint_ids[side] = self._joint_ids[side]
            else:
                self._jacobi_body_idx[side] = body_ids[0]
                self._jacobi_joint_ids[side] = self._joint_ids[side] + 6

            self._ik[side] = DifferentialIKController(cfg=ik_cfg, num_envs=self.num_envs, device=self.device)

        # Per-side goal-cycling state — same structure as
        # validation/integration_validation/eval_full_demo.py's _ArmAttemptTracker.
        self._goal_local = {s: torch.zeros(self.num_envs, 3, device=self.device) for s in self._sides}
        self._attempt_steps = {s: torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for s in self._sides}
        self._min_dist = {s: torch.full((self.num_envs,), float("inf"), device=self.device) for s in self._sides}
        # Which side(s) this episode reaches with — resampled per env on reset.
        self._active = {s: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for s in self._sides}

        # Deployment gain matching (2026-07-14, Phase 1 of the integration-recovery plan):
        # when enabled, every episode reset writes each env's *actively reaching* arm(s) to
        # _DEPLOY_ACTIVE_ARM_GAIN and its held arm(s) back to the env cfg's own trained
        # gain, mirroring the per-arm split both deployment scripts apply. Off by default —
        # existing configs/checkpoints keep their historical behavior unchanged. Held-arm
        # values are captured from the live sim here (i.e. whatever the actuator cfg set,
        # 60/1.5 for the standing configs) rather than hardcoded, so a future cfg gain
        # change propagates automatically. Known small approximation: before enable_step
        # the "active" arm is only holding default but already carries the stiff gain until
        # its next reset — deployment-idle runs both arms soft. Accepted: the reaching
        # regime is the one being trained for, and pre-enable covers only the first 30k
        # env-steps of a run.
        self._match_deployment_arm_gains = bool(cfg.params.get("match_deployment_arm_gains", False))
        if self._match_deployment_arm_gains:
            self._held_arm_stiffness = {
                s: self._asset.data.joint_stiffness[:, self._joint_ids[s]].clone() for s in self._sides
            }
            self._held_arm_damping = {
                s: self._asset.data.joint_damping[:, self._joint_ids[s]].clone() for s in self._sides
            }

    @property
    def num_envs(self) -> int:
        return self._asset.num_instances

    def _ramp_fraction(self, env_step: int) -> float:
        """Fraction (start_fraction..1.0] applied to both reach speed and goal-box range
        right now — see class docstring. 0 before enable_step (arms held at default,
        unaffected by this), start_fraction right as it turns on, linearly reaching 1.0
        at ramp_full_step, then staying there. Same shape as
        g1_arm_env.py's _current_goal_curriculum_fraction."""
        if env_step < self.enable_step:
            return 0.0
        if self.ramp_full_step <= self.enable_step:
            return 1.0
        progress = min(1.0, (env_step - self.enable_step) / (self.ramp_full_step - self.enable_step))
        return self.start_fraction + (1.0 - self.start_fraction) * progress

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        self._targets[env_ids] = self._default[env_ids]
        self._env._standing_arm_motion_targets = self._targets

        # none / both / left-only / right-only, per env — see no_reach_prob/both_arms_prob.
        # "none" leaves both actives False: _step_side then holds both arms at default the
        # whole episode (torch.where(active, ...) falls through to the default pose), and
        # _apply_deployment_arm_gains gives both arms the cfg's own held gain — exactly
        # deployment's idle condition.
        roll = torch.rand(len(env_ids), device=self.device)
        none = roll < self.no_reach_prob
        both = ~none & (roll < self.no_reach_prob + self.both_arms_prob)
        reach_prob = 1.0 - self.no_reach_prob - self.both_arms_prob
        left_only = ~none & ~both & (roll < self.no_reach_prob + self.both_arms_prob + 0.5 * reach_prob)
        right_only = ~none & ~both & ~left_only
        self._active["left"][env_ids] = both | left_only
        self._active["right"][env_ids] = both | right_only

        if self._match_deployment_arm_gains:
            self._apply_deployment_arm_gains(env_ids)

        fraction = self._ramp_fraction(self._env.common_step_counter)
        for side in self._sides:
            self._resample_goal(side, env_ids, fraction)

    def _apply_deployment_arm_gains(self, env_ids: torch.Tensor):
        """Per-env, per-arm gain write on reset — active arm(s) at _DEPLOY_ACTIVE_ARM_GAIN,
        held arm(s) restored to the env cfg's own trained gain. See the
        match_deployment_arm_gains comment in __init__."""
        for side in self._sides:
            active = self._active[side][env_ids]
            active_ids = env_ids[active]
            held_ids = env_ids[~active]
            joint_ids = self._joint_ids[side]
            if active_ids.numel() > 0:
                self._asset.write_joint_stiffness_to_sim(
                    self._DEPLOY_ACTIVE_ARM_GAIN[0], joint_ids=joint_ids, env_ids=active_ids
                )
                self._asset.write_joint_damping_to_sim(
                    self._DEPLOY_ACTIVE_ARM_GAIN[1], joint_ids=joint_ids, env_ids=active_ids
                )
            if held_ids.numel() > 0:
                self._asset.write_joint_stiffness_to_sim(
                    self._held_arm_stiffness[side][held_ids], joint_ids=joint_ids, env_ids=held_ids
                )
                self._asset.write_joint_damping_to_sim(
                    self._held_arm_damping[side][held_ids], joint_ids=joint_ids, env_ids=held_ids
                )

    def _resample_goal(self, side: str, env_ids: torch.Tensor, fraction: float):
        bounds = self._goal_bounds[side]
        n = len(env_ids)
        for i, key in enumerate(("x", "y", "z")):
            lo, hi = bounds[key]
            centre = (lo + hi) * 0.5
            half_span = (hi - lo) * 0.5 * fraction
            self._goal_local[side][env_ids, i] = centre + (torch.rand(n, device=self.device) * 2 - 1) * half_span
        self._attempt_steps[side][env_ids] = 0
        self._min_dist[side][env_ids] = float("inf")

    def _jacobian_b(self, side: str) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat, quat_inv

        jacobian_w = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx[side], :, self._jacobi_joint_ids[side]
        ]
        base_rot_matrix = matrix_from_quat(quat_inv(self._asset.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
        return jacobian_b

    def _step_side(self, side: str, env: ManagerBasedEnv, fraction: float):
        from isaaclab.utils.math import subtract_frame_transforms

        robot = self._asset
        joint_ids = self._joint_ids[side]
        active = self._active[side]

        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_idx[side]]
        ee_quat_w = robot.data.body_quat_w[:, self._ee_body_idx[side]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
        )

        # Ground-referenced goal z -> root-frame z, recomputed every step from the
        # robot's *current* height — see class docstring.
        root_height_above_ground = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        target_b = self._goal_local[side].clone()
        target_b[:, 2] = target_b[:, 2] - root_height_above_ground

        self._ik[side].set_command(target_b, ee_quat=ee_quat_b)
        jacobian_b = self._jacobian_b(side)
        joint_pos = robot.data.joint_pos[:, joint_ids]
        joint_pos_des = self._ik[side].compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos)

        # Rate-limit to a realistic joint speed (ramped — see class docstring), then
        # clamp to real hardware limits — same two safety nets every deployment script in
        # this repo already applies to the RL arm policy's own output
        # (ARM_MAX_JOINT_DELTA_PER_STEP + soft_joint_pos_limits).
        max_delta = self.max_joint_delta_per_step * fraction
        delta = (joint_pos_des - joint_pos).clamp(-max_delta, max_delta)
        limits = robot.data.soft_joint_pos_limits[:, joint_ids]
        new_targets = (joint_pos + delta).clamp(limits[:, :, 0], limits[:, :, 1])

        col = self._col_idx[side]
        self._targets[:, col] = torch.where(active.unsqueeze(-1), new_targets, self._default[:, col])

        dist = torch.linalg.vector_norm(ee_pos_b - target_b, dim=-1)
        self._min_dist[side] = torch.minimum(self._min_dist[side], dist.detach())
        self._attempt_steps[side] += 1

        # Resample only on timeout, never as soon as goal_threshold_m is first crossed
        # (2026-07-12, was `active & (reached | timed_out)`). The old behavior meant this
        # disturbance never held a goal once reached — the instant _min_dist dipped under
        # 3cm, a brand new goal was sampled that same step, so standing never actually
        # trained against "the arm has arrived and is sitting there" — only against
        # continuous, ever-changing reach motion. That's a real, confirmed gap, found
        # while investigating why a checkpoint trained under this disturbance still fell
        # in deployment specifically a few seconds *after* a target was reached and held —
        # a scenario this training signal never produced. Now a goal simply persists for
        # the full max_steps_per_goal (15s) regardless of how quickly it's reached, so
        # fast reaches naturally get long holds and slow ones get short holds — a spread of
        # hold durations for free, rather than hardcoding one fixed dwell time.
        timed_out = self._attempt_steps[side] >= self.max_steps_per_goal
        needs_new_goal = active & timed_out
        resample_ids = needs_new_goal.nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_goal(side, resample_ids, fraction)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        enable_step: int | None = None,
        ramp_full_step: int | None = None,
        start_fraction: float | None = None,
        match_deployment_arm_gains: bool | None = None,
        no_reach_prob: float | None = None,
    ):
        # asset_cfg and the construction-time params are all consumed once in
        # __init__ (see there) — EventManager re-passes the full params dict as kwargs on
        # every single call though (same reason StandingArmTrajectoryDisturbance's own
        # phase_step_boundaries/phase_step_offset params are accepted here and ignored,
        # not just at construction), so these need to be accepted here too or a PLAY-style
        # cfg override crashes every step instead of just working.
        del asset_cfg, enable_step, ramp_full_step, start_fraction, match_deployment_arm_gains, no_reach_prob
        # Re-assert both attrs every call — see the __init__ comment (2026-07-16).
        self._env._standing_arm_motion_joint_ids = self._all_joint_ids
        fraction = self._ramp_fraction(env.common_step_counter)
        if fraction <= 0.0:
            self._targets[:] = self._default
            self._env._standing_arm_motion_targets = self._targets
            return

        for side in self._sides:
            self._step_side(side, env, fraction)


class StandingArmPolicyReachDisturbance(StandingArmIKReachDisturbance):
    """Same goal-sampling/dwell/curriculum-ramp scaffolding as StandingArmIKReachDisturbance
    (inherited unchanged — reset(), _resample_goal(), _ramp_fraction(), _jacobian_b() all
    apply as-is), but drives the arm(s) with the actual frozen, trained arm-IK RL policy
    instead of analytic IK. Only __init__ (loads the extra policy) and _step_side (runs it
    instead of IK) differ — see the override below for the frame/scale conversion, which
    matches every deployment script's own arm-policy handling exactly (eval_full_demo.py's
    _compute_arm_delta, g1_full_demo.py's _arm_action_override).

    StandingArmIKReachDisturbance was deliberately built on analytic IK instead of the
    trained arm policy (2026-07-10, see that class's docstring) specifically to avoid
    baking the arm policy's current quirks into standing's training and to survive future
    arm-policy retrains — that reasoning is still sound in general. This class exists
    because direct evidence now contradicts analytic IK as the right proxy for *this* gap:
    a checkpoint trained against it (with the dwell fix — goals now held until timeout
    instead of resampled instantly on arrival, see StandingArmIKReachDisturbance._step_side)
    scored fall_rate=0.13% in its own native eval under this exact disturbance (768
    episodes, 2026-07-13) but fall_rate=73%+ when the *same* checkpoint was deployed
    against the real, trained arm-IK policy in validation/integration_validation/
    eval_full_demo.py. Every other difference between those two environments (reset
    events, forced commands, action term, gains, termination conditions, observation
    terms, default pose) was individually checked and confirmed identical — the disturbance
    source is the one remaining, unverified difference. Manually driving the real policy in
    testing/general_testing/g1_full_demo.py also directly showed it visibly oscillating
    near a reached goal (a few cm back and forth, repeatedly) rather than settling —
    analytic IK never produces that; standing has never trained against it. This class
    closes that gap by using the actual policy as the disturbance source, jitter included.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        from rsl_rl.modules import ActorCritic
        from tensordict import TensorDict

        from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import G1ArmIKLeftPPORunnerCfg
        from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import mirror_arm_actions, mirror_arm_obs

        self._mirror_arm_obs = mirror_arm_obs
        self._mirror_arm_actions = mirror_arm_actions
        self._TensorDict = TensorDict

        checkpoint_path = cfg.params.get("arm_checkpoint")
        if not checkpoint_path:
            raise ValueError(
                "StandingArmPolicyReachDisturbance requires params={'arm_checkpoint': <path>, ...} "
                "— there is no sensible default checkpoint to fall back to."
            )
        # No dedicated right-arm checkpoint exists — the right arm is driven by mirroring
        # this same left-trained policy's observation/action, exactly like every other
        # deployment path in this repo (eval_full_demo.py, g1_full_demo.py,
        # mdp/symmetry.py's own module docstring explains the rationale).
        agent_cfg = G1ArmIKLeftPPORunnerCfg()
        obs_dim, action_dim = 28, 5
        dummy_obs = TensorDict(
            {"policy": torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)},
            batch_size=[1], device=self.device,
        )
        self._arm_actor_critic = ActorCritic(
            obs=dummy_obs,
            obs_groups=agent_cfg.obs_groups,
            num_actions=action_dim,
            actor_obs_normalization=agent_cfg.policy.actor_obs_normalization,
            critic_obs_normalization=agent_cfg.policy.critic_obs_normalization,
            actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
            activation=agent_cfg.policy.activation,
            init_noise_std=agent_cfg.policy.init_noise_std,
        ).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self._arm_actor_critic.load_state_dict(state["model_state_dict"])
        self._arm_actor_critic.eval()
        for param in self._arm_actor_critic.parameters():
            param.requires_grad_(False)

        # EMA action-filter state, one 5-D buffer per side per env — matches
        # ARM_ACTION_FILTER_ALPHA's role everywhere else this policy is deployed. Running
        # it unfiltered here would train standing against a signal no real deployment path
        # actually produces (every one of them filters).
        self._arm_action_filter_alpha = cfg.params.get("arm_action_filter_alpha", 0.25)
        self._filtered_delta = {s: torch.zeros(self.num_envs, 5, device=self.device) for s in self._sides}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        for side in self._sides:
            self._filtered_delta[side][env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        enable_step: int | None = None,
        ramp_full_step: int | None = None,
        start_fraction: float | None = None,
        match_deployment_arm_gains: bool | None = None,
        no_reach_prob: float | None = None,
        arm_checkpoint: str | None = None,
        arm_action_filter_alpha: float | None = None,
    ):
        # arm_checkpoint/arm_action_filter_alpha consumed once in __init__ — same
        # re-passed-every-call reasoning as the base class's own ramp params.
        del arm_checkpoint, arm_action_filter_alpha
        return super().__call__(
            env, env_ids, asset_cfg, enable_step, ramp_full_step, start_fraction,
            match_deployment_arm_gains, no_reach_prob,
        )

    def _step_side(self, side: str, env: ManagerBasedEnv, fraction: float):
        from isaaclab.utils.math import subtract_frame_transforms

        robot = self._asset
        joint_ids = self._joint_ids[side]
        active = self._active[side]

        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_idx[side]]
        ee_quat_w = robot.data.body_quat_w[:, self._ee_body_idx[side]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
        )
        del ee_quat_b  # unused here (analytic-IK-only need); kept for signature parity

        # Ground-referenced goal z -> root-frame z, recomputed every step — same as the
        # base class.
        root_height_above_ground = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        target_b = self._goal_local[side].clone()
        target_b[:, 2] = target_b[:, 2] - root_height_above_ground

        joint_pos = robot.data.joint_pos[:, joint_ids]
        joint_vel = robot.data.joint_vel[:, joint_ids]
        # Both ee_pos_b and target_b are already in the robot's *current* root frame
        # (subtract_frame_transforms / the ground-height conversion above), so the error
        # here needs no extra rotation — unlike eval_full_demo.py/g1_full_demo.py, which
        # start from world-frame quantities and must explicitly rotate into body frame to
        # match what g1_arm_env.py's fixed-root training produced. Same destination, this
        # class just starts one step closer to it.
        error_b = target_b - ee_pos_b

        obs = torch.cat(
            [
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                joint_pos,
                joint_vel,
                ee_pos_b,
                target_b,
                error_b,
            ],
            dim=-1,
        )
        if side == "right":
            obs = self._mirror_arm_obs(obs)

        with torch.inference_mode():
            td = self._TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=self.device)
            raw_delta = self._arm_actor_critic.act_inference(td)

        if side == "right":
            raw_delta = self._mirror_arm_actions(raw_delta)

        alpha = self._arm_action_filter_alpha
        self._filtered_delta[side] = alpha * raw_delta + (1.0 - alpha) * self._filtered_delta[side]

        # Same raw-output -> absolute-target conversion every deployment path uses
        # (filtered delta * scale, added to current position) — the shared rate-limit +
        # soft-joint-limit clamp right after this is identical to the base class's, so a
        # too-aggressive desired target still gets the same safety net either way.
        joint_pos_des = joint_pos + self._filtered_delta[side] * _ARM_ACTION_SCALE

        max_delta = self.max_joint_delta_per_step * fraction
        delta = (joint_pos_des - joint_pos).clamp(-max_delta, max_delta)
        limits = robot.data.soft_joint_pos_limits[:, joint_ids]
        new_targets = (joint_pos + delta).clamp(limits[:, :, 0], limits[:, :, 1])

        col = self._col_idx[side]
        self._targets[:, col] = torch.where(active.unsqueeze(-1), new_targets, self._default[:, col])

        dist = torch.linalg.vector_norm(error_b, dim=-1)
        self._min_dist[side] = torch.minimum(self._min_dist[side], dist.detach())
        self._attempt_steps[side] += 1

        # Same dwell-until-timeout behavior as the base class (2026-07-12 fix) — resample
        # only on timeout, never the instant the goal is first reached, so a hold actually
        # gets exercised. See the base class's _step_side for the full rationale.
        timed_out = self._attempt_steps[side] >= self.max_steps_per_goal
        needs_new_goal = active & timed_out
        resample_ids = needs_new_goal.nonzero(as_tuple=False).squeeze(-1)
        if resample_ids.numel() > 0:
            self._resample_goal(side, resample_ids, fraction)
            self._filtered_delta[side][resample_ids] = 0.0
        self._env._standing_arm_motion_targets = self._targets
