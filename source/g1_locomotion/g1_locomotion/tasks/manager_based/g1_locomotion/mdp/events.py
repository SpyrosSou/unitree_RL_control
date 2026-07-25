# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arm-motion disturbance curriculum for the G1 29dof unified stand+walk task.

Ported+adapted 2026-07-21 from ``g1_locomotion/tasks/manager_based/g1_locomotion/mdp/
events.py``'s ``StandingArmTrajectoryDisturbance`` (the 23dof-era standing-only version) —
see ``29dof_implementation_plan.md`` for the full rationale. Confirmed necessary, not
optional: the 23dof phase's own history (``phase_logs/phase_1.md``/``phase_2.md``) shows a
standing/walking policy trained with zero arm disturbance falls over the first time a real
reaching arm is introduced — this is exactly the gap Unitree's own stand+walk recipe has
(it has never faced an active arm-reach disturbance either).

**Only this scripted, goal-independent variant is ported today.** The 23dof-era codebase
also had two more sophisticated variants — ``StandingArmIKReachDisturbance`` (real x/y/z
reach targets solved via ``DifferentialIKController``) and ``StandingArmPolicyReachDisturbance``
(driven by the actual trained arm-IK RL policy) — both of which are BLOCKED on the 7-DOF arm
rewrite (Phase 3 of the plan): they need ``_GOAL_BOUNDS``/arm joint lists/end-effector body
names from ``g1_arm_env.py``, which doesn't exist for the 7-DOF arm yet. Port those once
Phase 3 lands, following the same pattern (see the 23dof-era ``events.py`` for the full
implementation to adapt).

Joint set change from the 23dof/5-DOF-arm version: 7 joints per arm now (shoulder
pitch/roll/yaw, elbow, wrist roll/pitch/yaw), not 5 (no wrist). Per-joint amplitude/rate
scaling below is a first-pass estimate for the 3 new wrist joints (smaller, faster joints
— scaled similarly to elbow_roll, the smallest-range joint in the old scheme) and has not
been visually verified via ``play.py`` — do that before trusting this for a real training
run, same standing instruction the 23dof-era version carried for every untested curriculum
change.

**Gated to (near-)standing envs only, 2026-07-22** (user request): "arms while walking"
is a deliberately deferred future feature (see ``29dof_implementation_plan.md``'s
deferred-items note), not something this curriculum should train toward yet. Envs
currently commanded to walk (``base_velocity`` command norm above
``_STANDING_CMD_THRESHOLD``) get their arm targets relaxed toward default instead of
disturbed, regardless of which curriculum phase is otherwise active — see the gate at
the end of ``__call__``.
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def reset_arm_motion_targets_to_default(
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

    if hasattr(env, "_arm_motion_targets"):
        env._arm_motion_targets[env_ids] = asset.data.default_joint_pos[env_ids][:, joint_ids_t].clone()


class ArmMotionDisturbance(ManagerTermBase):
    """Inject smooth random arm trajectories during unified stand+walk training.

    **Re-tuned 2026-07-22** (user request, replacing the untuned 23dof-era carryover
    values): the old 5-tier scheme topped out at 0.66 rad/step (~33 rad/s), deliberately
    chosen to sit just under the real G1 arm joint speed limit (N5020-16 group:
    ``velocity_limit_sim=37 rad/s`` in ``assets/robots/unitree.py`` — confirmed against
    the actual URDF, https://github.com/unitreerobotics/unitree_ros). Physically
    achievable, but not representative of anything the robot will actually do — that's a
    flailing speed, not a reach. Re-derived from the robot's own joint ranges instead
    (URDF ``<limit>`` values, cross-checked against ``velocity_limit_sim``):

    | Joint | Range (rad) | Range (deg) |
    |---|---|---|
    | shoulder_pitch | 5.7596 | 330 |
    | shoulder_roll  | 3.8397 | 220 |
    | shoulder_yaw   | 5.2360 | 300 |
    | elbow          | 3.1416 | 180 |
    | wrist_roll     | 3.9444 | 226 |
    | wrist_pitch/yaw| 3.2289 | 185 |

    Each phase is defined by "how many seconds would a full range-of-motion sweep take
    at this phase's max speed" — shoulder_pitch (the largest-range joint) sets the
    per-step delta directly; every other joint's delta is scaled by its own range ÷
    shoulder_pitch's range (replacing the old, explicitly-untuned-guess
    ``_joint_motion_scale`` table with a value actually derived from the robot's own
    geometry):
    - Phase 0: no arm motion (baseline warm-up + lets the policy learn to stand without
      any arm disturbance at all, not just as a transient warm-up — standing must work
      with zero disturbance before it's asked to work with any)
    - Phase 1 (mild): 8s full-sweep-equivalent speed (shoulder_pitch 0.72 rad/s)
    - Phase 2 (moderate): 4s full-sweep-equivalent speed (shoulder_pitch 1.44 rad/s)
    - Phase 3 (max): 2s full-sweep-equivalent speed (shoulder_pitch 2.88 rad/s) — still
      only ~5-8% of the real actuator speed ceiling (37/22 rad/s), comfortably below
      even a 1-second full sweep (which the user's own elbow math put at 3.14 rad/s and
      considered already aggressive).

    ``_MAX_AMPLITUDE_RAD``/``_NO_MOTION_PROB``/``_ASYMMETRY_PROB``/``_REVERSAL_PROB``
    keep the first 3 non-zero tiers of the old 5-tier progression unchanged (not part of
    this re-tune's scope — only speed/phase-count were revisited).

    **Phase mixing, added 2026-07-22** (user request, after diagnosing why phase 1/2
    fall rates weren't improving with more training: the phase you're in was a single
    value shared by every env, computed purely from elapsed step count — once training
    crosses a boundary, *every* env moves to the harder phase at once and nothing ever
    goes back. Phase 1 was only ever actively trained for ~500 iterations before the
    curriculum moved on; querying it later reflects a skill that stopped being practiced,
    not one that was never learned).

    Fix: each env now independently samples which phase's parameters to actually use at
    every ``reset()`` (not every step — that would make difficulty jitter within a single
    attempt, unrealistic and a worse training signal), from a distribution over
    ``{0, ..., frontier}`` weighted toward the frontier but with real mass on earlier
    tiers: ``weight(i) = 0.5 ** (frontier - i)``, normalized. At frontier=3 that's
    roughly 53/27/13/7% across phases 3/2/1/0 — mostly pushing the current edge, but
    every standing env still has a real chance of practicing an earlier tier on any given
    attempt, keeping it exercised instead of frozen the moment the curriculum advances.
    The frontier itself is still the same step-count-driven ``_phase_index`` as before
    (and still resume-safe via ``phase_step_offset``) — mixing only changes what happens
    *below* the frontier, not how the frontier itself advances.

    Incidental fix from the same refactor: the old code gated reversals behind
    ``if phase >= 3`` even though ``_REVERSAL_PROB[2]=0.02`` was defined — meaning phase 2
    never actually got reversals despite having a nonzero configured probability. The
    per-env vectorized version below applies ``reversal_prob`` directly per sampled phase,
    so phase 2's already-configured value now actually takes effect.
    """

    # Combined norm of the commanded (lin_vel_x, lin_vel_y, ang_vel_z) below which an
    # env counts as "standing" for gating purposes (2026-07-22, user request: "arms
    # while walking" is a deliberately deferred future feature — see
    # 29dof_implementation_plan.md's deferred-items note — this curriculum must not
    # train the policy to expect arm motion while a real walk command is active, only
    # while it's effectively stationary). Not zero-exactly: incidental small-but-
    # nonzero samples from the general command distribution (not just the dedicated
    # rel_standing_envs slice) should still count as "standing enough" — 0.1 is a
    # starting value, not derived from data; tighten/loosen if the resulting standing-
    # vs-walking split looks wrong once training data is in.
    _STANDING_CMD_THRESHOLD = 0.1

    # step boundaries chosen so a normal ~6000-iteration run (num_steps_per_env=24,
    # i.e. step = iteration * 24) reaches max difficulty by iteration ~1667 and gets
    # thousands of iterations of real dwell time there, instead of the old scheme's
    # ~1000-2900 (see phase-reset-on-resume bug fixed in scripts/rsl_rl/train.py the
    # same day — this pacing assumes that fix is in place and phase progress now
    # actually survives a --resume).
    _PHASE_STEP_BOUNDARIES = (3000, 15000, 40000)
    _NO_MOTION_PROB    = (1.00, 0.00, 0.00, 0.00)
    _MAX_DELTA_RAD     = (0.0000, 0.0144, 0.0288, 0.0576)  # shoulder_pitch rad/step; scaled per-joint below
    _MAX_AMPLITUDE_RAD = (0.00, 0.24, 0.45, 0.70)
    _ASYMMETRY_PROB    = (0.00, 0.08, 0.20, 0.45)
    _REVERSAL_PROB     = (0.00, 0.00, 0.02, 0.06)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        joint_ids, joint_names = self._asset.find_joints(_ARM_JOINT_NAMES)
        if len(joint_ids) == 0:
            raise RuntimeError("ArmMotionDisturbance could not find arm joints.")

        self._arm_joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self._arm_joint_names = list(joint_names)

        self._targets = self._asset.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._default = self._targets.clone()
        self._env._arm_motion_targets = self._targets
        self._env._arm_motion_joint_ids = self._arm_joint_ids

        # FIXED 2026-07-24: previously hard-set to the class default here regardless of
        # cfg.params, with a comment claiming __call__ "normally" sets these from cfg
        # params — true for every reset EXCEPT the very first one, since reset() fires
        # at env construction/env.reset(), before any "interval"-mode __call__ has ever
        # run. That meant eval_walking.py's --pin_disturbance_phase override (which sets
        # cfg.params["phase_step_boundaries"] before the env is built) was silently
        # ignored for every env's first episode — with common_step_counter starting
        # near 0, _phase_index() against the unpinned default boundaries (3000, ...)
        # always returns 0 regardless of what phase was requested. Confirmed via a
        # 4-way --pin_disturbance_phase 0/1/2/3 stand_still eval on a low-fall-rate
        # checkpoint returning bit-identical results for all four — a checkpoint that
        # rarely resets mid-eval spends nearly the whole eval window on this
        # always-phase-0 first episode no matter which phase was requested. Now reads
        # the same params __call__ would, so the first reset is pinned correctly too.
        self._phase_step_boundaries = tuple(
            int(v) for v in cfg.params.get("phase_step_boundaries", self._PHASE_STEP_BOUNDARIES)
        )
        self._phase_step_offset = int(cfg.params.get("phase_step_offset", 0))

        # Per-env difficulty-tier mix (2026-07-22 phase-mixing fix — see class
        # docstring). Sampled fresh at each env's reset(), held for that whole standing
        # bout, not resampled per-step.
        self._episode_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Phase-indexed constant tensors for per-env gather-indexing — self._episode_phase
        # now varies per env, so these can't stay plain python tuples indexed by one
        # shared scalar the way the pre-mixing version used them.
        self._no_motion_prob_t = torch.tensor(self._NO_MOTION_PROB, dtype=torch.float32, device=self.device)
        self._max_delta_rad_t = torch.tensor(self._MAX_DELTA_RAD, dtype=torch.float32, device=self.device)
        self._max_amplitude_rad_t = torch.tensor(self._MAX_AMPLITUDE_RAD, dtype=torch.float32, device=self.device)
        self._asymmetry_prob_t = torch.tensor(self._ASYMMETRY_PROB, dtype=torch.float32, device=self.device)
        self._reversal_prob_t = torch.tensor(self._REVERSAL_PROB, dtype=torch.float32, device=self.device)

        # Build left-right joint index pairs and mirror signs for symmetric motions.
        # Roll and yaw components are mirrored with opposite sign under left-right symmetry
        # — same convention as mdp/symmetry.py's _mirror_sign.
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

        # Per-joint scaling, re-derived 2026-07-22 (user request) from each joint's own
        # real range of motion (URDF <limit> values, cross-checked against
        # velocity_limit_sim in assets/robots/unitree.py) instead of the old guessed
        # 1.00/0.95/0.75/0.60/0.55 table — see this class's docstring for the full
        # ranges and methodology. scale[joint] = range[joint] / range[shoulder_pitch]
        # (shoulder_pitch is the largest-range joint and sets _MAX_DELTA_RAD directly).
        _RANGE_SCALE = {
            "shoulder_pitch": 1.000,  # 5.7596 rad range (reference joint)
            "shoulder_roll": 0.667,  # 3.8397 rad range
            "shoulder_yaw": 0.909,  # 5.2360 rad range
            "elbow": 0.545,  # 3.1416 rad range
            "wrist_roll": 0.685,  # 3.9444 rad range
            "wrist_pitch": 0.561,  # 3.2289 rad range
            "wrist_yaw": 0.561,  # 3.2289 rad range
        }
        scale = []
        for name in self._arm_joint_names:
            if "shoulder_pitch" in name:
                scale.append(_RANGE_SCALE["shoulder_pitch"])
            elif "shoulder_roll" in name:
                scale.append(_RANGE_SCALE["shoulder_roll"])
            elif "shoulder_yaw" in name:
                scale.append(_RANGE_SCALE["shoulder_yaw"])
            elif "elbow" in name:
                scale.append(_RANGE_SCALE["elbow"])
            elif "wrist_roll" in name:
                scale.append(_RANGE_SCALE["wrist_roll"])
            elif "wrist_pitch" in name:
                scale.append(_RANGE_SCALE["wrist_pitch"])
            elif "wrist_yaw" in name:
                scale.append(_RANGE_SCALE["wrist_yaw"])
            else:
                scale.append(1.00)
        self._joint_motion_scale = torch.tensor(scale, dtype=torch.float32, device=self.device).view(1, -1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        self._targets[env_ids] = self._default[env_ids]
        self._env._arm_motion_targets = self._targets
        self._episode_phase[env_ids] = self._sample_phase_mix(len(env_ids))

    def _sample_phase_mix(self, n: int) -> torch.Tensor:
        """Sample n per-env difficulty tiers from {0..frontier}, weighted toward the
        frontier (geometric decay, ratio 0.5) but with real mass on earlier tiers too —
        see class docstring's "Phase mixing" section."""
        frontier = self._phase_index(self._env.common_step_counter + self._phase_step_offset)
        weights = torch.tensor(
            [0.5 ** (frontier - i) for i in range(frontier + 1)], dtype=torch.float32, device=self.device
        )
        return torch.multinomial(weights, n, replacement=True)

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

        current = self._targets[env_ids]
        defaults = self._default[env_ids]
        relaxed = 0.9 * current + 0.1 * defaults

        # Per-env phase, sampled at each env's last reset() — NOT recomputed here, see
        # _sample_phase_mix / class docstring's "Phase mixing" section.
        phase_per_env = self._episode_phase[env_ids]  # (N,)

        delta_mag = self._max_delta_rad_t[phase_per_env].unsqueeze(-1)
        amp = self._max_amplitude_rad_t[phase_per_env].unsqueeze(-1)
        no_motion_prob = self._no_motion_prob_t[phase_per_env].unsqueeze(-1)
        asymmetry_prob = self._asymmetry_prob_t[phase_per_env].unsqueeze(-1)
        reversal_prob = self._reversal_prob_t[phase_per_env].unsqueeze(-1)

        delta = (2.0 * torch.rand_like(current) - 1.0) * delta_mag
        delta = delta * self._joint_motion_scale

        reversal_mask = torch.rand_like(delta) < reversal_prob
        delta = torch.where(reversal_mask, -2.0 * delta, delta)

        if self._pair_left is not None:
            symmetric_roll = torch.rand((len(env_ids), 1), device=self.device) > asymmetry_prob
            symmetric_mask = symmetric_roll.expand(-1, len(self._pair_left))
            mirrored_right = delta[:, self._pair_left] * self._pair_sign
            delta[:, self._pair_right] = torch.where(symmetric_mask, mirrored_right, delta[:, self._pair_right])

        no_motion_mask = torch.rand((len(env_ids), 1), device=self.device) < no_motion_prob
        updated = current + delta
        amp_vec = amp * self._joint_motion_scale
        updated = torch.clamp(updated, defaults - amp_vec, defaults + amp_vec)
        updated = torch.where(no_motion_mask, relaxed, updated)

        # Envs whose sampled phase is 0 (no disturbance): use the smooth relax-to-default
        # blend, not the hard "clamp to exactly default" that delta_mag=amp=0 would
        # otherwise produce in a single step — preserves the original phase-0 behavior
        # exactly now that phase 0 can be mixed in alongside higher phases in one batch.
        is_phase_0 = (phase_per_env == 0).unsqueeze(-1)
        updated = torch.where(is_phase_0, relaxed, updated)

        # Only actually disturb envs currently commanded to (near-)stand — "arms while
        # walking" is a deliberately deferred future feature (2026-07-22, user
        # request), not something this curriculum should train toward yet. Envs
        # currently commanded to walk get the same smooth relax-to-default blend the
        # phase-0/no-motion branches already use, regardless of which disturbance
        # phase is otherwise active — this is a gate on top of the phase logic, not a
        # replacement for it.
        command = env.command_manager.get_command("base_velocity")[env_ids]
        standing_mask = torch.linalg.vector_norm(command, dim=-1) < self._STANDING_CMD_THRESHOLD
        updated = torch.where(standing_mask.unsqueeze(-1), updated, relaxed)

        self._targets[env_ids] = updated
        self._env._arm_motion_targets = self._targets


class ArmPolicyDisturbance(ManagerTermBase):
    """Same standing-only gate / goal-cycling shape as ``ArmMotionDisturbance``, but drives
    the arm(s) with the actual trained 7-DOF arm RL policy instead of scripted random
    joint deltas — real x/y/z reach targets from ``g1_arm_env.py``'s own ``_GOAL_BOUNDS``,
    solved by running the frozen policy's forward pass every step, exactly the way it
    will really be used once arm commands are deployed.

    **Built 2026-07-22, not yet wired into any training task** (user request — the
    currently-training ``G1-Locomotion-Velocity-ArmDisturbance-v0`` run still uses the
    scripted ``ArmMotionDisturbance``; this class exists as a ready-to-test alternative
    for later, once the current run's results are in). Adapted from the 23dof-era
    ``StandingArmPolicyReachDisturbance``, but deliberately NOT inheriting that class's
    analytic-IK base (``StandingArmIKReachDisturbance``) — this class never needs a
    ``DifferentialIKController``/Jacobian at all (forward kinematics — ``body_pos_w`` —
    is enough to build the observation the policy expects), so porting the unused IK
    machinery would just be dead weight. The 23dof gain-matching mechanism
    (``match_deployment_arm_gains``, training gain vs. a stiffer "actively reaching" gain)
    is also deliberately dropped — that solved a 23dof train/deploy gain mismatch this
    project no longer has (29dof arm gains were already reconciled against ``deploy.yaml``
    directly, see ``assets/robots/unitree.py``'s own module docstring).

    Observation/action convention copied exactly from ``g1_arm_env.py`` (NOT re-derived —
    a mismatched convention here would just feed the policy garbage and produce a
    convincing-looking but meaningless disturbance):
    - 32-D single-arm observation: base_lin_vel_b(3) | base_ang_vel_b(3) |
      projected_gravity_b(3) | joint_pos(7) | joint_vel(7) | ee_pos_b(3) | goal_b(3) |
      error_b(3) — fed from the REAL locomotion robot's actual base motion here (not the
      synthetic wobble ``g1_arm_env.py`` trains against in its own fixed-base env) — same
      real-state-standing-in-for-synthetic-training-state translation the 23dof version
      already relied on.
    - 7-D action -> EMA-filtered (``action_filter_alpha=0.25``) -> scaled
      (``action_scale=0.5``) -> rate-limited (``max_action_delta_per_step=0.06``, itself
      scaled by the same enable/ramp curriculum fraction ``ArmMotionDisturbance`` uses) ->
      clamped to ``soft_joint_pos_limits``.
    - Right arm uses the same left-trained policy, mirrored via ``mirror_arm_obs``/
      ``mirror_arm_actions`` (``g1_arm/mdp/symmetry.py``) — no dedicated right-arm
      checkpoint exists, matching every other deployment path in this repo.

    **Not yet runtime-verified** — written directly against the actual current source
    (obs/action dims, checkpoint key, PPO cfg fields, symmetry functions all confirmed by
    reading them, not guessed) and syntax-checked, but GPU was occupied by the
    arm-disturbance training run this was built alongside, so it has not been exercised
    via ``play.py`` or an actual training step yet. Verify that before trusting it.

    Also not yet resume-safe the way ``ArmMotionDisturbance`` now is — ``scripts/rsl_rl/
    train.py``'s phase-persistence fix only looks for an event named
    ``arm_motion_disturbance``; extend that (or this class's own step-counting) before
    relying on a ``--resume`` of a task using this term to carry its ramp forward
    correctly.
    """

    _STANDING_CMD_THRESHOLD = 0.1  # same convention/value as ArmMotionDisturbance

    enable_step: int = 3000
    ramp_full_step: int = 15000
    start_fraction: float = 0.15
    both_arms_prob: float = 0.2
    no_reach_prob: float = 0.0
    max_steps_per_goal: int = 750  # 15s @ 50Hz
    action_filter_alpha: float = 0.25  # matches g1_arm_env.py's G1ArmEnvCfg.action_filter_alpha
    action_scale: float = 0.5  # matches g1_arm_env.py's G1ArmEnvCfg.action_scale
    max_action_delta_per_step: float = 0.06  # matches g1_arm_env.py's G1ArmEnvCfg.max_action_delta_per_step

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.enable_step = cfg.params.get("enable_step", self.enable_step)
        self.ramp_full_step = cfg.params.get("ramp_full_step", self.ramp_full_step)
        self.start_fraction = cfg.params.get("start_fraction", self.start_fraction)
        self.both_arms_prob = cfg.params.get("both_arms_prob", self.both_arms_prob)
        self.no_reach_prob = cfg.params.get("no_reach_prob", self.no_reach_prob)

        from rsl_rl.modules import ActorCritic
        from tensordict import TensorDict

        from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import G1ArmLeftPPORunnerCfg
        from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
            _GOAL_BOUNDS,
            _LEFT_ARM_JOINTS,
            _LEFT_EE_BODY,
            _RIGHT_ARM_JOINTS,
            _RIGHT_EE_BODY,
        )
        from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import mirror_arm_actions, mirror_arm_obs

        self._TensorDict = TensorDict
        self._mirror_arm_obs = mirror_arm_obs
        self._mirror_arm_actions = mirror_arm_actions
        self._goal_bounds = _GOAL_BOUNDS

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        self._sides = ("left", "right")
        self._arm_joint_names = {"left": _LEFT_ARM_JOINTS, "right": _RIGHT_ARM_JOINTS}
        self._ee_body_name = {"left": _LEFT_EE_BODY, "right": _RIGHT_EE_BODY}

        all_joint_ids, _ = self._asset.find_joints(_ARM_JOINT_NAMES)
        self._all_joint_ids = torch.tensor(all_joint_ids, dtype=torch.long, device=self.device)
        self._targets = self._asset.data.default_joint_pos[:, self._all_joint_ids].clone()
        self._default = self._targets.clone()
        self._env._arm_motion_targets = self._targets
        self._env._arm_motion_joint_ids = self._all_joint_ids

        id_to_col = {int(j): c for c, j in enumerate(self._all_joint_ids.tolist())}
        self._joint_ids: dict[str, torch.Tensor] = {}
        self._ee_body_idx: dict[str, int] = {}
        self._col_idx: dict[str, torch.Tensor] = {}
        for side in self._sides:
            joint_ids, _ = self._asset.find_joints(self._arm_joint_names[side])
            self._joint_ids[side] = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
            self._col_idx[side] = torch.tensor(
                [id_to_col[int(j)] for j in joint_ids], dtype=torch.long, device=self.device
            )
            body_ids, _ = self._asset.find_bodies(self._ee_body_name[side])
            self._ee_body_idx[side] = body_ids[0]

        self._goal_local = {s: torch.zeros(self.num_envs, 3, device=self.device) for s in self._sides}
        self._attempt_steps = {s: torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for s in self._sides}
        self._min_dist = {s: torch.full((self.num_envs,), float("inf"), device=self.device) for s in self._sides}
        self._active = {s: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for s in self._sides}
        self._filtered_delta = {s: torch.zeros(self.num_envs, 7, device=self.device) for s in self._sides}

        checkpoint_path = cfg.params.get("arm_checkpoint")
        if not checkpoint_path:
            raise ValueError(
                "ArmPolicyDisturbance requires params={'arm_checkpoint': <path>, ...} — "
                "there is no sensible default checkpoint to fall back to."
            )
        agent_cfg = G1ArmLeftPPORunnerCfg()
        obs_dim, action_dim = 32, 7
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
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._arm_actor_critic.load_state_dict(state["model_state_dict"])
        self._arm_actor_critic.eval()
        for param in self._arm_actor_critic.parameters():
            param.requires_grad_(False)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        self._targets[env_ids] = self._default[env_ids]
        self._env._arm_motion_targets = self._targets

        n = len(env_ids)
        no_reach = torch.rand(n, device=self.device) < self.no_reach_prob
        both = (~no_reach) & (torch.rand(n, device=self.device) < self.both_arms_prob)
        left_only = (~no_reach) & (~both) & (torch.rand(n, device=self.device) < 0.5)
        right_only = (~no_reach) & (~both) & (~left_only)
        self._active["left"][env_ids] = both | left_only
        self._active["right"][env_ids] = both | right_only

        for side in self._sides:
            self._filtered_delta[side][env_ids] = 0.0
            self._resample_goal(side, env_ids)

    def _resample_goal(self, side: str, env_ids: torch.Tensor) -> None:
        bounds = self._goal_bounds[side]
        n = len(env_ids)
        for i, axis in enumerate(("x", "y", "z")):
            lo, hi = bounds[axis]
            self._goal_local[side][env_ids, i] = torch.rand(n, device=self.device) * (hi - lo) + lo
        self._attempt_steps[side][env_ids] = 0
        self._min_dist[side][env_ids] = float("inf")

    def _ramp_fraction(self, step: int) -> float:
        if step < self.enable_step:
            return 0.0
        if step >= self.ramp_full_step:
            return 1.0
        span = max(self.ramp_full_step - self.enable_step, 1)
        t = (step - self.enable_step) / span
        return self.start_fraction + (1.0 - self.start_fraction) * t

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        enable_step: int | None = None,
        ramp_full_step: int | None = None,
        start_fraction: float | None = None,
        both_arms_prob: float | None = None,
        no_reach_prob: float | None = None,
        arm_checkpoint: str | None = None,
    ) -> None:
        # All __init__-only params re-passed every call by the manager — consumed once
        # above, ignored here (same reasoning as ArmMotionDisturbance's own params).
        del asset_cfg, enable_step, ramp_full_step, start_fraction, both_arms_prob, no_reach_prob, arm_checkpoint

        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        from isaaclab.utils.math import subtract_frame_transforms

        fraction = self._ramp_fraction(env.common_step_counter)
        current = self._targets[env_ids]
        defaults = self._default[env_ids]
        relaxed = 0.9 * current + 0.1 * defaults

        if fraction <= 0.0:
            self._targets[env_ids] = relaxed
            self._env._arm_motion_targets = self._targets
            return

        # Standing-only gate — same rule as ArmMotionDisturbance (2026-07-22, "arms while
        # walking" is a deliberately deferred future feature).
        command = env.command_manager.get_command("base_velocity")[env_ids]
        standing_mask = torch.linalg.vector_norm(command, dim=-1) < self._STANDING_CMD_THRESHOLD

        robot = self._asset
        updated = current.clone()
        root_height_above_ground = robot.data.root_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
        base_lin_vel_b = robot.data.root_lin_vel_b[env_ids]
        base_ang_vel_b = robot.data.root_ang_vel_b[env_ids]
        projected_gravity_b = robot.data.projected_gravity_b[env_ids]

        for side in self._sides:
            active = self._active[side][env_ids] & standing_mask
            if not torch.any(active):
                continue

            joint_ids = self._joint_ids[side]
            col = self._col_idx[side]

            ee_pos_w = robot.data.body_pos_w[env_ids][:, self._ee_body_idx[side]]
            ee_quat_w = robot.data.body_quat_w[env_ids][:, self._ee_body_idx[side]]
            ee_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], ee_pos_w, ee_quat_w
            )

            target_b = self._goal_local[side][env_ids].clone()
            target_b[:, 2] = target_b[:, 2] - root_height_above_ground
            error_b = target_b - ee_pos_b

            joint_pos = robot.data.joint_pos[env_ids][:, joint_ids]
            joint_vel = robot.data.joint_vel[env_ids][:, joint_ids]

            obs = torch.cat(
                [base_lin_vel_b, base_ang_vel_b, projected_gravity_b, joint_pos, joint_vel, ee_pos_b, target_b, error_b],
                dim=-1,
            )
            if side == "right":
                obs = self._mirror_arm_obs(obs)

            with torch.inference_mode():
                td = self._TensorDict({"policy": obs}, batch_size=[obs.shape[0]], device=self.device)
                raw_delta = self._arm_actor_critic.act_inference(td)

            if side == "right":
                raw_delta = self._mirror_arm_actions(raw_delta)

            alpha = self.action_filter_alpha
            fd = alpha * raw_delta + (1.0 - alpha) * self._filtered_delta[side][env_ids]
            self._filtered_delta[side][env_ids] = fd

            delta = fd * self.action_scale
            max_delta = self.max_action_delta_per_step * fraction
            delta = delta.clamp(-max_delta, max_delta)
            limits = robot.data.soft_joint_pos_limits[env_ids][:, joint_ids]
            new_targets_side = (joint_pos + delta).clamp(limits[:, :, 0], limits[:, :, 1])

            updated[:, col] = torch.where(active.unsqueeze(-1), new_targets_side, updated[:, col])

            dist = torch.linalg.vector_norm(error_b, dim=-1)
            self._min_dist[side][env_ids] = torch.where(
                active, torch.minimum(self._min_dist[side][env_ids], dist.detach()), self._min_dist[side][env_ids]
            )
            self._attempt_steps[side][env_ids] = torch.where(
                active, self._attempt_steps[side][env_ids] + 1, self._attempt_steps[side][env_ids]
            )

            timed_out = self._attempt_steps[side][env_ids] >= self.max_steps_per_goal
            resample_local = (active & timed_out).nonzero(as_tuple=False).squeeze(-1)
            if resample_local.numel() > 0:
                resample_ids = env_ids[resample_local]
                self._resample_goal(side, resample_ids)
                self._filtered_delta[side][resample_ids] = 0.0

        updated = torch.where(standing_mask.unsqueeze(-1), updated, relaxed)
        self._targets[env_ids] = updated
        self._env._arm_motion_targets = self._targets
        self._env._arm_motion_targets = self._targets
