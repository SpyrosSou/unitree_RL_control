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
