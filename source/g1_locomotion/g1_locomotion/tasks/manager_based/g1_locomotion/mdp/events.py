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

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device)

    joint_ids, _ = asset.find_joints(_ARM_JOINT_NAMES)
    joint_ids_t = torch.tensor(joint_ids, dtype=torch.long, device=asset.device)

    if hasattr(env, "_standing_arm_motion_targets"):
        env._standing_arm_motion_targets[env_ids] = asset.data.default_joint_pos[env_ids][:, joint_ids_t].clone()


class StandingArmTrajectoryDisturbance(ManagerTermBase):
    """Inject smooth random arm trajectories during standing-policy training.

    The curriculum is phased by global env step count:
    - Phase 0: no arm motion (standing baseline warm-up)
    - Phase 1: moderate motions
    - Phase 2: large motions
    - Phase 3: large + faster + asymmetric + occasional reversals
    """

    _PHASE_STEP_BOUNDARIES = (6000, 15000, 27000)
    _NO_MOTION_PROB = (1.00, 0.00, 0.00, 0.00)
    _MAX_DELTA_RAD = (0.00, 0.030, 0.070, 0.100)
    _MAX_AMPLITUDE_RAD = (0.00, 0.28, 0.60, 0.90)
    _ASYMMETRY_PROB = (0.00, 0.10, 0.35, 0.85)
    _REVERSAL_PROB = (0.00, 0.00, 0.05, 0.20)

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
            elif "shoulder_roll" in name:
                scale.append(0.95)
            elif "shoulder_yaw" in name:
                scale.append(0.95)
            elif "elbow_pitch" in name:
                scale.append(0.75)
            elif "elbow_roll" in name:
                scale.append(0.65)
            else:
                scale.append(1.00)
        self._joint_motion_scale = torch.tensor(scale, dtype=torch.float32, device=self.device).view(1, -1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._targets[env_ids] = self._default[env_ids]
        self._env._standing_arm_motion_targets = self._targets

    def _phase_index(self, common_step_counter: int) -> int:
        if common_step_counter < self._phase_step_boundaries[0]:
            return 0
        if common_step_counter < self._phase_step_boundaries[1]:
            return 1
        if common_step_counter < self._phase_step_boundaries[2]:
            return 2
        return 3

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        phase_step_boundaries: tuple[int, int, int] | None = None,
        phase_step_offset: int = 0,
    ) -> None:
        del asset_cfg

        if phase_step_boundaries is None:
            self._phase_step_boundaries = self._PHASE_STEP_BOUNDARIES
        else:
            self._phase_step_boundaries = tuple(int(v) for v in phase_step_boundaries)

        self._phase_step_offset = int(phase_step_offset)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, slice):
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
                symmetric_mask = (torch.rand((len(env_ids), 1), device=self.device) > self._ASYMMETRY_PROB[phase]).expand(
                    -1, len(self._pair_left)
                )
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
