# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action terms for the G1 29dof unified stand+walk task.

Ported+adapted 2026-07-21 from ``g1_locomotion/tasks/manager_based/g1_locomotion/mdp/
actions.py`` (the 23dof-era standing-only version) — see ``29dof_implementation_plan.md``.
Only change: the arm-joint substring match now also matches ``"wrist"`` (the 7-DOF arm
has 3 wrist joints the 23dof/5-DOF arm didn't), and the class/attribute names drop the
"Standing" prefix since the disturbance now applies to the unified stand+walk task, not a
dedicated standing-only policy.
"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class ArmDisturbanceBlendJointPositionAction(JointPositionAction):
    """Joint position action that overrides arm joints with the arm-motion-disturbance
    curriculum's targets (see ``events.py``'s ``ArmMotionDisturbance``), leaving every
    other joint (legs, waist) to the policy's own raw output unchanged.

    Keeps the action dimension unchanged while enforcing scripted arm motion — critically,
    this only overrides the *simulated* target (via ``set_joint_position_target``), not the
    raw action tensor itself, so the policy's own output for the arm columns still flows
    into next step's ``last_action`` observation exactly as trained (see the 23dof-era
    ``g1_full_demo.py``'s identical rationale for why this decoupling matters — overwriting
    the action tensor directly corrupts ``last_action`` in a way this class avoids).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        name_to_col = {name: idx for idx, name in enumerate(self._joint_names)}
        arm_cols = []
        for name in self._joint_names:
            is_arm_joint = ("shoulder" in name) or ("elbow" in name) or ("wrist" in name)
            if is_arm_joint:
                arm_cols.append(name_to_col[name])
        self._arm_action_cols = torch.tensor(arm_cols, dtype=torch.long, device=self.device) if arm_cols else None

    def apply_actions(self):
        targets = self.processed_actions.clone()

        arm_motion_targets = getattr(self._env, "_arm_motion_targets", None)
        arm_motion_joint_ids = getattr(self._env, "_arm_motion_joint_ids", None)

        if (
            self._arm_action_cols is not None
            and arm_motion_targets is not None
            and arm_motion_joint_ids is not None
            and len(self._arm_action_cols) > 0
        ):
            action_joint_ids = self._joint_ids
            if isinstance(action_joint_ids, slice):
                action_joint_ids = torch.arange(self._asset.num_joints, device=self.device)
            else:
                action_joint_ids = torch.tensor(action_joint_ids, dtype=torch.long, device=self.device)

            # Map each disturbed arm joint id onto the corresponding action-space column.
            id_to_action_col = {int(jid): idx for idx, jid in enumerate(action_joint_ids.tolist())}
            mapped_cols = []
            mapped_src = []
            for src_idx, joint_id in enumerate(arm_motion_joint_ids.tolist()):
                col = id_to_action_col.get(int(joint_id))
                if col is not None:
                    mapped_cols.append(col)
                    mapped_src.append(src_idx)
            if len(mapped_cols) > 0:
                col_t = torch.tensor(mapped_cols, dtype=torch.long, device=self.device)
                src_t = torch.tensor(mapped_src, dtype=torch.long, device=self.device)
                targets[:, col_t] = arm_motion_targets[:, src_t]

        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)
