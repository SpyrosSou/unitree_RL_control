# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the G1 29dof arm-policy observation and action
space.

Full rewrite, 2026-07-21, from the 23dof-era ``g1_arm/mdp/symmetry.py`` — that version's
hardcoded per-dimension sign vectors were sized for the 28-D/5-D single-arm layout (no
wrist); this task's per-arm block is 32-D/7-D (see ``g1_arm_env.py``'s module
docstring). Same approach as before (hardcoded, not built dynamically like the
locomotion task's symmetry code) since this task's observation/action layout is small
and fixed by construction, not derived from a variable joint set.

Used via ``RslRlSymmetryCfg.data_augmentation_func`` (see ``agents/rsl_rl_ppo_cfg.py``)
so the left-right mirror relationship a future arm-mirror-test script would rely on (to
run a left-trained policy on the right arm, or vice versa — see the 23dof-era
``testing/arm_testing/g1_arm_mirror_test.py`` for the pattern to adapt) is something the
policy was actually trained to satisfy, not an assumption that happens to look right
because the workspace bounds are symmetric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_arm_states", "mirror_arm_obs", "mirror_arm_actions"]

# Per-arm observation block (39-D, see g1_arm_env.py's module docstring):
#   [0:3]   base_lin_vel        — lateral (y) component flips
#   [3:6]   base_ang_vel        — roll-rate (x) and yaw-rate (z) flip, pitch-rate (y) doesn't
#   [6:9]   projected_gravity   — lateral (y) component flips
#   [9:16]  joint_pos  (7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#                            wrist_roll, wrist_pitch, wrist_yaw
#   [16:23] joint_vel  (7): same order as joint_pos
#   [23:26] ee_pos              — y flips
#   [26:29] goal                — y flips
#   [29:32] error               — y flips (= goal - ee_pos, consistent with both flipping)
#   [32:39] action_fb  (7): ADDED 2026-07-26 — same order/sign convention as joint_pos/
#                            joint_vel (it's a joint-position-target delta, same per-joint
#                            semantic), see g1_arm_env.py's _get_observations comment.
# Roll/yaw joints flip sign under a left-right mirror, pitch joints (and the single-hinge
# elbow) don't — same convention as g1_locomotion/mdp/symmetry.py.
_PER_ARM_OBS_DIM = 39
_PER_ARM_ACTION_DIM = 7

_OBS_SIGN = torch.tensor(
    [
        1.0, -1.0, 1.0,        # base_lin_vel
        -1.0, 1.0, -1.0,       # base_ang_vel
        1.0, -1.0, 1.0,        # projected_gravity
        # joint_pos: shoulder_pitch, shoulder_roll*, shoulder_yaw*, elbow,
        #            wrist_roll*, wrist_pitch, wrist_yaw*
        1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        # joint_vel: same order/signs as joint_pos
        1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        1.0, -1.0, 1.0,        # ee_pos
        1.0, -1.0, 1.0,        # goal
        1.0, -1.0, 1.0,        # error
        # action_fb: same order/signs as joint_pos/joint_vel
        1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    ],
    dtype=torch.float32,
)
_ACTION_SIGN = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float32)

assert _OBS_SIGN.numel() == _PER_ARM_OBS_DIM
assert _ACTION_SIGN.numel() == _PER_ARM_ACTION_DIM


def mirror_arm_obs(obs: torch.Tensor) -> torch.Tensor:
    total_dim = obs.shape[-1]
    sign = _OBS_SIGN.to(obs.device)

    if total_dim == _PER_ARM_OBS_DIM:
        # Single-arm mode: pure sign flip, no block swap needed.
        return obs * sign

    if total_dim == 2 * _PER_ARM_OBS_DIM:
        # Both-arms mode: mirroring swaps which arm is "left" and which is "right", so
        # the two 32-D blocks swap position *and* each gets sign-flipped.
        left, right = obs[:, :_PER_ARM_OBS_DIM], obs[:, _PER_ARM_OBS_DIM:]
        return torch.cat([right * sign, left * sign], dim=-1)

    raise RuntimeError(
        f"G1 29dof arm symmetry: observation dim {total_dim} is neither one arm "
        f"({_PER_ARM_OBS_DIM}) nor both arms ({2 * _PER_ARM_OBS_DIM}) — check this "
        "wasn't applied to a task with a different observation layout than "
        "g1_arm_env.py currently defines."
    )


def mirror_arm_actions(actions: torch.Tensor) -> torch.Tensor:
    total_dim = actions.shape[-1]
    sign = _ACTION_SIGN.to(actions.device)

    if total_dim == _PER_ARM_ACTION_DIM:
        return actions * sign

    if total_dim == 2 * _PER_ARM_ACTION_DIM:
        left, right = actions[:, :_PER_ARM_ACTION_DIM], actions[:, _PER_ARM_ACTION_DIM:]
        return torch.cat([right * sign, left * sign], dim=-1)

    raise RuntimeError(
        f"G1 29dof arm symmetry: action dim {total_dim} is neither one arm "
        f"({_PER_ARM_ACTION_DIM}) nor both arms ({2 * _PER_ARM_ACTION_DIM})."
    )


@torch.no_grad()
def compute_symmetric_arm_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with their left-right mirror (2x batch).

    Same call signature as ``RslRlSymmetryCfg.data_augmentation_func`` expects.
    """
    del env  # unused — the mirror here is purely a fixed index/sign transform, no scene lookup

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = mirror_arm_obs(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = mirror_arm_actions(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
