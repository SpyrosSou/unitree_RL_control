# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the G1 arm-IK observation and action space.

Used via ``RslRlSymmetryCfg.data_augmentation_func`` (see ``agents/rsl_rl_ppo_cfg.py``) so the
left-right mirror relationship ``testing/arm_testing/g1_arm_mirror_test.py`` already relies on at
deployment time (to run a left-trained policy on the right arm, or vice versa) is something the
policy was actually trained to satisfy, rather than an assumption that happens to look right
because the workspace bounds are symmetric. Every training experience is mirrored and trained on
alongside the original, so ``policy(mirror(obs)) == mirror(policy(obs))`` becomes a real, trained
property instead of an untested one.

Unlike ``g1_locomotion/mdp/symmetry.py`` (which builds its joint swap/sign maps dynamically from
the robot's full joint list, since walking's whole-body layout varies by task), this task's
observation/action layout is small and fixed by construction (see ``g1_arm_env.py``'s module
docstring), so the mirror maps here are simply hardcoded — same approach already used and
validated in ``g1_arm_mirror_test.py``'s ``_OBS_FLIP_IDX``/``_ACT_FLIP_IDX`` (this module's index
math is that same mapping, just re-expressed as dense per-dimension sign vectors covering the new,
larger per-arm block introduced in Phase 2 — the arm mirror test's own indices need updating to
match when this lands, per known_issues.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_arm_states", "mirror_arm_obs", "mirror_arm_actions"]

# Per-arm observation block (28-D, see g1_arm_env.py's module docstring):
#   [0:3]   base_lin_vel        — lateral (y) component flips
#   [3:6]   base_ang_vel        — roll-rate (x) and yaw-rate (z) flip, pitch-rate (y) doesn't
#   [6:9]   projected_gravity   — lateral (y) component flips
#   [9:14]  joint_pos  (5): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll
#   [14:19] joint_vel  (5): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, elbow_roll
#   [19:22] ee_pos              — y flips
#   [22:25] goal                — y flips
#   [25:28] error               — y flips (= goal - ee_pos, consistent with both flipping)
# Roll/yaw joints flip sign under a left-right mirror, pitch joints don't — same convention as
# g1_locomotion/mdp/symmetry.py and the arm mirror test.
#
# joint_vel was only 3-D (shoulder joints only) until 2026-07-08 — g1_arm_env.py's
# observation builder sliced it to jt[:3], silently dropping elbow_pitch/elbow_roll
# velocity. Fixed alongside that bug fix; see known_issues.md.
_PER_ARM_OBS_DIM = 28
_PER_ARM_ACTION_DIM = 5

_OBS_SIGN = torch.tensor(
    [
        1.0, -1.0, 1.0,        # base_lin_vel
        -1.0, 1.0, -1.0,       # base_ang_vel
        1.0, -1.0, 1.0,        # projected_gravity
        1.0, -1.0, -1.0, 1.0, -1.0,  # joint_pos: pitch, roll*, yaw*, pitch, roll*
        1.0, -1.0, -1.0, 1.0, -1.0,  # joint_vel: pitch, roll*, yaw*, pitch, roll*
        1.0, -1.0, 1.0,        # ee_pos
        1.0, -1.0, 1.0,        # goal
        1.0, -1.0, 1.0,        # error
    ],
    dtype=torch.float32,
)
_ACTION_SIGN = torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0], dtype=torch.float32)


def mirror_arm_obs(obs: torch.Tensor) -> torch.Tensor:
    total_dim = obs.shape[-1]
    sign = _OBS_SIGN.to(obs.device)

    if total_dim == _PER_ARM_OBS_DIM:
        # Single-arm mode: pure sign flip, no block swap needed (mirrors the arm's own
        # workspace onto itself — see module docstring for why this is still meaningful
        # training signal, not a no-op).
        return obs * sign

    if total_dim == 2 * _PER_ARM_OBS_DIM:
        # Both-arms mode: mirroring swaps which arm is "left" and which is "right", so the
        # two 26-D blocks swap position *and* each gets sign-flipped.
        left, right = obs[:, :_PER_ARM_OBS_DIM], obs[:, _PER_ARM_OBS_DIM:]
        return torch.cat([right * sign, left * sign], dim=-1)

    raise RuntimeError(
        f"G1 arm symmetry: observation dim {total_dim} is neither one arm ({_PER_ARM_OBS_DIM}) "
        f"nor both arms ({2 * _PER_ARM_OBS_DIM}) — check this wasn't applied to a task with a "
        "different observation layout than g1_arm_env.py currently defines."
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
        f"G1 arm symmetry: action dim {total_dim} is neither one arm ({_PER_ARM_ACTION_DIM}) "
        f"nor both arms ({2 * _PER_ARM_ACTION_DIM})."
    )


@torch.no_grad()
def compute_symmetric_arm_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with their left-right mirror (2x batch).

    Same call signature as ``RslRlSymmetryCfg.data_augmentation_func`` expects (and the same
    pattern as ``g1_locomotion/mdp/symmetry.py``'s ``compute_symmetric_states``), just for the
    arm task's fixed, smaller observation/action layout.
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
