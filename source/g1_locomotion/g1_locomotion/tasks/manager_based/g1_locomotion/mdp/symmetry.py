# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the G1 walking/standing observation and action space.

Unlike ANYmal (a quadruped, with left-right *and* front-back symmetry), G1 is a biped: the
only rigid-body symmetry available is left-right (mirroring through the sagittal plane), so
this produces a 2x data augmentation (original + mirrored), not ANYmal's 4x.

Used via ``RslRlSymmetryCfg.data_augmentation_func`` (see ``agents/rsl_rl_ppo_cfg.py``) to fix
a real, observed failure mode: nothing in the reward function cares which foot swings or how
high, only that steps alternate with roughly-correct timing — so plain PPO training is free to
(and did, in practice) settle into a lazy, asymmetric gait where one leg does a stereotyped high
lift and the other barely lifts at all. Mirroring every transition and training on both versions
structurally prevents that, since the same network now has to explain both the original and the
mirrored trajectory equally well.

Joint mirror-sign convention (which joint negates under a left-right mirror) matches the one
already validated in ``arm_testing/g1_arm_mirror_test.py`` for the arms — roll- and yaw-axis
joints flip sign, pitch-axis joints don't — extended here to the whole body (legs + torso).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Fixed-size observation prefix, in declaration order (see the base velocity ObservationsCfg.PolicyCfg):
# base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3) — then joint_pos(N),
# joint_vel(N), actions(N). These four sign patterns are generic rigid-body-mirroring formulas
# (same math ANYmal's symmetry module uses), not G1-specific.
_PREFIX_LEN = 12


def _mirror_sign(joint_name: str) -> float:
    """+1 if this joint's value is unchanged under a left-right mirror, -1 if it flips.

    Roll and yaw axes flip sign under mirroring (their rotation sense reverses); pitch axes
    don't. ``torso_joint`` (no left/right pair, single waist joint) is assumed to be a yaw
    axis — the common choice for a single-DOF humanoid waist joint — so it self-mirrors with
    a sign flip. Anything else not matching one of these patterns (e.g. finger joints) is
    treated as sign-preserving, since their exact axis doesn't materially affect gait and
    getting it wrong there is low-risk compared to getting a leg joint wrong.
    """
    if "_roll_" in joint_name or "_yaw_" in joint_name or joint_name == "torso_joint":
        return -1.0
    return 1.0


def _mirror_partner_name(joint_name: str) -> str:
    """The other side's joint name, or the same name if this joint has no left/right pair."""
    if joint_name.startswith("left_"):
        return joint_name.replace("left_", "right_", 1)
    if joint_name.startswith("right_"):
        return joint_name.replace("right_", "left_", 1)
    return joint_name


def _get_or_build_joint_mirror_maps(robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (and cache on the robot object) the swap-index and sign tensors for joint-shaped
    observation/action blocks (joint_pos, joint_vel, actions — all length N, same joint order).

    Cached rather than rebuilt every call since this runs every PPO mini-batch update.
    """
    cached = getattr(robot, "_g1_symmetry_lr_maps", None)
    if cached is not None:
        return cached

    joint_names = list(robot.data.joint_names)
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    swap_index = torch.zeros(len(joint_names), dtype=torch.long)
    sign = torch.zeros(len(joint_names), dtype=torch.float32)
    for i, name in enumerate(joint_names):
        partner = _mirror_partner_name(name)
        if partner not in name_to_idx:
            raise RuntimeError(
                f"G1 symmetry: joint '{name}' has no mirror partner '{partner}' in the robot's "
                "joint list. The left/right naming convention this transform relies on doesn't "
                "match this asset — check joint names before using symmetry-augmented training."
            )
        swap_index[i] = name_to_idx[partner]
        sign[i] = _mirror_sign(name)

    device = robot.data.default_joint_pos.device
    maps = (swap_index.to(device), sign.to(device).view(1, -1))
    robot._g1_symmetry_lr_maps = maps
    return maps


def _switch_g1_joints_left_right(
    joint_data: torch.Tensor, swap_index: torch.Tensor, sign: torch.Tensor
) -> torch.Tensor:
    return joint_data[:, swap_index] * sign


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device

    total_dim = obs.shape[-1]
    n_joints = (total_dim - _PREFIX_LEN) // 3
    if _PREFIX_LEN + 3 * n_joints != total_dim:
        raise RuntimeError(
            f"G1 symmetry: observation dim {total_dim} doesn't decompose into a {_PREFIX_LEN}-dim "
            "prefix + 3 equal joint blocks (joint_pos, joint_vel, actions). This transform assumes "
            "the flat-terrain policy observation layout (no height_scan) — check it wasn't applied "
            "to a rough-terrain/height-scan task by mistake."
        )

    robot = env.unwrapped.scene["robot"]
    swap_index, sign = _get_or_build_joint_mirror_maps(robot)

    # base_lin_vel: lateral (y) component flips.
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0], device=device)
    # base_ang_vel: roll-rate (x) and yaw-rate (z) flip, pitch-rate (y) doesn't.
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    # projected_gravity: lateral (y) component flips.
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1.0, -1.0, 1.0], device=device)
    # velocity_commands [vx, vy, wz]: lateral velocity and yaw-rate command flip.
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1.0, -1.0, -1.0], device=device)

    j0, j1 = _PREFIX_LEN, _PREFIX_LEN + n_joints
    obs[:, j0:j1] = _switch_g1_joints_left_right(obs[:, j0:j1], swap_index, sign)  # joint_pos
    obs[:, j1 : j1 + n_joints] = _switch_g1_joints_left_right(obs[:, j1 : j1 + n_joints], swap_index, sign)  # joint_vel
    obs[:, j1 + n_joints : j1 + 2 * n_joints] = _switch_g1_joints_left_right(
        obs[:, j1 + n_joints : j1 + 2 * n_joints], swap_index, sign
    )  # last actions

    return obs


def _transform_actions_left_right(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    swap_index, sign = _get_or_build_joint_mirror_maps(robot)
    actions = actions.clone()
    actions[:] = _switch_g1_joints_left_right(actions[:], swap_index, sign)
    return actions


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment observations/actions with their left-right mirror (2x batch, biped-only symmetry).

    Same call signature as ``RslRlSymmetryCfg.data_augmentation_func`` expects (and the same
    pattern as Isaac Lab's ``velocity/mdp/symmetry/anymal.py``), just for one mirror axis
    instead of four.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(env.unwrapped, obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _transform_actions_left_right(env, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug
