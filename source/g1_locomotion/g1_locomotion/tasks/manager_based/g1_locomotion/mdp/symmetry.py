# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for the G1 29dof unified stand+walk observation and
action space.

Adapted 2026-07-21 from ``g1_locomotion/tasks/manager_based/g1_locomotion/mdp/symmetry.py``
(the 23dof-era version) — see ``29dof_implementation_plan.md``. The joint-mirror-map logic
(swap-index/sign built dynamically from ``robot.data.joint_names`` rather than a hardcoded
list, and ``_mirror_sign()``'s substring match) needed **no change** — it already correctly
classifies the new 3-DOF waist (``waist_roll_joint``/``waist_yaw_joint`` flip,
``waist_pitch_joint`` doesn't) with no special case. The old single-DOF ``torso_joint`` name
check is removed as dead code.

**What DID need to change**: velocity_env_cfg.py's ``ObservationsCfg.PolicyCfg`` sets
``history_length = 5`` on the whole group — every term (not just this task's raw per-step
reading) is stacked over the last 5 control steps and flattened before concatenation (see
Isaac Lab's ``ObservationManager.compute_group``: each term gets its own ``(num_envs,
history_length, term_dim)`` circular buffer, flattened to ``(num_envs, history_length *
term_dim)`` in oldest-to-newest — or newest-to-oldest, doesn't matter for this transform —
row-major order, then all terms concatenate in declaration order). The 23dof-era version of
this file assumed single-step terms (a flat 9-D-prefix + 3 equal N-joint blocks); this
version instead treats each term as its own ``history_length``-repeated block and reshapes
to (env, H, D) to apply the same per-step channel transform to every history slice, since
the left-right symmetry itself is time-invariant (a joint's mirror relationship doesn't
change between now and 5 steps ago). ``_HISTORY_LENGTH`` below must stay in sync with
``ObservationsCfg.PolicyCfg.__post_init__``'s ``self.history_length`` — there is no way to
read it back from the flat observation tensor alone.

Unlike ANYmal (a quadruped, with left-right *and* front-back symmetry), G1 is a biped: the
only rigid-body symmetry available is left-right (mirroring through the sagittal plane), so
this produces a 2x data augmentation (original + mirrored), not ANYmal's 4x.

Used via ``RslRlSymmetryCfg.data_augmentation_func`` (see ``agents/rsl_rl_ppo_cfg.py``) to
prevent PPO settling into a lazy, asymmetric gait — nothing in the reward function cares
which foot swings or how high, only that steps alternate with roughly-correct timing, and a
real training run on this exact codebase (23dof phase, ``phase_logs/phase_1.md``) produced
a visibly lopsided gait before this fix was added.

Joint mirror-sign convention (which joint negates under a left-right mirror): roll- and
yaw-axis joints flip sign, pitch-axis joints don't — same convention the arm task's own
mirror code uses (``g1_arm/mdp/symmetry.py``).

**Not yet verified against a live env** — the block-size math (below) is derived from
reading ``observation_manager.py`` and ``velocity_env_cfg.py`` directly, not by running
training and inspecting a real observation tensor. Sanity-check ``total_dim`` against the
actual policy input width the very first time this runs (the ``RuntimeError`` in
``_transform_policy_obs_left_right`` will fire immediately if the assumed decomposition is
wrong, rather than silently producing a garbled mirror).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Must match velocity_env_cfg.py's ObservationsCfg.PolicyCfg.__post_init__
# (self.history_length = 5).
_HISTORY_LENGTH = 5

# Per-step term order, in declaration order (see velocity_env_cfg.py's
# ObservationsCfg.PolicyCfg): base_ang_vel(3), projected_gravity(3), velocity_commands(3)
# — then joint_pos(N), joint_vel(N), last_action(N). Unlike the 23dof-era policy
# observation, there is no base_lin_vel term here (Unitree's own recipe deliberately omits
# it from the policy group, only the critic sees it — see 29dof_implementation_plan.md's
# sim2real notes), so the fixed-size (pre-history) prefix is 9-D, not 12-D.
_PREFIX_LEN = 9


def _mirror_sign(joint_name: str) -> float:
    """+1 if this joint's value is unchanged under a left-right mirror, -1 if it flips.

    Roll and yaw axes flip sign under mirroring (their rotation sense reverses); pitch axes
    don't. Every joint in the 29dof G1 (legs, 3-DOF waist, 7-DOF arms with wrists) fits this
    substring-based classification with no special case needed — verified by construction
    from the joint names (``*_hip_roll_*``/``*_hip_yaw_*``, ``waist_roll_joint``/
    ``waist_yaw_joint``, ``*_shoulder_roll_*``/``*_shoulder_yaw_*``, ``*_wrist_roll_*``/
    ``*_wrist_yaw_*`` all flip; pitch axes and ``*_knee_*``/``*_elbow_*`` don't).
    """
    if "_roll_" in joint_name or "_yaw_" in joint_name:
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
    cached = getattr(robot, "_g1_29dof_symmetry_lr_maps", None)
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
                f"G1 29dof symmetry: joint '{name}' has no mirror partner '{partner}' in the "
                "robot's joint list. The left/right naming convention this transform relies on "
                "doesn't match this asset — check joint names before using symmetry-augmented "
                "training."
            )
        swap_index[i] = name_to_idx[partner]
        sign[i] = _mirror_sign(name)

    device = robot.data.default_joint_pos.device
    maps = (swap_index.to(device), sign.to(device).view(1, -1))
    robot._g1_29dof_symmetry_lr_maps = maps
    return maps


def _switch_g1_joints_left_right(
    joint_data: torch.Tensor, swap_index: torch.Tensor, sign: torch.Tensor
) -> torch.Tensor:
    return joint_data[:, swap_index] * sign


def _mirror_history_block(
    obs: torch.Tensor, block: slice, num_envs: int, term_dim: int, transform
) -> None:
    """Apply a per-step channel transform to a ``history_length``-flattened term block,
    in place. ``obs[:, block]`` has shape ``(num_envs, _HISTORY_LENGTH * term_dim)``, row-
    major (history, then channel) per Isaac Lab's ``CircularBuffer.buffer.reshape(num_envs,
    -1)`` — reshaping to ``(num_envs, _HISTORY_LENGTH, term_dim)`` recovers per-step
    channels, ``transform`` (a function ``(N, term_dim) -> (N, term_dim)``) applies
    identically to every history slice (the mirror relationship is time-invariant), and the
    result is flattened back. ``transform`` is called on a 2-D view with the history
    dimension folded into the batch dimension so it can reuse ``_switch_g1_joints_left_right``
    /simple sign-multiply unchanged.
    """
    chunk = obs[:, block].reshape(num_envs * _HISTORY_LENGTH, term_dim)
    obs[:, block] = transform(chunk).reshape(num_envs, _HISTORY_LENGTH * term_dim)


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    num_envs = obs.shape[0]

    total_dim = obs.shape[-1]
    if total_dim % _HISTORY_LENGTH != 0:
        raise RuntimeError(
            f"G1 29dof symmetry: observation dim {total_dim} isn't a multiple of "
            f"_HISTORY_LENGTH ({_HISTORY_LENGTH}) — check _HISTORY_LENGTH still matches "
            "ObservationsCfg.PolicyCfg.__post_init__'s self.history_length."
        )
    per_step_dim = total_dim // _HISTORY_LENGTH
    n_joints = (per_step_dim - _PREFIX_LEN) // 3
    if _PREFIX_LEN + 3 * n_joints != per_step_dim:
        raise RuntimeError(
            f"G1 29dof symmetry: per-step observation dim {per_step_dim} (total {total_dim} "
            f"/ history {_HISTORY_LENGTH}) doesn't decompose into a {_PREFIX_LEN}-dim prefix "
            "+ 3 equal joint blocks (joint_pos, joint_vel, last_action). This transform "
            "assumes velocity_env_cfg.py's PolicyCfg layout — check this wasn't applied to a "
            "different observation cfg, or that _HISTORY_LENGTH is still correct."
        )

    robot = env.unwrapped.scene["robot"]
    swap_index, sign = _get_or_build_joint_mirror_maps(robot)
    sign_flat = sign.view(-1)  # (n_joints,) — sign was cached as (1, n_joints) for the non-history path

    ang_vel_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)
    gravity_sign = torch.tensor([1.0, -1.0, 1.0], device=device)
    cmd_sign = torch.tensor([1.0, -1.0, -1.0], device=device)

    b = _HISTORY_LENGTH
    # Block boundaries in the flattened (history-major) layout: each per-step term occupies
    # _HISTORY_LENGTH * term_dim contiguous columns, in PolicyCfg's declaration order.
    ang_vel_block = slice(0, b * 3)
    gravity_block = slice(b * 3, b * 6)
    cmd_block = slice(b * 6, b * 9)
    jpos_block = slice(b * 9, b * 9 + b * n_joints)
    jvel_block = slice(jpos_block.stop, jpos_block.stop + b * n_joints)
    action_block = slice(jvel_block.stop, jvel_block.stop + b * n_joints)
    assert action_block.stop == total_dim

    _mirror_history_block(obs, ang_vel_block, num_envs, 3, lambda x: x * ang_vel_sign)
    _mirror_history_block(obs, gravity_block, num_envs, 3, lambda x: x * gravity_sign)
    _mirror_history_block(obs, cmd_block, num_envs, 3, lambda x: x * cmd_sign)
    _mirror_history_block(
        obs, jpos_block, num_envs, n_joints, lambda x: _switch_g1_joints_left_right(x, swap_index, sign_flat)
    )
    _mirror_history_block(
        obs, jvel_block, num_envs, n_joints, lambda x: _switch_g1_joints_left_right(x, swap_index, sign_flat)
    )
    _mirror_history_block(
        obs, action_block, num_envs, n_joints, lambda x: _switch_g1_joints_left_right(x, swap_index, sign_flat)
    )

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
