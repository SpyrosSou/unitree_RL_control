# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the G1 locomotion/standing tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .events import _ARM_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["standing_arm_motion_targets"]


def standing_arm_motion_targets(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The arm's currently-commanded joint targets, as offsets from the default pose (10-D).

    The "arm-intent" observation (2026-07-15, plan.md §4.5 option 2): until now the standing
    policy only ever *felt* the arm after the fact — joint positions/velocities, i.e. the
    consequences of motion that already happened — and never saw where the arm is being told
    to go next. That forces purely reactive balancing, and the observed result is a single
    fixed bracing posture (the consolidated checkpoint's persistently rotated torso, which
    swallowed the near-inner reach workspace) and one-sided solutions (each policy-driven-
    trained run "solves" exactly one arm). Deployed humanoid stacks put the upper-body
    command in the locomotion policy's observation for exactly this reason: anticipation is
    cheaper than robustness. This term is the minimal version of that — the same
    ``env._standing_arm_motion_targets`` buffer that already drives the simulated arm
    targets everywhere (training's disturbance classes, eval_full_demo.py's buckets,
    g1_full_demo.py), exposed as an observation.

    Encoding: ``targets - default_joint_pos``, in the fixed canonical joint order
    ``_ARM_JOINT_NAMES`` (left arm 5, then right arm 5) — zero when the arms are just being
    held at default, so "no arm activity" reads as a clean null signal. The env attribute's
    own column order is *not* assumed: deployment sets ``_standing_arm_motion_joint_ids`` in
    whatever order suits it (eval_full_demo.py's reach buckets put the active side first),
    so columns are permuted into the canonical order by joint id on every call (permutation
    cached until the id tensor changes).

    Falls back to zeros when the buffers don't exist — which is both the pre-init case
    (ObservationManager probes terms for their dims before the event manager may have
    created the buffers) and the walking case (g1_full_demo.py clears the attributes to
    None outside standing), matching training's "arms at default → zero offset" null.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.data.default_joint_pos.device

    cache = getattr(env, "_arm_intent_obs_cache", None)
    if cache is None:
        canonical_ids, _ = asset.find_joints(_ARM_JOINT_NAMES)
        cache = {
            "canonical": torch.tensor(canonical_ids, dtype=torch.long, device=device),
            "src_ids": None,
            "perm": None,
        }
        env._arm_intent_obs_cache = cache
    canonical = cache["canonical"]

    targets = getattr(env, "_standing_arm_motion_targets", None)
    src_ids = getattr(env, "_standing_arm_motion_joint_ids", None)
    if targets is None or src_ids is None:
        return torch.zeros(asset.num_instances, canonical.numel(), device=device)

    if cache["perm"] is None or not torch.equal(cache["src_ids"], src_ids):
        id_to_col = {int(j): i for i, j in enumerate(src_ids.tolist())}
        cache["perm"] = torch.tensor(
            [id_to_col[int(j)] for j in canonical.tolist()], dtype=torch.long, device=device
        )
        cache["src_ids"] = src_ids.clone()

    return targets[:, cache["perm"]] - asset.data.default_joint_pos[:, canonical]
