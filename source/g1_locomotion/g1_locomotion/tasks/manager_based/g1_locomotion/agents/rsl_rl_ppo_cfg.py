# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
    G1RoughPPORunnerCfg,
)

from ..mdp.symmetry import compute_symmetric_states

def _walking_symmetry_cfg() -> RslRlSymmetryCfg:
    """Fresh RslRlSymmetryCfg per runner cfg instance (not a shared module-level object).

    Flat-terrain walking observation has no height_scan, so the left-right mirror transform in
    mdp/symmetry.py applies directly. Not used for rough terrain (height_scan needs its own mirror
    handling, not implemented) or standing (its arm-motion disturbance curriculum intentionally
    injects asymmetric motions sometimes — combining that with symmetry-augmented training needs
    separate, careful design, not a drop-in reuse of this).
    """
    return RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=compute_symmetric_states,
    )


@configclass
class G1LocomotionFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """RSL-RL PPO config for G1 flat-terrain locomotion.

    Inherits network sizes [256, 128, 128], 1500 iterations, adaptive LR from Isaac Lab.
    Override fields here to tune, e.g.:
        self.max_iterations = 2000

    Trains with left-right symmetry data augmentation (see mdp/symmetry.py) — added after an
    observed asymmetric gait (one leg lifting much higher than the other) in an unaugmented run;
    nothing in the reward function penalizes gait asymmetry, so plain PPO is free to settle into
    one. Mirroring every transition removes that degree of freedom structurally.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "legs/g1_locomotion_flat"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()


@configclass
class G1LocomotionRoughPPORunnerCfg(G1RoughPPORunnerCfg):
    """RSL-RL PPO config for G1 rough-terrain locomotion.

    Inherits network sizes [512, 256, 128], 3000 iterations, adaptive LR from Isaac Lab.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "legs/g1_locomotion_rough"


@configclass
class G1LocomotionStandingFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """RSL-RL PPO config for the standing-only flat terrain task."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "standing/g1_locomotion_flat"


@configclass
class G1LocomotionFlatTransitionPPORunnerCfg(G1FlatPPORunnerCfg):
    """RSL-RL PPO config for transition-heavy walking policy.

    Same left-right symmetry augmentation as ``G1LocomotionFlatPPORunnerCfg`` — see there for why.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "legs/g1_locomotion_flat_transition"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()


