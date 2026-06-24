# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
    G1RoughPPORunnerCfg,
)


@configclass
class G1LocomotionFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """RSL-RL PPO config for G1 flat-terrain locomotion.

    Inherits network sizes [256, 128, 128], 1500 iterations, adaptive LR from Isaac Lab.
    Override fields here to tune, e.g.:
        self.max_iterations = 2000
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "legs/g1_locomotion_flat"


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
    """RSL-RL PPO config for transition-heavy walking policy."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "legs/g1_locomotion_flat_transition"


