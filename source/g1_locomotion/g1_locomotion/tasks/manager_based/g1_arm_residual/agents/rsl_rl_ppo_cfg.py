# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the G1 29dof arm residual-RL task.

Adapted from ``g1_arm/agents/rsl_rl_ppo_cfg.py``'s ``G1ArmPPORunnerCfg`` — same
network size/PPO hyperparameters (no reason to assume the residual task needs a
different network shape or algorithm tuning; revisit with real training data, not a
guess). The one deliberate difference: symmetry data augmentation is disabled — see
``g1_arm_residual_env.py``'s module docstring for why (``mdp/symmetry.py``'s
``mirror_arm_obs`` hard-codes the base task's observation layouts and doesn't know
about this task's extra 7-dim baseline-error block).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1ArmResidualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Shared RSL-RL PPO config for the residual-RL arm task (left/right vary only in
    ``experiment_name`` — see the public subclasses below)."""

    num_steps_per_env = 24
    max_iterations = 3000  # narrower task than full reaching (small bounded residual, not a from-scratch reach) -- revisit with real training-curve data, not a guess
    save_interval = 100

    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        # No symmetry_cfg set (defaults to None) -- see module docstring.


@configclass
class G1ArmResidualLeftPPORunnerCfg(G1ArmResidualPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/residual_left"


@configclass
class G1ArmResidualRightPPORunnerCfg(G1ArmResidualPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/residual_right"
