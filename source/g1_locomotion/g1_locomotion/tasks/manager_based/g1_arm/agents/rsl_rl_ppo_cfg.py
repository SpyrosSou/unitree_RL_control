# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1ArmIKPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO config for the G1 arm IK reaching task.

    A single runner config is shared for left/right arm; only ``experiment_name``
    differs in the two public subclasses below.
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100

    # Map the env's single "policy" observation group to both actor and critic
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
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


@configclass
class G1ArmIKLeftPPORunnerCfg(G1ArmIKPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left"


@configclass
class G1ArmIKRightPPORunnerCfg(G1ArmIKPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_right"


@configclass
class G1ArmIKBothPPORunnerCfg(G1ArmIKPPORunnerCfg):
    """Both arms: 34-D obs, 10-D actions. Needs training from scratch."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_both"
        # Slightly wider network to handle the larger input/output
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
