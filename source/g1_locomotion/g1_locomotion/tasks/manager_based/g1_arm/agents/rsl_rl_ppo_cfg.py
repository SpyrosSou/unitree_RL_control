# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configs for the G1 29dof arm policy.

Adapted 2026-07-21 from the 23dof-era ``g1_arm/agents/rsl_rl_ppo_cfg.py`` — see
``g1_arm_env.py``'s module docstring for the full rationale on what changed vs. carried
forward. Network size ([512,256,128]) and symmetry augmentation are the DEFAULT here
(not opt-in experimental variants the way the 23dof-era task first discovered them),
since both were confirmed-good findings from that phase, not open questions.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ..mdp.symmetry import compute_symmetric_arm_states


def _arm_symmetry_cfg() -> RslRlSymmetryCfg:
    """Fresh RslRlSymmetryCfg per runner cfg instance (not a shared module-level object)."""
    return RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=compute_symmetric_arm_states,
    )


@configclass
class G1ArmPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO config for the G1 29dof arm-reaching task.

    A single runner config is shared for left/right/both arm; only ``experiment_name``
    differs in the public subclasses below. Trains with left-right symmetry data
    augmentation (see ``mdp/symmetry.py``) from the start.
    """

    num_steps_per_env = 24
    max_iterations = 6000  # 2026-07-21, per user: match the walking overnight budget
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
        # REVERTED 2026-07-24 back to 0.0 (the original) — briefly bumped to 0.01
        # (matching walking) alongside a position_reward_exp_scale change (see
        # g1_arm_env.py) as a combined guess to fix a plateaued training run. That
        # combination caused Policy/mean_noise_std to explode (1.0 -> 57 over 2000
        # iterations) and Train/mean_reward to get steadily worse. Superseded by
        # G1ArmLeftAblationEntropyCoefPPORunnerCfg below, which tests this exact value
        # (0.01) in isolation, position_reward_exp_scale left at baseline 0.0.
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
        self.algorithm.symmetry_cfg = _arm_symmetry_cfg()


@configclass
class G1ArmLeftPPORunnerCfg(G1ArmPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # "arms/..." not the package name — parallel to "walking/..." (see the
        # locomotion task's identical rename, 2026-07-21 user request), so
        # logs/rsl_rl/ branches by mode (walking/ vs arms/), not by package name.
        self.experiment_name = "arms/left"


@configclass
class G1ArmLeftLockedWristPPORunnerCfg(G1ArmPPORunnerCfg):
    """Same as G1ArmLeftPPORunnerCfg, except symmetry augmentation is disabled.

    Found via a live sanity-check run (2026-07-24) that mdp/symmetry.py's
    mirror_arm_obs hard-codes an assumption that observations are 32-dim (one arm)
    or 64-dim (both) — it doesn't know about this variant's 28-dim (5 controlled
    joints, not 7) layout, and throws a RuntimeError rather than silently producing
    wrong mirrored data (a safe failure, not a bug in the check itself). Properly
    extending the mirroring math for the reduced joint set is real work not done
    under the time pressure of getting tonight's run started safely — disabling
    augmentation for this variant is the correct tradeoff here (a real but
    non-critical sample-efficiency cost) versus risking a rushed, wrong fix to the
    mirror math that wouldn't loudly fail the way this did.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/left_locked_wrist"
        self.algorithm.symmetry_cfg = None


@configclass
class G1ArmLeftAblationEntropyCoefPPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-24 ablation: entropy_coef=0.01 in isolation (position_reward_exp_scale
    stays at baseline 0.0 — see G1ArmLeftAblationExpScaleEnvCfg for that one tested
    alone). See the algorithm cfg's own comment above for the full rationale."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_entropy_coef"
        self.algorithm.entropy_coef = 0.01


@configclass
class G1ArmLeftAblationExpScalePPORunnerCfg(G1ArmPPORunnerCfg):
    """Runner cfg for the exp_scale ablation — same as G1ArmLeftPPORunnerCfg except
    experiment_name, so its logs land separately. entropy_coef stays at baseline 0.0;
    the actual change (position_reward_exp_scale) lives in the env cfg
    (G1ArmLeftAblationExpScaleEnvCfg), not here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_exp_scale"


@configclass
class G1ArmRightPPORunnerCfg(G1ArmPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/right"


@configclass
class G1ArmBothPPORunnerCfg(G1ArmPPORunnerCfg):
    """Both arms: 64-D obs, 14-D actions."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/both"
