# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``BasePPORunnerCfg`` is ported verbatim from ``unitree_rl_lab``, 2026-07-21 — see
``29dof_implementation_plan.md``. ``G1LocomotionArmDisturbancePPORunnerCfg`` adds this
project's own left-right symmetry data augmentation (``mdp/symmetry.py``) for the base
(no-disturbance) recipe only — deliberately NOT wired into the arm-disturbance variant,
matching the 23dof-era ``g1_locomotion`` task's own precedent (``_walking_symmetry_cfg``'s
docstring, preserved on the ``23_dof`` branch): the arm-motion curriculum intentionally
injects asymmetric motions sometimes, and combining that with symmetry-augmented
training needs separate, careful design rather than a drop-in reuse — not verified
either way for this task, so following the established, already-reasoned-through
precedent rather than silently assuming it's fine.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ..mdp.symmetry import compute_symmetric_states


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _walking_symmetry_cfg() -> RslRlSymmetryCfg:
    """Fresh RslRlSymmetryCfg per runner cfg instance (not a shared module-level object) —
    same reasoning as the 23dof-era task's identical helper."""
    return RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=compute_symmetric_states,
    )


@configclass
class G1LocomotionPPORunnerCfg(BasePPORunnerCfg):
    """``BasePPORunnerCfg`` (Unitree's own recipe) plus left-right symmetry data
    augmentation — for the no-arm-disturbance base task (``G1LocomotionEnvCfg``),
    matching the 23dof-era ``G1LocomotionFlatPPORunnerCfg``'s precedent (added there
    after an observed asymmetric gait — see that class's docstring)."""

    def __post_init__(self):
        super().__post_init__()
        # "walking/..." not the package name (2026-07-21, user request) — keeps this
        # task's logs grouped under logs/rsl_rl/walking/, parallel to logs/rsl_rl/arms/.
        self.experiment_name = "walking/base"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()


@configclass
class G1LocomotionArmDisturbancePPORunnerCfg(BasePPORunnerCfg):
    """``BasePPORunnerCfg`` for the arm-motion-disturbance variant
    (``G1LocomotionArmDisturbanceEnvCfg``) — deliberately NO symmetry augmentation wired
    in, see module docstring."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walking/arm_disturbance"


# 2026-07-23 walking-fix ablation runner cfgs — see the matching Ablation*EnvCfg classes
# in g1_locomotion_env_cfg.py for what each one changes. Distinct experiment_name per
# variant keeps their logs separated under logs/rsl_rl/walking/ablation_*/ for direct
# comparison. Same symmetry augmentation as G1LocomotionPPORunnerCfg (all on the base,
# no-disturbance task).
@configclass
class G1LocomotionAblationTermPenaltyPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walking/ablation_term_penalty"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()


@configclass
class G1LocomotionAblationCurriculumPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walking/ablation_curriculum"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()


@configclass
class G1LocomotionAblationRewardWeightsPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walking/ablation_reward_weights"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()


@configclass
class G1LocomotionAblationAllPPORunnerCfg(BasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "walking/ablation_all"
        self.algorithm.symmetry_cfg = _walking_symmetry_cfg()
