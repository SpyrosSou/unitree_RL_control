# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ..mdp.symmetry import compute_symmetric_arm_states


def _arm_symmetry_cfg() -> RslRlSymmetryCfg:
    """Fresh RslRlSymmetryCfg per runner cfg instance (not a shared module-level object).

    See ``mdp/symmetry.py`` for why this replaces the assumption
    ``testing/arm_testing/g1_arm_mirror_test.py``'s runtime mirror relies on with a trained-for
    property instead (Phase 2, finding #7 in the roadmap plan).
    """
    return RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func=compute_symmetric_arm_states,
    )


@configclass
class G1ArmIKPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO config for the G1 arm IK reaching task.

    A single runner config is shared for left/right arm; only ``experiment_name``
    differs in the two public subclasses below.

    Trains with left-right symmetry data augmentation (see ``mdp/symmetry.py``) so the
    left/right mirror deployment path is a trained-for property, not an assumption.
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
        # Tried 0.005 (2026-07-07, alongside the reward-shaping change reverted in
        # g1_arm_env.py — see known_issues.md) to keep exploration alive longer against
        # an apparent early plateau. That retrain regressed badly (success rate declined
        # over the run instead of improving), and since both changes were bundled into
        # the same retrain there's no way to tell whether entropy or the reward term was
        # responsible. Reverted to 0.0 (the confirmed-good baseline) — if revisited,
        # change this alone, not together with other reward/hyperparameter changes.
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
class G1ArmIKLeftPPORunnerCfg(G1ArmIKPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left"


# ---------------------------------------------------------------------------
# Overnight sweep (2026-07-07): 3 isolated arm-policy experiments vs. the baseline
# above, each changing exactly one thing. Distinct experiment_name per variant so
# each lands in its own logs/rsl_rl/arms/<name>/ subfolder — identifiable by path
# alone, not by timestamp. See known_issues.md for the plateau this targets and
# g1_arm_env.py for the matching env cfg variant (where one exists).
# ---------------------------------------------------------------------------


@configclass
class G1ArmIKLeftRewardShapePPORunnerCfg(G1ArmIKPPORunnerCfg):
    """Experiment 1/4: exponential proximity bonus (env cfg change only, see g1_arm_env.py).

    entropy_coef and network size stay at baseline — only pair with
    G1ArmIKLeftRewardShapeEnvCfg.
    """
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left_reward_shape"


@configclass
class G1ArmIKLeftGoalCurriculumPPORunnerCfg(G1ArmIKPPORunnerCfg):
    """Experiment 2/4: goal-difficulty curriculum (env cfg change only, see g1_arm_env.py).

    entropy_coef and network size stay at baseline — only pair with
    G1ArmIKLeftGoalCurriculumEnvCfg.
    """
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left_goal_curriculum"


@configclass
class G1ArmIKLeftStressRegionPPORunnerCfg(G1ArmIKPPORunnerCfg):
    """1b: isolated diagnostic on the elbow-extension stress region (env cfg change
    only, see g1_arm_env.py). Only pair with G1ArmIKLeftStressRegionEnvCfg.
    """
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left_stress_region"


@configclass
class G1ArmIKLeftWideNetPPORunnerCfg(G1ArmIKPPORunnerCfg):
    """Experiment 3/4: wider actor/critic network, isolated (pairs with the baseline env
    cfg — no env changes here, just more capacity to see if the plateau is a capacity
    limit rather than an exploration one).
    """
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left_wide_net"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class G1ArmIKLeftEntropyPPORunnerCfg(G1ArmIKPPORunnerCfg):
    """Experiment 4/4: higher entropy_coef, isolated (pairs with the baseline env cfg).

    Targets the leading plateau theory directly: entropy_coef=0.0 gives no exploration
    incentive, consistent with the observed early-and-flat training curve (see
    known_issues.md). A prior attempt at entropy_coef=0.005 was bundled with the reward-
    shaping change and regressed, but the two were never isolated from each other — this
    is the first time entropy alone is tested. 0.01 (not 0.005) to give this a clearly
    differentiated, less ambiguous test against that prior attempt.
    """
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left_entropy"
        self.algorithm.entropy_coef = 0.01


@configclass
class G1ArmIKLeftPPOTuningPPORunnerCfg(G1ArmIKPPORunnerCfg):
    """PPO hyperparameter tuning (2026-07-08, isolated, pairs with the baseline env cfg —
    no env changes here). Lowest-confidence of the three remaining experiments — no prior
    evidence points at this specifically, unlike the other two (entropy has a direct
    theory, wide-net has capacity reasoning); this is exploratory.

    num_learning_epochs 5->8: more thorough optimization per collected rollout. Given the
    training curve consistently shows fast early convergence into a hard plateau, this
    tests whether the plateau is partly an under-optimization artifact within each PPO
    update (not extracting enough signal from each rollout before moving on) rather than
    purely an exploration (entropy) or capacity (network size) limit — orthogonal to both
    of the other two experiments, safe to combine results with if it helps.
    """
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/g1_arm_ik_left_ppo_tuning"
        self.algorithm.num_learning_epochs = 8


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
