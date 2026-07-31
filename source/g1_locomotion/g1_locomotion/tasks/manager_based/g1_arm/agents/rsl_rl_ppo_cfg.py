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
class G1ArmLeftAblationEntropyCoefFixedSchedulePPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-25 ablation: entropy_coef=0.005 with schedule="fixed" instead of the
    default "adaptive" — isolates whether the adaptive KL-based LR schedule itself is
    what makes any nonzero entropy_coef unstable, rather than the coefficient's value.

    Read rsl_rl's own ppo.py directly: the adaptive schedule multiplies learning_rate by
    1.5x (capped at 1e-2, i.e. 10x the 1e-3 start) whenever kl_mean < desired_kl/2, and
    entropy_coef>0 pushes the policy toward higher variance every update via the loss
    term `- entropy_coef * entropy_batch.mean()`. If that variance increase produces a
    KL shift that's still under desired_kl/2 (plausible here — this task's reward is
    deeply negative and looks weak/noisy, so the surrogate-loss-driven KL is probably
    small on its own), the schedule keeps boosting the learning rate, which then makes
    the NEXT entropy-driven variance increase larger too — a real positive-feedback
    loop. This matches the actual observed shape for entropy_coef=0.003 (fixed
    schedule): Policy/mean_noise_std went 1.0 -> 1.5 -> 4.3 -> 7.5 -> 11.0 over 2000
    iterations, ACCELERATING, not leveling off — consistent with an LR-driven runaway,
    not simply "the coefficient was a bit too big." Testing a fixed, non-adaptive
    learning_rate removes the schedule's ability to amplify this, isolating whether
    entropy_coef itself (at a modest value) is viable at all once that's controlled for.
    entropy_coef chosen (0.005) sits between the two already-tested values (0.003, 0.01)
    — not because a value "in between" was expected to matter on its own (per-user
    correctly skeptical of that), but because it's a reasonable modest value to pair
    with the schedule change, not a retest of the same hypothesis."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_entropy_coef_fixed_schedule"
        self.algorithm.entropy_coef = 0.005
        self.algorithm.schedule = "fixed"


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
class G1ArmLeftAblationRateLimitPPORunnerCfg(G1ArmPPORunnerCfg):
    """Runner cfg for the rate-limit ablation — same as G1ArmLeftPPORunnerCfg except
    experiment_name, so its logs land separately. The actual change (action_filter_alpha,
    max_action_delta_per_step) lives in the env cfg (G1ArmLeftAblationRateLimitEnvCfg),
    not here; entropy_coef/exp_scale both stay at baseline (0.0)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_rate_limit"


@configclass
class G1ArmLeftAblationLowVelNoisePPORunnerCfg(G1ArmPPORunnerCfg):
    """Runner cfg for the low-vel-noise ablation — same as G1ArmLeftPPORunnerCfg except
    experiment_name, so its logs land separately. The actual change (joint_vel_noise)
    lives in the env cfg (G1ArmLeftAblationLowVelNoiseEnvCfg), not here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_low_vel_noise"


@configclass
class G1ArmLeftGain60Kd1p5PPORunnerCfg(G1ArmPPORunnerCfg):
    """Runner cfg for the 60/1.5-gain test — same as G1ArmLeftPPORunnerCfg except
    experiment_name, so its logs land separately. The actual change (actuator gain)
    lives in the env cfg (G1ArmLeftEnvCfg_Gain60Kd1p5), not here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/gain_60_kd1p5"


@configclass
class G1ArmLeftGain15Kd1PPORunnerCfg(G1ArmPPORunnerCfg):
    """Runner cfg for the 15/1-gain test — same as G1ArmLeftPPORunnerCfg except
    experiment_name, so its logs land separately. The actual change (actuator gain)
    lives in the env cfg (G1ArmLeftEnvCfg_Gain15Kd1), not here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/gain_15_kd1"


@configclass
class G1ArmLeftGoalCurriculumPPORunnerCfg(G1ArmPPORunnerCfg):
    """Runner cfg for the goal-distance-curriculum test — same as G1ArmLeftPPORunnerCfg
    except experiment_name, so its logs land separately. The actual change (curriculum)
    lives in the env cfg (G1ArmLeftEnvCfg_GoalCurriculum), not here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/goal_curriculum"


@configclass
class G1ArmLeftAblationPrivilegedCriticPPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-25 ablation: privileged (noise-free) critic, in isolation — same gain,
    same reward, same joint_vel_noise (1.5, unchanged), symmetry intact.

    The critic never deploys to real hardware — it exists purely to estimate value
    during training, so unlike the actor there's no reason it needs to cope with
    simulated sensor noise. Currently both actor and critic read the exact same noisy
    "policy" observation (obs_groups default: critic -> ["policy"]), meaning the value
    function has to estimate returns through unnecessary observation noise, which could
    produce noisier value estimates and hence noisier advantage estimates feeding every
    PPO update — a plausible independent contributor to instability on top of the
    entropy_coef/adaptive-schedule issues already found. g1_arm_env.py's
    _get_observations() now always emits a second, clean "critic" key (see its own
    comment) — this is the only cfg that actually points obs_groups at it; every other
    runner cfg's default (critic -> ["policy"]) is unaffected by that addition."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_privileged_critic"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}


@configclass
class G1ArmLeftAblationLogStdPPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-25 ablation: noise_std_type="log" instead of the default "scalar", in
    isolation — same gain, same reward, same joint_vel_noise, symmetry intact.

    rsl_rl's ActorCritic supports parameterizing exploration noise either as a raw
    scalar std (current default, directly optimized) or as log(std) (std = exp(log_std)).
    Log-parameterization of positive scale parameters is a standard stability technique
    generally — equal steps in log-space are multiplicative (percentage) changes in the
    actual value, avoiding the asymmetric gradient behavior a raw scalar can have near
    very small or very large values. Every entropy_coef test so far (0.0/0.003/0.005/0.01,
    adaptive or fixed LR schedule) used the default "scalar" type — this changes the
    parameterization itself, not the incentive (entropy_coef stays at baseline 0.0),
    testing whether noise dynamics behave differently (neither collapsing to ~0.09 nor
    exploding) purely from the reparameterization, independent of any entropy bonus."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_log_std"
        self.policy.noise_std_type = "log"


@configclass
class G1ArmLeftBestCombinedPPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-26: combines the two confirmed-non-harmful ablations from the 4-way sweep
    (privileged_critic +5pts, log_std +2pts over baseline's 24.83%/22.53%; the third,
    low_vel_noise, is confirmed HARMFUL — 2-3% — and is not included). Paired with the
    new action_fb observation (g1_arm_env.py's _get_observations — the policy previously
    had no way to observe its own action-filter's internal state, a real POMDP-vs-MDP
    gap for a memoryless policy with a rate-limiting/smoothing filter in the loop; that
    change is baked into the base env class so it applies here automatically, not a
    separate flag). Not re-isolated from each other first — given the time budget, this
    is the single best-guess combination for a longer, monitored run rather than another
    round of isolated 2k sanity checks; if it works we won't cleanly know which
    component mattered most, which is an accepted tradeoff here, not an oversight."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/best_combined"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.policy.noise_std_type = "log"


@configclass
class G1ArmLeftIntegratedPPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-29: runner for the integrated-action-target env (G1ArmLeftIntegratedEnvCfg
    — the static-torque-ceiling fix, probe-validated via
    testing/general_testing/check_arm_static_torque_ceiling.py). Same proven agent
    recipe as BestCombined (privileged critic + log_std; action_fb comes from the env
    base class), so the comparison against best_combined's 28.20%/32.85% isolates the
    env-side parameterization change, not an agent-side confound."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/integrated"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        self.policy.noise_std_type = "log"


@configclass
class G1ArmLeftIntegratedNoTermPPORunnerCfg(G1ArmLeftIntegratedPPORunnerCfg):
    """2026-07-30: same agent recipe as G1ArmLeftIntegratedPPORunnerCfg — only
    experiment_name differs, so this run's logs land separately (`arms/
    integrated_no_term`) rather than mixing with the first Integrated run. The actual
    change (terminate_on_success=False) lives in the env cfg
    (G1ArmLeftIntegratedNoTermEnvCfg), not here — see that class's docstring."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/integrated_no_term"


@configclass
class G1ArmLeftAblationEntropyCoefSmallPPORunnerCfg(G1ArmPPORunnerCfg):
    """2026-07-25 ablation: entropy_coef=0.003, a real gap between the two extremes
    already tested — 0.0 (baseline) collapses Mean action noise std to 0.09 by
    iteration 5999 (confirmed by reading arms/left/2026-07-23_22-54-45's own training
    log) while Mean reward stays flat/deeply negative the whole run: premature
    convergence, not a slow-but-steady climb, so more iterations at entropy_coef=0.0
    doesn't help. 0.01 overcorrected the other way (Policy/mean_noise_std exploded
    1.0 -> 39-57 within 1500-2000 iterations, see G1ArmLeftAblationEntropyCoefPPORunnerCfg).
    Neither extreme has been tested with anything in between. position_reward_exp_scale
    stays at baseline 0.0 — this isolates entropy_coef alone, since the exp_scale=3.0
    ablation showed the identical noise-collapse pattern (0.26 by iter 1500) and didn't
    address the actual mechanism, so reward-shape changes aren't implicated here."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "arms/ablation_entropy_coef_small"
        self.algorithm.entropy_coef = 0.003


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
