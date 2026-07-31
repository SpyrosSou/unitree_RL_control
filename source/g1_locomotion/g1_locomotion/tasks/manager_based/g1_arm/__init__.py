# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof arm policy — see 29dof_implementation_plan.md Phase 3.

Task ids deliberately don't say "IK" (2026-07-21, user request) — the deployed policy is
pure RL joint-space control, no inverse kinematics involved. See g1_arm_env.py's module
docstring for the full rationale and what changed from the 23dof-era 5-DOF/"Arm-IK"-named
task this was adapted from.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# Left arm  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-Left-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftPPORunnerCfg",
    },
)

# 2026-07-24: wrist_pitch/wrist_yaw locked at default, 5 controlled joints instead of
# 7 — see G1ArmLeftLockedWristEnvCfg's own docstring.
gym.register(
    id="G1-Arm-Left-LockedWrist-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftLockedWristEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftLockedWristPPORunnerCfg",
    },
)

# 2026-07-24 walking-fix-style ablations for the arm task — see G1ArmLeftAblation*
# classes in g1_arm_env.py/agents/rsl_rl_ppo_cfg.py for what each isolates.
gym.register(
    id="G1-Arm-Left-Ablation-EntropyCoef-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationEntropyCoefPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Ablation-ExpScale-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftAblationExpScaleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationExpScalePPORunnerCfg",
    },
)

# 2026-07-25: entropy_coef=0.003, a real gap between 0.0 (confirmed
# premature noise collapse, 0.09 by iter 5999) and 0.01 (confirmed
# divergence) — see G1ArmLeftAblationEntropyCoefSmallPPORunnerCfg's
# own docstring for the full reasoning.
gym.register(
    id="G1-Arm-Left-Ablation-EntropyCoefSmall-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationEntropyCoefSmallPPORunnerCfg",
    },
)

# 2026-07-25: relax action_filter_alpha/max_action_delta_per_step in
# isolation — see G1ArmLeftAblationRateLimitEnvCfg's own docstring for
# the full reasoning (Train/mean_episode_length stuck at ~285-297 across
# every 40/10-gain RL-hyperparameter variant tried so far).
gym.register(
    id="G1-Arm-Left-Ablation-RateLimit-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftAblationRateLimitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationRateLimitPPORunnerCfg",
    },
)

# 2026-07-25: entropy_coef=0.005 with schedule="fixed" instead of "adaptive" —
# see G1ArmLeftAblationEntropyCoefFixedSchedulePPORunnerCfg's own docstring
# for the full reasoning (adaptive LR schedule may be what makes any nonzero
# entropy_coef unstable, not the coefficient's value).
gym.register(
    id="G1-Arm-Left-Ablation-EntropyCoefFixedSchedule-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationEntropyCoefFixedSchedulePPORunnerCfg",
    },
)

# 2026-07-25: three isolated ablations testing whether sim2real/training-mechanics
# defaults borrowed wholesale from the locomotion recipe (never re-validated for arm
# reaching's own precision requirements) explain the 40/10-gain plateau. See each
# EnvCfg/PPORunnerCfg's own docstring for the full reasoning. Run alongside a plain
# G1-Arm-Left-v0 baseline (same iteration count, same seed) as the 4th, "all off" leg.
gym.register(
    id="G1-Arm-Left-Ablation-LowVelNoise-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftAblationLowVelNoiseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationLowVelNoisePPORunnerCfg",
    },
)

# 2026-07-26: kp=60, kd=1.5 — the gain Unitree's own dedicated arm-control SDK examples
# actually use, not the 40/10 this task inherited from a locomotion-context commit. See
# G1ArmLeftEnvCfg_Gain60Kd1p5's own docstring for the full reasoning.
gym.register(
    id="G1-Arm-Left-Gain60Kd1p5-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg_Gain60Kd1p5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftGain60Kd1p5PPORunnerCfg",
    },
)

# 2026-07-26: kp=15, kd=1 — matches unitree_rl_mjlab's actual RL-policy deploy config
# (kp=14.3-16.8, kd=0.9-1.1 for arms), not the scripted-motion SDK's 60/1.5. See
# G1ArmLeftEnvCfg_Gain15Kd1's own docstring for the full reasoning.
gym.register(
    id="G1-Arm-Left-Gain15Kd1-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg_Gain15Kd1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftGain15Kd1PPORunnerCfg",
    },
)

# 2026-07-26: goal-distance curriculum — start goals within a small fraction of the
# full workspace box, expand to 100% by common_step_counter=30_000. See
# G1ArmLeftEnvCfg_GoalCurriculum's own docstring / goal_curriculum_enabled's field
# comment on G1ArmEnvCfg for the full reasoning.
gym.register(
    id="G1-Arm-Left-GoalCurriculum-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg_GoalCurriculum",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftGoalCurriculumPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Ablation-PrivilegedCritic-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationPrivilegedCriticPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Ablation-LogStd-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationLogStdPPORunnerCfg",
    },
)

# 2026-07-26: combines the two non-harmful legs from the 4-way sweep (privileged_critic,
# log_std) — see G1ArmLeftBestCombinedPPORunnerCfg's own docstring for the full
# reasoning and why low_vel_noise is deliberately excluded (confirmed harmful).
gym.register(
    id="G1-Arm-Left-BestCombined-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftBestCombinedPPORunnerCfg",
    },
)

# 2026-07-28: pairs the new distance-adaptive rate limit (env side, see
# G1ArmLeftAdaptiveRateEnvCfg's own docstring) with the same proven agent recipe
# BestCombined uses (privileged_critic + log_std) rather than re-testing from a bare
# baseline — this is meant to combine with the best-known recipe, not replace it.
gym.register(
    id="G1-Arm-Left-AdaptiveRate-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftAdaptiveRateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftBestCombinedPPORunnerCfg",
    },
)

# 2026-07-29: integrated (persistent) action targets + env-local ee/goal observations —
# the static-torque-ceiling fix, probe-validated via
# testing/general_testing/check_arm_static_torque_ceiling.py (the legacy current+delta
# pipeline caps static holding torque at kp*0.06 = 2.4 Nm at the real 40/10 gain;
# typical goal-box postures need 4.5-5.7 Nm). See G1ArmLeftIntegratedEnvCfg's docstring.
# Obs is 46-D (target_fb added) — NOT loadable by/against any pre-existing checkpoint.
gym.register(
    id="G1-Arm-Left-Integrated-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftIntegratedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftIntegratedPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Integrated-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftIntegratedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftIntegratedPPORunnerCfg",
    },
)

# 2026-07-30: disables terminate_on_success — the hold-quality fix for the dithering
# problem found in the first Integrated run (6.5-9.4% legacy success despite solved
# reaching) — see G1ArmLeftIntegratedNoTermEnvCfg's own docstring and
# arms_policy_finalisation.md step 1.
gym.register(
    id="G1-Arm-Left-IntegratedNoTerm-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftIntegratedNoTermEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftIntegratedNoTermPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-IntegratedNoTerm-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftIntegratedNoTermEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftIntegratedNoTermPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Right arm  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-Right-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmRightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmRightPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Right-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmRightEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmRightPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Both arms  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-Both-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmBothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmBothPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Both-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmBothEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmBothPPORunnerCfg",
    },
)
