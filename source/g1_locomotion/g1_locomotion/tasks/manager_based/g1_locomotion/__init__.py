# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof unified stand+walk velocity-tracking task — see 29dof_implementation_plan.md.

Two gym tasks:
- ``G1-Locomotion-Velocity-v0`` — Unitree's ported base recipe, no arm disturbance.
  Train this first as the Phase 1.4 sanity-check baseline before anything else.
- ``G1-Locomotion-Velocity-ArmDisturbance-v0`` — the same recipe plus the arm-motion
  disturbance curriculum (Phase 2) — this is the one that actually needs to hold up once
  a reaching arm is introduced later.
Each has a ``-Play-v0`` sibling for interactive testing / eval (fewer envs, full command
range, arm-disturbance phase starts further along for quick visual inspection).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="G1-Locomotion-Velocity-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionPPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionPPORunnerCfg",
    },
)

# 2026-07-23 walking-fix ablation tasks — each trains the base (no-disturbance) recipe
# with exactly one candidate fix applied in isolation, plus one with all three combined,
# for a direct 1k-iteration comparison against the unmodified G1-Locomotion-Velocity-v0
# baseline. See the matching Ablation*EnvCfg classes' docstrings in g1_locomotion_env_cfg.py.
gym.register(
    id="G1-Locomotion-Velocity-Ablation-TermPenalty-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionAblationTermPenaltyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionAblationTermPenaltyPPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-Ablation-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionAblationCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionAblationCurriculumPPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-Ablation-RewardWeights-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionAblationRewardWeightsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionAblationRewardWeightsPPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-Ablation-All-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionAblationAllEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionAblationAllPPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-ArmDisturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmDisturbanceEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmDisturbanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionArmDisturbancePPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-ArmDisturbance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmDisturbanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionArmDisturbancePPORunnerCfg",
    },
)

# Same recipe as ArmDisturbance, plus a much higher rel_standing_envs — a dedicated
# fine-tuning phase for an existing ArmDisturbance run to resume into, giving the
# (gated to standing-only) arm-disturbance curriculum more of training to actually act
# on. See G1LocomotionArmDisturbanceStandingFocusEnvCfg's docstring, 2026-07-22.
gym.register(
    id="G1-Locomotion-Velocity-ArmDisturbance-StandingFocus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmDisturbanceStandingFocusEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmDisturbanceStandingFocusEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionArmDisturbancePPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-ArmDisturbance-StandingFocus-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmDisturbanceStandingFocusEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionArmDisturbancePPORunnerCfg",
    },
)

# Same recipe, but the arm-motion disturbance is driven by the actual trained 7-DOF arm
# RL policy (mdp.ArmPolicyDisturbance) instead of scripted random joint deltas. Built
# 2026-07-22, NOT currently used by any active training — the scripted-disturbance run
# above is still the one training. See ArmPolicyDisturbance's own docstring in mdp/
# events.py for the full rationale and the "not yet runtime-verified" caveat.
gym.register(
    id="G1-Locomotion-Velocity-ArmPolicyDisturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmPolicyDisturbanceEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmPolicyDisturbanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionArmDisturbancePPORunnerCfg",
    },
)

gym.register(
    id="G1-Locomotion-Velocity-ArmPolicyDisturbance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionArmPolicyDisturbanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionArmDisturbancePPORunnerCfg",
    },
)
