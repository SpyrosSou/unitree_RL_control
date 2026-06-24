# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# Flat terrain  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Locomotion-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------
# Rough terrain  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Locomotion-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionRoughPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionRoughPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_rough_ppo_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------
# Flat terrain standing-only policy
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Locomotion-Standing-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------
# Flat terrain transition-focused walking policy
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Locomotion-Flat-Transition-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionFlatTransitionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionFlatTransitionPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Flat-Transition-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionFlatTransitionEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionFlatTransitionPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

