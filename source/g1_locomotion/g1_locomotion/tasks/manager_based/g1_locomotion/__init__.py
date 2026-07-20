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

# Same policy/PPO config as G1-Locomotion-Standing-Flat-v0 (G1LocomotionStandingFlatPPORunnerCfg,
# unchanged) — only the arm-reach disturbance mechanism differs (real per-arm differential IK
# instead of a scripted joint-space trajectory). See G1LocomotionStandingFlatIKReachEnvCfg's
# docstring for why this is a separate task id rather than a change to the existing one.
gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Eval-only variant — disturbance fully on from step 0, no training curriculum delay.
# Used by validation/eval_standing_ikreach.py to measure an already-trained checkpoint
# against the actual disturbance it trained on (neither eval_standing.py nor
# eval_full_demo.py exercise mdp.StandingArmIKReachDisturbance at all — see that
# script's own docstring).
gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Same as G1-Locomotion-Standing-Flat-IKReach-v0 plus one change: a direct base_height_l2
# reward term (target_height=0.75, matching G1's own spawn height), added because nothing
# in the inherited reward stack penalizes hip-pitch/knee deviation at all — standing had
# been sinking into a deep, stable squat as a genuinely free way to buy stability, and
# nothing in training ever pushed back on it. See
# G1LocomotionStandingFlatIKReachHeightEnvCfg's docstring for the full mechanism.
gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Height-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachHeightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Height-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachHeightEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Same as G1-Locomotion-Standing-Flat-IKReach-Height-v0 plus one change: mdp.symmetry.leg_symmetry_l2
# (weight -2.0), penalizing left-right leg asymmetry directly — added after visually confirming a
# persistent, lopsided rest pose (tilt, uneven legs) with zero arm disturbance active. See
# G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg's docstring for the full mechanism.
gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Height-Symmetry-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Phase 1 consolidated config (plan.md §4): policy-driven disturbance + base_height_l2 +
# deployment-matched arm gains (active arm trains at 200/20, held at 60/1.5 — the one
# lever never tried; see G1LocomotionStandingFlatConsolidatedEnvCfg's docstring).
gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Consolidated + the arm-intent observation (10-D commanded arm targets appended to the
# policy obs) — posture *information* instead of posture penalty; see
# G1LocomotionStandingFlatConsolidatedIntentEnvCfg's docstring (2026-07-15).
gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-Intent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedIntentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-Intent-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedIntentEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Consolidated + no_reach_prob=0.15 (idle-episode slice), WITHOUT the intent observation —
# the control that separates the intent run's two bundled changes. See
# G1LocomotionStandingFlatConsolidatedNoReachEnvCfg's docstring (2026-07-15).
gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-NoReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedNoReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-NoReach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedNoReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Consolidated + joint_deviation_torso re-tightened to -0.05 — the "is the rotated-torso
# bracing posture load-bearing or just an unpenalized habit" experiment (2026-07-15). See
# G1LocomotionStandingFlatConsolidatedTorsoEnvCfg's docstring.
gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-Torso-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedTorsoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-Consolidated-Torso-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatConsolidatedTorsoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Height-Symmetry-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachHeightSymmetryEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Same as G1-Locomotion-Standing-Flat-IKReach-v0 plus one change: joint_deviation_torso's
# weight partially re-tightened (0.0 -> -0.05). See
# G1LocomotionStandingFlatIKReachTorsoEnvCfg's docstring — isolated single-variable
# experiment against the dwell-phase-fix baseline, run independently from the
# arm-policy-driven-disturbance experiment below so each change's effect is attributable.
gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Torso-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachTorsoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-IKReach-Torso-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatIKReachTorsoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

# Same as G1-Locomotion-Standing-Flat-IKReach-v0 but the arm-reach disturbance is driven
# by the actual trained arm-IK policy (chosen_checkpoints/arm_left_latest.pt) instead of
# analytic IK. See G1LocomotionStandingFlatPolicyReachEnvCfg's docstring for the evidence
# behind this — isolated single-variable experiment, run independently from the torso
# experiment above.
gym.register(
    id="G1-Locomotion-Standing-Flat-PolicyReach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatPolicyReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocomotionStandingFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="G1-Locomotion-Standing-Flat-PolicyReach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_locomotion_env_cfg:G1LocomotionStandingFlatPolicyReachEnvCfg_PLAY",
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

