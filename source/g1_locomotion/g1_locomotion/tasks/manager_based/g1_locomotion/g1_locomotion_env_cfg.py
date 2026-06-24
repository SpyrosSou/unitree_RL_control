# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 locomotion environment configurations.

These classes inherit from Isaac Lab's pre-built G1 velocity-tracking environments.
Override any field here to tune or extend the base configs.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1RoughEnvCfg,
    G1RoughEnvCfg_PLAY,
)


@configclass
class G1LocomotionFlatEnvCfg(G1FlatEnvCfg):
    """G1 flat-terrain locomotion (customisable).

    Inherits the full flat-ground G1 setup from Isaac Lab.
    Add per-project overrides below, e.g.:
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
    """

    def __post_init__(self):
        super().__post_init__()

        base_velocity = getattr(self.commands, "base_velocity", None)
        if base_velocity is not None:
            # Add frequent command changes and a small near-zero/zero slice.
            if hasattr(base_velocity, "resampling_time_range"):
                base_velocity.resampling_time_range = (0.9, 2.6)
            if hasattr(base_velocity, "rel_standing_envs"):
                base_velocity.rel_standing_envs = 0.10
            if hasattr(base_velocity, "ranges"):
                ranges = base_velocity.ranges
                if hasattr(ranges, "lin_vel_x"):
                    ranges.lin_vel_x = (-0.5, 1.0)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-0.6, 0.6)
                if hasattr(ranges, "ang_vel_z"):
                    ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1LocomotionFlatTransitionEnvCfg(G1FlatEnvCfg):
    """Flat locomotion config with transition-heavy command sampling.

    Designed to expose frequent stop/start, direction reversal, and short pulses.
    """

    def __post_init__(self):
        super().__post_init__()

        base_velocity = getattr(self.commands, "base_velocity", None)
        if base_velocity is not None:
            if hasattr(base_velocity, "rel_standing_envs"):
                base_velocity.rel_standing_envs = 0.30
            if hasattr(base_velocity, "resampling_time_range"):
                base_velocity.resampling_time_range = (0.5, 1.6)
            if hasattr(base_velocity, "ranges"):
                ranges = base_velocity.ranges
                if hasattr(ranges, "lin_vel_x"):
                    ranges.lin_vel_x = (-1.0, 1.0)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-0.9, 0.9)
                if hasattr(ranges, "ang_vel_z"):
                    ranges.ang_vel_z = (-1.2, 1.2)


@configclass
class G1LocomotionFlatEnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Play / evaluation variant of the flat env (fewer envs, no randomisation)."""

    pass


@configclass
class G1LocomotionFlatTransitionEnvCfg_PLAY(G1LocomotionFlatTransitionEnvCfg):
    """Play variant for transition-heavy walking policy."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class G1LocomotionRoughEnvCfg(G1RoughEnvCfg):
    """G1 rough-terrain locomotion (customisable).

    Inherits the full rough-terrain G1 setup from Isaac Lab.
    """

    pass


@configclass
class G1LocomotionRoughEnvCfg_PLAY(G1RoughEnvCfg_PLAY):
    """Play / evaluation variant of the rough env (fewer envs, no randomisation)."""

    pass


@configclass
class G1LocomotionStandingFlatEnvCfg(G1FlatEnvCfg):
    """Flat terrain standing-only policy configuration.

    Commands are mostly zero with tiny pulses so the policy can learn
    minimal corrective stepping instead of only in-place balancing.
    """

    def __post_init__(self):
        super().__post_init__()

        # Arm actuator gains are reduced for smoother, physically plausible trajectory tracking.
        if "arms" in self.scene.robot.actuators:
            self.scene.robot.actuators["arms"].stiffness = 25.0
            self.scene.robot.actuators["arms"].damping = 8.0

        # Keep action dimension unchanged, but override arm-joint targets from disturbance curriculum.
        if hasattr(self.actions, "joint_pos"):
            self.actions.joint_pos.class_type = mdp.StandingArmBlendJointPositionAction

        # Arm-motion curriculum event: starts with no motion, then small/slow, then larger/faster,
        # and finally asymmetric motions with occasional sudden reversals.
        self.events.standing_arm_motion_disturbance = EventTerm(
            func=mdp.StandingArmTrajectoryDisturbance,
            mode="interval",
            interval_range_s=(self.sim.dt * self.decimation, self.sim.dt * self.decimation),
            params={"asset_cfg": SceneEntityCfg("robot")},
            is_global_time=False,
        )

        # Ensure reset keeps the arm-disturbance trajectory anchored to default posture.
        self.events.standing_arm_motion_reset = EventTerm(
            func=mdp.reset_arm_targets_to_default,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "left_elbow_roll_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "right_elbow_roll_joint",
                    ],
                )
            },
        )

        # Apply mixed random pushes that are decoupled from arm-motion phase,
        # so the policy experiences easy/hard combinations across the whole curriculum.
        self.events.push_robot = EventTerm(
            func=mdp.StandingRandomPushDisturbance,
            mode="interval",
            interval_range_s=(0.8, 1.8),
            params={"asset_cfg": SceneEntityCfg("robot")},
            is_global_time=False,
        )

        base_velocity = getattr(self.commands, "base_velocity", None)
        if base_velocity is not None:
            if hasattr(base_velocity, "heading_command"):
                base_velocity.heading_command = False
            if hasattr(base_velocity, "rel_standing_envs"):
                # Keep standing dominant while reserving a small slice for
                # micro-corrections around zero velocity.
                base_velocity.rel_standing_envs = 0.92
            if hasattr(base_velocity, "resampling_time_range"):
                base_velocity.resampling_time_range = (0.8, 2.0)
            if hasattr(base_velocity, "ranges"):
                ranges = base_velocity.ranges
                if hasattr(ranges, "lin_vel_x"):
                    ranges.lin_vel_x = (-0.06, 0.06)
                if hasattr(ranges, "lin_vel_y"):
                    ranges.lin_vel_y = (-0.04, 0.04)
                if hasattr(ranges, "ang_vel_z"):
                    ranges.ang_vel_z = (-0.10, 0.10)

        rewards = getattr(self, "rewards", None)
        if rewards is not None:
            if hasattr(rewards, "lin_vel_z_l2"):
                rewards.lin_vel_z_l2.weight = -2.2
            if hasattr(rewards, "ang_vel_xy_l2"):
                rewards.ang_vel_xy_l2.weight = -0.12
            if hasattr(rewards, "action_rate_l2"):
                rewards.action_rate_l2.weight = -0.008
            if hasattr(rewards, "dof_acc_l2"):
                rewards.dof_acc_l2.weight = -1.5e-7
            if hasattr(rewards, "feet_air_time"):
                rewards.feet_air_time.weight = 0.0


@configclass
class G1LocomotionStandingFlatEnvCfg_PLAY(G1LocomotionStandingFlatEnvCfg):
    """Play variant for the standing flat terrain policy."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None

        # Keep perturbations in play mode, but slightly milder than training,
        # so policy behavior is visible without instantly destabilizing demos.
        if hasattr(self.events, "push_robot"):
            self.events.push_robot.interval_range_s = (1.0, 2.0)
            self.events.push_robot.params.update(
                {
                    "warmup_steps": 0,
                    "push_probability": 0.45,
                    "hard_push_probability": 0.12,
                    "easy_lin_speed_range": (0.05, 0.16),
                    "hard_lin_speed_range": (0.16, 0.30),
                    "yaw_vel_range_easy": (-0.15, 0.15),
                    "yaw_vel_range_hard": (-0.35, 0.35),
                }
            )

        # In play mode, begin arm-motion disturbances quickly for visualization.
        if hasattr(self.events, "standing_arm_motion_disturbance"):
            self.events.standing_arm_motion_disturbance.params["phase_step_boundaries"] = (20, 80, 180, 320)
            # Start play close to phase-3 so stronger motions appear quickly.
            self.events.standing_arm_motion_disturbance.params["phase_step_offset"] = 140