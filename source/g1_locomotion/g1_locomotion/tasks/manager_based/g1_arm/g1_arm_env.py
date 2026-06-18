# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 arm IK via RL — DirectRLEnv.

The policy learns to move the palm(s) to randomly sampled 3-D goal positions
using delta joint-position commands on the 5-DOF arm(s).

Single arm  (arm="left" or "right"):
    Observation (17-D): joint_pos(5) | joint_vel(3) | ee_pos(3) | goal(3) | error(3)
    Action      ( 5-D): delta joint targets for the active arm

Both arms   (arm="both"):
    Observation (34-D): [left 17-D] + [right 17-D]
    Action      (10-D): [left 5-D]  + [right 5-D]
    Episode terminates when BOTH palms reach their respective goals.
"""

from __future__ import annotations

from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import G1_CFG


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------


@configclass
class G1ArmIKEnvCfg(DirectRLEnvCfg):
    """Base configuration for the G1 arm IK reaching task."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=2)
    episode_length_s: float = 10.0
    decimation: int = 2  # 30 Hz control

    # Observation / action / state dims
    observation_space: int = 17
    action_space: int = 5
    state_space: int = 0  # no separate critic state; must be set (None crashes serialization)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0)

    # Which arm(s) to train: "left", "right", or "both"
    arm: str = "left"

    # Reward scales
    position_reward_scale: float = 10.0
    action_smoothness_scale: float = 0.01
    joint_limit_penalty_scale: float = 1.0
    goal_reached_bonus: float = 50.0
    goal_threshold: float = 0.02  # 2 cm
    action_scale: float = 0.5
    action_filter_alpha: float = 0.25
    max_action_delta_per_step: float = 0.06

    # Robot — G1 asset with prim_path set for multi-env cloning.
    # Arm actuator stiffness/damping are overridden in __post_init__ for
    # responsive position tracking (G1_CFG defaults are tuned for torque control).
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    def __post_init__(self):
        super().__post_init__()
        # Fix the pelvis to the world — legs don't matter for arm-only training
        self.robot.spawn.articulation_props.fix_root_link = True
        # Higher gains for position-control on the arm joints
        self.robot.actuators["arms"] = ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ],
            stiffness=200.0,
            damping=20.0,
        )
        # Contact sensors not needed for this task
        self.robot.spawn.activate_contact_sensors = False


@configclass
class G1ArmIKLeftEnvCfg(G1ArmIKEnvCfg):
    arm: str = "left"
    observation_space: int = 17
    action_space: int = 5


@configclass
class G1ArmIKRightEnvCfg(G1ArmIKEnvCfg):
    arm: str = "right"
    observation_space: int = 17
    action_space: int = 5


@configclass
class G1ArmIKBothEnvCfg(G1ArmIKEnvCfg):
    """Both arms trained simultaneously (34-D obs, 10-D actions)."""
    arm: str = "both"
    observation_space: int = 34
    action_space: int = 10


@configclass
class G1ArmIKLeftEnvCfg_PLAY(G1ArmIKLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


@configclass
class G1ArmIKRightEnvCfg_PLAY(G1ArmIKRightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


@configclass
class G1ArmIKBothEnvCfg_PLAY(G1ArmIKBothEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 20.0


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

# Joint names for each arm (must match the G1 USD articulation)
_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
]
_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
]
_LEFT_EE_BODY = "left_palm_link"
_RIGHT_EE_BODY = "right_palm_link"

# Goal workspace bounds in the robot's local frame [m]
# (env origin is added later to convert to world frame)
_GOAL_BOUNDS = {
    "left":  {"x": (0.1, 0.5), "y": (0.05, 0.45), "z": (0.9, 1.2)},
    "right": {"x": (0.1, 0.5), "y": (-0.45, -0.05), "z": (0.9, 1.2)},
}


class G1ArmIKEnv(DirectRLEnv):
    """G1 arm reaching environment using direct RL (``DirectRLEnv``).

    Supports arm="left", "right", or "both".  When arm="both" the observation
    and action spaces are doubled and both palms must reach their goals for the
    episode to terminate early.
    """

    cfg: G1ArmIKEnvCfg

    def __init__(self, cfg: G1ArmIKEnvCfg, render_mode: str | None = None, **kwargs):
        # Nothing arm-specific needed before super().__init__ now
        super().__init__(cfg, render_mode, **kwargs)

        # ------------------------------------------------------------------
        # Post-init: build arm-group list (safe after sim is running)
        # ------------------------------------------------------------------
        # Each entry: {joint_tensor, ee_idx, bounds}
        self._arm_groups: list[dict] = []
        if cfg.arm in ("left", "both"):
            ids, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            ee, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            self._arm_groups.append({
                "joint_tensor": torch.tensor(ids, dtype=torch.long, device=self.device),
                "ee_idx": ee[0],
                "bounds": _GOAL_BOUNDS["left"],
            })
        if cfg.arm in ("right", "both"):
            ids, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            ee, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            self._arm_groups.append({
                "joint_tensor": torch.tensor(ids, dtype=torch.long, device=self.device),
                "ee_idx": ee[0],
                "bounds": _GOAL_BOUNDS["right"],
            })

        self.n_arms = len(self._arm_groups)

        # Combined joint tensor (5 for single arm, 10 for both) — used for reset
        self.arm_joint_indices_tensor = torch.cat(
            [g["joint_tensor"] for g in self._arm_groups]
        )
        self.arm_joint_indices = self.arm_joint_indices_tensor.tolist()

        # Runtime buffers
        # goal_positions: (num_envs, n_arms, 3)
        self.goal_positions = torch.zeros((self.num_envs, self.n_arms, 3), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.filtered_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Goal visualisation markers
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/GoalMarkers"
        marker_cfg.markers["sphere"].radius = 0.03
        self.goal_markers = VisualizationMarkers(marker_cfg)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # Ground plane (global — not cloned per env)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        # Clone environments and filter inter-env collisions
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/defaultGroundPlane"])

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        # Filter and clamp action deltas to emulate actuator lag and avoid abrupt jumps.
        # Works for single arm (5 joints) and both arms (10 joints) identically.
        alpha = float(self.cfg.action_filter_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        self.filtered_actions = alpha * self.actions + (1.0 - alpha) * self.filtered_actions

        current = self.robot.data.joint_pos[:, self.arm_joint_indices_tensor]
        delta = self.filtered_actions * self.cfg.action_scale
        max_delta = float(self.cfg.max_action_delta_per_step)
        if max_delta > 0.0:
            delta = delta.clamp(min=-max_delta, max=max_delta)

        targets = current + delta
        limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_indices_tensor]
        targets = targets.clamp(limits[:, 0], limits[:, 1])
        self.robot.set_joint_position_target(targets, joint_ids=self.arm_joint_indices_tensor)
        self.previous_actions[:] = self.filtered_actions

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # Build 17-D block per arm, then concatenate (→ 17 or 34 total)
        parts = []
        for i, arm in enumerate(self._arm_groups):
            jt = arm["joint_tensor"]
            joint_pos = self.robot.data.joint_pos[:, jt]         # (N, 5)
            joint_vel = self.robot.data.joint_vel[:, jt[:3]]     # (N, 3)
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]  # (N, 3)
            goal = self.goal_positions[:, i, :]                  # (N, 3)
            error = goal - ee_pos                                 # (N, 3)
            parts.extend([joint_pos, joint_vel, ee_pos, goal, error])
        return {"policy": torch.cat(parts, dim=-1)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        total = torch.zeros(self.num_envs, device=self.device)
        all_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for i, arm in enumerate(self._arm_groups):
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]  # (N, 3)
            dist = torch.norm(self.goal_positions[:, i, :] - ee_pos, dim=-1)
            total += -dist * self.cfg.position_reward_scale
            reached = dist < self.cfg.goal_threshold
            total += reached.float() * self.cfg.goal_reached_bonus
            all_reached &= reached

        # Smoothness and joint-limit penalties (shared across all arm joints)
        total += -torch.norm(self.previous_actions, dim=-1) * self.cfg.action_smoothness_scale

        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices_tensor]
        limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_indices_tensor]
        margin = 0.05 * (limits[:, 1] - limits[:, 0])
        at_limit = ((joint_pos < limits[:, 0] + margin) | (joint_pos > limits[:, 1] - margin)).float().sum(-1)
        total += -at_limit * self.cfg.joint_limit_penalty_scale

        # Episode terminates when ALL arms have reached their goals
        self.successes = all_reached
        return total

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.successes.clone()
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int]):
        num_resets = len(env_ids)
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device) \
            if not isinstance(env_ids, torch.Tensor) else env_ids

        # Start from default standing pose, randomise only the arm joints
        joint_pos = self.robot.data.default_joint_pos[env_ids_tensor].clone()
        joint_vel = torch.zeros_like(joint_pos)

        for idx in self.arm_joint_indices:
            lo = self.robot.data.soft_joint_pos_limits[0, idx, 0]
            hi = self.robot.data.soft_joint_pos_limits[0, idx, 1]
            centre = (lo + hi) * 0.5
            half = (hi - lo) * 0.3  # ±30 % of range → 60 % total
            joint_pos[:, idx] = centre + (torch.rand(num_resets, device=self.device) * 2 - 1) * half

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)

        # Sample goals in robot-local frame, shift to world frame via env origins
        for i, arm in enumerate(self._arm_groups):
            self.goal_positions[env_ids_tensor, i, :] = (
                self._sample_goal_positions(num_resets, arm["bounds"])
                + self.scene.env_origins[env_ids_tensor]
            )

        # Reset buffers
        self.previous_actions[env_ids_tensor] = 0.0
        self.filtered_actions[env_ids_tensor] = 0.0
        self.successes[env_ids_tensor] = False
        self.episode_length_buf[env_ids_tensor] = 0

        self._update_goal_markers(env_ids_tensor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_position(self, arm_index: int = 0) -> torch.Tensor:
        """End-effector world position for one arm group (N, 3)."""
        return self.robot.data.body_pos_w[:, self._arm_groups[arm_index]["ee_idx"], :]

    def _sample_goal_positions(self, num_goals: int, bounds: dict) -> torch.Tensor:
        """Sample reachable goals in the robot's local frame for one arm."""
        goals = torch.zeros((num_goals, 3), device=self.device)
        goals[:, 0] = torch.rand(num_goals, device=self.device) * (bounds["x"][1] - bounds["x"][0]) + bounds["x"][0]
        goals[:, 1] = torch.rand(num_goals, device=self.device) * (bounds["y"][1] - bounds["y"][0]) + bounds["y"][0]
        goals[:, 2] = torch.rand(num_goals, device=self.device) * (bounds["z"][1] - bounds["z"][0]) + bounds["z"][0]
        return goals

    def _update_goal_markers(self, env_ids: torch.Tensor):
        # Flatten (n_reset_envs, n_arms, 3) → (n_reset_envs * n_arms, 3) for the marker visualizer
        self.goal_markers.visualize(self.goal_positions[env_ids].reshape(-1, 3))
