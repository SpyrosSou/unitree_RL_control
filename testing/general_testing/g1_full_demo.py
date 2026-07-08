"""
G1 full integrated demo — standing + arm control + walking.

Combines:
- Standing policy (zero-command balance stabiliser)
- Walking policy (velocity-commanded locomotion)
- Arm IK policy (goal-reaching for left / right / both arms)

Behaviour:
- Start in STANDING mode.  Arms actively reach their current targets.
- Press W / A / D / Q / E to command walking.  Arms freeze (locomotion policy controls them).
- Release commands (S or stop pressing keys) → robot decelerates, switches back to STANDING.
- When standing again, arm policy resumes.
- Press T to type a new arm target at the console (same pattern as testing/arm_testing).
- Press L / R to select which arm to address when arm_mode=both (default: left).
- Press C to toggle third-person / free camera.

Arm target format (robot-local frame, base at origin):
    x  forward     reachable 0.1 – 0.5 m
    y  lateral     left arm: 0.05 – 0.45 m   right arm: -0.05 – -0.45 m
    z  height      reachable 0.9 – 1.2 m

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Auto-load from testing/general_testing/checkpoints.yaml
    ~/Elm/Code/IsaacLab/isaaclab.sh -p testing/general_testing/g1_full_demo.py

    # Explicit checkpoints + initial arm target
    ~/Elm/Code/IsaacLab/isaaclab.sh -p testing/general_testing/g1_full_demo.py \\
        --standing_checkpoint logs/rsl_rl/standing/.../model_1499.pt \\
        --walking_checkpoint  logs/rsl_rl/legs/.../model_3149.pt \\
        --arm_checkpoint      logs/rsl_rl/arms/g1_arm_ik_left/.../model_4200.pt \\
        --arm left \\
        --target 0.3 0.2 1.0

    # Both arms
    ~/Elm/Code/IsaacLab/isaaclab.sh -p testing/general_testing/g1_full_demo.py \\
        --arm both \\
        --target 0.3 0.2 1.0
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be started before all other imports
# ---------------------------------------------------------------------------
import argparse
import importlib.util as _ilu
import os
import yaml

ISAACLAB_PATH = os.path.expanduser("~/Elm/Code/IsaacLab")
_spec = _ilu.spec_from_file_location(
    "cli_args",
    os.path.join(ISAACLAB_PATH, "scripts/reinforcement_learning/rsl_rl/cli_args.py"),
)
cli_args_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cli_args_mod)

from isaaclab.app import AppLauncher

_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints.yaml")

# ---- Defaults (used if CLI and YAML both absent) ----
_DEFAULT_STANDING = (
    "logs/rsl_rl/standing/g1_locomotion_flat/2026-06-16_12-26-58/model_1499.pt"
)
_DEFAULT_WALKING = (
    "logs/rsl_rl/legs/g1_locomotion_flat/2026-06-03_10-20-23/model_3149.pt"
)
_DEFAULT_ARM_LEFT = (
    "logs/rsl_rl/arms/g1_arm_ik_left/2026-06-11_09-10-01/model_4200.pt"
)
_DEFAULT_ARM_BOTH = (
    "logs/rsl_rl/arms/g1_arm_ik_both/2026-06-09_20-31-43/model_1000.pt"
)

parser = argparse.ArgumentParser(description="G1 full integrated demo: stand/walk + arm IK.")
cli_args_mod.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--standing_checkpoint", type=str, default=None)
parser.add_argument("--walking_checkpoint",  type=str, default=None)
parser.add_argument("--arm_checkpoint",      type=str, default=None,
                    help="Checkpoint for the arm IK policy (overrides YAML).")
parser.add_argument("--arm", type=str, default=None, choices=["left", "right", "both"],
                    help="Which arm(s) to control. Overrides YAML arm_mode.")
parser.add_argument("--target", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Initial arm target in robot-local frame (x y z).")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionFlatEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _LEFT_ARM_JOINTS,
    _RIGHT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_EE_BODY,
    _GOAL_BOUNDS,
)
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmIKLeftPPORunnerCfg,
    G1ArmIKRightPPORunnerCfg,
    G1ArmIKBothPPORunnerCfg,
)

# ---------------------------------------------------------------------------
# Switch demo constants (identical to g1_stand_walk_switch_demo.py)
# ---------------------------------------------------------------------------
LOCO_TASK            = "G1-Locomotion-Flat-Play-v0"
LIN_VEL              = 1.0
ANG_VEL              = 0.5
SWITCH_TO_WALK_THRESHOLD    = 0.12
SWITCH_TO_STAND_THRESHOLD   = 0.04
MIN_MODE_STEPS              = 20
CMD_FILTER_ALPHA            = 0.12
SWITCH_TO_STAND_SPEED_THRESHOLD = 0.22
TRANSITION_STEPS_TO_WALK    = 10
TRANSITION_STEPS_TO_STAND   = 22

# Arm action scale (must match what the arm policy was trained with)
ARM_ACTION_SCALE = 0.5
# Cap arm target changes to realistic joint speed for sim integration.
# 2.5 rad/s at 50 Hz policy rate => 0.05 rad per policy step.
ARM_MAX_JOINT_DELTA_PER_STEP = 0.05

# Goal-sphere colours
_RED_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/FullDemoTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaml_value(keys: list[str], default=None):
    """Traverse the checkpoints.yaml by a list of keys and return the leaf value."""
    if not os.path.isfile(_YAML_PATH):
        return default
    with open(_YAML_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k)
    return node if node is not None else default


def _resolve_checkpoint(cli_val: str | None, yaml_keys: list[str], hardcoded: str) -> str:
    """Return the best checkpoint path from CLI > YAML > hardcoded default."""
    path = cli_val or _yaml_value(yaml_keys, hardcoded)
    if path and not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_YAML_PATH))), path)
    return path


def _default_arm_target(arm_mode: str) -> torch.Tensor:
    """A safe default target in the centre of the workspace for the given arm."""
    if arm_mode in ("left", "both"):
        b = _GOAL_BOUNDS["left"]
    else:
        b = _GOAL_BOUNDS["right"]
    x = (b["x"][0] + b["x"][1]) / 2
    y = (b["y"][0] + b["y"][1]) / 2
    z = (b["z"][0] + b["z"][1]) / 2
    return torch.tensor([x, y, z], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class G1FullDemo:
    def __init__(self):
        # ------ resolve checkpoints ------
        self.standing_ckpt = _resolve_checkpoint(
            args_cli.standing_checkpoint, ["standing", "checkpoint"], _DEFAULT_STANDING
        )
        self.walking_ckpt = _resolve_checkpoint(
            args_cli.walking_checkpoint, ["walking", "checkpoint"], _DEFAULT_WALKING
        )

        # arm_mode: CLI > YAML arm_mode > "left"
        self.arm_mode = args_cli.arm or _yaml_value(["arm_mode"]) or "left"

        arm_ckpt_default = _DEFAULT_ARM_BOTH if self.arm_mode == "both" else _DEFAULT_ARM_LEFT
        self.arm_ckpt = _resolve_checkpoint(
            args_cli.arm_checkpoint,
            ["arm", self.arm_mode, "checkpoint"],
            arm_ckpt_default,
        )

        for name, path in [
            ("standing", self.standing_ckpt),
            ("walking", self.walking_ckpt),
            ("arm", self.arm_ckpt),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} checkpoint not found: {path}")

        print(f"[FullDemo] standing  : {self.standing_ckpt}")
        print(f"[FullDemo] walking   : {self.walking_ckpt}")
        print(f"[FullDemo] arm ({self.arm_mode:5s}): {self.arm_ckpt}")
        print(f"[FullDemo] arm_mode  : {self.arm_mode}")

        # ------ locomotion environment ------
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args_mod.parse_rsl_rl_cfg(
            LOCO_TASK, args_cli
        )
        env_cfg = G1LocomotionFlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None

        loco_env = ManagerBasedRLEnv(cfg=env_cfg)
        self.env = RslRlVecEnvWrapper(loco_env)
        self.device = loco_env.device
        self.robot = loco_env.scene["robot"]

        # ------ identify arm joint indices in the locomotion action vector ------
        # The locomotion env uses joint_names=[".*"] — all joints appear in the action.
        # We keep separate index sets for:
        #   - the arm(s) controlled by the IK policy
        #   - all arm joints, which must be held neutral while standing without a target
        self.arm_joint_ids_in_action, self.arm_joint_names_ordered = \
            self._find_arm_action_indices(include_all_arms=False)
        self.all_arm_joint_ids_in_action, _ = self._find_arm_action_indices(include_all_arms=True)

        # ------ load locomotion policies ------
        self.standing_policy = self._load_loco_policy(agent_cfg, self.standing_ckpt)
        self.walking_policy  = self._load_loco_policy(agent_cfg, self.walking_ckpt)

        # ------ load arm policy (separate runner, arm-shaped env for obs inference) ------
        self.arm_policy, self.arm_body_ids, self.arm_joint_ids_robot = \
            self._load_arm_policy()

        # ------ switch-demo state ------
        self.mode = "standing"
        self.mode_steps = 0
        self.cmd_filtered = torch.zeros(3, device=self.device)
        self._in_transition   = False
        self._transition_from = "standing"
        self._transition_to   = "standing"
        self._transition_step = 0
        self._transition_total_steps = TRANSITION_STEPS_TO_STAND

        # ------ arm target state ------
        # No arm target is active by default. The arm policy only engages after
        # the user explicitly provides a target via --target or the T-key prompt.
        n_arms = 2 if self.arm_mode == "both" else 1
        self.goal_pos_local = torch.zeros(n_arms, 3, device=self.device)
        self.arm_target_active = torch.zeros(n_arms, dtype=torch.bool, device=self.device)

        if args_cli.target is not None:
            init_tgt = torch.tensor(args_cli.target, dtype=torch.float32, device=self.device)
            self.goal_pos_local[0] = init_tgt
            self.arm_target_active[0] = True
            if self.arm_mode == "both":
                self.goal_pos_local[1] = torch.tensor(
                    [init_tgt[0].item(), -init_tgt[1].item(), init_tgt[2].item()],
                    device=self.device,
                )
                self.arm_target_active[1] = True

        # Which arm the T-key prompt edits (only relevant for both mode)
        self._active_arm_idx = 0  # 0=left, 1=right

        # Arm targets are specified in robot-local coordinates, but once placed
        # they stay fixed in world space until the robot respawns.
        self._goal_anchor_pos_w = self.robot.data.root_pos_w[0].clone()
        self._goal_anchor_quat_w = self.robot.data.root_quat_w[0].clone()

        # Goal visualisation
        self._goal_vis = VisualizationMarkers(_RED_SPHERE_CFG)
        self._update_goal_markers()

        # Prompt state for T-key arm targets.
        self._target_prompt_requested = False
        self._target_prompt_active = False

        self._create_camera()
        self._setup_keyboard()

        print("\n[FullDemo] Ready.")
        print("  W/A/D/Q/E  — walk commands    S  — stop")
        print("  T          — type new arm target (blocks simulation briefly)")
        print("  L / R      — switch active arm (only for arm_mode=both)")
        print("  C          — toggle camera\n")

    # ------------------------------------------------------------------ utils

    def _find_arm_action_indices(self, include_all_arms: bool = False) -> tuple[torch.Tensor, list[str]]:
        """Return (action_vector_indices, joint_names) for the arm joints.

        The locomotion action manager sorts joints by the order returned by
        robot.find_joints([".*"]).  We query each arm joint name and record
        its position in that ordering.
        """
        if include_all_arms:
            arm_names = list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS)
        elif self.arm_mode in ("left", "both"):
            arm_names = list(_LEFT_ARM_JOINTS)
        else:
            arm_names = list(_RIGHT_ARM_JOINTS)
        if self.arm_mode == "both" and not include_all_arms:
            arm_names += list(_RIGHT_ARM_JOINTS)

        # Get ALL joints in the same order the action manager uses
        all_joint_ids, all_joint_names = self.robot.find_joints([".*"])
        name_to_action_idx = {name: i for i, name in enumerate(all_joint_names)}

        ordered_names = []
        action_indices = []
        for jname in arm_names:
            if jname in name_to_action_idx:
                action_indices.append(name_to_action_idx[jname])
                ordered_names.append(jname)
            else:
                print(
                    f"[FullDemo] WARNING: arm joint '{jname}' not found in action space — "
                    "arm override will be partial."
                )

        return (
            torch.tensor(action_indices, dtype=torch.long, device=self.device),
            ordered_names,
        )

    def _load_loco_policy(self, agent_cfg: RslRlOnPolicyRunnerCfg, ckpt: str):
        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(ckpt)
        return runner.get_inference_policy(device=self.device)

    def _load_arm_policy(self):
        """Load the arm IK policy and return (policy_fn, ee_body_ids, joint_ids)."""
        # 28-D per arm (9-D base state prefix + 19-D arm state — joint_pos(5)/joint_vel(5)/
        # ee_pos(3)/goal(3)/error(3); see g1_arm_env.py's module docstring); 56-D for
        # "both". Was 26/52 before the 2026-07-08 joint_vel fix (was only 3-D, dropping
        # elbow_pitch/elbow_roll velocity), and 17/34 before Phase 2.
        if self.arm_mode == "left":
            arm_agent_cfg = G1ArmIKLeftPPORunnerCfg()
            obs_dim = 28
            action_dim = 5
        elif self.arm_mode == "right":
            arm_agent_cfg = G1ArmIKRightPPORunnerCfg()
            obs_dim = 28
            action_dim = 5
        else:
            arm_agent_cfg = G1ArmIKBothPPORunnerCfg()
            obs_dim = 56
            action_dim = 10

        arm_obs = TensorDict(
            {"policy": torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)},
            batch_size=[1],
            device=self.device,
        )
        actor_critic = ActorCritic(
            obs=arm_obs,
            obs_groups=arm_agent_cfg.obs_groups,
            num_actions=action_dim,
            actor_obs_normalization=arm_agent_cfg.policy.actor_obs_normalization,
            critic_obs_normalization=arm_agent_cfg.policy.critic_obs_normalization,
            actor_hidden_dims=arm_agent_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=arm_agent_cfg.policy.critic_hidden_dims,
            activation=arm_agent_cfg.policy.activation,
            init_noise_std=arm_agent_cfg.policy.init_noise_std,
        ).to(self.device)

        checkpoint = torch.load(self.arm_ckpt, map_location=self.device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        actor_critic.eval()

        def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            batch = next(iter(obs_dict.values())).shape[0]
            tensordict_obs = TensorDict(obs_dict, batch_size=[batch], device=self.device)
            with torch.inference_mode():
                return actor_critic.act_inference(tensordict_obs)

        # Get arm joint IDs and EE body IDs FROM THE LOCOMOTION ROBOT
        body_ids = []
        joint_ids = []

        if self.arm_mode in ("left", "both"):
            l_joints, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            l_bodies, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            joint_ids.extend(l_joints)
            body_ids.append(l_bodies[0])

        if self.arm_mode in ("right", "both"):
            r_joints, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_bodies, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            joint_ids.extend(r_joints)
            body_ids.append(r_bodies[0])

        return (
            policy,
            body_ids,
            torch.tensor(joint_ids, dtype=torch.long, device=self.device),
        )

    # ------------------------------------------------------------------ arm obs

    def _build_arm_obs(self) -> torch.Tensor:
        """Construct the 28-D (or 56-D) arm observation from the locomotion scene.

        Base-state prefix (base_lin_vel, base_ang_vel, projected_gravity — added Phase 2,
        see g1_arm_env.py) reads the *real* locomotion robot's current state here, unlike
        training's synthetic wobble signal (g1_arm_env.py's root is physically fixed
        during arm-only training, so it fakes this) — here the base is actually the real,
        moving standing/walking robot, so the real sensor values are the more faithful
        choice, not an approximation of one.
        """
        parts = []
        goal_world = self._goal_positions_world()

        base_lin_vel = self.robot.data.root_lin_vel_b[0].unsqueeze(0)   # (1, 3)
        base_ang_vel = self.robot.data.root_ang_vel_b[0].unsqueeze(0)   # (1, 3)
        projected_gravity = self.robot.data.projected_gravity_b[0].unsqueeze(0)  # (1, 3)

        arm_groups = []
        if self.arm_mode in ("left", "both"):
            l_ids, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            l_body, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            arm_groups.append({
                "joint_ids": torch.tensor(l_ids, device=self.device),
                "ee_idx": l_body[0],
                "goal": goal_world[0],
            })
        if self.arm_mode in ("right", "both"):
            r_ids, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_body, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            goal_idx = 1 if self.arm_mode == "both" else 0
            arm_groups.append({
                "joint_ids": torch.tensor(r_ids, device=self.device),
                "ee_idx": r_body[0],
                "goal": goal_world[goal_idx],
            })

        for g in arm_groups:
            jids = g["joint_ids"]
            joint_pos = self.robot.data.joint_pos[0, jids].unsqueeze(0)    # (1, 5)
            joint_vel = self.robot.data.joint_vel[0, jids].unsqueeze(0)    # (1, 5)
            ee_pos    = self.robot.data.body_pos_w[0, g["ee_idx"], :].unsqueeze(0)  # (1, 3)
            goal      = g["goal"].unsqueeze(0)                               # (1, 3)
            error     = goal - ee_pos                                        # (1, 3)
            # Base-state prefix duplicated per arm block, matching g1_arm_env.py's
            # layout (same convention "both" mode needs there too, for the mirror
            # transform to work block-by-block).
            parts.extend([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error])

        return torch.cat(parts, dim=-1)  # (1, 28) or (1, 56)

    # ------------------------------------------------------------------ arm delta → action override

    def _arm_action_override(self, loco_action: torch.Tensor) -> torch.Tensor:
        """Run arm policy and substitute its output into the locomotion action."""
        arm_obs = self._build_arm_obs()
        arm_delta = self.arm_policy({"policy": arm_obs})  # (1, 5) or (1, 10)

        # Compute new arm joint targets from current position + delta
        current = self.robot.data.joint_pos[0, self.arm_joint_ids_robot]  # (n_arm_joints,)
        limits  = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_ids_robot]  # (n, 2)
        delta = (arm_delta.squeeze(0) * ARM_ACTION_SCALE).clamp(
            -ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP
        )
        new_targets = (current + delta).clamp(
            limits[:, 0], limits[:, 1]
        )

        # Convert absolute targets → action-space offset:
        #   loco_action = (target - default_pos) / scale
        default = self.robot.data.default_joint_pos[0, self.arm_joint_ids_robot]
        action_delta = (new_targets - default) / ARM_ACTION_SCALE  # (n_arm_joints,)

        # Substitute into the locomotion action vector
        modified = loco_action.clone()
        modified[0, self.arm_joint_ids_in_action] = action_delta

        return modified

    def _hold_arms_at_default(self, loco_action: torch.Tensor) -> torch.Tensor:
        """Zero the arm slice so the locomotion action holds the default arm posture."""
        modified = loco_action.clone()
        modified[0, self.all_arm_joint_ids_in_action] = 0.0
        return modified

    def _has_active_arm_target(self) -> bool:
        if self.arm_mode == "both":
            return bool(torch.all(self.arm_target_active).item())
        return bool(self.arm_target_active[0].item())

    def _clear_arm_targets(self):
        if not bool(torch.any(self.arm_target_active).item()):
            return
        self.arm_target_active.zero_()
        self._update_goal_markers()
        print("[FullDemo] Cleared arm target(s).")

    def _goal_positions_world(self) -> torch.Tensor:
        """Transform anchored robot-local arm targets into world coordinates."""
        world_goals = []
        for target_local in self.goal_pos_local:
            world_goals.append(
                quat_apply(self._goal_anchor_quat_w, target_local) + self._goal_anchor_pos_w
            )
        return torch.stack(world_goals, dim=0)

    def _anchor_targets_to_current_base(self):
        self._goal_anchor_pos_w = self.robot.data.root_pos_w[0].clone()
        self._goal_anchor_quat_w = self.robot.data.root_quat_w[0].clone()
        self._update_goal_markers()

    def _handle_env_resets(self, dones: torch.Tensor):
        if not bool(torch.any(dones > 0).item()):
            return

        self.mode = "standing"
        self.mode_steps = 0
        self._in_transition = False
        self._transition_from = "standing"
        self._transition_to = "standing"
        self._transition_step = 0
        self.cmd_filtered.zero_()

        if self._has_active_arm_target():
            self._anchor_targets_to_current_base()
            print("[FullDemo] Respawn detected. Re-anchored arm target to robot base.")

    # ------------------------------------------------------------------ locomotion mode

    def _start_transition(self, new_mode: str, force_from: str | None = None, reason: str = "switch"):
        if new_mode == "walking":
            self._clear_arm_targets()
        self._in_transition = True
        self._transition_from = force_from if force_from is not None else self.mode
        self._transition_to   = new_mode
        self._transition_step = 0
        self._transition_total_steps = (
            TRANSITION_STEPS_TO_WALK if new_mode == "walking" else TRANSITION_STEPS_TO_STAND
        )
        self.mode = new_mode
        self.mode_steps = 0
        print(f"[FullDemo] transition {self._transition_from} → {self._transition_to} (reason={reason})")

    def _robot_planar_speed(self) -> float:
        vel = self.env.unwrapped.scene["robot"].data.root_lin_vel_w[0, :2]
        return torch.linalg.vector_norm(vel).item()

    def _maybe_switch_mode(self, cmd_mag: float):
        speed = self._robot_planar_speed()

        # As soon as the user starts commanding locomotion, drop any active arm
        # target so the standing arm override cannot keep fighting the walk handoff.
        if self.mode == "standing" and self._has_active_arm_target() and cmd_mag >= 0.02:
            self._clear_arm_targets()

        if self._in_transition:
            if self._transition_to == "standing" and cmd_mag >= SWITCH_TO_WALK_THRESHOLD:
                self._start_transition("walking", force_from="standing", reason="interrupt")
            return

        if self.mode == "standing":
            if self.mode_steps >= MIN_MODE_STEPS and cmd_mag >= SWITCH_TO_WALK_THRESHOLD:
                self._start_transition("walking", reason="threshold")
        else:
            if (
                self.mode_steps >= MIN_MODE_STEPS
                and cmd_mag <= SWITCH_TO_STAND_THRESHOLD
                and speed <= SWITCH_TO_STAND_SPEED_THRESHOLD
            ):
                self._start_transition("standing", reason="threshold")

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select locomotion action (blend during transition); overlay arm if standing."""
        if not self._in_transition:
            loco_action = (
                self.walking_policy(obs)
                if self.mode == "walking"
                else self.standing_policy(obs)
            )
        else:
            alpha = min((self._transition_step + 1) / float(self._transition_total_steps), 1.0)
            from_pol = self.walking_policy if self._transition_from == "walking" else self.standing_policy
            to_pol   = self.walking_policy if self._transition_to   == "walking" else self.standing_policy
            loco_action = (1.0 - alpha) * from_pol(obs) + alpha * to_pol(obs)

            self._transition_step += 1
            if self._transition_step >= self._transition_total_steps:
                self._in_transition = False
                print(f"[FullDemo] mode settled: {self.mode}")

        # Overlay arm policy when standing (or transitioning to stand)
        arm_active = self._has_active_arm_target() and (self.mode == "standing") and (
            not self._in_transition or self._transition_to == "standing"
        )
        if arm_active:
            loco_action = self._arm_action_override(loco_action)
        elif self.mode == "standing" or (self._in_transition and self._transition_to == "standing"):
            loco_action = self._hold_arms_at_default(loco_action)

        return loco_action

    # ------------------------------------------------------------------ goal markers

    def _update_goal_markers(self):
        world_goals = self._goal_positions_world()
        inactive = torch.logical_not(self.arm_target_active)
        if bool(torch.any(inactive).item()):
            world_goals = world_goals.clone()
            world_goals[inactive, 2] = -10.0
        self._goal_vis.visualize(world_goals)

    def _set_arm_target(self, arm_idx: int, target_local: torch.Tensor):
        self._anchor_targets_to_current_base()
        self.goal_pos_local[arm_idx] = target_local
        self.arm_target_active[arm_idx] = True

        # Reset just this arm's joints to their default pose before pursuing the new
        # target. Nothing else here ever resets the arm between targets, so without
        # this it just keeps going from wherever it physically ended up chasing the
        # *previous* one. If that previous target was out of range or poorly reached,
        # the arm could be sitting in an extreme, never-seen-in-training pose — and the
        # policy would try to reach the *new* target starting from that bad state,
        # producing motion that looks broken but has nothing to do with whether the new
        # target itself is reasonable. Only resets the targeted arm (relevant for
        # arm_mode="both"), not the other one mid-reach.
        n_per_arm = len(self.arm_joint_ids_robot) // (2 if self.arm_mode == "both" else 1)
        start = arm_idx * n_per_arm if self.arm_mode == "both" else 0
        joint_ids = self.arm_joint_ids_robot[start:start + n_per_arm]
        default_pos = self.robot.data.default_joint_pos[0:1, joint_ids]
        zero_vel = torch.zeros_like(default_pos)
        self.robot.write_joint_state_to_sim(
            default_pos, zero_vel,
            joint_ids=joint_ids, env_ids=torch.tensor([0], device=self.device),
        )

        self._update_goal_markers()
        arm_label = ["left", "right"][arm_idx] if self.arm_mode == "both" else self.arm_mode
        print(f"[FullDemo] Arm target ({arm_label}): {target_local.tolist()}")

    # ------------------------------------------------------------------ keyboard / target input

    def _prompt_target(self):
        """Request a blocking arm-target prompt on the main loop."""
        if self._target_prompt_active:
            print("[FullDemo] Target prompt already open.")
            return

        self._target_prompt_requested = True

    def _run_target_prompt(self):
        """Blocking console prompt for a new arm target on the main thread."""
        self._target_prompt_requested = False
        self._target_prompt_active = True

        arm_label = (
            ["left", "right"][self._active_arm_idx]
            if self.arm_mode == "both"
            else self.arm_mode
        )
        b = _GOAL_BOUNDS["left" if arm_label == "left" else "right"]
        prompt = (
            f"\n[FullDemo] New target for {arm_label} arm (x y z)\n"
            f"  x: {b['x']}, y: {b['y']}, z: {b['z']}\n"
            "  > "
        )
        while True:
            try:
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self._target_prompt_active = False
                return
            if not line:
                self._target_prompt_active = False
                return
            parts = line.split()
            if len(parts) != 3:
                print("  Need exactly 3 numbers: x y z")
                continue
            try:
                target = torch.tensor(
                    [float(p) for p in parts],
                    dtype=torch.float32,
                    device=self.device,
                )
                out_of_range = not (
                    b["x"][0] <= target[0].item() <= b["x"][1]
                    and b["y"][0] <= target[1].item() <= b["y"][1]
                    and b["z"][0] <= target[2].item() <= b["z"][1]
                )
                if out_of_range:
                    # Not just "harder" — the policy has never trained on anything
                    # outside this box, so it can produce genuinely erratic motion
                    # trying to reach one, not just fall a bit short.
                    confirm = input(
                        "  Outside the trained range — the policy has never seen this "
                        "and may behave strangely. Send anyway? [y/N] "
                    ).strip().lower()
                    if confirm != "y":
                        continue
                self._set_arm_target(self._active_arm_idx, target)
                self._target_prompt_active = False
                return
            except ValueError:
                print(f"  Could not parse: {line!r}")

    def _apply_pending_target(self):
        if not self._target_prompt_requested:
            return

        self._run_target_prompt()

    def _setup_keyboard(self):
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        import numpy as np

        kb_cfg = Se2KeyboardCfg(sim_device=str(self.device))
        self._keyboard = Se2Keyboard(kb_cfg)
        self._keyboard._INPUT_KEY_MAPPING = {
            "W": np.array([LIN_VEL,  0.0,     0.0]),
            "A": np.array([0.0,      0.0,  ANG_VEL]),
            "D": np.array([0.0,      0.0, -ANG_VEL]),
            "Q": np.array([0.0,  LIN_VEL,     0.0]),
            "E": np.array([0.0, -LIN_VEL,     0.0]),
        }
        self._keyboard.add_callback("S", self._keyboard.reset)
        self._keyboard.add_callback("C", self._toggle_camera)
        self._keyboard.add_callback("T", self._prompt_target)
        if self.arm_mode == "both":
            self._keyboard.add_callback("L", lambda: self._select_arm(0))
            self._keyboard.add_callback("R", lambda: self._select_arm(1))

    def _select_arm(self, idx: int):
        self._active_arm_idx = idx
        label = ["left", "right"][idx]
        print(f"[FullDemo] Active arm for T-key targeting: {label}")

    # ------------------------------------------------------------------ camera

    def _create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"

        cam = stage.DefinePrim(self.camera_path, "Camera")
        cam.GetAttribute("focalLength").Set(8.5)
        coi = cam.GetProperty("omni:kit:centerOfInterest")
        if not coi or not coi.IsValid():
            cam.CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.camera_path)

    def _toggle_camera(self):
        active = self.viewport.get_active_camera()
        self.viewport.set_active_camera(
            self.perspective_path if active == self.camera_path else self.camera_path
        )

    def update_camera(self):
        base_pos  = self.robot.data.root_pos_w[0]
        base_quat = self.robot.data.root_quat_w[0]
        offset = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        cam_pos = quat_apply(base_quat, offset) + base_pos

        state = ViewportCameraState(self.camera_path, self.viewport)
        state.set_position_world(
            Gf.Vec3d(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item()), True
        )
        state.set_target_world(
            Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6), True
        )


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def main():
    demo = G1FullDemo()
    obs, _ = demo.env.reset()
    step = 0

    while simulation_app.is_running():
        demo.update_camera()
        demo._apply_pending_target()

        with torch.inference_mode():
            cmd_raw = demo._keyboard.advance()
            demo.cmd_filtered = (
                (1.0 - CMD_FILTER_ALPHA) * demo.cmd_filtered
                + CMD_FILTER_ALPHA * cmd_raw
            )
            obs[:, 9:12] = demo.cmd_filtered.unsqueeze(0)

            cmd_mag = torch.linalg.vector_norm(demo.cmd_filtered).item()
            demo._maybe_switch_mode(cmd_mag)

            action = demo.select_action(obs)
            obs, _, dones, _ = demo.env.step(action)
            demo._handle_env_resets(dones)

        if step % 60 == 0:
            # Distance to arm target(s) for display
            target_world = demo._goal_positions_world()
            arm_groups = []
            if demo.arm_mode in ("left", "both"):
                arm_groups.append(("left",  0, demo.arm_body_ids[0]))
            if demo.arm_mode in ("right", "both"):
                arm_idx = 1 if demo.arm_mode == "both" else 0
                arm_groups.append(("right", arm_idx, demo.arm_body_ids[-1]))

            dist_str = ""
            for label, gi, body_id in arm_groups:
                ee  = demo.robot.data.body_pos_w[0, body_id, :]
                tgt = target_world[gi]
                d   = (tgt - ee).norm().item() * 100
                dist_str += f"  {label}: {d:.1f}cm"

            arm_active = demo._has_active_arm_target() and (demo.mode == "standing")

            print(
                f"[FullDemo] step={step:6d}  mode={demo.mode:8s}  "
                f"|cmd|={cmd_mag:.3f}  arm_active={arm_active}"
                + dist_str
            )

        demo.mode_steps += 1
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
