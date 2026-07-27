"""
G1 full integrated demo — unified stand+walk policy + arm control.

Rewritten 2026-07-21 for the 29dof pivot — replaces the 23dof-era version that used to
live at this exact path (preserved on the `23_dof` git branch) — see
`29dof_implementation_plan.md` Phase 4. Real architecture changes, not a line-by-line
port:

1. **One unified locomotion policy**, not separate standing/walking checkpoints — the
   29dof recipe covers both regimes with one policy (near-zero commanded velocity IS
   standing). This removes the ENTIRE mode-switch state machine the 23dof-era version
   needed (MIN_MODE_STEPS, SWITCH_TO_WALK/STAND_THRESHOLD, transition blending between
   two policies) — WASD now just feeds the velocity command straight to the one policy.
2. **7-DOF arm** (was 5-DOF) — joint lists/EE body names/observation width updated.
3. **Arm reaching is no longer gated to "standing only."** The 23dof-era version only
   allowed an active arm target while `mode == "standing"`, because the arm-disturbance
   training curriculum back then only ever ran during the dedicated standing task. The
   new `ArmMotionDisturbance` curriculum (mdp/events.py) doesn't distinguish — it
   injects arm motion regardless of commanded velocity — so gating this demo the same
   way would be testing a narrower condition than what training actually covers. Set a
   target any time; whether that's a good idea while walking fast is exactly what
   `walk_stop_reach` in the integration eval is starting to measure.
4. **`ArmDisturbanceBlendJointPositionAction` reused as-is** for the arm overlay — same
   mechanism the training curriculum uses (`env._arm_motion_targets`/
   `_arm_motion_joint_ids`), not a new action term.

Kept essentially unchanged (proven, still correct): the goal-frame convention (fixed
offset from the torso, recomputed from the live pose every call — never a world-fixed
point, see `_goal_positions_world`'s docstring for why that matters), the z-height
convention (ground-referenced, not root-referenced), the homing-between-targets
behavior, the mirror-testing mode (Y/U keys), and camera control (C/V).

Behaviour:
- Arms actively reach their current target(s), if any, at all times — no mode gating.
- WASD/QE command the locomotion policy directly (forward/strafe/turn).
- Press T to type a new arm target at the console.
- Press L / R to select which arm to address when arm_mode=both (default: left).
- Press Y (arm_mode=left only) to type a target for the RIGHT arm, driven by mirroring
  the left-trained policy (see mdp/symmetry.py) — no separate right-arm checkpoint
  needed. Shown as a blue marker, vs. the native target's red.
- Press U (arm_mode=left only) to type two targets, one per arm (left native, right
  mirrored) — tests whether the mirror-generalized policy holds up driving both arms
  simultaneously.
- Press C to toggle camera follow off/on (off lets you orbit freely with the mouse).
- Press V to reset the camera to the default chase view (and re-enable follow).

Arm target format (robot-local frame, base at origin — see g1_arm_env.py's
_GOAL_BOUNDS, the authoritative source; these numbers are just a quick reference):
    x  forward     reachable 0.20 – 0.42 m
    y  lateral     left arm: 0.08 – 0.40 m   right arm: -0.40 – -0.08 m
    z  height      reachable 0.9 – 1.15 m  (ground = 0)

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Auto-load from testing/general_testing/checkpoints.yaml
    python3 testing/visual_testing/full_demo/g1_full_demo_legacy32.py

    # Explicit checkpoints + initial arm target
    python3 testing/visual_testing/full_demo/g1_full_demo_legacy32.py \\
        --loco_checkpoint logs/rsl_rl/walking/base/.../model_5999.pt \\
        --arm_checkpoint  logs/rsl_rl/arms/left/.../model_5999.pt \\
        --arm left \\
        --target 0.3 0.2 1.0

    # Both arms
    python3 testing/visual_testing/full_demo/g1_full_demo_legacy32.py \\
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
_DEFAULT_LOCO = "logs/rsl_rl/walking/base/CHANGEME/model_CHANGEME.pt"
_DEFAULT_ARM_LEFT = "logs/rsl_rl/arms/left/CHANGEME/model_CHANGEME.pt"
_DEFAULT_ARM_BOTH = "logs/rsl_rl/arms/both/CHANGEME/model_CHANGEME.pt"

parser = argparse.ArgumentParser(description="G1 full integrated demo: unified stand+walk policy + arm control.")
cli_args_mod.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--loco_checkpoint", type=str, default=None,
                    help="Checkpoint for the unified stand+walk policy (overrides YAML).")
parser.add_argument("--arm_checkpoint", type=str, default=None,
                    help="Checkpoint for the arm policy (overrides YAML).")
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
import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmBothPPORunnerCfg,
    G1ArmLeftPPORunnerCfg,
    G1ArmRightPPORunnerCfg,
)
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_ARM_JOINTS,
    _RIGHT_EE_BODY,
)
from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import (
    mirror_arm_actions,
    mirror_arm_obs,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp import ArmDisturbanceBlendJointPositionAction
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# ---------------------------------------------------------------------------
# Demo constants
# ---------------------------------------------------------------------------
LOCO_TASK  = "G1-Locomotion-Velocity-Play-v0"
LIN_VEL    = 1.0
ANG_VEL    = 0.5
CMD_FILTER_ALPHA = 0.12

# Arm action conversion — must match what the arm policy was trained with
# (g1_arm_env.py's action_scale/max_action_delta_per_step/action_filter_alpha).
ARM_ACTION_SCALE = 0.5
ARM_MAX_JOINT_DELTA_PER_STEP = 0.06
ARM_ACTION_FILTER_ALPHA = 0.25
# When a new target is set, the arm first *homes* back to its default pose through the
# same clamped-delta path the reach targets use, then hands over to the arm policy —
# avoids starting a fresh reach from a possibly-extreme previous pose (a nonphysical
# teleport would be a CoM shock nothing in training produces). Homing ends when every
# joint is within this tolerance of default.
ARM_HOMING_TOL_RAD = 0.05

# Arm PD gain: matches g1_arm_env.py's training gain while actively reach-driven, the
# loco env's own stock gain otherwise (matches
# g1_locomotion.assets.robots.unitree.UNITREE_G1_29DOF_CFG's N5020-16/W4010-25 groups —
# 40/10, uniform across all 7 arm joints). Only two gains now, not three: the 23dof-era
# version also had a separate standing-vs-walking split because those were different
# policies with different trained arm gains; one unified policy means one held gain.
# FIXED 2026-07-23: g1_arm_env.py's own training gain was changed 2026-07-22 from
# 200/20 (an unverified 23dof-era experimental value) to 40/10 (the real hardware gain
# — deliberately made equal to the held gain below, since that fix's whole point was to
# stop training against a softer gain than deployment will ever provide). This constant
# was left at the stale 200/20 value until now — this demo's "active reach" gain never
# actually matched what the checkpoint was trained under.
_GAIN_ARM_ACTIVE = (40.0, 10.0)
_GAIN_ARM_HELD = (40.0, 10.0)

# Goal-sphere colours. Red = native (left-trained-policy-driven) target. Blue = the
# mirror-testing target.
_RED_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/FullDemoTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)
_BLUE_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/FullDemoMirrorTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 1.0)),
        )
    },
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaml_value(keys: list[str], default=None):
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
    path = cli_val or _yaml_value(yaml_keys, hardcoded)
    if path and not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_YAML_PATH)))), path)
    return path


def _default_arm_target(arm_mode: str) -> torch.Tensor:
    b = _GOAL_BOUNDS["left"] if arm_mode in ("left", "both") else _GOAL_BOUNDS["right"]
    x = (b["x"][0] + b["x"][1]) / 2
    y = (b["y"][0] + b["y"][1]) / 2
    z = (b["z"][0] + b["z"][1]) / 2
    return torch.tensor([x, y, z], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class G1FullDemo:
    def __init__(self):
        self.loco_ckpt = _resolve_checkpoint(
            args_cli.loco_checkpoint, ["loco", "checkpoint"], _DEFAULT_LOCO
        )
        self.arm_mode = args_cli.arm or _yaml_value(["arm_mode"]) or "left"
        arm_ckpt_default = _DEFAULT_ARM_BOTH if self.arm_mode == "both" else _DEFAULT_ARM_LEFT
        self.arm_ckpt = _resolve_checkpoint(
            args_cli.arm_checkpoint, ["arm", self.arm_mode, "checkpoint"], arm_ckpt_default,
        )

        for name, path in [("loco", self.loco_ckpt), ("arm", self.arm_ckpt)]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} checkpoint not found: {path}")

        print(f"[FullDemo] loco     : {self.loco_ckpt}")
        print(f"[FullDemo] arm ({self.arm_mode:5s}): {self.arm_ckpt}")

        # ------ locomotion environment ------
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args_mod.parse_rsl_rl_cfg(LOCO_TASK, args_cli)
        env_cfg = G1LocomotionEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None
        env_cfg.observations.policy.enable_corruption = True
        # Never auto-resample the velocity command — this demo drives it entirely from
        # the keyboard every step (see _command_term below); a periodic random resample
        # would otherwise transiently clobber the keyboard-driven value once every
        # resampling_time_range.
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        # ArmDisturbanceBlendJointPositionAction overrides only the *simulated* arm
        # target (env._arm_motion_targets/_arm_motion_joint_ids), leaving the policy's
        # own raw output flowing into last_action untouched — exactly matching
        # training. Same action term the training curriculum uses, reused unchanged.
        if hasattr(env_cfg.actions, "JointPositionAction"):
            env_cfg.actions.JointPositionAction.class_type = ArmDisturbanceBlendJointPositionAction

        loco_env = ManagerBasedRLEnv(cfg=env_cfg)
        self.env = RslRlVecEnvWrapper(loco_env)
        self.device = loco_env.device
        self.robot = loco_env.scene["robot"]
        # Ground-level world z for this env slot — arm target z is "height above
        # ground" (matching g1_arm_env.py's _GOAL_BOUNDS convention), not "height above
        # the root link" (the root sits ~0.75-0.8m off the ground for a standing G1).
        self._ground_z_w = loco_env.scene.env_origins[0, 2].clone()

        all_arm_ids, _ = self.robot.find_joints(list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS))
        self.all_arm_joint_ids_robot = torch.tensor(all_arm_ids, dtype=torch.long, device=self.device)

        self.loco_policy = self._load_loco_policy(agent_cfg, self.loco_ckpt)
        self.arm_policy, self.arm_body_ids, self.arm_joint_ids_robot = self._load_arm_policy()
        self._active_arm_cols = self._cols_in_all_arm(self.arm_joint_ids_robot)

        # ------ command state ------
        # Write the keyboard-driven command straight into the command term's own live
        # buffer (vel_command_b), NOT the flat observation tensor directly — the
        # policy's observation group uses history_length=5 (every term, including
        # velocity_commands, is stacked over the last 5 control steps and flattened;
        # see mdp/symmetry.py's identical note), so "the velocity_commands columns" are
        # not a fixed 3-wide slice of the flat obs the way they were in the 23dof-era
        # single-step layout — hand-patching the wrong slice there would silently feed
        # the policy a garbled command. Writing vel_command_b instead lets the normal
        # observation pipeline (mdp.generated_commands -> the history buffer) pick up
        # the new value through its own correct mechanism; effect is visible from the
        # NEXT control step (~20ms latency at 50Hz), imperceptible for keyboard control.
        self._command_term = loco_env.command_manager.get_term("base_velocity")
        self.cmd_filtered = torch.zeros(3, device=self.device)

        # ------ arm target state ------
        n_arms = 2 if self.arm_mode == "both" else 1
        self.goal_pos_local = torch.zeros(n_arms, 3, device=self.device)
        self.arm_target_active = torch.zeros(n_arms, dtype=torch.bool, device=self.device)
        n_joints_per_arm = len(_LEFT_ARM_JOINTS)
        self._filtered_arm_delta = torch.zeros(1, n_joints_per_arm * n_arms, device=self.device)
        self._arm_homing = torch.zeros(n_arms, dtype=torch.bool, device=self.device)

        if args_cli.target is not None:
            init_tgt = torch.tensor(args_cli.target, dtype=torch.float32, device=self.device)
            self.goal_pos_local[0] = init_tgt
            self.arm_target_active[0] = True
            if self.arm_mode == "both":
                self.goal_pos_local[1] = torch.tensor(
                    [init_tgt[0].item(), -init_tgt[1].item(), init_tgt[2].item()], device=self.device,
                )
                self.arm_target_active[1] = True

        self._active_arm_idx = 0  # 0=left, 1=right — which arm the T-key prompt edits

        self._goal_vis = VisualizationMarkers(_RED_SPHERE_CFG)
        self._update_goal_markers()

        # ------ mirror-testing state (right arm, driven by mirroring the LEFT-trained
        # policy). Only meaningful when arm_mode="left". ------
        self._mirror_enabled = self.arm_mode == "left"
        self._mirror_homing = False
        if self._mirror_enabled:
            r_joints, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_bodies, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            self._mirror_joint_ids_robot = torch.tensor(r_joints, dtype=torch.long, device=self.device)
            self._mirror_ee_body_id = r_bodies[0]
            self._mirror_arm_cols = self._cols_in_all_arm(self._mirror_joint_ids_robot)

            self.mirror_goal_pos_local = torch.zeros(3, device=self.device)
            self.mirror_target_active = False
            self._filtered_mirror_delta = torch.zeros(1, n_joints_per_arm, device=self.device)

            self._mirror_goal_vis = VisualizationMarkers(_BLUE_SPHERE_CFG)
            self._update_mirror_goal_marker()

        self._target_prompt_requested = False
        self._target_prompt_active = False
        self._pending_prompt_kind = "left"

        self._create_camera()
        self._setup_keyboard()

        print("\n[FullDemo] Ready.")
        print("  W/A/D/Q/E  — velocity command    S  — stop")
        print("  T          — type new arm target (blocks simulation briefly)")
        print("  L / R      — switch active arm (only for arm_mode=both)")
        if self._mirror_enabled:
            print("  Y          — type a target for the right arm (mirrored via the left policy)")
            print("  U          — type two targets, one per arm (left native, right mirrored)")
        print("  C          — toggle camera follow (off = orbit freely with the mouse)")
        print("  V          — reset camera to the default chase view\n")

    # ------------------------------------------------------------------ utils

    def _cols_in_all_arm(self, joint_ids: torch.Tensor) -> torch.Tensor:
        id_to_col = {int(j): i for i, j in enumerate(self.all_arm_joint_ids_robot.tolist())}
        cols = [id_to_col[int(j)] for j in joint_ids.tolist()]
        return torch.tensor(cols, dtype=torch.long, device=self.device)

    def _load_loco_policy(self, agent_cfg: RslRlOnPolicyRunnerCfg, ckpt: str):
        """Network shape (including the critic's) and obs-normalization flags are all
        inferred directly from the checkpoint's own state_dict, not trusted from
        agent_cfg — 2026-07-22 bug fix. This task's CriticCfg is genuinely wider than
        PolicyCfg (495 vs 480 columns: it adds a privileged base_lin_vel term the actor
        never sees — see g1_locomotion_env_cfg.py), so the old fallback
        ``obs_groups = {"policy": [...], "critic": ["policy"]}`` silently built a critic
        sized for the actor's (narrower) input, causing a load_state_dict shape
        mismatch on critic.0.weight. agent_cfg.policy.actor_obs_normalization/
        critic_obs_normalization also didn't reliably match what the checkpoint was
        actually trained with (BasePPORunnerCfg never sets them explicitly, unlike the
        arm task's PPO cfg) — checking for the normalizer buffers' presence in the
        checkpoint directly sidesteps needing to know why that mismatch happens."""
        state = torch.load(ckpt, map_location=self.device)["model_state_dict"]
        in_dim = state["actor.0.weight"].shape[1]
        critic_in_dim = state["critic.0.weight"].shape[1]
        num_actions = state[f"actor.{2 * len(agent_cfg.policy.actor_hidden_dims)}.weight"].shape[0]
        has_actor_norm = "actor_obs_normalizer._mean" in state
        has_critic_norm = "critic_obs_normalizer._mean" in state

        dummy_obs = TensorDict(
            {
                "policy": torch.zeros((1, in_dim), dtype=torch.float32, device=self.device),
                "critic": torch.zeros((1, critic_in_dim), dtype=torch.float32, device=self.device),
            },
            batch_size=[1], device=self.device,
        )
        actor_critic = ActorCritic(
            obs=dummy_obs,
            obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=num_actions,
            actor_obs_normalization=has_actor_norm,
            critic_obs_normalization=has_critic_norm,
            actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
            activation=agent_cfg.policy.activation,
            init_noise_std=agent_cfg.policy.init_noise_std,
        ).to(self.device)
        actor_critic.load_state_dict(state)
        actor_critic.eval()

        def policy(obs) -> torch.Tensor:
            obs_tensor = obs["policy"] if isinstance(obs, TensorDict) else obs
            td = TensorDict({"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]], device=self.device)
            with torch.inference_mode():
                return actor_critic.act_inference(td)

        return policy

    def _load_arm_policy(self):
        n_joints_per_arm = len(_LEFT_ARM_JOINTS)
        if self.arm_mode == "left":
            arm_agent_cfg = G1ArmLeftPPORunnerCfg()
            obs_dim, action_dim = 32, n_joints_per_arm
        elif self.arm_mode == "right":
            arm_agent_cfg = G1ArmRightPPORunnerCfg()
            obs_dim, action_dim = 32, n_joints_per_arm
        else:
            arm_agent_cfg = G1ArmBothPPORunnerCfg()
            obs_dim, action_dim = 64, n_joints_per_arm * 2

        arm_obs = TensorDict(
            {"policy": torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)},
            batch_size=[1], device=self.device,
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

        return policy, body_ids, torch.tensor(joint_ids, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ arm obs

    def _build_arm_obs(self) -> torch.Tensor:
        """32-D (or 64-D) arm observation from the locomotion scene — base-state prefix
        reads the *real* robot's current state (unlike training's synthetic wobble
        signal, since here the base is actually moving)."""
        parts = []
        goal_world = self._goal_positions_world()

        base_lin_vel = self.robot.data.root_lin_vel_b[0].unsqueeze(0)
        base_ang_vel = self.robot.data.root_ang_vel_b[0].unsqueeze(0)
        projected_gravity = self.robot.data.projected_gravity_b[0].unsqueeze(0)

        arm_groups = []
        if self.arm_mode in ("left", "both"):
            l_ids, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            l_body, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            arm_groups.append({"joint_ids": torch.tensor(l_ids, device=self.device), "ee_idx": l_body[0], "goal": goal_world[0]})
        if self.arm_mode in ("right", "both"):
            r_ids, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_body, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            goal_idx = 1 if self.arm_mode == "both" else 0
            arm_groups.append({"joint_ids": torch.tensor(r_ids, device=self.device), "ee_idx": r_body[0], "goal": goal_world[goal_idx]})

        root_quat = self.robot.data.root_quat_w[0].unsqueeze(0)
        for g in arm_groups:
            jids = g["joint_ids"]
            joint_pos = self.robot.data.joint_pos[0, jids].unsqueeze(0)
            joint_vel = self.robot.data.joint_vel[0, jids].unsqueeze(0)
            ee_pos = self.robot.data.body_pos_w[0, g["ee_idx"], :].unsqueeze(0)
            goal = g["goal"].unsqueeze(0)
            # error must be in the robot's current body frame, not raw world — training
            # runs with a physically fixed, never-rotating root, so world frame is body
            # frame throughout training; here the robot can actually turn.
            error = quat_apply_inverse(root_quat, goal - ee_pos)
            parts.extend([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error])

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ arm delta -> simulated target

    def _compute_arm_targets(self) -> torch.Tensor:
        arm_obs = self._build_arm_obs()
        arm_delta = self.arm_policy({"policy": arm_obs})

        self._filtered_arm_delta = (
            ARM_ACTION_FILTER_ALPHA * arm_delta + (1.0 - ARM_ACTION_FILTER_ALPHA) * self._filtered_arm_delta
        )
        arm_delta = self._filtered_arm_delta

        current = self.robot.data.joint_pos[0, self.arm_joint_ids_robot]
        limits = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_ids_robot]
        delta = (arm_delta.squeeze(0) * ARM_ACTION_SCALE).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
        new_targets = (current + delta).clamp(limits[:, 0], limits[:, 1])
        return new_targets.unsqueeze(0)

    # ------------------------------------------------------------------ mirror testing

    def _build_mirror_source_obs(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b[0].unsqueeze(0)
        base_ang_vel = self.robot.data.root_ang_vel_b[0].unsqueeze(0)
        projected_gravity = self.robot.data.projected_gravity_b[0].unsqueeze(0)

        joint_pos = self.robot.data.joint_pos[0, self._mirror_joint_ids_robot].unsqueeze(0)
        joint_vel = self.robot.data.joint_vel[0, self._mirror_joint_ids_robot].unsqueeze(0)
        ee_pos = self.robot.data.body_pos_w[0, self._mirror_ee_body_id, :].unsqueeze(0)
        goal = self._mirror_goal_world().unsqueeze(0)
        root_quat = self.robot.data.root_quat_w[0].unsqueeze(0)
        error = quat_apply_inverse(root_quat, goal - ee_pos)

        return torch.cat([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error], dim=-1)

    def _compute_mirror_targets(self) -> torch.Tensor:
        raw_obs = self._build_mirror_source_obs()
        mirrored_obs = mirror_arm_obs(raw_obs)
        delta_left_frame = self.arm_policy({"policy": mirrored_obs})
        delta = mirror_arm_actions(delta_left_frame)

        self._filtered_mirror_delta = (
            ARM_ACTION_FILTER_ALPHA * delta + (1.0 - ARM_ACTION_FILTER_ALPHA) * self._filtered_mirror_delta
        )
        delta = self._filtered_mirror_delta.squeeze(0)

        current = self.robot.data.joint_pos[0, self._mirror_joint_ids_robot]
        limits = self.robot.data.soft_joint_pos_limits[0, self._mirror_joint_ids_robot]
        delta = (delta * ARM_ACTION_SCALE).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
        new_targets = (current + delta).clamp(limits[:, 0], limits[:, 1])
        return new_targets.unsqueeze(0)

    # ------------------------------------------------------------------ simulated arm targets + gain

    def _update_arm_sim_targets(self):
        """Set env._arm_motion_targets/_arm_motion_joint_ids every step — always active
        (no mode gating, see module docstring point 3). Both arms start held at
        default; the actively-controlled arm(s) then overwrite their own columns."""
        env = self.env.unwrapped
        env._arm_motion_joint_ids = self.all_arm_joint_ids_robot
        targets = self.robot.data.default_joint_pos[0:1, self.all_arm_joint_ids_robot].clone()

        arm_active = self._has_active_arm_target()
        if arm_active or bool(self._arm_homing.any().item()):
            arm_targets = self._compute_arm_targets() if arm_active else targets[:, self._active_arm_cols].clone()
            n_per_arm = len(self.arm_joint_ids_robot) // self._arm_homing.numel()
            for arm_idx in range(self._arm_homing.numel()):
                if not bool(self._arm_homing[arm_idx]):
                    continue
                sl = slice(arm_idx * n_per_arm, (arm_idx + 1) * n_per_arm)
                homing_target = self._homing_step(self.arm_joint_ids_robot[sl])
                if homing_target is None:
                    self._arm_homing[arm_idx] = False
                    self._filtered_arm_delta[0, sl] = 0.0
                    print("[FullDemo] Arm homed to default — arm policy engaging.")
                else:
                    arm_targets[:, sl] = homing_target
            targets[:, self._active_arm_cols] = arm_targets

        mirror_active = self._mirror_enabled and self.mirror_target_active
        if mirror_active or self._mirror_homing:
            if self._mirror_homing:
                homing_target = self._homing_step(self._mirror_joint_ids_robot)
                if homing_target is None:
                    self._mirror_homing = False
                    self._filtered_mirror_delta[:] = 0.0
                    print("[FullDemo] Mirror arm homed to default — mirrored policy engaging.")
                else:
                    targets[:, self._mirror_arm_cols] = homing_target
            if mirror_active and not self._mirror_homing:
                targets[:, self._mirror_arm_cols] = self._compute_mirror_targets()

        env._arm_motion_targets = targets
        self._update_arm_gains(arm_active, mirror_active)

    def _homing_step(self, joint_ids: torch.Tensor) -> torch.Tensor | None:
        current = self.robot.data.joint_pos[0, joint_ids]
        default = self.robot.data.default_joint_pos[0, joint_ids]
        err = default - current
        if bool((err.abs().max() <= ARM_HOMING_TOL_RAD).item()):
            return None
        delta = err.clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
        return (current + delta).unsqueeze(0)

    def _update_arm_gains(self, arm_active: bool, mirror_active: bool):
        """Whichever arm(s) are actively driven (native or mirrored) get the arm
        policy's own training gain; everything else gets the loco env's stock gain.
        Only two gains now (not three) — one unified loco policy means one held gain,
        not a standing-vs-walking split."""
        active_cols = []
        if arm_active:
            active_cols.append(self._active_arm_cols)
        if mirror_active:
            active_cols.append(self._mirror_arm_cols)
        active_ids = (
            self.all_arm_joint_ids_robot[torch.cat(active_cols)] if active_cols
            else self.all_arm_joint_ids_robot[:0]
        )
        held_mask = torch.ones(self.all_arm_joint_ids_robot.numel(), dtype=torch.bool, device=self.device)
        if active_cols:
            held_mask[torch.cat(active_cols)] = False
        held_ids = self.all_arm_joint_ids_robot[held_mask]

        if active_ids.numel() > 0:
            self.robot.write_joint_stiffness_to_sim(_GAIN_ARM_ACTIVE[0], joint_ids=active_ids)
            self.robot.write_joint_damping_to_sim(_GAIN_ARM_ACTIVE[1], joint_ids=active_ids)
        if held_ids.numel() > 0:
            self.robot.write_joint_stiffness_to_sim(_GAIN_ARM_HELD[0], joint_ids=held_ids)
            self.robot.write_joint_damping_to_sim(_GAIN_ARM_HELD[1], joint_ids=held_ids)

    def _mirror_goal_world(self) -> torch.Tensor:
        pos = self.robot.data.root_pos_w[0].clone()
        pos[2] = self._ground_z_w
        return quat_apply(self.robot.data.root_quat_w[0], self.mirror_goal_pos_local) + pos

    def _update_mirror_goal_marker(self):
        pos = self._mirror_goal_world().unsqueeze(0).clone()
        if not self.mirror_target_active:
            pos[0, 2] = -10.0
        self._mirror_goal_vis.visualize(pos)

    def _set_mirror_target(self, target_local: torch.Tensor):
        self.mirror_goal_pos_local = target_local
        self.mirror_target_active = True
        with torch.inference_mode():
            self._mirror_homing = True
            self._filtered_mirror_delta[:] = 0.0
        self._update_mirror_goal_marker()
        print(f"[FullDemo] Mirror target (right, via left policy): {target_local.tolist()}")

    def _clear_mirror_target(self):
        if not self.mirror_target_active:
            return
        self._mirror_homing = True
        self.mirror_target_active = False
        self._update_mirror_goal_marker()
        print("[FullDemo] Cleared mirror target — homing to default.")

    def _has_active_arm_target(self) -> bool:
        if self.arm_mode == "both":
            return bool(torch.all(self.arm_target_active).item())
        return bool(self.arm_target_active[0].item())

    def _goal_positions_world(self) -> torch.Tensor:
        """Fixed offset from the torso, recomputed from the robot's *current* pose
        every call — NOT a world-fixed point snapshotted once. A world-fixed goal would
        force the arm to compensate for base tilt/drift it was never trained to
        handle, which feeds back into more tilt — a runaway feedback loop this
        convention avoids by construction (the goal only moves because the robot
        pursuing it does)."""
        pos = self.robot.data.root_pos_w[0].clone()
        pos[2] = self._ground_z_w
        quat = self.robot.data.root_quat_w[0]
        return torch.stack([quat_apply(quat, t) + pos for t in self.goal_pos_local], dim=0)

    def _update_goal_markers(self):
        world_goals = self._goal_positions_world()
        inactive = torch.logical_not(self.arm_target_active)
        if bool(torch.any(inactive).item()):
            world_goals = world_goals.clone()
            world_goals[inactive, 2] = -10.0
        self._goal_vis.visualize(world_goals)

    def _set_arm_target(self, arm_idx: int, target_local: torch.Tensor):
        self.goal_pos_local[arm_idx] = target_local
        self.arm_target_active[arm_idx] = True

        n_per_arm = len(self.arm_joint_ids_robot) // (2 if self.arm_mode == "both" else 1)
        start = arm_idx * n_per_arm if self.arm_mode == "both" else 0
        with torch.inference_mode():
            self._arm_homing[arm_idx if self.arm_mode == "both" else 0] = True
            self._filtered_arm_delta[0, start:start + n_per_arm] = 0.0

        self._update_goal_markers()
        arm_label = ["left", "right"][arm_idx] if self.arm_mode == "both" else self.arm_mode
        print(f"[FullDemo] Arm target ({arm_label}): {target_local.tolist()}")

    def _clear_arm_targets(self):
        if not bool(torch.any(self.arm_target_active).item()):
            return
        self._arm_homing |= self.arm_target_active
        self.arm_target_active.zero_()
        self._update_goal_markers()
        print("[FullDemo] Cleared arm target(s) — homing to default.")

    # ------------------------------------------------------------------ keyboard / target input

    def _prompt_target(self, kind: str = "left"):
        if self._target_prompt_active:
            print("[FullDemo] Target prompt already open.")
            return
        self._pending_prompt_kind = kind
        self._target_prompt_requested = True

    def _parse_and_confirm_target(self, parts: list[str], bounds: dict) -> torch.Tensor | None:
        target = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=self.device)
        out_of_range = not (
            bounds["x"][0] <= target[0].item() <= bounds["x"][1]
            and bounds["y"][0] <= target[1].item() <= bounds["y"][1]
            and bounds["z"][0] <= target[2].item() <= bounds["z"][1]
        )
        if out_of_range:
            confirm = input(
                "  Outside the trained range — the policy has never seen this "
                "and may behave strangely. Send anyway? [y/N] "
            ).strip().lower()
            if confirm != "y":
                return None
        return target

    def _run_target_prompt(self):
        self._target_prompt_requested = False
        self._target_prompt_active = True
        kind = self._pending_prompt_kind

        if kind == "both":
            self._run_both_target_prompt()
            return

        if kind == "mirror":
            arm_label, b = "right (mirrored via left policy)", _GOAL_BOUNDS["right"]
        else:
            arm_label = ["left", "right"][self._active_arm_idx] if self.arm_mode == "both" else self.arm_mode
            b = _GOAL_BOUNDS["left" if arm_label == "left" else "right"]

        prompt = f"\n[FullDemo] New target for {arm_label} (x y z)\n  x: {b['x']}, y: {b['y']}, z: {b['z']}\n  > "
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
                target = self._parse_and_confirm_target(parts, b)
                if target is None:
                    continue
                if kind == "mirror":
                    if self._mirror_enabled:
                        self._set_mirror_target(target)
                    else:
                        print("  [FullDemo] Mirror control only available when arm_mode=left.")
                else:
                    self._set_arm_target(self._active_arm_idx, target)
                self._target_prompt_active = False
                return
            except ValueError:
                print(f"  Could not parse: {line!r}")

    def _run_both_target_prompt(self):
        if not self._mirror_enabled:
            print("  [FullDemo] Mirror control only available when arm_mode=left.")
            self._target_prompt_active = False
            return

        b_left, b_right = _GOAL_BOUNDS["left"], _GOAL_BOUNDS["right"]
        left_target = None
        while left_target is None:
            prompt = f"\n[FullDemo] New target for LEFT arm (x y z)\n  x: {b_left['x']}, y: {b_left['y']}, z: {b_left['z']}\n  > "
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
                left_target = self._parse_and_confirm_target(parts, b_left)
            except ValueError:
                print(f"  Could not parse: {line!r}")

        right_target = None
        while right_target is None:
            prompt = f"\n[FullDemo] New target for RIGHT arm (mirrored via left policy) (x y z)\n  x: {b_right['x']}, y: {b_right['y']}, z: {b_right['z']}\n  > "
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
                right_target = self._parse_and_confirm_target(parts, b_right)
            except ValueError:
                print(f"  Could not parse: {line!r}")

        self._set_arm_target(0, left_target)
        self._set_mirror_target(right_target)
        self._target_prompt_active = False

    def _apply_pending_target(self):
        if not self._target_prompt_requested:
            return
        self._run_target_prompt()

    def _setup_keyboard(self):
        import numpy as np

        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg

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
        self._keyboard.add_callback("C", self._toggle_camera_follow)
        self._keyboard.add_callback("V", self._reset_camera)
        self._keyboard.add_callback("T", self._prompt_target)
        if self.arm_mode == "both":
            self._keyboard.add_callback("L", lambda: self._select_arm(0))
            self._keyboard.add_callback("R", lambda: self._select_arm(1))
        if self._mirror_enabled:
            self._keyboard.add_callback("Y", lambda: self._prompt_target("mirror"))
            self._keyboard.add_callback("U", lambda: self._prompt_target("both"))

    def _select_arm(self, idx: int):
        self._active_arm_idx = idx
        print(f"[FullDemo] Active arm for T-key targeting: {['left', 'right'][idx]}")

    # ------------------------------------------------------------------ camera

    _CAMERA_OFFSET = (-2.5, 0.0, 0.8)
    _CAMERA_TARGET_HEIGHT_OFFSET = 0.6

    def _create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"

        cam = stage.DefinePrim(self.camera_path, "Camera")
        cam.GetAttribute("focalLength").Set(8.5)
        coi = cam.GetProperty("omni:kit:centerOfInterest")
        if not coi or not coi.IsValid():
            cam.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.camera_path)
        self._camera_follow = True

    def _toggle_camera_follow(self):
        self._camera_follow = not self._camera_follow
        print(f"[FullDemo] Camera follow: {'ON' if self._camera_follow else 'OFF (orbit freely with the mouse)'}")

    def _reset_camera(self):
        self._camera_follow = True
        self._position_camera()
        print("[FullDemo] Camera reset to chase view.")

    def _position_camera(self):
        base_pos = self.robot.data.root_pos_w[0]
        base_quat = self.robot.data.root_quat_w[0]
        offset = torch.tensor(self._CAMERA_OFFSET, device=self.device)
        cam_pos = quat_apply(base_quat, offset) + base_pos

        state = ViewportCameraState(self.camera_path, self.viewport)
        state.set_position_world(Gf.Vec3d(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item()), True)
        state.set_target_world(
            Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + self._CAMERA_TARGET_HEIGHT_OFFSET), True,
        )

    def update_camera(self):
        if self._camera_follow:
            self._position_camera()

    # ------------------------------------------------------------------ main step

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Unified policy: no mode blending, no standing/walking split — the command
        drives the one loco policy directly."""
        loco_action = self.loco_policy(obs)
        self._update_arm_sim_targets()
        return loco_action

    def _handle_env_resets(self, dones: torch.Tensor):
        if not bool(torch.any(dones > 0).item()):
            return
        self.cmd_filtered.zero_()
        self._arm_homing.zero_()
        if self._mirror_enabled:
            self._mirror_homing = False
        # No re-anchoring needed on respawn — targets are a fixed offset from the
        # torso, recomputed from the live pose every call.


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def main():
    demo = G1FullDemo()
    obs, _ = demo.env.reset()
    step = 0

    while simulation_app.is_running():
        demo.update_camera()
        demo._update_goal_markers()
        if demo._mirror_enabled:
            demo._update_mirror_goal_marker()
        demo._apply_pending_target()

        with torch.inference_mode():
            cmd_raw = demo._keyboard.advance()
            demo.cmd_filtered = (1.0 - CMD_FILTER_ALPHA) * demo.cmd_filtered + CMD_FILTER_ALPHA * cmd_raw
            # Write the live command buffer, not the flat obs tensor — see
            # self._command_term's own comment for why (history_length=5 stacking).
            demo._command_term.vel_command_b[:] = demo.cmd_filtered.unsqueeze(0)
            # FOUND 2026-07-23 (user report: WASD does nothing all session): IsaacLab's
            # UniformVelocityCommand._update_command() runs every step as part of
            # env.step() and force-zeroes vel_command_b for any env flagged
            # is_standing_env — a per-env coin flip (probability rel_standing_envs)
            # rolled once at reset. Since this demo disables periodic resampling
            # (resampling_time_range set to ~infinite, specifically so a random
            # resample can't clobber the keyboard-driven command), that flag is never
            # re-rolled either — a single unlucky roll at startup silently zeroes every
            # keyboard command for the rest of the session, with no visible symptom
            # other than "WASD does nothing." Force it off every step (cheap, and
            # correct regardless of whether it was ever true) — this demo's whole
            # point is direct keyboard control, "standing" should mean "not pressing
            # keys," never an invisible sticky flag from a training-time mechanism.
            demo._command_term.is_standing_env[:] = False

            action = demo.select_action(obs)
            obs, _, dones, _ = demo.env.step(action)
            demo._handle_env_resets(dones)

        if step % 60 == 0:
            target_world = demo._goal_positions_world()
            arm_groups = []
            if demo.arm_mode in ("left", "both"):
                arm_groups.append(("left", 0, demo.arm_body_ids[0]))
            if demo.arm_mode in ("right", "both"):
                arm_idx = 1 if demo.arm_mode == "both" else 0
                arm_groups.append(("right", arm_idx, demo.arm_body_ids[-1]))

            dist_str = ""
            for label, gi, body_id in arm_groups:
                ee = demo.robot.data.body_pos_w[0, body_id, :]
                tgt = target_world[gi]
                d = (tgt - ee).norm().item() * 100
                dist_str += f"  {label}: {d:.1f}cm"

            cmd_mag = torch.linalg.vector_norm(demo.cmd_filtered).item()
            # 2026-07-23 (user request): actual world-frame root position, added
            # specifically to give an unambiguous, numeric answer to "is the robot
            # really translating" — the arm-distance columns above can't answer that
            # (the arm goal is a fixed offset from the robot's own current torso,
            # recomputed every frame, so it reads the same whether the robot is
            # walking or standing still).
            root_xy = demo.robot.data.root_pos_w[0, :2]
            print(
                f"[FullDemo] step={step:6d}  |cmd|={cmd_mag:.3f}  pos=({root_xy[0].item():+.2f}, {root_xy[1].item():+.2f})  "
                f"arm_active={demo._has_active_arm_target()}" + dist_str
            )

        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
