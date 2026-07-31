"""
G1 walking-only demo — arms held rigid at default, locomotion control isolated.

Derived 2026-07-31 from `g1_full_demo.py` by removing everything arm-related (T/Y/U
target prompting, mirror-testing, goal markers, arm-policy loading/observation/action
code) and permanently holding both arms at their default joint positions every step,
instead of letting an arm policy drive them. WASD/QE locomotion, the loco-policy
loading (including its own obs-dim/normalizer inference), the velocity-command term,
and camera control (C/V) are copied verbatim from `g1_full_demo.py`. Kept in sync
manually — if you touch the locomotion control logic in `g1_full_demo.py`, check
whether this file needs the same change.

Built from the PLAIN `G1LocomotionEnvCfg_PLAY` (not the `-ArmDisturbance` family) —
deliberately: that family's own `ArmMotionDisturbance` event scripts extra arm motion
into `env._arm_motion_targets` every step, which is exactly what this file doesn't
want. The `ArmDisturbanceBlendJointPositionAction` action class is still swapped in
(same mechanism `g1_full_demo.py` uses for its arm overlay) purely so this file can
pin `env._arm_motion_targets` to the default pose every step — the walking
checkpoint's own action-space/observation-space is identical between the base and
`-ArmDisturbance` task families, so evaluating it against the plain env is already a
well-established pattern in this repo (see `validation/eval_walking.py`'s own
`--arm_disturbance`-optional handling).

Behaviour (same as `g1_full_demo.py`'s locomotion half):
- WASD/QE command the locomotion policy directly (forward/strafe/turn).
- Press S to zero the command (stop).
- Press C to toggle camera follow off/on (off lets you orbit freely with the mouse).
- Press V to reset the camera to the default chase view (and re-enable follow).
- Arms never move — held at default_joint_pos the whole session.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Auto-load from testing/general_testing/checkpoints.yaml (this dir's own,
    # shared with g1_full_demo.py — same "loco" key, "arm"/"arm_mode" are unused)
    python3 testing/visual_testing/full_demo/walking_full_demo.py

    # Explicit checkpoint
    python3 testing/visual_testing/full_demo/walking_full_demo.py \\
        --loco_checkpoint chosen_checkpoints/walking_latest.pt
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

parser = argparse.ArgumentParser(description="G1 walking-only demo: arms held at default, locomotion isolated.")
cli_args_mod.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--loco_checkpoint", type=str, default=None,
                    help="Checkpoint for the unified stand+walk policy (overrides YAML).")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import _LEFT_ARM_JOINTS, _RIGHT_ARM_JOINTS
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp import ArmDisturbanceBlendJointPositionAction
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# ---------------------------------------------------------------------------
# Demo constants
# ---------------------------------------------------------------------------
LOCO_TASK  = "G1-Locomotion-Velocity-Play-v0"
LIN_VEL    = 1.0
ANG_VEL    = 0.5
CMD_FILTER_ALPHA = 0.12

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


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_YAML_PATH))))


def _resolve_checkpoint(cli_val: str | None, yaml_keys: list[str], hardcoded: str) -> str:
    path = cli_val or _yaml_value(yaml_keys, hardcoded)
    if path and not os.path.isabs(path):
        path = os.path.join(_REPO_ROOT, path)
    return path


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class G1WalkingFullDemo:
    def __init__(self):
        self.loco_ckpt = _resolve_checkpoint(
            args_cli.loco_checkpoint, ["loco", "checkpoint"], _DEFAULT_LOCO
        )
        if not os.path.isfile(self.loco_ckpt):
            raise FileNotFoundError(f"loco checkpoint not found: {self.loco_ckpt}")
        print(f"[WalkingDemo] loco: {self.loco_ckpt}")

        # ------ locomotion environment ------
        # Plain (non-ArmDisturbance) task deliberately — that family's own
        # ArmMotionDisturbance event scripts extra arm motion every step, which is
        # exactly what this demo doesn't want. See module docstring.
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args_mod.parse_rsl_rl_cfg(LOCO_TASK, args_cli)
        env_cfg = G1LocomotionEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None
        env_cfg.observations.policy.enable_corruption = True
        # Never auto-resample the velocity command — driven entirely from the
        # keyboard every step (see _command_term below).
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        # Swapped in purely so this demo can pin env._arm_motion_targets to the
        # default pose every step (see _hold_arms_at_default) — same mechanism
        # g1_full_demo.py uses for its actual arm overlay, here just holding
        # unconditionally. Leaves the policy's own raw arm-column output flowing
        # into last_action untouched, exactly matching training/deployment.
        if hasattr(env_cfg.actions, "JointPositionAction"):
            env_cfg.actions.JointPositionAction.class_type = ArmDisturbanceBlendJointPositionAction

        loco_env = ManagerBasedRLEnv(cfg=env_cfg)
        self.env = RslRlVecEnvWrapper(loco_env)
        self.device = loco_env.device
        self.robot = loco_env.scene["robot"]

        all_arm_ids, _ = self.robot.find_joints(list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS))
        self.all_arm_joint_ids_robot = torch.tensor(all_arm_ids, dtype=torch.long, device=self.device)

        self.loco_policy = self._load_loco_policy(agent_cfg, self.loco_ckpt)

        # ------ command state ------
        # Write the keyboard-driven command straight into the command term's own live
        # buffer (vel_command_b), NOT the flat observation tensor directly — see
        # g1_full_demo.py's identical comment for the full rationale
        # (history_length=5 stacking means "the velocity_commands columns" aren't a
        # fixed slice of the flat obs).
        self._command_term = loco_env.command_manager.get_term("base_velocity")
        self.cmd_filtered = torch.zeros(3, device=self.device)

        self._create_camera()
        self._setup_keyboard()

        print("\n[WalkingDemo] Ready. Arms are held rigid at default the whole session.")
        print("  W/A/D/Q/E  — velocity command    S  — stop")
        print("  C          — toggle camera follow (off = orbit freely with the mouse)")
        print("  V          — reset camera to the default chase view\n")

    # ------------------------------------------------------------------ loco policy

    def _load_loco_policy(self, agent_cfg: RslRlOnPolicyRunnerCfg, ckpt: str):
        """Network shape (including the critic's) and obs-normalization flags are all
        inferred directly from the checkpoint's own state_dict — verbatim copy of
        g1_full_demo.py's identical method; see its own docstring for the full
        rationale (this task's CriticCfg is wider than PolicyCfg, a privileged
        base_lin_vel term the actor never sees)."""
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

    # ------------------------------------------------------------------ arms held at default

    def _hold_arms_at_default(self):
        """Pin every arm joint's simulated target to default_joint_pos, every step —
        the walking-only analog of g1_full_demo.py's _update_arm_sim_targets (which
        this always takes the "no active arm target" branch of). No homing/blend
        logic needed since there's never a target to reach."""
        env = self.env.unwrapped
        env._arm_motion_joint_ids = self.all_arm_joint_ids_robot
        env._arm_motion_targets = self.robot.data.default_joint_pos[0:1, self.all_arm_joint_ids_robot].clone()

    # ------------------------------------------------------------------ keyboard

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
        print(f"[WalkingDemo] Camera follow: {'ON' if self._camera_follow else 'OFF (orbit freely with the mouse)'}")

    def _reset_camera(self):
        self._camera_follow = True
        self._position_camera()
        print("[WalkingDemo] Camera reset to chase view.")

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
        loco_action = self.loco_policy(obs)
        self._hold_arms_at_default()
        return loco_action

    def _handle_env_resets(self, dones: torch.Tensor):
        if not bool(torch.any(dones > 0).item()):
            return
        self.cmd_filtered.zero_()


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def main():
    demo = G1WalkingFullDemo()
    obs, _ = demo.env.reset()
    step = 0

    while simulation_app.is_running():
        demo.update_camera()

        with torch.inference_mode():
            cmd_raw = demo._keyboard.advance()
            demo.cmd_filtered = (1.0 - CMD_FILTER_ALPHA) * demo.cmd_filtered + CMD_FILTER_ALPHA * cmd_raw
            demo._command_term.vel_command_b[:] = demo.cmd_filtered.unsqueeze(0)
            # See g1_full_demo.py's identical line for the full "WASD does nothing"
            # bug-fix rationale (is_standing_env is a per-env coin-flip rolled once at
            # reset that force-zeroes the command; force it off every step since this
            # demo's whole point is direct keyboard control).
            demo._command_term.is_standing_env[:] = False

            action = demo.select_action(obs)
            obs, _, dones, _ = demo.env.step(action)
            demo._handle_env_resets(dones)

        if step % 60 == 0:
            cmd_mag = torch.linalg.vector_norm(demo.cmd_filtered).item()
            root_xy = demo.robot.data.root_pos_w[0, :2]
            print(
                f"[WalkingDemo] step={step:6d}  |cmd|={cmd_mag:.3f}  "
                f"pos=({root_xy[0].item():+.2f}, {root_xy[1].item():+.2f})"
            )

        step += 1


if __name__ == "__main__":
    main()
