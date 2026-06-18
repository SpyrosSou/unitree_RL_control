"""
G1 flat-terrain locomotion demo with keyboard control (simplest setup).

One robot, flat ground, no height scan.

By default uses the pre-trained RSL-RL checkpoint for Isaac-Velocity-Flat-G1-v0,
downloaded automatically from NVIDIA Nucleus on first run.

Pass --checkpoint to use a locally trained checkpoint from this repo instead.

Usage (from project root):
    # Pre-trained Isaac Lab checkpoint (default)
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python walking_testing/g1_locomotion_demo.py

    # Locally trained checkpoint
    python walking_testing/g1_locomotion_demo.py \\
        --checkpoint logs/rsl_rl/g1_locomotion_flat/YYYY-MM-DD_HH-MM-SS/model_1500.pt

Controls:
    W  -- walk forward
    S  -- stop
    A  -- turn left (while walking)
    D  -- turn right (while walking)
    Q  -- strafe left
    E  -- strafe right
    C  -- toggle third-person / free camera
"""

# -- Isaac Sim must be launched before any other imports ---------------------
import argparse
import importlib.util as _ilu
import os

# Load IsaacLab rsl_rl CLI helpers by file path to avoid collision with
# ROS 2's "scripts" Python package that lives on the same sys.path.
ISAACLAB_PATH = os.path.expanduser("~/Elm/Code/IsaacLab")
_spec = _ilu.spec_from_file_location(
    "cli_args",
    os.path.join(ISAACLAB_PATH, "scripts/reinforcement_learning/rsl_rl/cli_args.py"),
)
cli_args = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cli_args)

from isaaclab.app import AppLauncher

import os
import yaml

_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints.yaml")


def _yaml_walking_checkpoint() -> str | None:
    """Return the 'walking' checkpoint from checkpoints.yaml, or None."""
    if not os.path.isfile(_YAML_PATH):
        return None
    with open(_YAML_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    entry = cfg.get("walking", {}) or {}
    ckpt = entry.get("checkpoint")
    if not ckpt:
        return None
    if not os.path.isabs(ckpt):
        repo_root = os.path.dirname(os.path.dirname(_YAML_PATH))
        ckpt = os.path.join(repo_root, ckpt)
    return ckpt


parser = argparse.ArgumentParser(
    description="Interactive G1 flat-terrain locomotion demo with keyboard teleoperation."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- Everything else after sim is up ----------------------------------------
import torch
from rsl_rl.runners import OnPolicyRunner

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg_PLAY,
)

TASK = "Isaac-Velocity-Flat-G1-v0"
RL_LIBRARY = "rsl_rl"

LIN_VEL = 1.0   # m/s  forward / strafe speed
ANG_VEL = 0.5   # rad/s turn rate


class G1FlatDemo:
    """Single-robot G1 flat-terrain demo with WASD keyboard control."""

    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)

        # Resolution order: --checkpoint CLI > checkpoints.yaml > pretrained download
        if getattr(args_cli, "checkpoint", None):
            checkpoint = args_cli.checkpoint
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            print(f"[G1Demo] Using local checkpoint: {checkpoint}")
        elif (yaml_ckpt := _yaml_walking_checkpoint()) and os.path.isfile(yaml_ckpt):
            checkpoint = yaml_ckpt
            print(f"[G1Demo] Using YAML checkpoint: {checkpoint}")
        else:
            checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)
            if checkpoint is None:
                raise RuntimeError(
                    f"Could not fetch pre-trained checkpoint for {TASK}. "
                    "Check your internet / Nucleus connection."
                )
            print(f"[G1Demo] Using pre-trained Isaac Lab checkpoint: {checkpoint}")

        env_cfg = G1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint)
        self.policy = runner.get_inference_policy(device=self.device)

        self._create_camera()
        self._setup_keyboard()

    # -- Camera ---------------------------------------------------------------

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

    def update_camera(self):
        """Keep third-person camera behind the robot."""
        base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0]
        base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[0]
        offset = torch.tensor([-2.5, 0.0, 0.8], device=self.device)
        cam_pos = quat_apply(base_quat, offset) + base_pos

        state = ViewportCameraState(self.camera_path, self.viewport)
        state.set_position_world(
            Gf.Vec3d(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item()), True
        )
        state.set_target_world(
            Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6), True
        )

    # -- Keyboard (via Isaac Lab's Se2Keyboard device) -------------------------
    # Se2Keyboard handles the carb API correctly and is the proven Isaac Lab way.
    # We override the key bindings to use WASD instead of arrow/numpad keys.

    def _setup_keyboard(self):
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        import numpy as np

        kb_cfg = Se2KeyboardCfg(sim_device=str(self.device))
        self._keyboard = Se2Keyboard(kb_cfg)

        # WASD bindings: press accumulates, release subtracts (Se2Keyboard style)
        self._keyboard._INPUT_KEY_MAPPING = {
            "W": np.array([ LIN_VEL,  0.0,     0.0]),
            "A": np.array([ 0.0,      0.0,  ANG_VEL]),
            "D": np.array([ 0.0,      0.0, -ANG_VEL]),
            "Q": np.array([ 0.0,  LIN_VEL,     0.0]),
            "E": np.array([ 0.0, -LIN_VEL,     0.0]),
        }
        # S stops: reset accumulated command to zero
        self._keyboard.add_callback("S", self._keyboard.reset)
        # C toggles camera
        self._keyboard.add_callback("C", self._toggle_camera)

    def _toggle_camera(self):
        active = self.viewport.get_active_camera()
        self.viewport.set_active_camera(
            self.perspective_path if active == self.camera_path else self.camera_path
        )


def main():
    demo = G1FlatDemo()
    obs, _ = demo.env.reset()
    step = 0

    while simulation_app.is_running():
        demo.update_camera()
        with torch.inference_mode():
            action = demo.policy(obs)
            obs, _, _, _ = demo.env.step(action)
            # Pull velocity command from keyboard device [vx, vy, wz]
            vel_cmd = demo._keyboard.advance()
            obs[:, 9:12] = vel_cmd.unsqueeze(0)
            # Debug: print command every 60 steps so you can see WASD is working
            if step % 60 == 0:
                print(f"[G1] cmd: vx={vel_cmd[0]:.2f}  vy={vel_cmd[1]:.2f}  wz={vel_cmd[2]:.2f}")
            step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
