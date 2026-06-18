"""
G1 stand/walk policy switch demo (flat terrain).

Loads two policies at once:
- standing policy for zero-command stabilization
- walking policy for commanded locomotion

The script switches between policies online using keyboard command magnitude
with hysteresis and a minimum dwell time to avoid chattering.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Defaults to the provided standing + walking checkpoints
    python walking_testing/g1_stand_walk_switch_demo.py

    # Override checkpoints
    python walking_testing/g1_stand_walk_switch_demo.py \
        --standing_checkpoint /path/to/standing.pt \
        --walking_checkpoint /path/to/walking.pt

Controls:
    W  forward
    S  stop (zero command)
    A  turn left
    D  turn right
    Q  strafe left
    E  strafe right
    C  toggle third-person / free camera
"""

import argparse
import importlib.util as _ilu
import os
import yaml

ISAACLAB_PATH = os.path.expanduser("~/Elm/Code/IsaacLab")
_spec = _ilu.spec_from_file_location(
    "cli_args",
    os.path.join(ISAACLAB_PATH, "scripts/reinforcement_learning/rsl_rl/cli_args.py"),
)
cli_args = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cli_args)

from isaaclab.app import AppLauncher

DEFAULT_STANDING_CHECKPOINT = (
    "/home/spyros/Elm/Code/g1_locomotion/logs/rsl_rl/standing/"
    "g1_locomotion_flat/2026-06-16_12-26-58/model_1499.pt"
)
DEFAULT_WALKING_CHECKPOINT = (
    "/home/spyros/Elm/Code/g1_locomotion/logs/rsl_rl/legs/"
    "g1_locomotion_flat/2026-06-03_10-20-23/model_3149.pt"
)

TASK = "G1-Locomotion-Flat-Play-v0"

LIN_VEL = 1.0
ANG_VEL = 0.5

# Hysteresis thresholds on ||[vx, vy, wz]||.
SWITCH_TO_WALK_THRESHOLD = 0.12
SWITCH_TO_STAND_THRESHOLD = 0.04
MIN_MODE_STEPS = 20

# Smoothing knobs.
CMD_FILTER_ALPHA = 0.12
SWITCH_TO_STAND_SPEED_THRESHOLD = 0.22
TRANSITION_STEPS_TO_WALK = 10
TRANSITION_STEPS_TO_STAND = 22

parser = argparse.ArgumentParser(description="Interactive G1 stand/walk policy switch demo.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--standing_checkpoint",
    type=str,
    default=None,
    help="Path to standing policy checkpoint. If omitted, walking_testing/checkpoints.yaml is used.",
)
parser.add_argument(
    "--walking_checkpoint",
    type=str,
    default=None,
    help="Path to walking policy checkpoint. If omitted, walking_testing/checkpoints.yaml is used.",
)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionFlatEnvCfg_PLAY,
)


_YAML_PATH = os.path.join(os.path.dirname(__file__), "checkpoints.yaml")


def _yaml_checkpoint(key: str) -> str | None:
    """Return checkpoint path for *key* from checkpoints.yaml, or None."""
    if not os.path.isfile(_YAML_PATH):
        return None
    with open(_YAML_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    entry = cfg.get(key, {}) or {}
    ckpt = entry.get("checkpoint")
    if not ckpt:
        return None
    if not os.path.isabs(ckpt):
        repo_root = os.path.dirname(os.path.dirname(_YAML_PATH))
        ckpt = os.path.join(repo_root, ckpt)
    return ckpt


class G1StandWalkSwitchDemo:
    def __init__(self):
        self.standing_checkpoint = (
            args_cli.standing_checkpoint
            or _yaml_checkpoint("standing")
            or DEFAULT_STANDING_CHECKPOINT
        )
        self.walking_checkpoint = (
            args_cli.walking_checkpoint
            or _yaml_checkpoint("walking")
            or DEFAULT_WALKING_CHECKPOINT
        )

        if not os.path.isfile(self.standing_checkpoint):
            raise FileNotFoundError(f"Standing checkpoint not found: {self.standing_checkpoint}")
        if not os.path.isfile(self.walking_checkpoint):
            raise FileNotFoundError(f"Walking checkpoint not found: {self.walking_checkpoint}")

        print(f"[SwitchDemo] Standing checkpoint: {self.standing_checkpoint}")
        print(f"[SwitchDemo] Walking checkpoint : {self.walking_checkpoint}")

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)

        env_cfg = G1LocomotionFlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        self.standing_policy = self._load_policy(agent_cfg, self.standing_checkpoint)
        self.walking_policy = self._load_policy(agent_cfg, self.walking_checkpoint)

        self.mode = "standing"
        self.mode_steps = 0
        self.cmd_filtered = torch.zeros(3, device=self.device)

        self._in_transition = False
        self._transition_from = "standing"
        self._transition_to = "standing"
        self._transition_step = 0
        self._transition_total_steps = TRANSITION_STEPS_TO_STAND

        self._create_camera()
        self._setup_keyboard()

    def _load_policy(self, agent_cfg: RslRlOnPolicyRunnerCfg, checkpoint: str):
        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint)
        return runner.get_inference_policy(device=self.device)

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

    def _setup_keyboard(self):
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        import numpy as np

        kb_cfg = Se2KeyboardCfg(sim_device=str(self.device))
        self._keyboard = Se2Keyboard(kb_cfg)
        self._keyboard._INPUT_KEY_MAPPING = {
            "W": np.array([LIN_VEL, 0.0, 0.0]),
            "A": np.array([0.0, 0.0, ANG_VEL]),
            "D": np.array([0.0, 0.0, -ANG_VEL]),
            "Q": np.array([0.0, LIN_VEL, 0.0]),
            "E": np.array([0.0, -LIN_VEL, 0.0]),
        }
        self._keyboard.add_callback("S", self._keyboard.reset)
        self._keyboard.add_callback("C", self._toggle_camera)

    def _toggle_camera(self):
        active = self.viewport.get_active_camera()
        self.viewport.set_active_camera(
            self.perspective_path if active == self.camera_path else self.camera_path
        )

    def _start_transition(self, new_mode: str, force_from: str | None = None, reason: str = "switch"):
        self._in_transition = True
        self._transition_from = force_from if force_from is not None else self.mode
        self._transition_to = new_mode
        self._transition_step = 0
        self._transition_total_steps = (
            TRANSITION_STEPS_TO_WALK if new_mode == "walking" else TRANSITION_STEPS_TO_STAND
        )
        self.mode = new_mode
        self.mode_steps = 0
        print(
            f"[SwitchDemo] transition {self._transition_from} -> {self._transition_to} "
            f"({self._transition_total_steps} steps, reason={reason})"
        )

    def _robot_planar_speed(self) -> float:
        root_lin = self.env.unwrapped.scene["robot"].data.root_lin_vel_w[0, :2]
        return torch.linalg.vector_norm(root_lin).item()

    def _maybe_switch_mode(self, cmd_mag: float):
        robot_speed = self._robot_planar_speed()

        if self._in_transition:
            # Critical safety behavior: if user re-commands walk while we are settling to stand,
            # immediately reverse transition to avoid a brittle low-speed handoff.
            if self._transition_to == "standing" and cmd_mag >= SWITCH_TO_WALK_THRESHOLD:
                self._start_transition("walking", force_from="standing", reason="interrupt")
                return
            return

        if self.mode == "standing":
            if self.mode_steps >= MIN_MODE_STEPS and cmd_mag >= SWITCH_TO_WALK_THRESHOLD:
                self._start_transition("walking", reason="threshold")
        else:
            if (
                self.mode_steps >= MIN_MODE_STEPS
                and cmd_mag <= SWITCH_TO_STAND_THRESHOLD
                and robot_speed <= SWITCH_TO_STAND_SPEED_THRESHOLD
            ):
                self._start_transition("standing", reason="threshold")

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._in_transition:
            policy = self.walking_policy if self.mode == "walking" else self.standing_policy
            return policy(obs)

        alpha = min((self._transition_step + 1) / float(self._transition_total_steps), 1.0)
        from_policy = (
            self.walking_policy if self._transition_from == "walking" else self.standing_policy
        )
        to_policy = self.walking_policy if self._transition_to == "walking" else self.standing_policy

        from_action = from_policy(obs)
        to_action = to_policy(obs)
        action = (1.0 - alpha) * from_action + alpha * to_action

        self._transition_step += 1
        if self._transition_step >= self._transition_total_steps:
            self._in_transition = False
            print(f"[SwitchDemo] mode settled: {self.mode}")

        return action


def main():
    demo = G1StandWalkSwitchDemo()
    obs, _ = demo.env.reset()
    step = 0

    print(
        "[SwitchDemo] Ready. W/A/D/Q/E to command walking, S to return to zero command."
    )

    while simulation_app.is_running():
        demo.update_camera()
        with torch.inference_mode():
            cmd_raw = demo._keyboard.advance()
            demo.cmd_filtered = (1.0 - CMD_FILTER_ALPHA) * demo.cmd_filtered + CMD_FILTER_ALPHA * cmd_raw
            obs[:, 9:12] = demo.cmd_filtered.unsqueeze(0)

            cmd_mag = torch.linalg.vector_norm(demo.cmd_filtered).item()
            demo._maybe_switch_mode(cmd_mag)

            action = demo.select_action(obs)
            obs, _, _, _ = demo.env.step(action)

            # Keep command channels aligned for the next policy step.
            obs[:, 9:12] = demo.cmd_filtered.unsqueeze(0)

            if step % 60 == 0:
                print(
                    f"[SwitchDemo] mode={demo.mode:8s} "
                    f"cmd_raw=({cmd_raw[0]:+.2f}, {cmd_raw[1]:+.2f}, {cmd_raw[2]:+.2f}) "
                    f"cmd_filt=({demo.cmd_filtered[0]:+.2f}, {demo.cmd_filtered[1]:+.2f}, {demo.cmd_filtered[2]:+.2f}) "
                    f"|cmd|={cmd_mag:.3f}"
                )

            demo.mode_steps += 1
            step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
