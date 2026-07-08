"""
G1 standing-only demo (flat terrain).

Loads a trained standing policy and runs it with zero velocity command.
This is intended to validate balance quality before integrating stand/walk switching.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Auto-pick latest standing checkpoint
    python testing/walking_testing/g1_standing_demo.py

    # Explicit checkpoint
    python testing/walking_testing/g1_standing_demo.py \
        --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt
"""

import argparse
import glob
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

parser = argparse.ArgumentParser(description="Interactive G1 standing-only demo.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
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
    G1LocomotionStandingFlatEnvCfg_PLAY,
)

TASK = "G1-Locomotion-Standing-Flat-Play-v0"
PREFERRED_STANDING_CHECKPOINT = (
    "/home/spyros/Elm/Code/g1_locomotion/logs/rsl_rl/standing/"
    "g1_locomotion_flat/2026-06-16_12-26-58/model_1499.pt"
)


def _yaml_checkpoint(yaml_path: str, key: str) -> str | None:
    """Return the checkpoint path for *key* from a YAML file, or None."""
    if not os.path.isfile(yaml_path):
        return None
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}
    entry = cfg.get(key, {}) or {}
    ckpt = entry.get("checkpoint")
    if not ckpt:
        return None
    if not os.path.isabs(ckpt):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(yaml_path)))
        ckpt = os.path.join(repo_root, ckpt)
    return ckpt


def _resolve_default_checkpoint() -> str:
    # 1. Try checkpoints.yaml in the same directory
    yaml_ckpt = _yaml_checkpoint(
        os.path.join(os.path.dirname(__file__), "checkpoints.yaml"), "standing"
    )
    if yaml_ckpt and os.path.isfile(yaml_ckpt):
        return yaml_ckpt

    # 2. Fall back to hard-coded preferred path
    if os.path.isfile(PREFERRED_STANDING_CHECKPOINT):
        return PREFERRED_STANDING_CHECKPOINT

    # 3. Glob for any local standing checkpoint
    pattern = "logs/rsl_rl/standing/g1_locomotion_flat/*/model_*.pt"
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not checkpoints:
        raise FileNotFoundError(
            "No local standing checkpoints found. Train first with:\n"
            "  python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-v0 --headless"
        )
    return checkpoints[-1]


class G1StandingDemo:
    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)

        if getattr(args_cli, "checkpoint", None):
            checkpoint = args_cli.checkpoint
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        else:
            checkpoint = _resolve_default_checkpoint()

        print(f"[G1Standing] Using checkpoint: {checkpoint}")

        env_cfg = G1LocomotionStandingFlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint)
        self.policy = runner.get_inference_policy(device=self.device)

        self._create_camera()

    def _create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"

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


def main():
    demo = G1StandingDemo()
    obs, _ = demo.env.reset()

    while simulation_app.is_running():
        demo.update_camera()
        with torch.inference_mode():
            action = demo.policy(obs)
            obs, _, _, _ = demo.env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
