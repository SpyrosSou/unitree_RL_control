# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils.math import quat_apply

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import g1_locomotion.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class StandingMetricsCsvWrapper(gym.Wrapper):
    """Collect per-episode standing stability metrics into a CSV file.

    The wrapper is intentionally lightweight: it records episode summaries per
    environment instance and appends them to a single CSV in the run directory.
    """

    _RECOVERY_TILT_DEG = 12.0

    def __init__(self, env: gym.Env, log_dir: str):
        super().__init__(env)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, "standing_metrics.csv")
        self._file = open(self.csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        if self._file.tell() == 0:
            self._writer.writeheader()
            self._file.flush()

        self._device = torch.device(self.unwrapped.device)
        self._num_envs = int(self.unwrapped.num_envs)
        self._episode_index = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_return = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_tilt = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_min_height = torch.full((self._num_envs,), float("inf"), dtype=torch.float32, device=self._device)
        self._episode_max_lin_speed = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_ang_speed = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_action_abs = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_action_delta_abs = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_push_lin_speed = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_push_yaw_speed = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._prev_actions: torch.Tensor | None = None

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "episode_index",
            "env_id",
            "episode_steps",
            "episode_return",
            "mean_reward",
            "outcome",
            "fell",
            "timed_out",
            "recovery_success",
            "max_tilt_deg",
            "min_base_height_m",
            "max_lin_speed_m_s",
            "max_ang_speed_rad_s",
            "max_action_abs",
            "max_action_delta_abs",
            "max_push_lin_speed_m_s",
            "max_push_yaw_speed_rad_s",
            "recovery_tilt_threshold_deg",
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_buffers(torch.arange(self._num_envs, device=self._episode_steps.device))
        self._prev_actions = None
        return obs, info

    def close(self):
        try:
            if hasattr(self, "_file") and not self._file.closed:
                self._file.flush()
                self._file.close()
        finally:
            return super().close()

    def step(self, action):
        env = self.unwrapped
        action_tensor = torch.as_tensor(action, device=env.device, dtype=torch.float32)

        self._update_step_metrics(env, action_tensor)

        obs, rew, terminated, truncated, info = self.env.step(action)
        self._episode_return += torch.as_tensor(rew, device=self._episode_return.device, dtype=torch.float32)

        done_mask = torch.as_tensor(terminated | truncated, device=env.device)
        if bool(done_mask.any().item()):
            self._flush_finished_episodes(done_mask, terminated, truncated)

        return obs, rew, terminated, truncated, info

    def _reset_buffers(self, env_ids: torch.Tensor):
        self._episode_steps[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._episode_max_tilt[env_ids] = 0.0
        self._episode_min_height[env_ids] = float("inf")
        self._episode_max_lin_speed[env_ids] = 0.0
        self._episode_max_ang_speed[env_ids] = 0.0
        self._episode_max_action_abs[env_ids] = 0.0
        self._episode_max_action_delta_abs[env_ids] = 0.0
        self._episode_max_push_lin_speed[env_ids] = 0.0
        self._episode_max_push_yaw_speed[env_ids] = 0.0
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0.0

    def _ensure_prev_actions(self, action_tensor: torch.Tensor):
        if self._prev_actions is None or self._prev_actions.shape != action_tensor.shape:
            self._prev_actions = torch.zeros_like(action_tensor)

    def _update_step_metrics(self, env, action_tensor: torch.Tensor):
        self._ensure_prev_actions(action_tensor)
        robot = env.scene["robot"]
        root_quat = robot.data.root_quat_w
        root_lin_vel = robot.data.root_lin_vel_w
        root_ang_vel = robot.data.root_ang_vel_w
        root_pos = robot.data.root_pos_w

        up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=root_quat.dtype).view(1, 3)
        body_z = quat_apply(root_quat, up_vec.expand(root_quat.shape[0], 3))
        tilt_cos = body_z[:, 2].clamp(-1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(tilt_cos))
        planar_speed = torch.linalg.vector_norm(root_lin_vel[:, :2], dim=-1)
        ang_speed = torch.linalg.vector_norm(root_ang_vel[:, :2], dim=-1)
        action_abs = action_tensor.abs().amax(dim=-1)
        action_delta = (action_tensor - self._prev_actions).abs().amax(dim=-1)

        push_lin_speed = getattr(env, "_standing_push_lin_speed", None)
        if push_lin_speed is None:
            push_lin_speed_t = torch.zeros_like(planar_speed)
        else:
            push_lin_speed_t = torch.as_tensor(push_lin_speed, device=env.device, dtype=torch.float32)

        push_yaw_speed = getattr(env, "_standing_push_yaw_speed", None)
        if push_yaw_speed is None:
            push_yaw_speed_t = torch.zeros_like(planar_speed)
        else:
            push_yaw_speed_t = torch.as_tensor(push_yaw_speed, device=env.device, dtype=torch.float32)

        self._episode_steps += 1
        self._episode_max_tilt = torch.maximum(self._episode_max_tilt, tilt_deg.detach().float())
        self._episode_min_height = torch.minimum(self._episode_min_height, root_pos[:, 2].detach().float())
        self._episode_max_lin_speed = torch.maximum(self._episode_max_lin_speed, planar_speed.detach().float())
        self._episode_max_ang_speed = torch.maximum(self._episode_max_ang_speed, ang_speed.detach().float())
        self._episode_max_action_abs = torch.maximum(self._episode_max_action_abs, action_abs.detach().float())
        self._episode_max_action_delta_abs = torch.maximum(self._episode_max_action_delta_abs, action_delta.detach().float())
        self._episode_max_push_lin_speed = torch.maximum(self._episode_max_push_lin_speed, push_lin_speed_t.detach().float())
        self._episode_max_push_yaw_speed = torch.maximum(self._episode_max_push_yaw_speed, push_yaw_speed_t.detach().float())
        self._prev_actions = action_tensor.detach().clone()

    def _flush_finished_episodes(self, done_mask: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        for env_id in env_ids.tolist():
            fell = bool(torch.as_tensor(terminated[env_id]).item())
            timed_out = bool(torch.as_tensor(truncated[env_id]).item())
            max_tilt = float(self._episode_max_tilt[env_id].item())
            recovery_success = int((not fell) and (max_tilt >= self._RECOVERY_TILT_DEG))
            row = {
                "episode_index": int(self._episode_index[env_id].item()),
                "env_id": env_id,
                "episode_steps": int(self._episode_steps[env_id].item()),
                "episode_return": float(self._episode_return[env_id].item()),
                "mean_reward": float(self._episode_return[env_id].item() / max(int(self._episode_steps[env_id].item()), 1)),
                "outcome": "fall" if fell else "timeout",
                "fell": int(fell),
                "timed_out": int(timed_out),
                "recovery_success": recovery_success,
                "max_tilt_deg": max_tilt,
                "min_base_height_m": float(self._episode_min_height[env_id].item()),
                "max_lin_speed_m_s": float(self._episode_max_lin_speed[env_id].item()),
                "max_ang_speed_rad_s": float(self._episode_max_ang_speed[env_id].item()),
                "max_action_abs": float(self._episode_max_action_abs[env_id].item()),
                "max_action_delta_abs": float(self._episode_max_action_delta_abs[env_id].item()),
                "max_push_lin_speed_m_s": float(self._episode_max_push_lin_speed[env_id].item()),
                "max_push_yaw_speed_rad_s": float(self._episode_max_push_yaw_speed[env_id].item()),
                "recovery_tilt_threshold_deg": self._RECOVERY_TILT_DEG,
            }
            self._writer.writerow(row)
            self._file.flush()
            self._episode_index[env_id] += 1
        self._reset_buffers(env_ids)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if "Standing" in args_cli.task:
        env = StandingMetricsCsvWrapper(env, log_dir)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint):
            # direct file path supplied via --checkpoint; skip directory scanning
            resume_path = agent_cfg.load_checkpoint
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
