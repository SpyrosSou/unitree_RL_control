# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
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
parser.add_argument(
    "--resume_new_dir",
    action="store_true",
    default=False,
    help=(
        "When used with --resume, start a fresh timestamped log folder instead of the default "
        "(continue writing checkpoints/CSV/TensorBoard into the resumed run's existing folder)."
    ),
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

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import g1_locomotion.tasks  # noqa: F401
from g1_locomotion.utils.metrics_wrappers import (
    ArmMetricsCsvWrapper,
    StandingMetricsCsvWrapper,
    WalkingMetricsCsvWrapper,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _archive_if_exists(path: str):
    """Rename an existing file out of the way with a timestamp suffix, if present.

    Used before re-dumping params/env.yaml and params/agent.yaml on a resume-in-place,
    so the config of an earlier segment isn't silently lost if this resume changed it.
    """
    if os.path.isfile(path):
        base, ext = os.path.splitext(path)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(path, f"{base}.before_{stamp}{ext}")


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

    # Resolve the resume checkpoint path early (moved up from below) so we can decide,
    # before creating the env/wrappers, whether to reuse its run folder or start fresh.
    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint):
            # direct file path supplied via --checkpoint; skip directory scanning
            resume_path = agent_cfg.load_checkpoint
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if resume_path is not None and not args_cli.resume_new_dir:
        # Default: continue writing checkpoints/CSV/TensorBoard into the same run folder
        # the resumed checkpoint came from, instead of starting a new timestamped one.
        # rsl_rl's OnPolicyRunner already restores the iteration counter from the checkpoint
        # (see load()/learn() in on_policy_runner.py) and saves model_<it>.pt by absolute
        # iteration number, so this does not collide with or overwrite earlier checkpoints.
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] --resume: continuing in existing run directory: {log_dir}")
    else:
        # specify directory for logging runs: {time-stamp}_{run_name}
        run_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
        # change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {run_dir_name}")
        if agent_cfg.run_name:
            run_dir_name += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, run_dir_name)

    # mdp.ArmMotionDisturbance's difficulty phase is keyed on env.common_step_counter, a
    # plain in-memory attribute (isaaclab ManagerBasedRLEnv.__init__ sets it to 0, never
    # restored from a checkpoint — RSL-RL's checkpoint only has model/optimizer state).
    # Left alone, every --resume silently restarts the disturbance curriculum from phase 0
    # instead of continuing from where the resumed checkpoint's training had reached (found
    # 2026-07-22 investigating why the standing-focus resume's disturbance behavior looked
    # off). Fix: peek at the resumed checkpoint's saved iteration count and convert it to an
    # equivalent step count (iterations * num_steps_per_env, matching how common_step_counter
    # itself accumulates — one increment per env.step() call, num_steps_per_env calls per
    # training iteration), then feed it in via the same phase_step_offset param the _PLAY
    # configs already use for a different purpose (jumping ahead for quick visual checks).
    # `iter` is the absolute iteration count since the lineage's true start (RSL-RL keeps
    # numbering checkpoints absolutely across resumes), so this is correct even for a resume
    # of a resume.
    if resume_path is not None and hasattr(env_cfg, "events") and hasattr(env_cfg.events, "arm_motion_disturbance"):
        resumed_iter = int(torch.load(resume_path, map_location="cpu", weights_only=False).get("iter", 0))
        step_offset = resumed_iter * agent_cfg.num_steps_per_env
        env_cfg.events.arm_motion_disturbance.params["phase_step_offset"] = step_offset
        print(
            f"[INFO] --resume: carrying arm-disturbance curriculum forward — resumed at iteration "
            f"{resumed_iter}, phase_step_offset={step_offset}"
        )

    # mdp.lin_vel_cmd_levels's current command range lives only in the same kind of
    # mutable, never-checkpointed object as arm_motion_disturbance's phase above (see
    # that fix's own comment) — found 2026-07-24 when a resumed run's curriculum
    # visibly dropped from its grown range back to the narrow (-0.1, 0.1) starting
    # point. Unlike the phase curriculum, this one isn't a pure function of step count
    # (it grows empirically, gated on reward exceeding a threshold each check), so it
    # can't be replayed the same way — but the only value it ever logs,
    # Curriculum/lin_vel_cmd_levels (= env.commands.base_velocity.ranges.lin_vel_x[1],
    # see mdp/curriculums.py's own return statement), is enough to exactly reconstruct
    # the full state: both lin_vel_x bounds and lin_vel_y's bounds all grow by the same
    # +/-0.1 step in lockstep on every promotion, from the same symmetric starting
    # point, each independently clamped to its own limit_ranges — so the shared
    # promotion count derived from lin_vel_x's logged upper bound is enough to
    # reconstruct every other bound exactly.
    if (
        resume_path is not None
        and hasattr(env_cfg, "curriculum")
        and getattr(env_cfg.curriculum, "lin_vel_cmd_levels", None) is not None
    ):
        from tensorboard.backend.event_processing import event_accumulator

        # FIXED 2026-07-24: was reading from `log_dir`, which is only the SOURCE run's
        # directory for a same-task, in-place resume (no --resume_new_dir) — the case
        # this was originally written and tested against. For a cross-task warm start
        # (--resume --resume_new_dir, e.g. loading a plain-recipe walking checkpoint
        # into a fresh arm-disturbance run), log_dir is a brand-new, EMPTY directory
        # with no tensorboard log at all — EventAccumulator.Reload() raised an
        # uncaught DirectoryDeletedError there, crashing the whole training process
        # instantly (confirmed the hard way: killed an overnight run within 5 seconds
        # of starting). The actual curriculum history always lives alongside the
        # checkpoint being resumed, i.e. os.path.dirname(resume_path) — which is
        # identical to log_dir in the in-place-resume case anyway, so this is a
        # strictly more correct fix, not just a narrower one. Also now wrapped in
        # try/except so any other unreadable-log edge case degrades to "skip this
        # optimization" instead of taking the whole run down with it.
        source_log_dir = os.path.dirname(resume_path)
        try:
            ea = event_accumulator.EventAccumulator(source_log_dir, size_guidance={"scalars": 0})
            ea.Reload()
            has_curriculum_log = "Curriculum/lin_vel_cmd_levels" in ea.Tags()["scalars"]
        except Exception as e:
            print(f"[WARN] --resume: couldn't read lin_vel_cmd_levels history from {source_log_dir} ({e}) — "
                  f"leaving the command-range curriculum at its default starting range.")
            has_curriculum_log = False
        if has_curriculum_log:
            last_upper_x = ea.Scalars("Curriculum/lin_vel_cmd_levels")[-1].value
            ranges = env_cfg.commands.base_velocity.ranges
            limit_ranges = env_cfg.commands.base_velocity.limit_ranges
            initial_upper_x = ranges.lin_vel_x[1]  # env_cfg's own fresh default, e.g. 0.1

            if last_upper_x >= limit_ranges.lin_vel_x[1] - 1e-6:
                # Fully promoted (upper pinned at its limit) — x's own recorded upper
                # can't reveal how many promotions actually happened past that point,
                # but every other axis caps at a smaller promotion count than x does
                # (x has the widest limit_ranges of the three), so "x is capped" alone
                # guarantees every axis is also fully capped — just use limits directly.
                new_x = (limit_ranges.lin_vel_x[0], limit_ranges.lin_vel_x[1])
                new_y = (limit_ranges.lin_vel_y[0], limit_ranges.lin_vel_y[1])
            else:
                num_promotions = round((last_upper_x - initial_upper_x) / 0.1)

                def _reconstruct(initial_upper: float, limits) -> tuple[float, float]:
                    raw = initial_upper + 0.1 * num_promotions
                    return (max(-raw, limits[0]), min(raw, limits[1]))

                new_x = _reconstruct(initial_upper_x, limit_ranges.lin_vel_x)
                new_y = _reconstruct(ranges.lin_vel_y[1], limit_ranges.lin_vel_y)

            ranges.lin_vel_x = list(new_x)
            ranges.lin_vel_y = list(new_y)
            print(
                f"[INFO] --resume: carrying lin_vel_cmd_levels forward — logged upper was "
                f"{last_upper_x:.3f}, reconstructed ranges.lin_vel_x={new_x}, ranges.lin_vel_y={new_y}"
            )

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
    elif "Locomotion" in args_cli.task:
        env = WalkingMetricsCsvWrapper(env, log_dir)
    elif "Arm" in args_cli.task:
        # "G1-Arm-*" (2026-07-21 — deliberately not named "IK": the deployed policy is
        # pure RL joint-space deltas, no inverse kinematics involved at all; "IK" is
        # reserved for the actual analytic-IK-based training disturbance generator
        # elsewhere (g1_locomotion/mdp/events.py's StandingArmIKReachDisturbance-
        # equivalent, not yet ported — see 29dof_implementation_plan.md).
        env = ArmMetricsCsvWrapper(env, log_dir)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    # dump the configuration into log-directory. When continuing an existing run
    # directory (resume without --resume_new_dir), the previous params files describe
    # the *original* segment's config — archive them with a timestamp instead of
    # silently overwriting, in case this resume changed --num_envs/--max_iterations/etc.
    if resume_path is not None and not args_cli.resume_new_dir:
        _archive_if_exists(os.path.join(log_dir, "params", "env.yaml"))
        _archive_if_exists(os.path.join(log_dir, "params", "agent.yaml"))
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
