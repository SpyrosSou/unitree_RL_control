# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode metrics-CSV gym wrappers shared by train.py and the validation/ eval scripts.

Lives in the installed ``g1_locomotion`` package (rather than next to a single script) so it's
importable by name (``from g1_locomotion.utils.metrics_wrappers import ...``) from any entry
point regardless of its directory — train.py, and validation/eval_standing.py and
validation/eval_walking.py all use it. train.py and play.py-style scripts execute
AppLauncher/argparse at module level (not guarded by ``__name__ == "__main__"``), so importing
*them* directly would re-launch Isaac Sim and re-parse argv — this module has no such side
effects, so it's safe to import from anywhere, as long as the import happens after AppLauncher
has already been launched by the importing script itself (same rule as any other isaaclab.*
import).

Each wrapper writes two files per run: a "detailed" CSV with every column, and a slimmer
"summary" CSV containing only the columns that answer "is training converging" — see
``logging_reference.md`` for the full column-by-column writeup and how to read them.
"""

import csv
import os

import gymnasium as gym
import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

from g1_locomotion.tasks.manager_based.g1_locomotion.mdp.events import StandingArmTrajectoryDisturbance


def _next_episode_index(csv_path: str) -> int:
    """Starting ``episode_index`` for a CSV wrapper, continuing a pre-existing file.

    When resuming training into the same run directory, a fresh process would otherwise
    restart every wrapper's episode counter at 0, producing duplicate ``episode_index``
    values alongside rows already in the file from the earlier segment. Reading back the
    max existing index (if any) and starting from there keeps the column monotonically
    increasing across a resume, instead of resetting.
    """
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    return max(int(r["episode_index"]) for r in rows) + 1


def _arm_disturbance_phase(common_step_counter: int) -> int:
    """Which arm-motion-disturbance curriculum phase a given env-step falls in.

    Mirrors ``StandingArmTrajectoryDisturbance._phase_index`` using the class's default
    phase boundaries (the ones actually used during training; only the *_PLAY* configs
    override them for faster visual demos). Kept here instead of read off the live event
    term instance so a finished-episode row can be labelled without holding a reference
    to the event manager.
    """
    for i, boundary in enumerate(StandingArmTrajectoryDisturbance._PHASE_STEP_BOUNDARIES):
        if common_step_counter < boundary:
            return i
    return len(StandingArmTrajectoryDisturbance._PHASE_STEP_BOUNDARIES)


class _DualCsvWriter:
    """Writes each finished-episode row to a detailed CSV (every column) and a slimmer
    summary CSV (just the convergence-relevant subset), so a quick glance doesn't require
    wading through every diagnostic column. Same episode_index/env_id/env_step key in
    both files, so they can be joined back together (e.g. in pandas) if needed.
    """

    def __init__(
        self,
        log_dir: str,
        name: str,
        detailed_fieldnames: list[str],
        summary_fieldnames: list[str],
        write_summary: bool = True,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.detailed_path = os.path.join(log_dir, f"{name}_detailed.csv")
        self._summary_fieldnames = summary_fieldnames
        self._write_summary = write_summary

        self._detailed_file = open(self.detailed_path, "a", newline="")
        self._detailed_writer = csv.DictWriter(self._detailed_file, fieldnames=detailed_fieldnames)
        if self._detailed_file.tell() == 0:
            self._detailed_writer.writeheader()
            self._detailed_file.flush()

        if self._write_summary:
            self.summary_path = os.path.join(log_dir, f"{name}_summary.csv")
            self._summary_file = open(self.summary_path, "a", newline="")
            self._summary_writer = csv.DictWriter(self._summary_file, fieldnames=summary_fieldnames)
            if self._summary_file.tell() == 0:
                self._summary_writer.writeheader()
                self._summary_file.flush()
        else:
            self.summary_path = None
            self._summary_file = None
            self._summary_writer = None

    def next_episode_index(self) -> int:
        # The detailed file is a strict superset of the summary file's rows, so its
        # episode_index history is the one to continue from.
        return _next_episode_index(self.detailed_path)

    def write_row(self, row: dict):
        self._detailed_writer.writerow(row)
        self._detailed_file.flush()
        if self._write_summary:
            self._summary_writer.writerow({k: row[k] for k in self._summary_fieldnames})
            self._summary_file.flush()

    def close(self):
        for f in (self._detailed_file, self._summary_file):
            if f is not None and not f.closed:
                f.flush()
                f.close()


class StandingMetricsCsvWrapper(gym.Wrapper):
    """Collect per-episode standing stability metrics into detailed + summary CSVs."""

    _RECOVERY_TILT_DEG = 12.0
    # A foot must have been airborne at least this long for a touchdown to count as a
    # deliberate corrective step, rather than contact-sensor noise/jitter.
    _STEP_AIR_TIME_THRESHOLD_S = 0.05
    _FOOT_BODY_NAME_PATTERN = ".*_ankle_roll_link"

    _SUMMARY_FIELDS = [
        "episode_index",
        "env_id",
        "env_step",
        "episode_steps",
        "episode_return",
        "mean_reward",
        "outcome",
        "fell",
        "step_count",
        "arm_disturbance_phase",
    ]

    def __init__(self, env: gym.Env, log_dir: str, write_summary: bool = True):
        super().__init__(env)
        self.log_dir = log_dir
        self._csv = _DualCsvWriter(log_dir, "standing", self._fieldnames(), self._SUMMARY_FIELDS, write_summary)
        self.csv_path = self._csv.detailed_path  # kept for validation/eval_standing.py

        self._device = torch.device(self.unwrapped.device)
        self._num_envs = int(self.unwrapped.num_envs)
        self._episode_index = torch.full(
            (self._num_envs,), self._csv.next_episode_index(), dtype=torch.long, device=self._device
        )
        self._episode_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_return = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_tilt = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_min_height = torch.full((self._num_envs,), float("inf"), dtype=torch.float32, device=self._device)
        self._episode_max_lin_speed = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_ang_speed = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_action_abs = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_action_delta_abs = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_step_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_max_foot_air_time = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._prev_actions: torch.Tensor | None = None
        self._foot_body_ids: list[int] | None = None

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "episode_index",
            "env_id",
            "env_step",
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
            "step_count",
            "max_foot_air_time_s",
            "arm_disturbance_phase",
            "recovery_tilt_threshold_deg",
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_buffers(torch.arange(self._num_envs, device=self._episode_steps.device))
        self._prev_actions = None
        return obs, info

    def close(self):
        try:
            if hasattr(self, "_csv"):
                self._csv.close()
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
        self._episode_step_count[env_ids] = 0
        self._episode_max_foot_air_time[env_ids] = 0.0
        if self._prev_actions is not None:
            self._prev_actions[env_ids] = 0.0

    def _ensure_prev_actions(self, action_tensor: torch.Tensor):
        if self._prev_actions is None or self._prev_actions.shape != action_tensor.shape:
            self._prev_actions = torch.zeros_like(action_tensor)

    def _update_stepping_metrics(self, env):
        """Detect corrective steps from the ankle contact sensor's air-time tracking.

        A step is counted when a tracked foot has *just* touched back down
        (``current_contact_time`` is within one control step of zero) after having been
        airborne for longer than ``_STEP_AIR_TIME_THRESHOLD_S`` (``last_air_time`` holds
        the air-time value at the moment contact resumes). This distinguishes an actual
        corrective step from sensor noise around continuous ground contact.
        """
        contact_sensor = env.scene.sensors.get("contact_forces")
        if contact_sensor is None or contact_sensor.data.last_air_time is None:
            return

        if self._foot_body_ids is None:
            body_ids, _ = contact_sensor.find_bodies(self._FOOT_BODY_NAME_PATTERN)
            self._foot_body_ids = list(body_ids)

        last_air_time = contact_sensor.data.last_air_time[:, self._foot_body_ids]
        current_contact_time = contact_sensor.data.current_contact_time[:, self._foot_body_ids]
        control_dt = env.step_dt

        just_touched_down = current_contact_time <= control_dt
        was_a_real_step = last_air_time > self._STEP_AIR_TIME_THRESHOLD_S
        step_this_tick = (just_touched_down & was_a_real_step).any(dim=-1)
        self._episode_step_count += step_this_tick.long()

        current_air_time = contact_sensor.data.current_air_time[:, self._foot_body_ids]
        self._episode_max_foot_air_time = torch.maximum(
            self._episode_max_foot_air_time, current_air_time.amax(dim=-1).detach().float()
        )

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

        self._episode_steps += 1
        self._episode_max_tilt = torch.maximum(self._episode_max_tilt, tilt_deg.detach().float())
        self._episode_min_height = torch.minimum(self._episode_min_height, root_pos[:, 2].detach().float())
        self._episode_max_lin_speed = torch.maximum(self._episode_max_lin_speed, planar_speed.detach().float())
        self._episode_max_ang_speed = torch.maximum(self._episode_max_ang_speed, ang_speed.detach().float())
        self._episode_max_action_abs = torch.maximum(self._episode_max_action_abs, action_abs.detach().float())
        self._episode_max_action_delta_abs = torch.maximum(
            self._episode_max_action_delta_abs, action_delta.detach().float()
        )
        self._update_stepping_metrics(env)
        self._prev_actions = action_tensor.detach().clone()

    def _flush_finished_episodes(self, done_mask: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        env_step = int(self.unwrapped.common_step_counter)
        phase = _arm_disturbance_phase(env_step)
        for env_id in env_ids.tolist():
            fell = bool(torch.as_tensor(terminated[env_id]).item())
            timed_out = bool(torch.as_tensor(truncated[env_id]).item())
            max_tilt = float(self._episode_max_tilt[env_id].item())
            recovery_success = int((not fell) and (max_tilt >= self._RECOVERY_TILT_DEG))
            row = {
                "episode_index": int(self._episode_index[env_id].item()),
                "env_id": env_id,
                "env_step": env_step,
                "episode_steps": int(self._episode_steps[env_id].item()),
                "episode_return": float(self._episode_return[env_id].item()),
                "mean_reward": float(
                    self._episode_return[env_id].item() / max(int(self._episode_steps[env_id].item()), 1)
                ),
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
                "step_count": int(self._episode_step_count[env_id].item()),
                "max_foot_air_time_s": float(self._episode_max_foot_air_time[env_id].item()),
                "arm_disturbance_phase": phase,
                "recovery_tilt_threshold_deg": self._RECOVERY_TILT_DEG,
            }
            self._csv.write_row(row)
            self._episode_index[env_id] += 1
        self._reset_buffers(env_ids)


class WalkingMetricsCsvWrapper(gym.Wrapper):
    """Collect per-episode command-tracking, foot-slip, and drift metrics for the walking task.

    Mirrors ``StandingMetricsCsvWrapper``'s structure but tracks what matters for a
    velocity-tracking gait instead of static balance: how well the policy follows the
    commanded velocity, how much the feet slide while in contact (vs. clean lift/plant
    steps), and how much the robot drifts off the path implied by its own commanded
    velocity (see the drift fields below).
    """

    _FOOT_BODY_NAME_PATTERN = ".*_ankle_roll_link"

    _SUMMARY_FIELDS = [
        "episode_index",
        "env_id",
        "env_step",
        "episode_steps",
        "episode_return",
        "mean_reward",
        "outcome",
        "fell",
        "mean_lin_vel_track_err_m_s",
        "heading_drift_deg",
    ]

    def __init__(self, env: gym.Env, log_dir: str, write_summary: bool = True):
        super().__init__(env)
        self.log_dir = log_dir
        self._csv = _DualCsvWriter(log_dir, "walking", self._fieldnames(), self._SUMMARY_FIELDS, write_summary)
        self.csv_path = self._csv.detailed_path

        self._device = torch.device(self.unwrapped.device)
        self._num_envs = int(self.unwrapped.num_envs)
        self._episode_index = torch.full(
            (self._num_envs,), self._csv.next_episode_index(), dtype=torch.long, device=self._device
        )
        self._episode_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_return = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_sum_lin_err = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_sum_ang_err = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_tilt = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_sum_foot_slip = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        # Drift bookkeeping: "expected" pose is what the robot's *own commanded* body-frame
        # velocity implies, integrated from its actual pose at the start of the episode.
        # Drift = how far actual pose has diverged from that, i.e. motion the command didn't
        # ask for (gait asymmetry, foot slip, etc. pushing it off its own intended path) —
        # not just "did it turn", since turning is expected whenever ang_vel_z is commanded.
        self._episode_start_pos = torch.zeros(self._num_envs, 2, dtype=torch.float32, device=self._device)
        self._episode_start_yaw = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_expected_pos = torch.zeros(self._num_envs, 2, dtype=torch.float32, device=self._device)
        self._episode_cmd_yaw_integral = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._last_pos = torch.zeros(self._num_envs, 2, dtype=torch.float32, device=self._device)
        self._last_yaw = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._foot_body_ids_robot: list[int] | None = None
        self._foot_body_ids_sensor: list[int] | None = None

    @staticmethod
    def _fieldnames() -> list[str]:
        return [
            "episode_index",
            "env_id",
            "env_step",
            "episode_steps",
            "episode_return",
            "mean_reward",
            "outcome",
            "fell",
            "timed_out",
            "max_tilt_deg",
            "mean_lin_vel_track_err_m_s",
            "mean_ang_vel_track_err_rad_s",
            "mean_foot_slip_speed_m_s",
            "heading_drift_deg",
            "lateral_drift_m",
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_buffers(torch.arange(self._num_envs, device=self._episode_steps.device))
        return obs, info

    def close(self):
        try:
            if hasattr(self, "_csv"):
                self._csv.close()
        finally:
            return super().close()

    def step(self, action):
        env = self.unwrapped
        self._update_step_metrics(env)

        obs, rew, terminated, truncated, info = self.env.step(action)
        self._episode_return += torch.as_tensor(rew, device=self._episode_return.device, dtype=torch.float32)

        done_mask = torch.as_tensor(terminated | truncated, device=env.device)
        if bool(done_mask.any().item()):
            self._flush_finished_episodes(done_mask, terminated, truncated)

        return obs, rew, terminated, truncated, info

    def _reset_buffers(self, env_ids: torch.Tensor):
        self._episode_steps[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._episode_sum_lin_err[env_ids] = 0.0
        self._episode_sum_ang_err[env_ids] = 0.0
        self._episode_max_tilt[env_ids] = 0.0
        self._episode_sum_foot_slip[env_ids] = 0.0

        # self.unwrapped's robot state here already reflects the *new* episode's initial
        # pose: reset_buffers is only ever called right after the underlying env has
        # actually reset those env_ids (either the global env.reset(), or Isaac Lab's
        # auto-reset of just-finished envs inside env.step()).
        robot = self.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w[env_ids, :2].detach().float()
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w[env_ids])
        yaw = yaw.detach().float()

        self._episode_start_pos[env_ids] = pos
        self._episode_start_yaw[env_ids] = yaw
        self._episode_expected_pos[env_ids] = pos.clone()
        self._episode_cmd_yaw_integral[env_ids] = 0.0
        self._last_pos[env_ids] = pos.clone()
        self._last_yaw[env_ids] = yaw.clone()

    def _update_step_metrics(self, env):
        robot = env.scene["robot"]
        root_quat = robot.data.root_quat_w

        up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=root_quat.dtype).view(1, 3)
        body_z = quat_apply(root_quat, up_vec.expand(root_quat.shape[0], 3))
        tilt_deg = torch.rad2deg(torch.acos(body_z[:, 2].clamp(-1.0, 1.0)))

        command = env.command_manager.get_command("base_velocity")
        lin_vel_b = robot.data.root_lin_vel_b[:, :2]
        ang_vel_b = robot.data.root_ang_vel_b[:, 2]
        lin_err = torch.linalg.vector_norm(command[:, :2] - lin_vel_b, dim=-1)
        ang_err = (command[:, 2] - ang_vel_b).abs()

        self._episode_steps += 1
        self._episode_max_tilt = torch.maximum(self._episode_max_tilt, tilt_deg.detach().float())
        self._episode_sum_lin_err += lin_err.detach().float()
        self._episode_sum_ang_err += ang_err.detach().float()
        self._episode_sum_foot_slip += self._foot_slip_speed(env, robot).detach().float()
        self._update_drift_metrics(env, robot, command)

    def _update_drift_metrics(self, env, robot, command: torch.Tensor):
        dt = env.step_dt
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        yaw = yaw.detach().float()
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        # Rotate the commanded body-frame velocity into world frame using the robot's actual
        # current heading, then integrate: "where would this step have taken it, given how
        # it's actually oriented right now."
        lin_cmd_b = command[:, :2].detach().float()
        cmd_vel_w_x = lin_cmd_b[:, 0] * cos_y - lin_cmd_b[:, 1] * sin_y
        cmd_vel_w_y = lin_cmd_b[:, 0] * sin_y + lin_cmd_b[:, 1] * cos_y
        self._episode_expected_pos[:, 0] += cmd_vel_w_x * dt
        self._episode_expected_pos[:, 1] += cmd_vel_w_y * dt
        self._episode_cmd_yaw_integral += command[:, 2].detach().float() * dt

        self._last_pos = robot.data.root_pos_w[:, :2].detach().float().clone()
        self._last_yaw = yaw.clone()

    def _foot_slip_speed(self, env, robot) -> torch.Tensor:
        contact_sensor = env.scene.sensors.get("contact_forces")
        if contact_sensor is None or contact_sensor.data.current_contact_time is None:
            return torch.zeros(env.num_envs, device=env.device)

        if self._foot_body_ids_robot is None:
            self._foot_body_ids_robot, _ = robot.find_bodies(self._FOOT_BODY_NAME_PATTERN)
            self._foot_body_ids_sensor, _ = contact_sensor.find_bodies(self._FOOT_BODY_NAME_PATTERN)

        in_contact = contact_sensor.data.current_contact_time[:, self._foot_body_ids_sensor] > 0.0
        foot_vel = robot.data.body_lin_vel_w[:, self._foot_body_ids_robot, :2]
        return (foot_vel.norm(dim=-1) * in_contact).sum(dim=-1)

    def _drift_for_env(self, env_id: int) -> tuple[float, float]:
        """(heading_drift_deg, lateral_drift_m) for one just-finished episode.

        heading_drift_deg: how much the actual heading has diverged from "start heading +
        everything the ang_vel_z command asked for" — i.e. rotation the command didn't
        cause. lateral_drift_m: the sideways (relative to the episode's starting heading)
        component of how far actual position has diverged from where the commanded
        body-frame velocity, integrated against the actual heading each step, implied it
        should be. Both are ~0 for a policy that tracks its own commands cleanly; nonzero
        even when a "go straight" command (vy=0, wz=0) is held the whole episode is exactly
        the drift failure mode this was added to catch.
        """
        yaw0 = self._episode_start_yaw[env_id]
        expected_yaw = yaw0 + self._episode_cmd_yaw_integral[env_id]
        heading_drift = wrap_to_pi((self._last_yaw[env_id] - expected_yaw).unsqueeze(0))[0]
        heading_drift_deg = float(torch.rad2deg(heading_drift).item())

        pos_err = self._last_pos[env_id] - self._episode_expected_pos[env_id]
        lateral_drift_m = float((-pos_err[0] * torch.sin(yaw0) + pos_err[1] * torch.cos(yaw0)).item())
        return heading_drift_deg, lateral_drift_m

    def _flush_finished_episodes(self, done_mask: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        env_step = int(self.unwrapped.common_step_counter)
        for env_id in env_ids.tolist():
            fell = bool(torch.as_tensor(terminated[env_id]).item())
            timed_out = bool(torch.as_tensor(truncated[env_id]).item())
            steps = max(int(self._episode_steps[env_id].item()), 1)
            heading_drift_deg, lateral_drift_m = self._drift_for_env(env_id)
            row = {
                "episode_index": int(self._episode_index[env_id].item()),
                "env_id": env_id,
                "env_step": env_step,
                "episode_steps": steps,
                "episode_return": float(self._episode_return[env_id].item()),
                "mean_reward": float(self._episode_return[env_id].item() / steps),
                "outcome": "fall" if fell else "timeout",
                "fell": int(fell),
                "timed_out": int(timed_out),
                "max_tilt_deg": float(self._episode_max_tilt[env_id].item()),
                "mean_lin_vel_track_err_m_s": float(self._episode_sum_lin_err[env_id].item() / steps),
                "mean_ang_vel_track_err_rad_s": float(self._episode_sum_ang_err[env_id].item() / steps),
                "mean_foot_slip_speed_m_s": float(self._episode_sum_foot_slip[env_id].item() / steps),
                "heading_drift_deg": heading_drift_deg,
                "lateral_drift_m": lateral_drift_m,
            }
            self._csv.write_row(row)
            self._episode_index[env_id] += 1
        self._reset_buffers(env_ids)


class ArmMetricsCsvWrapper(gym.Wrapper):
    """Collect per-episode reach-quality metrics for the arm IK task."""

    _SUMMARY_FIELDS = [
        "episode_index",
        "env_id",
        "env_step",
        "episode_steps",
        "episode_return",
        "mean_reward",
        "outcome",
        "success",
        "min_dist_to_goal_cm",
        "wobble_active",
    ]

    def __init__(self, env: gym.Env, log_dir: str, write_summary: bool = True):
        super().__init__(env)
        self.log_dir = log_dir
        self._write_summary = write_summary
        self._device = torch.device(self.unwrapped.device)
        self._num_envs = int(self.unwrapped.num_envs)

        # Joint configuration at the episode's closest approach (2026-07-08, joint-config
        # correlation check — see known_issues.md). Answers "does failure correlate with a
        # specific joint being pinned near its limit, or a specific multi-joint combination"
        # rather than just goal difficulty — a question the reachability-workspace fix
        # couldn't answer on its own. Only wired up for the arm task (needs
        # arm_joint_indices_tensor and joint_names); silently skipped (empty columns) for
        # envs that don't expose it. Computed before _fieldnames()/_csv construction below
        # since the field list depends on it.
        self._arm_joint_ids = getattr(self.unwrapped, "arm_joint_indices_tensor", None)
        if self._arm_joint_ids is not None:
            all_names = self.unwrapped.robot.data.joint_names
            self._arm_joint_names = [all_names[i] for i in self._arm_joint_ids.tolist()]
            self._episode_joint_pos_at_min_dist = torch.zeros(
                (self._num_envs, len(self._arm_joint_ids)), dtype=torch.float32, device=self._device
            )
        else:
            self._arm_joint_names = []
        self._log_goal_position = hasattr(self.unwrapped, "goal_positions") and hasattr(
            self.unwrapped.scene, "env_origins"
        )

        self._csv = _DualCsvWriter(log_dir, "arm", self._fieldnames(), self._SUMMARY_FIELDS, self._write_summary)
        self.csv_path = self._csv.detailed_path

        self._episode_index = torch.full(
            (self._num_envs,), self._csv.next_episode_index(), dtype=torch.long, device=self._device
        )
        self._episode_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_return = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_min_dist = torch.full((self._num_envs,), float("inf"), dtype=torch.float32, device=self._device)
        self._episode_max_dist = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        # Closest end-effector approach to torso_link this episode — tracks whether the
        # torso-proximity penalty / self-collision fix (Phase 2) is actually relevant,
        # i.e. whether the arm is routinely getting close enough to matter. None until
        # the wrapped env exposes _torso_body_idx (added alongside these Phase 2 changes;
        # older checkpoints/envs without it just won't get this column populated).
        self._episode_min_torso_dist = torch.full(
            (self._num_envs,), float("inf"), dtype=torch.float32, device=self._device
        )

    def _fieldnames(self) -> list[str]:
        base = [
            "episode_index",
            "env_id",
            "env_step",
            "episode_steps",
            "episode_return",
            "mean_reward",
            "outcome",
            "success",
            "min_dist_to_goal_cm",
            "max_dist_to_goal_cm",
            "wobble_active",
            "min_torso_dist_cm",
        ]
        base += [f"{name}_deg_at_min_dist" for name in self._arm_joint_names]
        if self._log_goal_position:
            base += ["goal_x_m", "goal_y_m", "goal_z_m"]
        return base

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_buffers(torch.arange(self._num_envs, device=self._episode_steps.device))
        return obs, info

    def close(self):
        try:
            if hasattr(self, "_csv"):
                self._csv.close()
        finally:
            return super().close()

    def step(self, action):
        env = self.unwrapped
        self._update_step_metrics(env)

        obs, rew, terminated, truncated, info = self.env.step(action)
        self._episode_return += torch.as_tensor(rew, device=self._episode_return.device, dtype=torch.float32)

        done_mask = torch.as_tensor(terminated | truncated, device=env.device)
        if bool(done_mask.any().item()):
            self._flush_finished_episodes(done_mask, terminated, truncated)

        return obs, rew, terminated, truncated, info

    def _reset_buffers(self, env_ids: torch.Tensor):
        self._episode_steps[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._episode_min_dist[env_ids] = float("inf")
        self._episode_max_dist[env_ids] = 0.0
        self._episode_min_torso_dist[env_ids] = float("inf")
        if self._arm_joint_ids is not None:
            self._episode_joint_pos_at_min_dist[env_ids] = 0.0

    def _update_step_metrics(self, env):
        worst_dist = torch.zeros(env.num_envs, device=env.device)
        torso_body_idx = getattr(env, "_torso_body_idx", None)
        closest_torso_dist = torch.full((env.num_envs,), float("inf"), device=env.device)
        for i, arm in enumerate(env._arm_groups):
            ee_pos = env.robot.data.body_pos_w[:, arm["ee_idx"], :]
            dist = torch.linalg.vector_norm(env.goal_positions[:, i, :] - ee_pos, dim=-1)
            worst_dist = torch.maximum(worst_dist, dist)
            if torso_body_idx is not None:
                torso_pos = env.robot.data.body_pos_w[:, torso_body_idx, :]
                torso_dist = torch.linalg.vector_norm(ee_pos - torso_pos, dim=-1)
                closest_torso_dist = torch.minimum(closest_torso_dist, torso_dist)

        self._episode_steps += 1
        worst_dist = worst_dist.detach().float()
        # Capture the joint configuration exactly on the step(s) where a new per-episode
        # minimum distance is set, before overwriting _episode_min_dist below — this is
        # "the pose at closest approach", the thing the joint-config correlation check
        # actually wants (not the pose at episode end, which for a timeout is often just
        # wherever the policy happened to drift to after already passing its best shot).
        if self._arm_joint_ids is not None:
            is_new_min = worst_dist < self._episode_min_dist
            if bool(is_new_min.any().item()):
                current_joint_pos = env.robot.data.joint_pos[:, self._arm_joint_ids]
                self._episode_joint_pos_at_min_dist[is_new_min] = current_joint_pos[is_new_min].detach().float()
        self._episode_min_dist = torch.minimum(self._episode_min_dist, worst_dist)
        self._episode_max_dist = torch.maximum(self._episode_max_dist, worst_dist)
        if torso_body_idx is not None:
            self._episode_min_torso_dist = torch.minimum(
                self._episode_min_torso_dist, closest_torso_dist.detach().float()
            )

    def _flush_finished_episodes(self, done_mask: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor):
        env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        env = self.unwrapped
        env_step = int(env.common_step_counter)
        wobble_cfg = getattr(env, "cfg", None)
        wobble_active = int(
            bool(getattr(wobble_cfg, "enable_root_wobble", False))
            and env_step >= getattr(wobble_cfg, "root_wobble_enable_step", float("inf"))
        )
        for env_id in env_ids.tolist():
            # Use `terminated` (the value returned by this step, captured before any
            # reset happened) rather than re-reading env.successes here — DirectRLEnv
            # auto-resets any env that just finished *inside* self.env.step() itself
            # (before control returns to this wrapper), and _reset_idx explicitly zeroes
            # env.successes as part of that reset. Reading the live attribute after the
            # fact meant `success`/`outcome` were wrong for every single episode that
            # ever succeeded (always read back as False, i.e. "timeout") — a real bug
            # that predates Phase 2, found 2026-07-07 while sanity-checking training
            # data: outcome was "timeout" for 100% of ~710k episodes despite ~46% of
            # them recording a min_dist_to_goal_cm under the 2cm success threshold.
            success = bool(torch.as_tensor(terminated[env_id]).item())
            steps = max(int(self._episode_steps[env_id].item()), 1)
            min_torso_dist = float(self._episode_min_torso_dist[env_id].item())
            row = {
                "episode_index": int(self._episode_index[env_id].item()),
                "env_id": env_id,
                "env_step": env_step,
                "episode_steps": steps,
                "episode_return": float(self._episode_return[env_id].item()),
                "mean_reward": float(self._episode_return[env_id].item() / steps),
                "outcome": "success" if success else "timeout",
                "success": int(success),
                "min_dist_to_goal_cm": float(self._episode_min_dist[env_id].item()) * 100.0,
                "max_dist_to_goal_cm": float(self._episode_max_dist[env_id].item()) * 100.0,
                "wobble_active": wobble_active,
                "min_torso_dist_cm": min_torso_dist * 100.0 if min_torso_dist != float("inf") else "",
            }
            if self._arm_joint_ids is not None:
                joint_pos_deg = torch.rad2deg(self._episode_joint_pos_at_min_dist[env_id]).tolist()
                for name, deg in zip(self._arm_joint_names, joint_pos_deg):
                    row[f"{name}_deg_at_min_dist"] = deg
            if self._log_goal_position:
                # 2026-07-08: goal position wasn't logged at all before — the
                # joint-config-at-min-dist columns above show *what pose the arm ended
                # up in*, but not *what goal it was trying to reach*, so there was no way
                # to check whether a geometric goal-box region (e.g. "far x") actually
                # corresponds to the poses that trigger joint-limit involvement, or
                # whether that assumption was wrong (see known_issues.md — it was).
                # Local frame (env origin subtracted), matching _GOAL_BOUNDS's own
                # convention, so it's directly comparable to the goal-box definition.
                goal_local = (env.goal_positions[env_id, 0, :] - env.scene.env_origins[env_id]).tolist()
                row["goal_x_m"], row["goal_y_m"], row["goal_z_m"] = goal_local
            self._csv.write_row(row)
            self._episode_index[env_id] += 1
        self._reset_buffers(env_ids)
