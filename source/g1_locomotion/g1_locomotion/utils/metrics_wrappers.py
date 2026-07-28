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
import math
import os

import gymnasium as gym
import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

from g1_locomotion.tasks.manager_based.g1_locomotion.mdp.events import ArmMotionDisturbance


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

    Mirrors ``ArmMotionDisturbance._phase_index`` using the class's default
    phase boundaries (the ones actually used during training; only the *_PLAY* configs
    override them for faster visual demos). Kept here instead of read off the live event
    term instance so a finished-episode row can be labelled without holding a reference
    to the event manager.
    """
    for i, boundary in enumerate(ArmMotionDisturbance._PHASE_STEP_BOUNDARIES):
        if common_step_counter < boundary:
            return i
    return len(ArmMotionDisturbance._PHASE_STEP_BOUNDARIES)


def _check_header_matches(path: str, expected_fieldnames: list[str]) -> None:
    """Guard against appending rows with a different schema than what's already on disk.

    2026-07-28: found the hard way — these wrappers open existing CSVs in append mode and
    only write a header when the file is empty, so a column added to a wrapper's
    fieldnames after a checkpoint's eval directory already has data in it (these live at
    a fixed, checkpoint-keyed path, e.g. `<checkpoint_dir>/walking_eval/`, reused across
    every eval invocation against that checkpoint, not a fresh timestamped directory —
    intentional, so episode counts can accumulate across multiple eval sessions) means the
    file's header silently stays stale forever. A reader (`csv.DictReader`, which takes the
    header from row 1) then either KeyErrors on the new column or, worse, silently
    misaligns if the column *count* happens to still match. Fail loudly here instead of
    letting either of those happen quietly.
    """
    with open(path, newline="") as f:
        existing_header = next(csv.reader(f), None)
    if existing_header is not None and existing_header != expected_fieldnames:
        raise RuntimeError(
            f"{path} already exists with a different column schema than the current code "
            f"produces (existing: {existing_header}; expected: {expected_fieldnames}). This "
            "file is at a fixed, checkpoint-keyed path reused across eval runs — it almost "
            "certainly predates a metrics/fieldname change. Delete or move the containing "
            "eval directory and re-run rather than appending mismatched rows into it."
        )


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
        else:
            _check_header_matches(self.detailed_path, detailed_fieldnames)

        if self._write_summary:
            self.summary_path = os.path.join(log_dir, f"{name}_summary.csv")
            self._summary_file = open(self.summary_path, "a", newline="")
            self._summary_writer = csv.DictWriter(self._summary_file, fieldnames=summary_fieldnames)
            if self._summary_file.tell() == 0:
                self._summary_writer.writeheader()
                self._summary_file.flush()
            else:
                _check_header_matches(self.summary_path, summary_fieldnames)
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
        # Torso-joint rotation (2026-07-15): added after visually confirming the Phase 1
        # consolidated checkpoint holds a rotated torso from the start as its bracing
        # strategy — which displaces the shoulder relative to the pelvis-frame goal box
        # and swallowed the near-inner reach workspace (goals 100% reachable at 1.8cm
        # under a nominal posture timed out at 15-20cm under the rotated one). These
        # columns quantify what until now was only a visual impression in g1_full_demo.
        self._episode_max_abs_torso = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_sum_abs_torso = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._torso_joint_id: int | None = None
        self._prev_actions: torch.Tensor | None = None
        self._foot_body_ids: list[int] | None = None

        # Per-joint diagnostics (2026-07-20): run-wide (NOT per-episode-reset) max |joint
        # position| across every joint, written to a separate joint_diagnostics.csv on
        # close() — added after a checkpoint was found pinning torso_joint at its exact
        # hardware limit (150 deg) for entire episodes, discovered only because that one
        # joint already had its own dedicated column; this generalizes the check to every
        # joint instead of relying on having hand-picked the right one to look at. Reads
        # straight from robot.data.joint_pos/joint_names/soft_joint_pos_limits, which are
        # guaranteed mutually consistent (same asset, same indexing) by construction —
        # deliberately NOT correlated against the raw action tensor's own column order,
        # which comes from a separate resolve (the action term's own find_joints call) and
        # is not guaranteed to match robot.data's ordering (see events.py's _col_idx fix
        # earlier the same day: assuming two independently-resolved joint orderings agree
        # was exactly the bug there). Position alone answers "which joint is pinned,"
        # without needing two orderings to agree.
        self._joint_names: list[str] | None = None
        self._episode_max_abs_pos_per_joint: torch.Tensor | None = None

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
            "max_abs_torso_deg",
            "mean_abs_torso_deg",
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
            self._write_joint_diagnostics()
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
        self._episode_max_abs_torso[env_ids] = 0.0
        self._episode_sum_abs_torso[env_ids] = 0.0
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

        # 2026-07-28 fix: current_contact_time reads exactly 0 BOTH the instant contact
        # resumes AND every tick a foot is still fully airborne (there's no "time in
        # contact" while not in contact) — `<= control_dt` alone can't tell those apart,
        # and matched the latter far more often in practice, overcounting massively
        # (verified against real per-tick sensor data via
        # testing/general_testing/check_step_count_metric.py: current_contact_time stuck
        # at 0.0 for many consecutive ticks while current_air_time grew the whole time).
        # Requiring > 0.0 too narrows the window to the single tick contact genuinely
        # just began.
        just_touched_down = (current_contact_time > 0.0) & (current_contact_time <= control_dt)
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
        if self._torso_joint_id is None:
            # "torso_joint" (2026-07-21): the 29dof asset has no single torso joint —
            # replaced by a 3-DOF waist (waist_yaw/roll/pitch). waist_yaw is the direct
            # analog of the old joint (the 23dof-era code's own comment already
            # described "torso_joint" as "a waist YAW joint" — this column always meant
            # torso *twist* specifically, not general waist motion), so this column
            # keeps its original meaning and name unchanged, just repointed to the
            # equivalent joint on the new asset. See 29dof_implementation_plan.md.
            torso_ids, _ = robot.find_joints("waist_yaw_joint")
            self._torso_joint_id = torso_ids[0]
        abs_torso = robot.data.joint_pos[:, self._torso_joint_id].abs().detach().float()
        self._episode_max_abs_torso = torch.maximum(self._episode_max_abs_torso, abs_torso)
        self._episode_sum_abs_torso += abs_torso

        if self._joint_names is None:
            self._joint_names = list(robot.data.joint_names)
            self._episode_max_abs_pos_per_joint = torch.zeros(
                (self._num_envs, len(self._joint_names)), dtype=torch.float32, device=self._device
            )
        self._episode_max_abs_pos_per_joint = torch.maximum(
            self._episode_max_abs_pos_per_joint, robot.data.joint_pos.abs().detach().float()
        )

        self._update_stepping_metrics(env)
        self._prev_actions = action_tensor.detach().clone()

    def _write_joint_diagnostics(self):
        """One row per joint, run-wide max |position| across every env/episode this
        wrapper has seen, vs. that joint's own soft/hard limits — see the __init__
        comment on _episode_max_abs_pos_per_joint for why this exists and why it's
        position-based rather than correlated against the action tensor."""
        if self._joint_names is None:
            return
        robot = self.unwrapped.scene["robot"]
        joint_ids = list(range(len(self._joint_names)))
        soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids]  # (N, 2), same for every env
        hard_limits = robot.data.joint_pos_limits[0, joint_ids]
        run_max_abs_pos = self._episode_max_abs_pos_per_joint.amax(dim=0)  # worst env per joint

        path = os.path.join(self.log_dir, "joint_diagnostics.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "joint_id", "joint_name", "max_abs_pos_rad", "max_abs_pos_deg",
                "soft_limit_lower_deg", "soft_limit_upper_deg",
                "hard_limit_lower_deg", "hard_limit_upper_deg",
                "frac_of_soft_limit_used",
            ])
            for i, name in enumerate(self._joint_names):
                max_pos = float(run_max_abs_pos[i].item())
                soft_lo, soft_hi = float(soft_limits[i, 0].item()), float(soft_limits[i, 1].item())
                hard_lo, hard_hi = float(hard_limits[i, 0].item()), float(hard_limits[i, 1].item())
                soft_extent = max(abs(soft_lo), abs(soft_hi), 1e-6)
                writer.writerow([
                    i, name,
                    f"{max_pos:.4f}", f"{math.degrees(max_pos):.2f}",
                    f"{math.degrees(soft_lo):.2f}", f"{math.degrees(soft_hi):.2f}",
                    f"{math.degrees(hard_lo):.2f}", f"{math.degrees(hard_hi):.2f}",
                    f"{max_pos / soft_extent:.3f}",
                ])
        print(f"[Metrics] Per-joint diagnostics: {path}")

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
                "max_abs_torso_deg": float(
                    torch.rad2deg(self._episode_max_abs_torso[env_id]).item()
                ),
                "mean_abs_torso_deg": float(
                    torch.rad2deg(
                        self._episode_sum_abs_torso[env_id] / max(int(self._episode_steps[env_id].item()), 1)
                    ).item()
                ),
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
    _LEFT_FOOT_BODY_NAME_PATTERN = "left_ankle_roll_link"
    _RIGHT_FOOT_BODY_NAME_PATTERN = "right_ankle_roll_link"
    _LEFT_KNEE_JOINT_NAME = "left_knee_joint"
    _RIGHT_KNEE_JOINT_NAME = "right_knee_joint"

    # Same threshold/mechanism as StandingMetricsCsvWrapper._update_stepping_metrics —
    # distinguishes a real corrective step (foot genuinely airborne) from contact-sensor
    # noise around continuous ground contact.
    _STEP_AIR_TIME_THRESHOLD_S = 0.05

    # 2026-07-28: per-side breakdown, added because every metric above (knee angle, step
    # count, air time) collapses left+right into one number (mean/max across both feet),
    # which makes a real, user-visually-suspected asymmetry ("left leg favored,
    # causing isolated imbalances") impossible to confirm or rule out after the fact —
    # see the g1_rl_control/policy_status discussion 2026-07-28. Left/right here always
    # means the robot's own left/right (not screen-relative).
    _DRIFT_SNAPSHOT_INTERVAL_S = 2.0
    _MAX_DRIFT_SNAPSHOTS = 10  # 20s episode / 2s cadence — see episode_length_s in the env cfg

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
        "step_count",
        "left_step_count",
        "right_step_count",
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

        # 2026-07-27: knee-angle tracking, added to check a visually-observed hypothesis
        # (round-2 drift checkpoint) that near-extended knees during straight-line walking
        # correlate with the asymmetric-step/foot-crossing corrections that precede bad
        # heading drift — see deferred_items_2026-07-21.md item 8. Knee joint angle
        # convention: 0 rad = fully extended/straight leg, larger = more bent (default
        # pose is 0.3 rad, see UNITREE_G1_29DOF_CFG). min_knee_angle_deg is the single
        # most-extended moment reached during the episode (both knees, whole episode).
        self._knee_joint_ids: list[int] | None = None
        self._episode_sum_knee_angle = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_min_knee_angle = torch.full(
            (self._num_envs,), float("inf"), dtype=torch.float32, device=self._device
        )

        # 2026-07-28: real stepping-event count, added because drift (net displacement)
        # and foot-slip (sliding while in contact) both miss the specific failure the user
        # visually flagged — a foot lifting and re-planting nearby, which can look like
        # "stationary stepping" while netting out to good drift numbers. Same
        # air-time-threshold mechanism as StandingMetricsCsvWrapper._update_stepping_metrics
        # (ported here rather than switching eval_walking.py to that class, since this one
        # already carries the drift/knee-angle tracking that class doesn't have).
        self._episode_step_count = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_max_foot_air_time = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

        # Per-side breakdown (see class docstring note above) — mirrors the combined
        # fields above but split left/right instead of pooled.
        self._left_foot_body_ids_robot: list[int] | None = None
        self._left_foot_body_ids_sensor: list[int] | None = None
        self._right_foot_body_ids_robot: list[int] | None = None
        self._right_foot_body_ids_sensor: list[int] | None = None
        self._left_knee_joint_ids: list[int] | None = None
        self._right_knee_joint_ids: list[int] | None = None
        self._episode_sum_knee_angle_left = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_sum_knee_angle_right = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_min_knee_angle_left = torch.full(
            (self._num_envs,), float("inf"), dtype=torch.float32, device=self._device
        )
        self._episode_min_knee_angle_right = torch.full(
            (self._num_envs,), float("inf"), dtype=torch.float32, device=self._device
        )
        self._episode_step_count_left = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_step_count_right = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._episode_max_foot_air_time_left = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._episode_max_foot_air_time_right = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)

        # 2026-07-28: fixed-cadence heading-drift snapshots, added because the final
        # per-episode drift number alone can't distinguish "one sudden shock mid-episode
        # that the robot never recovers a clean heading from" vs. "slow continuous yaw-rate
        # bias accumulating the whole time" — both produce the same final number. NaN
        # means the episode ended (fell) before reaching that checkpoint, not "drift was
        # exactly zero" — see _flush_finished_episodes.
        self._drift_snapshots = torch.full(
            (self._num_envs, self._MAX_DRIFT_SNAPSHOTS), float("nan"), dtype=torch.float32, device=self._device
        )

    @staticmethod
    def _fieldnames() -> list[str]:
        snapshot_cols = [
            f"heading_drift_deg_t{int(round((i + 1) * WalkingMetricsCsvWrapper._DRIFT_SNAPSHOT_INTERVAL_S))}s"
            for i in range(WalkingMetricsCsvWrapper._MAX_DRIFT_SNAPSHOTS)
        ]
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
            "mean_knee_angle_deg",
            "min_knee_angle_deg",
            "left_knee_angle_deg",
            "right_knee_angle_deg",
            "min_left_knee_angle_deg",
            "min_right_knee_angle_deg",
            "step_count",
            "max_foot_air_time_s",
            "left_step_count",
            "right_step_count",
            "left_max_foot_air_time_s",
            "right_max_foot_air_time_s",
        ] + snapshot_cols

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
        self._episode_sum_knee_angle[env_ids] = 0.0
        self._episode_min_knee_angle[env_ids] = float("inf")
        self._episode_step_count[env_ids] = 0
        self._episode_max_foot_air_time[env_ids] = 0.0
        self._episode_sum_knee_angle_left[env_ids] = 0.0
        self._episode_sum_knee_angle_right[env_ids] = 0.0
        self._episode_min_knee_angle_left[env_ids] = float("inf")
        self._episode_min_knee_angle_right[env_ids] = float("inf")
        self._episode_step_count_left[env_ids] = 0
        self._episode_step_count_right[env_ids] = 0
        self._episode_max_foot_air_time_left[env_ids] = 0.0
        self._episode_max_foot_air_time_right[env_ids] = 0.0
        self._drift_snapshots[env_ids] = float("nan")

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
        self._update_knee_metrics(robot)
        self._update_stepping_metrics(env)
        self._update_drift_snapshot(env)

    def _update_knee_metrics(self, robot):
        if self._knee_joint_ids is None:
            self._knee_joint_ids, _ = robot.find_joints([".*_knee_joint"])
        knee_pos = robot.data.joint_pos[:, self._knee_joint_ids].detach().float()
        self._episode_sum_knee_angle += knee_pos.mean(dim=-1)
        self._episode_min_knee_angle = torch.minimum(self._episode_min_knee_angle, knee_pos.amin(dim=-1))

        if self._left_knee_joint_ids is None:
            self._left_knee_joint_ids, _ = robot.find_joints([self._LEFT_KNEE_JOINT_NAME])
            self._right_knee_joint_ids, _ = robot.find_joints([self._RIGHT_KNEE_JOINT_NAME])
        left_knee_pos = robot.data.joint_pos[:, self._left_knee_joint_ids].detach().float().squeeze(-1)
        right_knee_pos = robot.data.joint_pos[:, self._right_knee_joint_ids].detach().float().squeeze(-1)
        self._episode_sum_knee_angle_left += left_knee_pos
        self._episode_sum_knee_angle_right += right_knee_pos
        self._episode_min_knee_angle_left = torch.minimum(self._episode_min_knee_angle_left, left_knee_pos)
        self._episode_min_knee_angle_right = torch.minimum(self._episode_min_knee_angle_right, right_knee_pos)

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

    def _update_stepping_metrics(self, env):
        """Count real corrective steps (foot lift + re-plant), independent of whether they
        net out to ~0 displacement — see StandingMetricsCsvWrapper._update_stepping_metrics
        for the identical mechanism this mirrors. Net drift alone can look fine even while
        the robot is visibly taking steps in place; this catches that directly."""
        contact_sensor = env.scene.sensors.get("contact_forces")
        if contact_sensor is None or contact_sensor.data.last_air_time is None:
            return

        if self._foot_body_ids_sensor is None:
            self._foot_body_ids_robot, _ = env.scene["robot"].find_bodies(self._FOOT_BODY_NAME_PATTERN)
            self._foot_body_ids_sensor, _ = contact_sensor.find_bodies(self._FOOT_BODY_NAME_PATTERN)

        last_air_time = contact_sensor.data.last_air_time[:, self._foot_body_ids_sensor]
        current_contact_time = contact_sensor.data.current_contact_time[:, self._foot_body_ids_sensor]
        control_dt = env.step_dt

        # 2026-07-28 fix: current_contact_time reads exactly 0 BOTH the instant contact
        # resumes AND every tick a foot is still fully airborne (there's no "time in
        # contact" while not in contact) — `<= control_dt` alone can't tell those apart,
        # and matched the latter far more often in practice, overcounting massively
        # (verified against real per-tick sensor data via
        # testing/general_testing/check_step_count_metric.py: current_contact_time stuck
        # at 0.0 for many consecutive ticks while current_air_time grew the whole time).
        # Requiring > 0.0 too narrows the window to the single tick contact genuinely
        # just began.
        just_touched_down = (current_contact_time > 0.0) & (current_contact_time <= control_dt)
        was_a_real_step = last_air_time > self._STEP_AIR_TIME_THRESHOLD_S
        step_this_tick = (just_touched_down & was_a_real_step).any(dim=-1)
        self._episode_step_count += step_this_tick.long()

        current_air_time = contact_sensor.data.current_air_time[:, self._foot_body_ids_sensor]
        self._episode_max_foot_air_time = torch.maximum(
            self._episode_max_foot_air_time, current_air_time.amax(dim=-1).detach().float()
        )

        # Per-side breakdown — identical mechanism, just one foot at a time instead of
        # pooled across both (see class docstring note on why this was added).
        if self._left_foot_body_ids_sensor is None:
            self._left_foot_body_ids_robot, _ = env.scene["robot"].find_bodies(self._LEFT_FOOT_BODY_NAME_PATTERN)
            self._left_foot_body_ids_sensor, _ = contact_sensor.find_bodies(self._LEFT_FOOT_BODY_NAME_PATTERN)
            self._right_foot_body_ids_robot, _ = env.scene["robot"].find_bodies(self._RIGHT_FOOT_BODY_NAME_PATTERN)
            self._right_foot_body_ids_sensor, _ = contact_sensor.find_bodies(self._RIGHT_FOOT_BODY_NAME_PATTERN)

        for side, sensor_ids, sum_count, sum_air_time in (
            ("left", self._left_foot_body_ids_sensor, "_episode_step_count_left", "_episode_max_foot_air_time_left"),
            ("right", self._right_foot_body_ids_sensor, "_episode_step_count_right", "_episode_max_foot_air_time_right"),
        ):
            # Re-slice from the sensor's raw (full-width) data, NOT from
            # last_air_time/current_contact_time/current_air_time above — those are
            # already narrowed to the combined 2-wide foot index space
            # (self._foot_body_ids_sensor), so indexing them with sensor_ids (indices
            # into the sensor's FULL body list) is an out-of-bounds access whenever the
            # sensor tracks more than just the two feet (confirmed via a real CUDA
            # device-side assert, 2026-07-28 — see g1_rl_control chat notes).
            side_contact = contact_sensor.data.current_contact_time[:, sensor_ids]
            side_last_air = contact_sensor.data.last_air_time[:, sensor_ids]
            side_air_now = contact_sensor.data.current_air_time[:, sensor_ids]
            side_touched_down = (side_contact > 0.0) & (side_contact <= control_dt)
            side_real_step = side_last_air > self._STEP_AIR_TIME_THRESHOLD_S
            side_step_this_tick = (side_touched_down & side_real_step).any(dim=-1)
            setattr(self, sum_count, getattr(self, sum_count) + side_step_this_tick.long())
            setattr(
                self, sum_air_time,
                torch.maximum(getattr(self, sum_air_time), side_air_now.amax(dim=-1).detach().float()),
            )

    def _update_drift_snapshot(self, env):
        """Record heading_drift_deg at a fixed cadence during the episode (not just the
        final value) — see class docstring on _DRIFT_SNAPSHOT_INTERVAL_S for why: the
        final number alone can't tell a sudden mid-episode shock apart from slow
        continuous drift, and both look identical in the existing single-column output."""
        interval_steps = max(1, round(self._DRIFT_SNAPSHOT_INTERVAL_S / env.step_dt))
        due = (self._episode_steps % interval_steps == 0) & (self._episode_steps > 0)
        if not bool(due.any()):
            return
        for env_id in due.nonzero(as_tuple=False).squeeze(-1).tolist():
            slot = int(self._episode_steps[env_id].item()) // interval_steps - 1
            if slot >= self._MAX_DRIFT_SNAPSHOTS:
                continue
            heading_drift_deg, _ = self._drift_for_env(env_id)
            self._drift_snapshots[env_id, slot] = heading_drift_deg

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
                "mean_knee_angle_deg": math.degrees(self._episode_sum_knee_angle[env_id].item() / steps),
                "min_knee_angle_deg": math.degrees(self._episode_min_knee_angle[env_id].item()),
                "left_knee_angle_deg": math.degrees(self._episode_sum_knee_angle_left[env_id].item() / steps),
                "right_knee_angle_deg": math.degrees(self._episode_sum_knee_angle_right[env_id].item() / steps),
                "min_left_knee_angle_deg": math.degrees(self._episode_min_knee_angle_left[env_id].item()),
                "min_right_knee_angle_deg": math.degrees(self._episode_min_knee_angle_right[env_id].item()),
                "step_count": int(self._episode_step_count[env_id].item()),
                "max_foot_air_time_s": float(self._episode_max_foot_air_time[env_id].item()),
                "left_step_count": int(self._episode_step_count_left[env_id].item()),
                "right_step_count": int(self._episode_step_count_right[env_id].item()),
                "left_max_foot_air_time_s": float(self._episode_max_foot_air_time_left[env_id].item()),
                "right_max_foot_air_time_s": float(self._episode_max_foot_air_time_right[env_id].item()),
            }
            for i in range(self._MAX_DRIFT_SNAPSHOTS):
                col = f"heading_drift_deg_t{int(round((i + 1) * self._DRIFT_SNAPSHOT_INTERVAL_S))}s"
                val = self._drift_snapshots[env_id, i].item()
                row[col] = "" if math.isnan(val) else val
            self._csv.write_row(row)
            self._episode_index[env_id] += 1
        self._reset_buffers(env_ids)


class ArmMetricsCsvWrapper(gym.Wrapper):
    """Collect per-episode reach-quality metrics for the arm IK task."""

    # 2026-07-28: distance-to-goal over time, added to directly test a real hypothesis
    # (g1_full_demo.py visual observation) that failed episodes are running out of the
    # episode's time budget while still closing in on the goal, rather than plateauing
    # somewhere and never improving further — min_dist_to_goal_cm alone can't distinguish
    # these (both produce "min distance stayed above 2cm"). final_dist_to_goal_cm (the
    # LAST tick's distance, not the best-ever) plus fixed-cadence snapshots give the
    # actual trajectory shape: final_dist_to_goal_cm close to min_dist_to_goal_cm AND
    # still dropping in the last snapshot(s) means "still converging at timeout" (the
    # speed hypothesis); final_dist_to_goal_cm much larger than the min (or flat across
    # the last several snapshots) means "reached its best point then stalled/overshot",
    # a different failure mode entirely.
    _DIST_SNAPSHOT_INTERVAL_S = 1.0

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

        # Sized dynamically off the wrapped env's own step_dt/max_episode_length rather
        # than a hardcoded episode length, so this works unchanged whether it's wrapping
        # the standard 10s/300-step task, GoalCurriculum, or the 45s LongHold eval class.
        self._dist_snapshot_interval_steps = max(
            1, round(self._DIST_SNAPSHOT_INTERVAL_S / self.unwrapped.step_dt)
        )
        self._max_dist_snapshots = max(
            1, int(self.unwrapped.max_episode_length) // self._dist_snapshot_interval_steps
        )
        self._episode_last_dist = torch.zeros(self._num_envs, dtype=torch.float32, device=self._device)
        self._dist_snapshots = torch.full(
            (self._num_envs, self._max_dist_snapshots), float("nan"), dtype=torch.float32, device=self._device
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
            "final_dist_to_goal_cm",
        ]
        base += [f"{name}_deg_at_min_dist" for name in self._arm_joint_names]
        if self._log_goal_position:
            base += ["goal_x_m", "goal_y_m", "goal_z_m"]
        base += [
            f"dist_to_goal_cm_t{round((i + 1) * self._DIST_SNAPSHOT_INTERVAL_S, 1)}s"
            for i in range(self._max_dist_snapshots)
        ]
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
        self._episode_last_dist[env_ids] = 0.0
        self._dist_snapshots[env_ids] = float("nan")
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
        self._episode_last_dist = worst_dist
        if torso_body_idx is not None:
            self._episode_min_torso_dist = torch.minimum(
                self._episode_min_torso_dist, closest_torso_dist.detach().float()
            )
        self._update_dist_snapshot()

    def _update_dist_snapshot(self):
        """Record dist-to-goal at a fixed cadence during the episode — see class
        docstring on _DIST_SNAPSHOT_INTERVAL_S for why: min/max distance alone can't show
        the trajectory shape (still converging vs. plateaued vs. overshot)."""
        due = (self._episode_steps % self._dist_snapshot_interval_steps == 0) & (self._episode_steps > 0)
        if not bool(due.any()):
            return
        for env_id in due.nonzero(as_tuple=False).squeeze(-1).tolist():
            slot = int(self._episode_steps[env_id].item()) // self._dist_snapshot_interval_steps - 1
            if slot >= self._max_dist_snapshots:
                continue
            self._dist_snapshots[env_id, slot] = self._episode_last_dist[env_id]

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
                "final_dist_to_goal_cm": float(self._episode_last_dist[env_id].item()) * 100.0,
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
            for i in range(self._max_dist_snapshots):
                col = f"dist_to_goal_cm_t{round((i + 1) * self._DIST_SNAPSHOT_INTERVAL_S, 1)}s"
                val = self._dist_snapshots[env_id, i].item()
                row[col] = "" if math.isnan(val) else val * 100.0
            self._csv.write_row(row)
            self._episode_index[env_id] += 1
        self._reset_buffers(env_ids)
