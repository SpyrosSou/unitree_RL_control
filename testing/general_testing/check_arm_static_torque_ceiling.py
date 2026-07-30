# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone diagnostic (2026-07-29): can the arm task's action pipeline physically HOLD
a gravity-loaded posture at the real hardware gain (40/10), independent of any policy?

Theory under test (see policy_status.md / 2026-07-29 arm analysis): g1_arm_env.py's
``_apply_action`` re-anchors the commanded target to the MEASURED joint position every
control step (``targets = current + delta``, ``|delta| <= max_action_delta_per_step``,
0.06 rad). At any static equilibrium (joint velocity ~0, kd term ~0) the torque an
implicit PD actuator can sustain is therefore capped at ``kp * 0.06`` — 2.4 Nm at kp=40
vs 12 Nm at kp=200 — regardless of what any policy commands. This is a property of the
equilibrium, not of a specific control law: no policy operating through this pipeline can
exceed it while stationary. URDF analysis says typical goal-box reach postures need
4.5-5.7 Nm at shoulder pitch/roll, so most of the box should be un-holdable at 40/10.

Four modes, each trying to hold the same fixed postures for --duration_s from a
zero-velocity teleported start (no RL anywhere — scripted control laws only):

  pipeline_40_10   — the training env's exact mechanism (target = current + clamped
                     delta) driven by a saturating proportional law: emulates the best
                     any policy could do statically within the pipeline.
  integrated_40_10 — the PROPOSED FIX: a persistent target state, rate-limited by the
                     same 0.06 rad/step, that integrates measured error — the target can
                     drift beyond the posture to build up gravity-holding bias (exactly
                     what a policy can learn under absolute/integrated parameterization,
                     and what real deployment's ``default + scale*action`` allows).
  absolute_40_10   — naive fixed target = posture: shows plain PD sag (tau/kp), i.e.
                     what an un-biased absolute target gives. Expect sag ~tau/40.
  pipeline_200_20  — the pipeline at the old 200/20 gain (reference: why the 99.98%
                     checkpoint's gain works — ceiling 12 Nm, above every posture here).

Postures (left arm, [sh_pitch, sh_roll, sh_yaw, elbow, wr_roll, wr_pitch, wr_yaw]) with
their URDF-derived max proximal gravity torque:
  3 "high-tau" reach postures (4.9-5.7 Nm — above the 2.4 Nm 40/10 pipeline ceiling,
  below the 12 Nm 200/20 one) + 1 "control" posture (1.4 Nm — below BOTH ceilings, so
  pipeline_40_10 should succeed there; makes the test falsifiable in both directions).

Expected if the theory is right:
  pipeline_40_10   fails high-tau postures (ee error >> 2cm), holds the control posture
  integrated_40_10 holds everything at the same 40/10 gain
  absolute_40_10   sags by ~tau/kp on high-tau postures
  pipeline_200_20  holds everything
If pipeline_40_10 holds the high-tau postures fine, the theory is WRONG — do not
implement the parameterization change on the strength of the analysis alone.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python testing/general_testing/check_arm_static_torque_ceiling.py --headless
    # or without --headless to watch the sag directly in the viewport
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--duration_s", type=float, default=6.0, help="hold time per posture/mode")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    G1ArmLeftEnvCfg,
)

# Control-rate constants matching the training env (sim dt 1/60, decimation 2 -> 30 Hz).
SIM_DT = 1.0 / 60.0
DECIMATION = 2
MAX_DELTA = 0.06  # rad/control-step — the env's max_action_delta_per_step

# name -> (7-joint posture, URDF-derived max proximal |gravity torque| in Nm, for print)
POSTURES = {
    "forward_reach   (5.7 Nm)": ([-1.05, 0.20, 0.00, 0.50, 0.0, 0.0, 0.0], 5.7),
    "goalbox_typical (5.1 Nm)": ([-0.79, 0.30, 0.20, 1.00, 0.0, 0.0, 0.0], 5.1),
    "lateral_reach   (4.9 Nm)": ([0.00, 1.40, 0.00, 0.30, 0.0, 0.0, 0.0], 4.9),
    "control_low_tau (1.4 Nm)": ([0.10, 0.20, 0.00, 1.50, 0.0, 0.0, 0.0], 1.4),
}

MODES = [
    ("pipeline_40_10", 40.0, 10.0),
    ("integrated_40_10", 40.0, 10.0),
    ("absolute_40_10", 40.0, 10.0),
    ("pipeline_200_20", 200.0, 20.0),
]


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=args_cli.device))

    # The arm task's own robot config (40/10 arm gains, fix_root_link=True, self-collisions
    # on) — single robot, no env/DR/wobble/reset machinery anywhere in this script.
    cfg = G1ArmLeftEnvCfg()
    robot = Articulation(cfg.robot.replace(prim_path="/World/Robot"))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    sim.reset()

    # preserve_order=True so the returned ids match _LEFT_ARM_JOINTS' declaration order —
    # the POSTURES tuples are written in that order, and find_joints' default returns
    # articulation (breadth-first) order instead (lessons_learned #1).
    arm_ids, _ = robot.find_joints(_LEFT_ARM_JOINTS, preserve_order=True)
    arm_ids_t = torch.tensor(arm_ids, dtype=torch.long, device=robot.device)
    ee_ids, _ = robot.find_bodies(_LEFT_EE_BODY)
    ee_idx = ee_ids[0]
    hw_limits = robot.data.soft_joint_pos_limits[0, arm_ids_t].clone()  # (7, 2)
    default_pos = robot.data.default_joint_pos.clone()  # (1, n)

    def step_sim(n_steps: int):
        for _ in range(n_steps):
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)

    def teleport(q_star_arm: torch.Tensor):
        q_full = default_pos.clone()
        q_full[:, arm_ids_t] = q_star_arm
        robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full))
        return q_full

    def kinematic_reference_ee(q_star_arm: torch.Tensor) -> torch.Tensor:
        """Exact ee position for the posture: re-teleport every step for a few steps so
        gravity cannot sag it, then read the ee body position (pure FK via PhysX)."""
        q_full = teleport(q_star_arm)
        for _ in range(5):
            robot.write_joint_state_to_sim(q_full, torch.zeros_like(q_full))
            step_sim(1)
        return robot.data.body_pos_w[0, ee_idx].clone()

    def set_gains(kp: float, kd: float):
        n = len(arm_ids)
        robot.write_joint_stiffness_to_sim(
            torch.full((1, n), kp, device=robot.device), joint_ids=arm_ids
        )
        robot.write_joint_damping_to_sim(
            torch.full((1, n), kd, device=robot.device), joint_ids=arm_ids
        )

    def run_hold(mode: str, kp: float, kd: float, q_star_arm: torch.Tensor, ee_star: torch.Tensor):
        set_gains(kp, kd)
        teleport(q_star_arm)
        step_sim(1)  # let PhysX ingest the teleported state before controlling

        persistent_target = q_star_arm.clone()
        n_ticks = int(args_cli.duration_s / (SIM_DT * DECIMATION))
        last_second_err = []
        for tick in range(n_ticks):
            q_arm = robot.data.joint_pos[:, arm_ids_t]
            if mode.startswith("pipeline"):
                # The env's mechanism: target re-anchored to MEASURED position, delta
                # capped. High-gain saturating P-law, NOT plain clamp(q*-q): a trained
                # policy can hold a delta bias up to the full 0.06 even at near-zero
                # position error (filtered_actions saturate independently of error), so
                # a fair emulation must be able to do the same — k=10 means errors down
                # to 6 mrad still command the full bias. The static ceiling argument is
                # law-independent either way: at equilibrium tau = kp*delta <= kp*0.06.
                delta = (10.0 * (q_star_arm - q_arm)).clamp(-MAX_DELTA, MAX_DELTA)
                target = q_arm + delta
            elif mode.startswith("integrated"):
                # Proposed fix: same per-step rate limit, but applied to a persistent
                # target state that integrates measured error — can build a gravity bias.
                persistent_target = persistent_target + (0.5 * (q_star_arm - q_arm)).clamp(
                    -MAX_DELTA, MAX_DELTA
                )
                target = persistent_target
            else:  # absolute
                target = q_star_arm.clone()
            target = target.clamp(hw_limits[:, 0], hw_limits[:, 1])

            robot.set_joint_position_target(default_pos.clone())  # hold rest of the body
            robot.set_joint_position_target(target, joint_ids=arm_ids)
            step_sim(DECIMATION)

            if tick >= n_ticks - int(1.0 / (SIM_DT * DECIMATION)):  # last 1 s
                ee = robot.data.body_pos_w[0, ee_idx]
                last_second_err.append(torch.norm(ee - ee_star).item())

        q_arm = robot.data.joint_pos[:, arm_ids_t]
        joint_err_deg = torch.rad2deg((q_arm - q_star_arm).abs()).max().item()
        # Implied static holding torque per joint = kp * (target - q); max over the 4
        # proximal joints — in pipeline mode this should saturate at kp * 0.06.
        implied_tau = (kp * (target - q_arm).abs())[0, :4].max().item()
        ee_err_cm = 100.0 * sum(last_second_err) / len(last_second_err)
        return ee_err_cm, joint_err_deg, implied_tau

    print(f"[Probe] duration per hold: {args_cli.duration_s:.1f}s | control 30 Hz | "
          f"max_delta {MAX_DELTA} rad | success threshold reference: 2 cm")
    print(f"[Probe] static-torque ceilings: kp=40 -> {40*MAX_DELTA:.1f} Nm, "
          f"kp=200 -> {200*MAX_DELTA:.1f} Nm\n")

    results: dict[str, dict[str, tuple]] = {}
    with torch.inference_mode():
        for pname, (q_list, _tau) in POSTURES.items():
            q_star = torch.tensor([q_list], device=robot.device)
            ee_star = kinematic_reference_ee(q_star)
            results[pname] = {}
            for mode, kp, kd in MODES:
                ee_err_cm, joint_err_deg, implied_tau = run_hold(mode, kp, kd, q_star, ee_star)
                results[pname][mode] = (ee_err_cm, joint_err_deg, implied_tau)

    header = f"{'posture':<26}" + "".join(f"{m:>20}" for m, _, _ in MODES)
    print(header)
    print("-" * len(header))
    print("ee error, cm (mean over final 1 s; >2 cm = cannot hold the success zone)")
    for pname, per_mode in results.items():
        print(f"{pname:<26}" + "".join(f"{per_mode[m][0]:>20.2f}" for m, _, _ in MODES))
    print("\nmax per-joint error at end, deg")
    for pname, per_mode in results.items():
        print(f"{pname:<26}" + "".join(f"{per_mode[m][1]:>20.2f}" for m, _, _ in MODES))
    print("\nimplied static holding torque at end, Nm (max over 4 proximal joints)")
    for pname, per_mode in results.items():
        print(f"{pname:<26}" + "".join(f"{per_mode[m][2]:>20.2f}" for m, _, _ in MODES))

    high_tau = [p for p in POSTURES if "control" not in p]
    control = [p for p in POSTURES if "control" in p]
    pipeline_fails_high = all(results[p]["pipeline_40_10"][0] > 2.0 for p in high_tau)
    pipeline_holds_ctrl = all(results[p]["pipeline_40_10"][0] < 2.0 for p in control)
    integrated_holds_all = all(results[p]["integrated_40_10"][0] < 2.0 for p in POSTURES)
    ref_holds_all = all(results[p]["pipeline_200_20"][0] < 2.0 for p in POSTURES)

    print("\n[Probe] === VERDICT ===")
    print(f"[Probe] pipeline_40_10 fails all high-tau postures : {pipeline_fails_high}")
    print(f"[Probe] pipeline_40_10 holds the low-tau control   : {pipeline_holds_ctrl}")
    print(f"[Probe] integrated_40_10 holds ALL postures        : {integrated_holds_all}")
    print(f"[Probe] pipeline_200_20 holds ALL postures         : {ref_holds_all}")
    if pipeline_fails_high and integrated_holds_all and ref_holds_all:
        print("[Probe] THEORY CONFIRMED: the current+delta pipeline is the binding "
              "constraint at 40/10; the integrated-target parameterization removes it "
              "at the same gain. Proceed with the env change.")
    elif not pipeline_fails_high:
        print("[Probe] THEORY NOT CONFIRMED: pipeline_40_10 held high-torque postures. "
              "Do NOT change the action parameterization on this basis — re-examine.")
    else:
        print("[Probe] MIXED RESULT — inspect the tables above before acting.")


if __name__ == "__main__":
    main()
    simulation_app.close()
