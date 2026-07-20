# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FK-verified accuracy test for the analytic-IK arm driving used in standing training.

Motivation (2026-07-16, see definitive_next_steps.md Step 1): `event_arm_goals.csv`
logging revealed that `StandingArmIKReachDisturbance` — the training disturbance every
standing checkpoint since the IKReach era was trained against, and (via
`--arm_driver event`) the trusted deployment arm driver — misses its reach goals by
p50 ≈ 39-54cm with ~0% of attempts within the 3cm threshold. The standing policies were
therefore trained against arm-WAVING, not arm-REACHING, and the reach deliverable is not
met. Prime suspect: the floating-base Jacobian indexing in `_jacobian_b`
(`_jacobi_joint_ids = joint_ids + 6`, body index unshifted), vs. PD tracking droop at the
soft 60/1.5 arm gain (which should explain ~10cm at most, not 50).

Method: drives the REAL `StandingArmIKReachDisturbance` class — not a copy — against a
minimal standalone scene (a duck-typed env shim provides the few attributes the term
reads: `scene`, `device`, `common_step_counter`), so the exact training/deployment code
path is what gets measured, byte for byte. The G1 is spawned either fixed-base
(exercises the term's fixed-base Jacobian branch) or floating-base with the root
kinematically re-pinned every physics step (exercises the floating-base `+6` branch that
training actually uses, without turning this into a balance task). Goals cycle exactly
as in training (full `_GOAL_BOUNDS` range, 15s per goal at 50Hz control, 0.06 rad/step
rate limit), and forward kinematics — palm position read straight from the sim — gives
the ground-truth hand-to-goal distance the disturbance itself never checked.

The `--track kinematic` mode bypasses arm dynamics entirely (the IK term's rate-limited
joint targets are written directly as joint state each control step, i.e. perfect
tracking), isolating the IK solve's correctness from PD droop. Decision matrix:

    fixed+kinematic GOOD, floating+kinematic BAD  -> floating-base Jacobian indexing bug
                                                     (fix events.py's `else` branch; the
                                                     fix propagates to training AND
                                                     deployment automatically)
    kinematic BAD everywhere                      -> controller usage/settings wrong
                                                     (DLS + rate limit combination, frame
                                                     conventions, ...)
    kinematic GOOD, pd BAD at 60/1.5              -> tracking droop, not solver error:
                                                     compare --arm_gain 200 20 to size
                                                     the gain contribution

Usage (each run ~2-3 min):
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python3 validation/check_ik_accuracy.py --base fixed    --track kinematic --headless
    python3 validation/check_ik_accuracy.py --base floating --track kinematic --headless
    python3 validation/check_ik_accuracy.py --base floating --track pd --headless
    python3 validation/check_ik_accuracy.py --base floating --track pd --arm_gain 200 20 --headless

Output: printed per-arm reach stats (p50/p90, % within 3/5/10cm, per-axis error bias,
rate-limit saturation at goal end), a per-goal CSV, and run_meta.yaml, all under
validation/ik_accuracy/<timestamp>_<base>_<track>/.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FK-verified accuracy test for StandingArmIKReachDisturbance's IK.")
parser.add_argument("--base", type=str, default="floating", choices=["fixed", "floating"],
                    help="Root mode: 'floating' (root kinematically pinned; the Jacobian branch training uses) or "
                         "'fixed' (fix_root_link=True; the term's fixed-base branch).")
parser.add_argument("--track", type=str, default="pd", choices=["pd", "kinematic"],
                    help="'pd': arm PD-tracks the IK targets at --arm_gain (training/deployment realism). "
                         "'kinematic': IK targets written directly as joint state (perfect tracking — isolates "
                         "the IK solve from PD droop; --arm_gain is irrelevant).")
parser.add_argument("--arm_gain", type=float, nargs=2, default=[60.0, 1.5], metavar=("KP", "KD"),
                    help="Arm actuator stiffness/damping in pd mode. 60/1.5 = the standing env's trained gain "
                         "(the condition the 39-54cm misses were measured under); 200/20 = deployment's active-arm gain.")
parser.add_argument("--arm", type=str, default="both", choices=["left", "right", "both"],
                    help="Which arm(s) to drive. Default both (they are independent chains).")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs (each samples its own goals).")
parser.add_argument("--num_goals", type=int, default=4, help="Goal rounds per env (each held --seconds_per_goal).")
parser.add_argument("--seconds_per_goal", type=float, default=15.0,
                    help="Seconds per goal at 50Hz control — matches training's max_steps_per_goal (750 = 15s).")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import csv
import datetime
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG

from g1_locomotion.tasks.manager_based.g1_locomotion.mdp.events import StandingArmIKReachDisturbance
from g1_locomotion.utils.eval_meta import write_eval_meta

# Standing training's control timing (velocity_env_cfg.py: sim.dt=0.005, decimation=4 ->
# 50Hz control; the disturbance fires once per control step, and its
# max_steps_per_goal=750 is explicitly "15s @ 50Hz").
_SIM_DT = 0.005
_DECIMATION = 4
_CONTROL_HZ = 1.0 / (_SIM_DT * _DECIMATION)


@configclass
class _IKAccuracySceneCfg(InteractiveSceneCfg):
    """Ground + light + robot. Ground kept so 'height above env origin' matches the
    standing env's ground-referenced goal-z convention (and default-pose foot contact
    matches training geometry)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = None  # filled in main() (depends on CLI args)


class _EnvShim:
    """The few attributes StandingArmIKReachDisturbance actually reads from its env
    (ManagerTermBase.__init__ just stores cfg/env; device comes from env.device, the
    asset from env.scene[...], env origins from env.scene.env_origins, and the ramp from
    env.common_step_counter). Duck-typing this lets the test run the REAL term class
    with zero copied code."""

    def __init__(self, scene: InteractiveScene, device: str):
        self.scene = scene
        self.device = device
        self.num_envs = scene.num_envs
        self.common_step_counter = 0


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = G1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Same asset-level overrides the standing/arm envs apply.
    robot_cfg.spawn.articulation_props.enabled_self_collisions = True
    robot_cfg.spawn.articulation_props.fix_root_link = args_cli.base == "fixed"
    robot_cfg.spawn.activate_contact_sensors = False
    # Arms at the CLI gain (60/1.5 = standing's trained value). Replacing the stock
    # "arms" group drops finger actuation (it covers shoulder/elbow AND fingers), so
    # fingers get their own modest hold — same fix g1_arm_env.py documents.
    robot_cfg.actuators["arms"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_pitch_joint",
            ".*_elbow_roll_joint",
        ],
        stiffness=args_cli.arm_gain[0],
        damping=args_cli.arm_gain[1],
    )
    robot_cfg.actuators["fingers"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint",
            ".*_zero_joint", ".*_one_joint", ".*_two_joint",
        ],
        stiffness=20.0,
        damping=2.0,
        armature=0.001,
    )
    return robot_cfg


def _measure(term: StandingArmIKReachDisturbance, robot: Articulation, scene: InteractiveScene, side: str):
    """FK hand-to-goal error, computed exactly the way _step_side frames it: palm pose in
    the root frame vs. the goal with its ground-referenced z converted to root frame from
    the robot's current height."""
    ee_pos_w = robot.data.body_pos_w[:, term._ee_body_idx[side]]
    ee_quat_w = robot.data.body_quat_w[:, term._ee_body_idx[side]]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w)
    root_height_above_ground = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
    target_b = term._goal_local[side].clone()
    target_b[:, 2] = target_b[:, 2] - root_height_above_ground
    err = ee_pos_b - target_b  # root-frame error vector, signed
    dist = torch.linalg.vector_norm(err, dim=-1)
    return err, dist


def _percentile(t: torch.Tensor, q: float) -> float:
    return torch.quantile(t, q).item()


def main():
    torch.manual_seed(args_cli.seed)

    sim = SimulationContext(SimulationCfg(dt=_SIM_DT, render_interval=_DECIMATION, device=args_cli.device))
    scene_cfg = _IKAccuracySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.robot = _make_robot_cfg()
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    device = str(sim.device)
    shim = _EnvShim(scene, device)

    # The real training term, driven directly. enable_step=0/ramp_full_step=0 forces the
    # disturbance fully on from step 0 (fraction=1.0: full goal box, full reach speed) —
    # the same override mechanism the term's own docstring describes for PLAY configs.
    term_cfg = EventTermCfg(
        func=StandingArmIKReachDisturbance,
        mode="interval",
        interval_range_s=(_SIM_DT * _DECIMATION, _SIM_DT * _DECIMATION),
        params={"asset_cfg": SceneEntityCfg("robot"), "enable_step": 0, "ramp_full_step": 0},
    )
    term = StandingArmIKReachDisturbance(term_cfg, shim)

    steps_per_goal = int(round(args_cli.seconds_per_goal * _CONTROL_HZ))
    term.max_steps_per_goal = steps_per_goal

    is_floating = args_cli.base == "floating"
    print(f"[IKAccuracy] base={args_cli.base} track={args_cli.track} arm_gain={tuple(args_cli.arm_gain)} "
          f"num_envs={args_cli.num_envs} goals/env={args_cli.num_goals} steps/goal={steps_per_goal} "
          f"(robot.is_fixed_base={robot.is_fixed_base})")

    # Root pin for floating mode: default pose at each env origin, zero velocity,
    # re-written every physics step — the root gets the floating-base Jacobian structure
    # without this becoming a balance task (same kinematic-hold idea g1_arm_env.py's root
    # wobble uses).
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, :3] += scene.env_origins
    root_vel = torch.zeros((scene.num_envs, 6), device=device)

    sides = ("left", "right") if args_cli.arm == "both" else (args_cli.arm,)

    # One reset samples goals for everyone (fraction=1.0), then actives are forced to the
    # requested arms — deterministic, instead of the term's random per-env reach-mode roll.
    term.reset(None)
    for side in ("left", "right"):
        term._active[side][:] = side in sides

    default_joint_pos = robot.data.default_joint_pos.clone()
    arm_joint_ids = term._all_joint_ids
    zero_arm_vel = torch.zeros((scene.num_envs, arm_joint_ids.numel()), device=device)
    prev_targets = term._targets.clone()

    # Results: per side, lists of per-round tensors.
    finals: dict[str, list[torch.Tensor]] = {s: [] for s in sides}
    mins: dict[str, list[torch.Tensor]] = {s: [] for s in sides}
    errs: dict[str, list[torch.Tensor]] = {s: [] for s in sides}
    slews: dict[str, list[torch.Tensor]] = {s: [] for s in sides}

    with torch.inference_mode():
        for rnd in range(args_cli.num_goals):
            for step in range(steps_per_goal):
                if step == steps_per_goal - 1:
                    # Round verdict, taken just before the term's final call for this
                    # goal (which resamples on timeout): FK error after (steps-1) steps
                    # of motion — one control step short of the full hold, negligible.
                    target_slew = (term._targets - prev_targets).abs()
                    for side in sides:
                        err, dist = _measure(term, robot, scene, side)
                        finals[side].append(dist.clone())
                        mins[side].append(torch.minimum(term._min_dist[side], dist).clone())
                        errs[side].append(err.clone())
                        # Still slewing at the rate limit at goal end = never converged
                        # (vs. converged-but-wrong, which sits still at the miss).
                        side_slew = target_slew[:, term._col_slice[side]].amax(dim=-1)
                        slews[side].append((side_slew >= 0.9 * term.max_joint_delta_per_step).clone())

                prev_targets.copy_(term._targets)
                term(shim, None)  # the real training/deployment code path: IK solve + rate limit + clamp

                # Hold everything at default, arms at the term's targets — the same
                # "arm columns come from the term" contract the blend action term applies
                # in training.
                full_target = default_joint_pos.clone()
                full_target[:, arm_joint_ids] = term._targets
                robot.set_joint_position_target(full_target)
                if args_cli.track == "kinematic":
                    robot.write_joint_state_to_sim(term._targets, zero_arm_vel, joint_ids=arm_joint_ids)

                for _ in range(_DECIMATION):
                    if is_floating:
                        robot.write_root_pose_to_sim(root_pose)
                        robot.write_root_velocity_to_sim(root_vel)
                    scene.write_data_to_sim()
                    sim.step(render=not args_cli.headless)
                    scene.update(_SIM_DT)
                shim.common_step_counter += 1

            print(f"[IKAccuracy] goal round {rnd + 1}/{args_cli.num_goals} done")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("validation", "ik_accuracy", f"{stamp}_{args_cli.base}_{args_cli.track}")
    os.makedirs(out_dir, exist_ok=True)
    write_eval_meta(out_dir, args_cli, __file__)

    csv_path = os.path.join(out_dir, "goal_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "round", "env", "final_dist_m", "min_dist_m",
                         "err_x_m", "err_y_m", "err_z_m", "slewing_at_end"])
        for side in sides:
            for rnd in range(args_cli.num_goals):
                fd, md, ev, sl = finals[side][rnd], mins[side][rnd], errs[side][rnd], slews[side][rnd]
                for env_i in range(scene.num_envs):
                    writer.writerow([side, rnd, env_i, f"{fd[env_i]:.4f}", f"{md[env_i]:.4f}",
                                     f"{ev[env_i, 0]:.4f}", f"{ev[env_i, 1]:.4f}", f"{ev[env_i, 2]:.4f}",
                                     int(sl[env_i])])
    print(f"[IKAccuracy] Per-goal results: {csv_path}")

    print(f"\n[IKAccuracy] ===== SUMMARY (base={args_cli.base}, track={args_cli.track}, "
          f"gain={tuple(args_cli.arm_gain) if args_cli.track == 'pd' else 'n/a'}) =====")
    for side in sides:
        fd = torch.cat(finals[side])
        md = torch.cat(mins[side])
        ev = torch.cat(errs[side], dim=0)
        sl = torch.cat(slews[side]).float()
        n = fd.numel()
        print(f"\n  {side} arm — {n} goals:")
        print(f"    final dist:  p50={_percentile(fd, 0.5) * 100:6.1f}cm  p90={_percentile(fd, 0.9) * 100:6.1f}cm  "
              f"mean={fd.mean() * 100:6.1f}cm  max={fd.max() * 100:6.1f}cm")
        print(f"    min dist:    p50={_percentile(md, 0.5) * 100:6.1f}cm  p90={_percentile(md, 0.9) * 100:6.1f}cm")
        for tol_cm in (3.0, 5.0, 10.0):
            frac = (fd <= tol_cm / 100.0).float().mean().item() * 100.0
            print(f"    within {tol_cm:4.0f}cm at goal end: {frac:5.1f}%")
        print(f"    per-axis error (root frame, mean signed / mean abs):")
        for i, ax in enumerate(("x", "y", "z")):
            print(f"      {ax}: {ev[:, i].mean() * 100:+6.1f}cm / {ev[:, i].abs().mean() * 100:5.1f}cm")
        print(f"    still slewing at rate limit when goal ended: {sl.mean() * 100:5.1f}% of goals")

    print("\n[IKAccuracy] Interpretation: compare this run against the other base/track "
          "combinations (see the module docstring's decision matrix).")


if __name__ == "__main__":
    main()
    simulation_app.close()
    # Kit teardown can hang after close() in this standalone-scene setup (observed
    # 2026-07-20, fixed+kinematic run: all output written, summary printed, process just
    # never exits). Everything is flushed to disk by this point — force the exit.
    os._exit(0)
