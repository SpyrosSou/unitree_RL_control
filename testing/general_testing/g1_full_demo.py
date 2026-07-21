"""
G1 full integrated demo — standing + arm control + walking.

Combines:
- Standing policy (zero-command balance stabiliser)
- Walking policy (velocity-commanded locomotion)
- Arm IK policy (goal-reaching for left / right / both arms)

Behaviour:
- Start in STANDING mode.  Arms actively reach their current targets.
- Press W / A / D / Q / E to command walking.  Arms freeze (locomotion policy controls them).
- Release commands (S or stop pressing keys) → robot decelerates, switches back to STANDING.
- When standing again, arm policy resumes.
- Press T to type a new arm target at the console (same pattern as testing/arm_testing).
- Press L / R to select which arm to address when arm_mode=both (default: left).
- Press Y (arm_mode=left only) to type a target for the RIGHT arm, driven by mirroring
  the left-trained policy (see mdp/symmetry.py) — no separate right-arm checkpoint
  needed. Shown as a blue marker, vs. the native target's red.
- Press U (arm_mode=left only) to type two independent targets, one per arm (left
  native, right mirrored) — tests whether the mirror-generalized policy holds up
  driving both arms simultaneously, not just one at a time. Two separate targets, not
  one auto-mirrored position, since sending both arms to the same physical point makes
  them collide.
- Press C to toggle camera follow off/on (off lets you orbit freely with the mouse).
- Press V to reset the camera to the default chase view (and re-enable follow).

Arm target format (robot-local frame, base at origin — see g1_arm_env.py's _GOAL_BOUNDS,
the authoritative source; these numbers are just a quick reference and can drift out of
date if that box is reshaped again, always trust the live prompt output over this):
    x  forward     reachable 0.20 – 0.42 m
    y  lateral     left arm: 0.08 – 0.40 m   right arm: -0.40 – -0.08 m
    z  height      reachable 0.9 – 1.15 m  (ground = 0)

--wait_arm_rest (2026-07-20): by default, a walk command clears the active arm target and
starts homing it back to default, but the standing->walking mode switch itself doesn't
wait for that homing move to finish — usually fine (homing is fast relative to
MIN_MODE_STEPS) but can still show a brief instability (e.g. a big first step) if the arm
was far from default when walk was commanded. Pass this flag to hold off the actual mode
switch until the arm has fully homed, at the cost of walk feeling less immediately
responsive. Off by default — opt in to compare the two behaviors, not a strict
improvement to force on always.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    # Auto-load from testing/general_testing/checkpoints.yaml
    python3 testing/general_testing/g1_full_demo.py

    # Explicit checkpoints + initial arm target
    python3 testing/general_testing/g1_full_demo.py \\
        --standing_checkpoint logs/rsl_rl/standing/.../model_1499.pt \\
        --walking_checkpoint  logs/rsl_rl/legs/.../model_3149.pt \\
        --arm_checkpoint      logs/rsl_rl/arms/g1_arm_ik_left/.../model_4200.pt \\
        --arm left \\
        --target 0.3 0.2 1.0

    # Both arms
    python3 testing/general_testing/g1_full_demo.py \\
        --arm both \\
        --target 0.3 0.2 1.0

    # Wait for the arm to finish homing before a walk command actually engages
    python3 testing/general_testing/g1_full_demo.py \\
        --arm left --target 0.3 0.2 1.0 --wait_arm_rest
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be started before all other imports
# ---------------------------------------------------------------------------
import argparse
import importlib.util as _ilu
import math
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
_DEFAULT_STANDING = (
    "logs/rsl_rl/standing/g1_locomotion_flat/2026-06-16_12-26-58/model_1499.pt"
)
_DEFAULT_WALKING = (
    "logs/rsl_rl/legs/g1_locomotion_flat/2026-06-03_10-20-23/model_3149.pt"
)
_DEFAULT_ARM_LEFT = (
    "logs/rsl_rl/arms/g1_arm_ik_left/2026-06-11_09-10-01/model_4200.pt"
)
_DEFAULT_ARM_BOTH = (
    "logs/rsl_rl/arms/g1_arm_ik_both/2026-06-09_20-31-43/model_1000.pt"
)

parser = argparse.ArgumentParser(description="G1 full integrated demo: stand/walk + arm IK.")
cli_args_mod.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--standing_checkpoint", type=str, default=None)
parser.add_argument("--walking_checkpoint",  type=str, default=None)
parser.add_argument("--arm_checkpoint",      type=str, default=None,
                    help="Checkpoint for the arm IK policy (overrides YAML).")
parser.add_argument("--arm", type=str, default=None, choices=["left", "right", "both"],
                    help="Which arm(s) to control. Overrides YAML arm_mode.")
parser.add_argument("--target", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="Initial arm target in robot-local frame (x y z).")
parser.add_argument(
    "--torso_clip_deg", type=float, default=None,
    help="Match the standing checkpoint's OWN torso_joint action clip, in +/-degrees "
    "(2026-07-21). This script builds its env from G1LocomotionFlatEnvCfg_PLAY (walking "
    "lineage), same as validation/integration_validation/eval_full_demo.py's "
    "_build_env_cfg — which never inherits the standing lineage's torso clip either. "
    "Omitting this for a torso-clip/torso-lock-trained checkpoint visualizes it under an "
    "UNCLIPPED action space it was never trained on (the exact bug eval_full_demo.py had "
    "until this same fix was added there). Pass the SAME value the checkpoint's own "
    "training config used (30 for TorsoClip, 0 for TorsoLock); omit for any checkpoint "
    "trained without a torso clip.",
)
parser.add_argument(
    "--wait_arm_rest", action="store_true",
    help="Delay the actual standing->walking mode switch until the arm(s) have finished "
    "homing back to their default pose, instead of only starting the homing move and "
    "switching to walking regardless of whether it's finished (2026-07-20). Off by "
    "default: the un-delayed switch is very often fine (homing is fast relative to "
    "MIN_MODE_STEPS) and this flag makes every walk command feel less responsive, so it's "
    "opt-in for comparing the two behaviors, not a strict improvement to force on always. "
    "Pass the bare flag to enable (no value) — e.g. --wait_arm_rest, not --wait_arm_rest True.",
)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import (
    G1ArmIKBothPPORunnerCfg,
    G1ArmIKLeftPPORunnerCfg,
    G1ArmIKRightPPORunnerCfg,
)
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_ARM_JOINTS,
    _RIGHT_EE_BODY,
)
from g1_locomotion.tasks.manager_based.g1_arm.mdp.symmetry import (
    mirror_arm_actions,
    mirror_arm_obs,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionFlatEnvCfg_PLAY,
)
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp import StandingArmBlendJointPositionAction
from g1_locomotion.tasks.manager_based.g1_locomotion.mdp.observations import standing_arm_motion_targets
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# ---------------------------------------------------------------------------
# Switch demo constants (identical to g1_stand_walk_switch_demo.py)
# ---------------------------------------------------------------------------
LOCO_TASK            = "G1-Locomotion-Flat-Play-v0"
LIN_VEL              = 1.0
ANG_VEL              = 0.5
SWITCH_TO_WALK_THRESHOLD    = 0.12
SWITCH_TO_STAND_THRESHOLD   = 0.04
MIN_MODE_STEPS              = 20
CMD_FILTER_ALPHA            = 0.12
SWITCH_TO_STAND_SPEED_THRESHOLD = 0.22
TRANSITION_STEPS_TO_WALK    = 10
TRANSITION_STEPS_TO_STAND   = 22

# Arm action scale (must match what the arm policy was trained with)
ARM_ACTION_SCALE = 0.5
# Matches G1ArmIKEnvCfg.max_action_delta_per_step (was 0.05 here — a stale mismatch, not
# a deliberate choice; g1_arm_env.py's actual training value is 0.06).
ARM_MAX_JOINT_DELTA_PER_STEP = 0.06
# Matches G1ArmIKEnvCfg.action_filter_alpha. Training never hands the raw policy output
# straight to the actuator: g1_arm_env.py's _pre_physics_step does
# filtered = alpha*raw + (1-alpha)*filtered_prev, THEN scales/clamps that — an EMA low-
# pass meant to emulate actuator lag and avoid abrupt jumps. This script was applying the
# raw, unfiltered policy output directly instead (found 2026-07-09 chasing why
# validation/integration_validation/eval_full_demo.py's standing_arm_reach fall rate was
# far worse than casual manual testing here suggested).
ARM_ACTION_FILTER_ALPHA = 0.25
# When a new target is set, the arm first *homes* back to its default pose through the
# same clamped-delta path the IK targets use (≤ ARM_MAX_JOINT_DELTA_PER_STEP per step),
# then hands over to the arm-IK policy. Replaces the previous write_joint_state_to_sim
# teleport (2026-07-14): an instantaneous joint-state jump is a nonphysical CoM/momentum
# shock the standing policy never sees in training (nothing in mdp/events.py ever writes
# joint state mid-episode) and can't happen on real hardware — but the reason the reset
# existed is still real (a previous out-of-range target can leave the arm in an extreme,
# never-trained pose the policy shouldn't have to start a fresh reach from), so the
# return-to-default is kept, just as a physical motion instead of a teleport. Homing ends
# when every joint is within this tolerance of default.
ARM_HOMING_TOL_RAD = 0.05

# Per-mode arm PD gain (2026-07-12, replaces one static 200/20 applied for the whole
# demo regardless of mode) — matches eval_full_demo.py's _STANDING_ARM_GAIN/_ARM_IK_GAIN/
# _WALKING_ARM_GAIN split: arm-IK reach precision and standing's compliant-arm balance
# strategy need genuinely different gains (confirmed 2026-07-12, arm-IK reach success
# regressed 86%->30% when retrained under standing's softer value), and walking was never
# touched at all so it should see its own stock training gain, not either policy's. The
# demo switches live between these with write_joint_stiffness_to_sim/write_joint_damping_to_sim
# (see G1FullDemo._update_arm_gains) since which policy/overlay is in control changes
# every step based on user input.
_GAIN_STANDING = (60.0, 1.5)   # matches g1_locomotion_env_cfg.py's standing training gain
_GAIN_ARM_IK = (200.0, 20.0)   # matches g1_arm_env.py's arm-IK training gain
_GAIN_WALKING = (40.0, 10.0)   # stock Isaac Lab default — walking never overrides "arms"

# Goal-sphere colours. Red = native (left-trained-policy-driven) target. Blue = the
# mirror-testing target — right arm, driven by mirroring the same left-trained policy
# (see mdp/symmetry.py) — kept visually distinct so it's obvious at a glance which
# target is "the real thing" vs. "the mirror-generalization experiment".
_RED_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/FullDemoTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
    },
)
_BLUE_SPHERE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/FullDemoMirrorTargets",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 1.0)),
        )
    },
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yaml_value(keys: list[str], default=None):
    """Traverse the checkpoints.yaml by a list of keys and return the leaf value."""
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


def _resolve_checkpoint(cli_val: str | None, yaml_keys: list[str], hardcoded: str) -> str:
    """Return the best checkpoint path from CLI > YAML > hardcoded default."""
    path = cli_val or _yaml_value(yaml_keys, hardcoded)
    if path and not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_YAML_PATH))), path)
    return path


def _default_arm_target(arm_mode: str) -> torch.Tensor:
    """A safe default target in the centre of the workspace for the given arm."""
    if arm_mode in ("left", "both"):
        b = _GOAL_BOUNDS["left"]
    else:
        b = _GOAL_BOUNDS["right"]
    x = (b["x"][0] + b["x"][1]) / 2
    y = (b["y"][0] + b["y"][1]) / 2
    z = (b["z"][0] + b["z"][1]) / 2
    return torch.tensor([x, y, z], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class G1FullDemo:
    def __init__(self):
        # ------ resolve checkpoints ------
        self.standing_ckpt = _resolve_checkpoint(
            args_cli.standing_checkpoint, ["standing", "checkpoint"], _DEFAULT_STANDING
        )
        self.walking_ckpt = _resolve_checkpoint(
            args_cli.walking_checkpoint, ["walking", "checkpoint"], _DEFAULT_WALKING
        )

        # arm_mode: CLI > YAML arm_mode > "left"
        self.arm_mode = args_cli.arm or _yaml_value(["arm_mode"]) or "left"

        arm_ckpt_default = _DEFAULT_ARM_BOTH if self.arm_mode == "both" else _DEFAULT_ARM_LEFT
        self.arm_ckpt = _resolve_checkpoint(
            args_cli.arm_checkpoint,
            ["arm", self.arm_mode, "checkpoint"],
            arm_ckpt_default,
        )

        for name, path in [
            ("standing", self.standing_ckpt),
            ("walking", self.walking_ckpt),
            ("arm", self.arm_ckpt),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} checkpoint not found: {path}")

        print(f"[FullDemo] standing  : {self.standing_ckpt}")
        print(f"[FullDemo] walking   : {self.walking_ckpt}")
        print(f"[FullDemo] arm ({self.arm_mode:5s}): {self.arm_ckpt}")
        print(f"[FullDemo] arm_mode  : {self.arm_mode}")

        # ------ locomotion environment ------
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args_mod.parse_rsl_rl_cfg(
            LOCO_TASK, args_cli
        )
        env_cfg = G1LocomotionFlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1_000_000
        env_cfg.curriculum = None

        # Restore training-time observation noise — the _PLAY config disables it for clean
        # visualization. Found 2026-07-14 that this script never restored it (unlike
        # eval_full_demo.py's _build_env_cfg, which does, for the same reason): running a
        # policy trained with observation noise against noise-free input isn't the usual
        # direction of sim-to-real risk, but it's still a real train/deploy mismatch and an
        # easy one to just not have.
        env_cfg.observations.policy.enable_corruption = True

        # Use the same action term standing's own training env uses (2026-07-12) — see
        # validation/integration_validation/eval_full_demo.py's _build_env_cfg for the full
        # rationale. In short: overwriting loco_action's arm columns before handing it to
        # .step() (the previous approach here, in _hold_arms_at_default/_arm_action_override/
        # _mirror_action_override) also corrupts the "last_action" observation the standing
        # policy reads back next step — ActionManager stores whatever raw tensor is passed
        # to .step() verbatim, independent of what an action term actually simulates.
        # StandingArmBlendJointPositionAction decouples the two correctly: it only
        # overrides the *simulated* arm target (via env._standing_arm_motion_targets/
        # _standing_arm_motion_joint_ids, set live in _update_arm_sim_targets), while the
        # policy's raw output keeps flowing into last_action untouched, exactly matching
        # training. Falls through to identical stock behavior when those two env
        # attributes are unset (walking mode — see _update_arm_sim_targets).
        env_cfg.actions.joint_pos.class_type = StandingArmBlendJointPositionAction

        if args_cli.torso_clip_deg is not None:
            env_cfg.actions.joint_pos.clip = {
                "torso_joint": (-math.radians(args_cli.torso_clip_deg), math.radians(args_cli.torso_clip_deg))
            }

        loco_env = ManagerBasedRLEnv(cfg=env_cfg)
        self.env = RslRlVecEnvWrapper(loco_env)
        self.device = loco_env.device
        self.robot = loco_env.scene["robot"]
        # Ground-level world z for this env slot. Arm targets' z is "height above ground"
        # (same convention g1_arm_env.py's _GOAL_BOUNDS and g1_arm_reach_test.py use — the
        # latter adds local targets straight onto env_origins, whose z is ground level),
        # not "height above the robot's root link" — the root sits ~0.75-0.8m off the
        # ground for a standing G1, so anchoring z to root_pos_w directly (as this file
        # used to) silently added that offset on top of an already-ground-referenced z,
        # placing every target far too high to ever reach.
        self._ground_z_w = loco_env.scene.env_origins[0, 2].clone()

        # ------ identify arm joint indices on the robot asset (not the action vector) ------
        # StandingArmBlendJointPositionAction (installed above) overrides simulated arm
        # targets by asset joint id, via env._standing_arm_motion_joint_ids — see
        # _update_arm_sim_targets. all_arm_joint_ids_robot covers both arms, used as the
        # "hold at default" baseline whenever standing without an active target.
        all_arm_ids, _ = self.robot.find_joints(list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS))
        self.all_arm_joint_ids_robot = torch.tensor(all_arm_ids, dtype=torch.long, device=self.device)

        # ------ load locomotion policies ------
        self.standing_policy = self._load_loco_policy(agent_cfg, self.standing_ckpt)
        self.walking_policy  = self._load_loco_policy(agent_cfg, self.walking_ckpt)

        # ------ load arm policy (separate runner, arm-shaped env for obs inference) ------
        self.arm_policy, self.arm_body_ids, self.arm_joint_ids_robot = \
            self._load_arm_policy()
        # Column of each actively-controlled arm joint within all_arm_joint_ids_robot —
        # env._standing_arm_motion_targets is always laid out in that fixed order, so this
        # is where _update_arm_sim_targets writes the IK policy's output.
        self._active_arm_cols = self._cols_in_all_arm(self.arm_joint_ids_robot)

        # ------ switch-demo state ------
        self.mode = "standing"
        self.mode_steps = 0
        self.cmd_filtered = torch.zeros(3, device=self.device)
        self._in_transition   = False
        self._transition_from = "standing"
        self._transition_to   = "standing"
        self._transition_step = 0
        self._transition_total_steps = TRANSITION_STEPS_TO_STAND
        self._walk_wait_notified = False  # --wait_arm_rest console-spam guard, see _maybe_switch_mode

        # ------ arm target state ------
        # No arm target is active by default. The arm policy only engages after
        # the user explicitly provides a target via --target or the T-key prompt.
        n_arms = 2 if self.arm_mode == "both" else 1
        self.goal_pos_local = torch.zeros(n_arms, 3, device=self.device)
        self.arm_target_active = torch.zeros(n_arms, dtype=torch.bool, device=self.device)
        # EMA action-filter state (see ARM_ACTION_FILTER_ALPHA) — (1, 5) per arm, matching
        # g1_arm_env.py's own self.filtered_actions. Reset per-arm whenever that arm gets a
        # new target (_set_arm_target), same as training resets it per-env on every episode
        # reset — a fresh target shouldn't start pre-biased by whatever the previous
        # target's filter had converged to.
        self._filtered_arm_delta = torch.zeros(1, 5 * n_arms, device=self.device)
        # Per-arm homing flags — see ARM_HOMING_TOL_RAD's comment. Set by _set_arm_target,
        # consumed (and cleared on arrival) by _update_arm_sim_targets each step.
        self._arm_homing = torch.zeros(n_arms, dtype=torch.bool, device=self.device)

        if args_cli.target is not None:
            init_tgt = torch.tensor(args_cli.target, dtype=torch.float32, device=self.device)
            self.goal_pos_local[0] = init_tgt
            self.arm_target_active[0] = True
            if self.arm_mode == "both":
                self.goal_pos_local[1] = torch.tensor(
                    [init_tgt[0].item(), -init_tgt[1].item(), init_tgt[2].item()],
                    device=self.device,
                )
                self.arm_target_active[1] = True

        # Which arm the T-key prompt edits (only relevant for both mode)
        self._active_arm_idx = 0  # 0=left, 1=right

        # Arm targets are specified in robot-local coordinates and stay a fixed offset
        # from the torso — _goal_positions_world() recomputes the world position from the
        # robot's *current* pose every call, it does not snapshot one. See that method's
        # docstring for why (2026-07-09 fix: a world-fixed goal forced the arm to
        # compensate for base tilt/drift it was never trained to handle, which fed back
        # into the tilt that caused it).

        # Goal visualisation
        self._goal_vis = VisualizationMarkers(_RED_SPHERE_CFG)
        self._update_goal_markers()

        # ------ mirror-testing state (right arm, driven by mirroring the LEFT-trained
        # policy — known_issues.md flags two-arm generalization via this exact mirror
        # transform as unverified; Y/U let you test it directly against the real robot
        # instead of only in isolation via g1_arm_mirror_test.py). Only meaningful when
        # arm_mode="left" — arm_mode="right"/"both" already have their own native
        # control path for the right arm (once a real checkpoint exists for them). ------
        self._mirror_enabled = self.arm_mode == "left"
        # Read unconditionally by _update_arm_sim_targets/_arm_ready_for_walk regardless of
        # arm_mode (mirror_active is already gated on _mirror_enabled there, but `mirror_active
        # or self._mirror_homing` still evaluates the attribute even when mirror_active is
        # False) — must exist even when mirroring itself is inactive, not just initialized
        # inside the _mirror_enabled block below.
        self._mirror_homing = False
        if self._mirror_enabled:
            r_joints, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_bodies, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            self._mirror_joint_ids_robot = torch.tensor(r_joints, dtype=torch.long, device=self.device)
            self._mirror_ee_body_id = r_bodies[0]
            self._mirror_arm_cols = self._cols_in_all_arm(self._mirror_joint_ids_robot)

            self.mirror_goal_pos_local = torch.zeros(3, device=self.device)
            self.mirror_target_active = False
            self._filtered_mirror_delta = torch.zeros(1, 5, device=self.device)
            # Same "fixed offset from the torso, not from the world" treatment as the
            # primary target — see _mirror_goal_world()'s docstring.

            self._mirror_goal_vis = VisualizationMarkers(_BLUE_SPHERE_CFG)
            self._update_mirror_goal_marker()

        # Prompt state for T/Y/U-key arm targets.
        self._target_prompt_requested = False
        self._target_prompt_active = False
        self._pending_prompt_kind = "left"

        self._create_camera()
        self._setup_keyboard()

        print("\n[FullDemo] Ready.")
        print("  W/A/D/Q/E  — walk commands    S  — stop")
        print("  T          — type new arm target (blocks simulation briefly)")
        print("  L / R      — switch active arm (only for arm_mode=both)")
        if self._mirror_enabled:
            print("  Y          — type a target for the right arm (mirrored via the left policy)")
            print("  U          — type two targets, one per arm (left native, right mirrored)")
        print("  C          — toggle camera follow (off = orbit freely with the mouse)")
        print("  V          — reset camera to the default chase view\n")

    # ------------------------------------------------------------------ utils

    def _cols_in_all_arm(self, joint_ids: torch.Tensor) -> torch.Tensor:
        """Column indices of `joint_ids` within self.all_arm_joint_ids_robot — that
        tensor's order is env._standing_arm_motion_targets' fixed column layout, so this
        is how _update_arm_sim_targets locates where to write a given arm's targets."""
        id_to_col = {int(j): i for i, j in enumerate(self.all_arm_joint_ids_robot.tolist())}
        cols = [id_to_col[int(j)] for j in joint_ids.tolist()]
        return torch.tensor(cols, dtype=torch.long, device=self.device)

    def _load_loco_policy(self, agent_cfg: RslRlOnPolicyRunnerCfg, ckpt: str):
        """Input width inferred from the checkpoint, not the env (2026-07-15) — same
        change/rationale as eval_full_demo.py's _load_loco_policy: the arm-intent standing
        task appends a 10-D block to the policy obs (75 -> 85), this demo's env is the
        75-D walking cfg shared by both policies, and OnPolicyRunner sizes from the env.
        The closure auto-extends a too-narrow obs with mdp.standing_arm_motion_targets —
        which reads env._standing_arm_motion_targets, the exact buffer this demo already
        maintains every step (_update_arm_sim_targets while standing, cleared to None
        while walking, where the function falls back to zeros = "arms at default")."""
        state = torch.load(ckpt, map_location=self.device)["model_state_dict"]
        in_dim = state["actor.0.weight"].shape[1]
        num_actions = state[f"actor.{2 * len(agent_cfg.policy.actor_hidden_dims)}.weight"].shape[0]

        # Locomotion runner cfgs never define obs_groups (dataclasses.MISSING) — same fix
        # as eval_full_demo.py's _load_loco_policy, found 2026-07-15 when the first
        # integration eval of the new loader crashed on exactly this.
        obs_groups = agent_cfg.obs_groups
        if not isinstance(obs_groups, dict):
            obs_groups = {"policy": ["policy"], "critic": ["policy"]}

        dummy_obs = TensorDict(
            {"policy": torch.zeros((1, in_dim), dtype=torch.float32, device=self.device)},
            batch_size=[1], device=self.device,
        )
        actor_critic = ActorCritic(
            obs=dummy_obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_obs_normalization=agent_cfg.policy.actor_obs_normalization,
            critic_obs_normalization=agent_cfg.policy.critic_obs_normalization,
            actor_hidden_dims=agent_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=agent_cfg.policy.critic_hidden_dims,
            activation=agent_cfg.policy.activation,
            init_noise_std=agent_cfg.policy.init_noise_std,
        ).to(self.device)
        actor_critic.load_state_dict(state)
        actor_critic.eval()

        env_unwrapped = self.env.unwrapped

        def policy(obs) -> torch.Tensor:
            # RslRlVecEnvWrapper hands back a TensorDict — same normalization as
            # eval_full_demo.py's _load_loco_policy, found 2026-07-15 (a TensorDict's
            # .shape is its batch size, and torch.cat on [TensorDict, Tensor] crashes).
            obs_tensor = obs["policy"] if isinstance(obs, TensorDict) else obs
            if obs_tensor.shape[-1] < in_dim:
                obs_tensor = torch.cat([obs_tensor, standing_arm_motion_targets(env_unwrapped)], dim=-1)
            td = TensorDict({"policy": obs_tensor}, batch_size=[obs_tensor.shape[0]], device=self.device)
            with torch.inference_mode():
                return actor_critic.act_inference(td)

        return policy

    def _load_arm_policy(self):
        """Load the arm IK policy and return (policy_fn, ee_body_ids, joint_ids)."""
        # 28-D per arm (9-D base state prefix + 19-D arm state — joint_pos(5)/joint_vel(5)/
        # ee_pos(3)/goal(3)/error(3); see g1_arm_env.py's module docstring); 56-D for
        # "both". Was 26/52 before the 2026-07-08 joint_vel fix (was only 3-D, dropping
        # elbow_pitch/elbow_roll velocity), and 17/34 before Phase 2.
        if self.arm_mode == "left":
            arm_agent_cfg = G1ArmIKLeftPPORunnerCfg()
            obs_dim = 28
            action_dim = 5
        elif self.arm_mode == "right":
            arm_agent_cfg = G1ArmIKRightPPORunnerCfg()
            obs_dim = 28
            action_dim = 5
        else:
            arm_agent_cfg = G1ArmIKBothPPORunnerCfg()
            obs_dim = 56
            action_dim = 10

        arm_obs = TensorDict(
            {"policy": torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)},
            batch_size=[1],
            device=self.device,
        )
        actor_critic = ActorCritic(
            obs=arm_obs,
            obs_groups=arm_agent_cfg.obs_groups,
            num_actions=action_dim,
            actor_obs_normalization=arm_agent_cfg.policy.actor_obs_normalization,
            critic_obs_normalization=arm_agent_cfg.policy.critic_obs_normalization,
            actor_hidden_dims=arm_agent_cfg.policy.actor_hidden_dims,
            critic_hidden_dims=arm_agent_cfg.policy.critic_hidden_dims,
            activation=arm_agent_cfg.policy.activation,
            init_noise_std=arm_agent_cfg.policy.init_noise_std,
        ).to(self.device)

        checkpoint = torch.load(self.arm_ckpt, map_location=self.device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        actor_critic.eval()

        def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
            batch = next(iter(obs_dict.values())).shape[0]
            tensordict_obs = TensorDict(obs_dict, batch_size=[batch], device=self.device)
            with torch.inference_mode():
                return actor_critic.act_inference(tensordict_obs)

        # Get arm joint IDs and EE body IDs FROM THE LOCOMOTION ROBOT
        body_ids = []
        joint_ids = []

        if self.arm_mode in ("left", "both"):
            l_joints, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            l_bodies, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            joint_ids.extend(l_joints)
            body_ids.append(l_bodies[0])

        if self.arm_mode in ("right", "both"):
            r_joints, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_bodies, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            joint_ids.extend(r_joints)
            body_ids.append(r_bodies[0])

        return (
            policy,
            body_ids,
            torch.tensor(joint_ids, dtype=torch.long, device=self.device),
        )

    # ------------------------------------------------------------------ arm obs

    def _build_arm_obs(self) -> torch.Tensor:
        """Construct the 28-D (or 56-D) arm observation from the locomotion scene.

        Base-state prefix (base_lin_vel, base_ang_vel, projected_gravity — added Phase 2,
        see g1_arm_env.py) reads the *real* locomotion robot's current state here, unlike
        training's synthetic wobble signal (g1_arm_env.py's root is physically fixed
        during arm-only training, so it fakes this) — here the base is actually the real,
        moving standing/walking robot, so the real sensor values are the more faithful
        choice, not an approximation of one.
        """
        parts = []
        goal_world = self._goal_positions_world()

        base_lin_vel = self.robot.data.root_lin_vel_b[0].unsqueeze(0)   # (1, 3)
        base_ang_vel = self.robot.data.root_ang_vel_b[0].unsqueeze(0)   # (1, 3)
        projected_gravity = self.robot.data.projected_gravity_b[0].unsqueeze(0)  # (1, 3)

        arm_groups = []
        if self.arm_mode in ("left", "both"):
            l_ids, _ = self.robot.find_joints(_LEFT_ARM_JOINTS)
            l_body, _ = self.robot.find_bodies(_LEFT_EE_BODY)
            arm_groups.append({
                "joint_ids": torch.tensor(l_ids, device=self.device),
                "ee_idx": l_body[0],
                "goal": goal_world[0],
            })
        if self.arm_mode in ("right", "both"):
            r_ids, _ = self.robot.find_joints(_RIGHT_ARM_JOINTS)
            r_body, _ = self.robot.find_bodies(_RIGHT_EE_BODY)
            goal_idx = 1 if self.arm_mode == "both" else 0
            arm_groups.append({
                "joint_ids": torch.tensor(r_ids, device=self.device),
                "ee_idx": r_body[0],
                "goal": goal_world[goal_idx],
            })

        root_quat = self.robot.data.root_quat_w[0].unsqueeze(0)  # (1, 4)
        for g in arm_groups:
            jids = g["joint_ids"]
            joint_pos = self.robot.data.joint_pos[0, jids].unsqueeze(0)    # (1, 5)
            joint_vel = self.robot.data.joint_vel[0, jids].unsqueeze(0)    # (1, 5)
            ee_pos    = self.robot.data.body_pos_w[0, g["ee_idx"], :].unsqueeze(0)  # (1, 3)
            goal      = g["goal"].unsqueeze(0)                               # (1, 3)
            # Rotate into the robot's current body frame before handing to the policy.
            # g1_arm_env.py trains with a physically fixed (never-rotating) root, so its
            # world frame *is* body frame throughout training — error's x/y/z always mean
            # forward/lateral/up relative to the robot. Here the robot can actually turn
            # (walking, or just reset_base's spawn-yaw randomization), so a raw world-frame
            # difference stops meaning that the moment the robot isn't at ~0 yaw — the
            # policy would be reaching in a direction rotated by whatever the robot's
            # current heading is. Only correcting `error` (not ee_pos/goal individually)
            # keeps this a minimal, targeted fix — error is what actually drives reaching.
            error = quat_apply_inverse(root_quat, goal - ee_pos)             # (1, 3)
            # Base-state prefix duplicated per arm block, matching g1_arm_env.py's
            # layout (same convention "both" mode needs there too, for the mirror
            # transform to work block-by-block).
            parts.extend([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error])

        return torch.cat(parts, dim=-1)  # (1, 28) or (1, 56)

    # ------------------------------------------------------------------ arm delta → simulated target

    def _compute_arm_targets(self) -> torch.Tensor:
        """Run the arm-IK policy and return new absolute joint targets (1, n_arm_joints)
        for the actively-controlled arm(s). Used to overlay onto
        env._standing_arm_motion_targets in _update_arm_sim_targets — this no longer
        touches the locomotion action tensor at all (see that method's docstring)."""
        arm_obs = self._build_arm_obs()
        arm_delta = self.arm_policy({"policy": arm_obs})  # (1, 5) or (1, 10)

        # EMA low-pass on the raw policy output, matching g1_arm_env.py's own
        # self.filtered_actions — see ARM_ACTION_FILTER_ALPHA.
        self._filtered_arm_delta = (
            ARM_ACTION_FILTER_ALPHA * arm_delta + (1.0 - ARM_ACTION_FILTER_ALPHA) * self._filtered_arm_delta
        )
        arm_delta = self._filtered_arm_delta

        # Compute new arm joint targets from current position + delta
        current = self.robot.data.joint_pos[0, self.arm_joint_ids_robot]  # (n_arm_joints,)
        limits  = self.robot.data.soft_joint_pos_limits[0, self.arm_joint_ids_robot]  # (n, 2)
        delta = (arm_delta.squeeze(0) * ARM_ACTION_SCALE).clamp(
            -ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP
        )
        new_targets = (current + delta).clamp(
            limits[:, 0], limits[:, 1]
        )
        return new_targets.unsqueeze(0)

    # ------------------------------------------------------------------ mirror testing
    # (right arm, driven by mirroring the left-trained policy — see mdp/symmetry.py and
    # the module docstring's note on why this exists.) Only ever constructed/called when
    # self._mirror_enabled (arm_mode="left").

    def _build_mirror_source_obs(self) -> torch.Tensor:
        """Right arm's own real (unmirrored) 28-D observation block — same layout
        _build_arm_obs() builds for the left arm, just sourced from the right arm's own
        joints/end-effector/goal."""
        base_lin_vel = self.robot.data.root_lin_vel_b[0].unsqueeze(0)
        base_ang_vel = self.robot.data.root_ang_vel_b[0].unsqueeze(0)
        projected_gravity = self.robot.data.projected_gravity_b[0].unsqueeze(0)

        joint_pos = self.robot.data.joint_pos[0, self._mirror_joint_ids_robot].unsqueeze(0)
        joint_vel = self.robot.data.joint_vel[0, self._mirror_joint_ids_robot].unsqueeze(0)
        ee_pos = self.robot.data.body_pos_w[0, self._mirror_ee_body_id, :].unsqueeze(0)
        goal = self._mirror_goal_world().unsqueeze(0)
        # See _build_arm_obs()'s identical comment — error must be in the robot's current
        # body frame, not raw world, or reaching direction is wrong whenever the robot
        # isn't at ~0 yaw.
        root_quat = self.robot.data.root_quat_w[0].unsqueeze(0)
        error = quat_apply_inverse(root_quat, goal - ee_pos)

        return torch.cat(
            [base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, ee_pos, goal, error], dim=-1
        )

    def _compute_mirror_targets(self) -> torch.Tensor:
        """Run the left-trained arm policy on a mirrored right-arm observation, then
        mirror its output back — same transform g1_arm_mirror_test.py already validates
        in isolation, applied here against the real, standing robot. Returns new absolute
        joint targets (1, n_arm_joints) for the mirror-driven right arm; see
        _compute_arm_targets for why this no longer touches the action tensor."""
        raw_obs = self._build_mirror_source_obs()
        mirrored_obs = mirror_arm_obs(raw_obs)
        delta_left_frame = self.arm_policy({"policy": mirrored_obs})
        delta = mirror_arm_actions(delta_left_frame)

        # EMA low-pass on the raw (mirrored-back) policy output, matching g1_arm_env.py's
        # own self.filtered_actions — see ARM_ACTION_FILTER_ALPHA. Filtering here (post
        # mirror) vs. pre-mirror is equivalent since the mirror is a fixed linear sign
        # flip, but this keeps the persisted buffer directly interpretable as "the
        # effective smoothed right-arm delta," matching _filtered_arm_delta's convention.
        self._filtered_mirror_delta = (
            ARM_ACTION_FILTER_ALPHA * delta + (1.0 - ARM_ACTION_FILTER_ALPHA) * self._filtered_mirror_delta
        )
        delta = self._filtered_mirror_delta.squeeze(0)

        current = self.robot.data.joint_pos[0, self._mirror_joint_ids_robot]
        limits = self.robot.data.soft_joint_pos_limits[0, self._mirror_joint_ids_robot]
        delta = (delta * ARM_ACTION_SCALE).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
        new_targets = (current + delta).clamp(limits[:, 0], limits[:, 1])
        return new_targets.unsqueeze(0)

    # ------------------------------------------------------------------ simulated arm targets + gain

    def _update_arm_sim_targets(self, arm_active: bool, mirror_active: bool):
        """Set env._standing_arm_motion_targets/_standing_arm_motion_joint_ids so
        StandingArmBlendJointPositionAction drives the *simulated* arm targets — called
        every step while standing_ish (see select_action). Both arms start held at
        default; the actively-controlled arm(s) then overwrite their own columns. The
        locomotion policy's raw action output is never touched by this — it keeps
        flowing into the env's stored action, and therefore into next step's last_action
        observation, exactly as it did during training. See env_cfg.actions.joint_pos's
        assignment in __init__ for the full rationale."""
        env = self.env.unwrapped
        env._standing_arm_motion_joint_ids = self.all_arm_joint_ids_robot
        targets = self.robot.data.default_joint_pos[0:1, self.all_arm_joint_ids_robot].clone()

        # Gated on "arm_active OR still homing" (2026-07-20), not just arm_active — a walk
        # command clears arm_target_active immediately (see _maybe_switch_mode) well
        # before the mode itself flips to "walking", but _arm_homing/_mirror_homing can
        # still be set from that same clear (_clear_arm_targets/_clear_mirror_target).
        # Gating purely on arm_active/mirror_active here would skip this whole block the
        # step after a clear — including the homing loop below it — letting targets fall
        # straight through to plain default_joint_pos with no rate limit at all. See
        # _clear_arm_targets's docstring for the full story.
        if arm_active or bool(self._arm_homing.any().item()):
            # Only run the arm-IK policy while there's an actual target to chase — a
            # homing-only pass (arm_active False) starts from the already-default slice
            # `targets` was initialized to, letting the loop below fill in the rate-
            # limited step instead of a fresh (and here, meaningless) policy inference.
            arm_targets = self._compute_arm_targets() if arm_active else targets[:, self._active_arm_cols].clone()
            # Any arm slot still homing overrides its slice of the policy output with the
            # scripted return-to-default step — see ARM_HOMING_TOL_RAD's comment.
            n_per_arm = len(self.arm_joint_ids_robot) // self._arm_homing.numel()
            for arm_idx in range(self._arm_homing.numel()):
                if not bool(self._arm_homing[arm_idx]):
                    continue
                sl = slice(arm_idx * n_per_arm, (arm_idx + 1) * n_per_arm)
                homing_target = self._homing_step(self.arm_joint_ids_robot[sl])
                if homing_target is None:
                    self._arm_homing[arm_idx] = False
                    # The EMA filter accumulated policy output during homing that was never
                    # applied — zero it so the policy engages fresh from the default pose,
                    # same as training's per-episode filter reset.
                    self._filtered_arm_delta[0, sl] = 0.0
                    print("[FullDemo] Arm homed to default — arm-IK policy engaging.")
                else:
                    arm_targets[:, sl] = homing_target
            targets[:, self._active_arm_cols] = arm_targets
        if mirror_active or self._mirror_homing:
            if self._mirror_homing:
                homing_target = self._homing_step(self._mirror_joint_ids_robot)
                if homing_target is None:
                    self._mirror_homing = False
                    self._filtered_mirror_delta[:] = 0.0
                    print("[FullDemo] Mirror arm homed to default — mirrored policy engaging.")
                else:
                    targets[:, self._mirror_arm_cols] = homing_target
            if mirror_active and not self._mirror_homing:
                targets[:, self._mirror_arm_cols] = self._compute_mirror_targets()
        env._standing_arm_motion_targets = targets

    def _homing_step(self, joint_ids: torch.Tensor) -> torch.Tensor | None:
        """One clamped-delta step of the scripted return-to-default move (see
        ARM_HOMING_TOL_RAD). Returns the (1, n) joint target for this step, or None once
        every joint is within tolerance of default (homing complete)."""
        current = self.robot.data.joint_pos[0, joint_ids]
        default = self.robot.data.default_joint_pos[0, joint_ids]
        err = default - current
        if bool((err.abs().max() <= ARM_HOMING_TOL_RAD).item()):
            return None
        delta = err.clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
        return (current + delta).unsqueeze(0)

    def _clear_arm_sim_targets(self):
        """Walking mode: let StandingArmBlendJointPositionAction fall through to stock
        JointPositionAction behavior (whatever the walking policy itself outputs for the
        arm columns), same as before this refactor — only standing ever held arms at a
        fixed pose."""
        env = self.env.unwrapped
        env._standing_arm_motion_targets = None
        env._standing_arm_motion_joint_ids = None

    def _update_arm_gains(self, standing_ish: bool, arm_active: bool, mirror_active: bool):
        """Live per-mode arm PD gain (2026-07-12) — see _GAIN_STANDING/_GAIN_ARM_IK/
        _GAIN_WALKING's module-level comment for why one static gain doesn't serve every
        policy. Splits the two arms independently: whichever arm(s) are actively
        IK-driven (native or mirrored) get the arm-IK gain, everything else gets whichever
        locomotion policy currently holds control's own trained gain."""
        active_cols = []
        if arm_active:
            active_cols.append(self._active_arm_cols)
        if mirror_active:
            active_cols.append(self._mirror_arm_cols)
        active_ids = (
            self.all_arm_joint_ids_robot[torch.cat(active_cols)] if active_cols
            else self.all_arm_joint_ids_robot[:0]
        )
        held_mask = torch.ones(self.all_arm_joint_ids_robot.numel(), dtype=torch.bool, device=self.device)
        if active_cols:
            held_mask[torch.cat(active_cols)] = False
        held_ids = self.all_arm_joint_ids_robot[held_mask]

        held_gain = _GAIN_STANDING if standing_ish else _GAIN_WALKING
        if active_ids.numel() > 0:
            self.robot.write_joint_stiffness_to_sim(_GAIN_ARM_IK[0], joint_ids=active_ids)
            self.robot.write_joint_damping_to_sim(_GAIN_ARM_IK[1], joint_ids=active_ids)
        if held_ids.numel() > 0:
            self.robot.write_joint_stiffness_to_sim(held_gain[0], joint_ids=held_ids)
            self.robot.write_joint_damping_to_sim(held_gain[1], joint_ids=held_ids)

    def _mirror_goal_world(self) -> torch.Tensor:
        """Recomputed every call from the robot's *current* pose — see
        _goal_positions_world()'s docstring for why this must not snapshot a pose once
        and hold it. Same fix, mirror-path copy."""
        pos = self.robot.data.root_pos_w[0].clone()
        pos[2] = self._ground_z_w
        return quat_apply(self.robot.data.root_quat_w[0], self.mirror_goal_pos_local) + pos

    def _update_mirror_goal_marker(self):
        pos = self._mirror_goal_world().unsqueeze(0).clone()
        if not self.mirror_target_active:
            pos[0, 2] = -10.0
        self._mirror_goal_vis.visualize(pos)

    def _set_mirror_target(self, target_local: torch.Tensor):
        self.mirror_goal_pos_local = target_local
        self.mirror_target_active = True

        # Same return-to-default-between-targets treatment as _set_arm_target — smooth
        # homing move, not a teleport (see that method's comment, 2026-07-14).
        with torch.inference_mode():
            self._mirror_homing = True
            # Reset the mirror path's EMA action filter too — same "must stay inside
            # inference_mode" restriction as _set_arm_target's identical pattern.
            self._filtered_mirror_delta[:] = 0.0

        self._update_mirror_goal_marker()
        print(f"[FullDemo] Mirror target (right, via left policy): {target_local.tolist()}")

    def _clear_mirror_target(self):
        if not self.mirror_target_active:
            return
        # Same fix and rationale as _clear_arm_targets: mark for homing instead of just
        # dropping the target, or the mirror arm's simulated position target jumps
        # straight to default with no rate limit on the next _update_arm_sim_targets call.
        self._mirror_homing = True
        self.mirror_target_active = False
        self._update_mirror_goal_marker()
        print("[FullDemo] Cleared mirror target — homing to default before walk.")

    def _has_active_arm_target(self) -> bool:
        if self.arm_mode == "both":
            return bool(torch.all(self.arm_target_active).item())
        return bool(self.arm_target_active[0].item())

    def _clear_arm_targets(self):
        """Clears any active reach target(s) — called right as a walk command is first
        detected (see _maybe_switch_mode), well before the mode actually flips to
        "walking". Marks whichever arm(s) were active for the SAME rate-limited homing
        move _set_arm_target already uses between reach targets (see ARM_HOMING_TOL_RAD),
        rather than just dropping the target — 2026-07-20, found via live testing: without
        this, arm_target_active going straight to all-False makes _update_arm_sim_targets
        skip its "arm_active" branch (and with it the homing loop) entirely on the very
        next step, so the simulated arm TARGET jumps straight from wherever the arm
        currently is to default_joint_pos with no rate limit at all — not a physical
        teleport (the 60/1.5 PD gain still has to physically catch up), but exactly the
        kind of abrupt, unclamped target change this project has repeatedly found
        destabilizing elsewhere (the same class of bug the "no teleports" homing move was
        originally built to avoid, just at a different call site). Observed as
        "walking becomes slightly unstable if the arm isn't already home" when a walk
        command interrupts an active reach.
        """
        if self._mirror_enabled:
            self._clear_mirror_target()
        if not bool(torch.any(self.arm_target_active).item()):
            return
        self._arm_homing |= self.arm_target_active
        self.arm_target_active.zero_()
        self._update_goal_markers()
        print("[FullDemo] Cleared arm target(s) — homing to default before walk.")

    def _goal_positions_world(self) -> torch.Tensor:
        """Transform robot-local arm targets into world coordinates, using the robot's
        *current* pose — recomputed on every call, not a pose snapshotted once when the
        target was set (found 2026-07-09 to be a real bug, not just a simplification: a
        goal fixed in world space forces the arm to compensate for base tilt/drift it was
        never trained to handle — g1_arm_env.py trains with fix_root_link=True, so the
        policy has zero experience with a moving base. As reaching itself perturbed the
        base, the fixed-world goal effectively receded, the policy pushed the arm further
        to chase it — out-of-distribution behavior, not considered compensation — adding
        more destabilizing torque and tilting the base further: a runaway feedback loop.
        Treating the target as a fixed offset from the torso instead (recomputed from the
        live root pose every call) reproduces what training's fixed root gave the policy
        for free: a goal that only moves because the robot pursuing it does.

        x/y and orientation track the robot's actual current base; z stays
        ground-referenced (see the _ground_z_w comment in __init__), not tied to the root
        link's own height, which bounces/settles independent of where a target "on the
        wall at 1.0m" should be."""
        pos = self.robot.data.root_pos_w[0].clone()
        pos[2] = self._ground_z_w
        quat = self.robot.data.root_quat_w[0]
        return torch.stack([quat_apply(quat, target_local) + pos for target_local in self.goal_pos_local], dim=0)

    def _handle_env_resets(self, dones: torch.Tensor):
        if not bool(torch.any(dones > 0).item()):
            return

        self.mode = "standing"
        self.mode_steps = 0
        self._in_transition = False
        self._transition_from = "standing"
        self._transition_to = "standing"
        self._transition_step = 0
        self.cmd_filtered.zero_()
        self._walk_wait_notified = False

        # A respawn resets the arm joints to default via the env's own reset — nothing
        # left to home back from.
        self._arm_homing.zero_()
        if self._mirror_enabled:
            self._mirror_homing = False

        # No re-anchoring needed on respawn any more — targets are a fixed offset from
        # the torso, recomputed from the live pose every call (_goal_positions_world /
        # _mirror_goal_world), so a respawn's new pose is picked up automatically on the
        # very next read, same as any other step's base movement.

    # ------------------------------------------------------------------ locomotion mode

    def _start_transition(self, new_mode: str, force_from: str | None = None, reason: str = "switch"):
        if new_mode == "walking":
            self._clear_arm_targets()
        self._in_transition = True
        self._transition_from = force_from if force_from is not None else self.mode
        self._transition_to   = new_mode
        self._transition_step = 0
        self._transition_total_steps = (
            TRANSITION_STEPS_TO_WALK if new_mode == "walking" else TRANSITION_STEPS_TO_STAND
        )
        self.mode = new_mode
        self.mode_steps = 0
        print(f"[FullDemo] transition {self._transition_from} → {self._transition_to} (reason={reason})")

    def _robot_planar_speed(self) -> float:
        vel = self.env.unwrapped.scene["robot"].data.root_lin_vel_w[0, :2]
        return torch.linalg.vector_norm(vel).item()

    def _arm_ready_for_walk(self) -> bool:
        """True once no arm is still mid-homing. Only consulted when --wait_arm_rest is
        set (2026-07-20) — gates the actual standing->walking mode switch so walking
        never begins while an arm is still partway through its return-to-default move,
        the residual instability the un-delayed switch can still show for a large
        excursion within a short MIN_MODE_STEPS window (see _clear_arm_targets)."""
        return not bool(self._arm_homing.any().item()) and not self._mirror_homing

    def _maybe_switch_mode(self, cmd_mag: float):
        speed = self._robot_planar_speed()

        # As soon as the user starts commanding locomotion, drop any active arm
        # target so the standing arm override cannot keep fighting the walk handoff.
        if self.mode == "standing" and self._has_active_arm_target() and cmd_mag >= 0.02:
            self._clear_arm_targets()

        can_start_walk = (not args_cli.wait_arm_rest) or self._arm_ready_for_walk()
        if not can_start_walk:
            if not self._walk_wait_notified:
                self._walk_wait_notified = True
                print("[FullDemo] Walk requested — waiting for arm(s) to finish homing before starting (--wait_arm_rest).")
        else:
            self._walk_wait_notified = False

        if self._in_transition:
            if self._transition_to == "standing" and cmd_mag >= SWITCH_TO_WALK_THRESHOLD and can_start_walk:
                self._start_transition("walking", force_from="standing", reason="interrupt")
            return

        if self.mode == "standing":
            if self.mode_steps >= MIN_MODE_STEPS and cmd_mag >= SWITCH_TO_WALK_THRESHOLD and can_start_walk:
                self._start_transition("walking", reason="threshold")
        else:
            if (
                self.mode_steps >= MIN_MODE_STEPS
                and cmd_mag <= SWITCH_TO_STAND_THRESHOLD
                and speed <= SWITCH_TO_STAND_SPEED_THRESHOLD
            ):
                self._start_transition("standing", reason="threshold")

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select locomotion action (blend during transition); overlay arm if standing."""
        if not self._in_transition:
            loco_action = (
                self.walking_policy(obs)
                if self.mode == "walking"
                else self.standing_policy(obs)
            )
        else:
            alpha = min((self._transition_step + 1) / float(self._transition_total_steps), 1.0)
            from_pol = self.walking_policy if self._transition_from == "walking" else self.standing_policy
            to_pol   = self.walking_policy if self._transition_to   == "walking" else self.standing_policy
            from_action = from_pol(obs)
            to_action = to_pol(obs)
            loco_action = (1.0 - alpha) * from_action + alpha * to_action

            # Arm columns during a transition come from the WALKING side only, never from
            # the blend (2026-07-15, the "arms flail wildly the moment walking starts"
            # fix): the standing policy's raw arm-column output is meaningless — its
            # training always overrides arm targets with the disturbance mechanism, so
            # nothing ever shapes what the network itself emits there (see the long
            # comment below). While standing_ish that garbage never reaches the sim (the
            # blend action term overrides the simulated arm targets), but a transition
            # *to walking* clears that override (standing_ish False), so for
            # TRANSITION_STEPS_TO_WALK steps the arms were driven by a blend that is
            # part unshaped garbage — visible as violent arm swings right at walk onset,
            # a real observed fall source. Every transition in this demo has walking on
            # exactly one side, so its arm output — the only trained arm signal in the
            # blend — is used unblended. (Transitions *to standing* keep the override
            # active, so this only changes what the sim sees in the to-walking case;
            # last_action sees it in both, which is closer to each policy's training
            # than blended garbage was.)
            walking_action = from_action if self._transition_from == "walking" else to_action
            loco_action = loco_action.clone()
            loco_action[:, self.all_arm_joint_ids_robot] = walking_action[:, self.all_arm_joint_ids_robot]

            self._transition_step += 1
            if self._transition_step >= self._transition_total_steps:
                self._in_transition = False
                print(f"[FullDemo] mode settled: {self.mode}")

        # Whenever standing, hold *both* arms at their default pose first — the standing
        # policy's own raw action output for arm joints is never meaningful (standing
        # trains with a mechanism that always overrides arm targets with a scripted
        # disturbance signal, so nothing ever shapes what the policy itself would output
        # there) — then overlay the arm-IK policy on top for whichever arm(s) actually
        # have an active target. Any arm *not* currently being driven by the arm-IK policy
        # (e.g. the right arm while only the left has a target) still gets held at default.
        #
        # This is now done entirely through env._standing_arm_motion_targets (see
        # _update_arm_sim_targets), never by editing loco_action itself. Editing the action
        # tensor before returning it used to also corrupt the "last_action" observation the
        # locomotion policy reads back next step (ActionManager stores whatever raw tensor
        # is passed to .step() verbatim, independent of what an action term actually
        # simulates) — found 2026-07-12 chasing why validation/integration_validation/
        # eval_full_demo.py's standing_still bucket (zero commanded velocity, zero arm
        # activity) still had a ~100% fall rate despite the standing policy being flawless
        # in its own native eval. loco_action below is returned completely untouched,
        # whatever the (possibly transition-blended) locomotion policy actually output.
        standing_ish = self.mode == "standing" or (self._in_transition and self._transition_to == "standing")
        arm_active = self._has_active_arm_target() and (self.mode == "standing") and (
            not self._in_transition or self._transition_to == "standing"
        )
        mirror_active = self._mirror_enabled and self.mirror_target_active and standing_ish
        if standing_ish:
            self._update_arm_sim_targets(arm_active, mirror_active)
        else:
            self._clear_arm_sim_targets()
        self._update_arm_gains(standing_ish, arm_active, mirror_active)

        return loco_action

    # ------------------------------------------------------------------ goal markers

    def _update_goal_markers(self):
        world_goals = self._goal_positions_world()
        inactive = torch.logical_not(self.arm_target_active)
        if bool(torch.any(inactive).item()):
            world_goals = world_goals.clone()
            world_goals[inactive, 2] = -10.0
        self._goal_vis.visualize(world_goals)

    def _set_arm_target(self, arm_idx: int, target_local: torch.Tensor):
        self.goal_pos_local[arm_idx] = target_local
        self.arm_target_active[arm_idx] = True

        # Return this arm to its default pose before pursuing the new target — via the
        # smooth homing move (_homing_step / ARM_HOMING_TOL_RAD's comment), not the old
        # write_joint_state_to_sim teleport (replaced 2026-07-14: an instantaneous
        # joint-state jump is a nonphysical CoM shock the standing policy never trains
        # against and real hardware can't do). The *reason* for resetting is unchanged:
        # if the previous target was out of range or poorly reached, the arm could be
        # sitting in an extreme, never-seen-in-training pose, and the policy would start
        # the new reach from that bad state. Only homes the targeted arm (relevant for
        # arm_mode="both"), not the other one mid-reach.
        n_per_arm = len(self.arm_joint_ids_robot) // (2 if self.arm_mode == "both" else 1)
        start = arm_idx * n_per_arm if self.arm_mode == "both" else 0
        # In-place writes to inference tensors (the homing flag / EMA filter buffers get
        # reassigned inside the main loop's inference_mode-wrapped step) are only allowed
        # from inside inference_mode, and this runs in main()'s loop *before* that block —
        # same restriction noted in validation/eval_standing.py (found 2026-07-12).
        with torch.inference_mode():
            self._arm_homing[arm_idx if self.arm_mode == "both" else 0] = True
            # Reset this arm's slice of the EMA action filter, same as training resets it
            # per-episode. (It's zeroed again when homing completes — this one covers the
            # case where the arm is already at default and homing finishes instantly.)
            self._filtered_arm_delta[0, start:start + n_per_arm] = 0.0

        self._update_goal_markers()
        arm_label = ["left", "right"][arm_idx] if self.arm_mode == "both" else self.arm_mode
        print(f"[FullDemo] Arm target ({arm_label}): {target_local.tolist()}")

    # ------------------------------------------------------------------ keyboard / target input

    def _prompt_target(self, kind: str = "left"):
        """Request a blocking arm-target prompt on the main loop.

        kind:
            "left"   — T key. Sets the arm_mode-selected slot (unchanged from before).
            "mirror" — Y key. Sets the mirrored right-arm target, right-arm coordinate
                       frame (only when arm_mode="left" — see self._mirror_enabled).
            "both"   — U key. One left-frame target, applied to the left arm directly
                       and mirrored (y negated) to the right arm at the same time —
                       the "does the mirror-generalized policy work on both arms
                       simultaneously" test.
        """
        if self._target_prompt_active:
            print("[FullDemo] Target prompt already open.")
            return

        self._pending_prompt_kind = kind
        self._target_prompt_requested = True

    def _parse_and_confirm_target(self, parts: list[str], bounds: dict) -> torch.Tensor | None:
        """Parse 3 numbers, confirm if out of *bounds*. Returns None to re-prompt."""
        target = torch.tensor([float(p) for p in parts], dtype=torch.float32, device=self.device)
        out_of_range = not (
            bounds["x"][0] <= target[0].item() <= bounds["x"][1]
            and bounds["y"][0] <= target[1].item() <= bounds["y"][1]
            and bounds["z"][0] <= target[2].item() <= bounds["z"][1]
        )
        if out_of_range:
            # Not just "harder" — the policy has never trained on anything outside this
            # box, so it can produce genuinely erratic motion trying to reach one, not
            # just fall a bit short.
            confirm = input(
                "  Outside the trained range — the policy has never seen this "
                "and may behave strangely. Send anyway? [y/N] "
            ).strip().lower()
            if confirm != "y":
                return None
        return target

    def _run_target_prompt(self):
        """Blocking console prompt for a new arm target on the main thread."""
        self._target_prompt_requested = False
        self._target_prompt_active = True
        kind = self._pending_prompt_kind

        if kind == "both":
            self._run_both_target_prompt()
            return

        if kind == "mirror":
            arm_label, b = "right (mirrored via left policy)", _GOAL_BOUNDS["right"]
        else:
            arm_label = (
                ["left", "right"][self._active_arm_idx]
                if self.arm_mode == "both"
                else self.arm_mode
            )
            b = _GOAL_BOUNDS["left" if arm_label == "left" else "right"]

        prompt = (
            f"\n[FullDemo] New target for {arm_label} (x y z)\n"
            f"  x: {b['x']}, y: {b['y']}, z: {b['z']}\n"
            "  > "
        )
        while True:
            try:
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self._target_prompt_active = False
                return
            if not line:
                self._target_prompt_active = False
                return
            parts = line.split()
            if len(parts) != 3:
                print("  Need exactly 3 numbers: x y z")
                continue
            try:
                target = self._parse_and_confirm_target(parts, b)
                if target is None:
                    continue

                if kind == "mirror":
                    if self._mirror_enabled:
                        self._set_mirror_target(target)
                    else:
                        print("  [FullDemo] Mirror control only available when arm_mode=left.")
                else:
                    self._set_arm_target(self._active_arm_idx, target)

                self._target_prompt_active = False
                return
            except ValueError:
                print(f"  Could not parse: {line!r}")

    def _run_both_target_prompt(self):
        """U — two independent targets, one per arm (left native, right mirrored via
        the left-trained policy). Two separate triplets, not one auto-mirrored target —
        sending the same (mirrored) position to both arms means they reach for the same
        physical point and collide."""
        if not self._mirror_enabled:
            print("  [FullDemo] Mirror control only available when arm_mode=left.")
            self._target_prompt_active = False
            return

        b_left, b_right = _GOAL_BOUNDS["left"], _GOAL_BOUNDS["right"]
        left_target = None
        while left_target is None:
            prompt = (
                f"\n[FullDemo] New target for LEFT arm (x y z)\n"
                f"  x: {b_left['x']}, y: {b_left['y']}, z: {b_left['z']}\n"
                "  > "
            )
            try:
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self._target_prompt_active = False
                return
            if not line:
                self._target_prompt_active = False
                return
            parts = line.split()
            if len(parts) != 3:
                print("  Need exactly 3 numbers: x y z")
                continue
            try:
                left_target = self._parse_and_confirm_target(parts, b_left)
            except ValueError:
                print(f"  Could not parse: {line!r}")

        right_target = None
        while right_target is None:
            prompt = (
                f"\n[FullDemo] New target for RIGHT arm (mirrored via left policy) (x y z)\n"
                f"  x: {b_right['x']}, y: {b_right['y']}, z: {b_right['z']}\n"
                "  > "
            )
            try:
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                self._target_prompt_active = False
                return
            if not line:
                self._target_prompt_active = False
                return
            parts = line.split()
            if len(parts) != 3:
                print("  Need exactly 3 numbers: x y z")
                continue
            try:
                right_target = self._parse_and_confirm_target(parts, b_right)
            except ValueError:
                print(f"  Could not parse: {line!r}")

        self._set_arm_target(0, left_target)
        self._set_mirror_target(right_target)
        self._target_prompt_active = False

    def _apply_pending_target(self):
        if not self._target_prompt_requested:
            return

        self._run_target_prompt()

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
        self._keyboard.add_callback("T", self._prompt_target)
        if self.arm_mode == "both":
            self._keyboard.add_callback("L", lambda: self._select_arm(0))
            self._keyboard.add_callback("R", lambda: self._select_arm(1))
        if self._mirror_enabled:
            self._keyboard.add_callback("Y", lambda: self._prompt_target("mirror"))
            self._keyboard.add_callback("U", lambda: self._prompt_target("both"))

    def _select_arm(self, idx: int):
        self._active_arm_idx = idx
        label = ["left", "right"][idx]
        print(f"[FullDemo] Active arm for T-key targeting: {label}")

    # ------------------------------------------------------------------ camera

    # Chase-cam offset from the robot base, robot-local frame (behind + above).
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
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.camera_path)

        # When True, update_camera() re-locks the camera behind the robot every frame
        # (the original always-on behavior). When False, update_camera() is a no-op, so
        # the camera stays exactly where the mouse last left it — that's what makes
        # manual orbiting possible at all (a camera whose transform gets overwritten
        # every frame can never hold a manual drag).
        self._camera_follow = True

    def _toggle_camera_follow(self):
        self._camera_follow = not self._camera_follow
        print(f"[FullDemo] Camera follow: {'ON' if self._camera_follow else 'OFF (orbit freely with the mouse)'}")

    def _reset_camera(self):
        """Snap back to the default chase framing and re-enable follow."""
        self._camera_follow = True
        self._position_camera()
        print("[FullDemo] Camera reset to chase view.")

    def _position_camera(self):
        base_pos  = self.robot.data.root_pos_w[0]
        base_quat = self.robot.data.root_quat_w[0]
        offset = torch.tensor(self._CAMERA_OFFSET, device=self.device)
        cam_pos = quat_apply(base_quat, offset) + base_pos

        state = ViewportCameraState(self.camera_path, self.viewport)
        state.set_position_world(
            Gf.Vec3d(cam_pos[0].item(), cam_pos[1].item(), cam_pos[2].item()), True
        )
        state.set_target_world(
            Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + self._CAMERA_TARGET_HEIGHT_OFFSET),
            True,
        )

    def update_camera(self):
        if self._camera_follow:
            self._position_camera()


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def main():
    demo = G1FullDemo()
    obs, _ = demo.env.reset()
    # No re-anchoring needed here any more (a --target-supplied goal used to be anchored
    # to the robot's pose at construction time, before this reset's randomized spawn was
    # known) — targets are now a fixed offset from the torso, recomputed from the live
    # pose every call, so this reset's real spawn pose is picked up automatically.
    step = 0

    while simulation_app.is_running():
        demo.update_camera()
        demo._update_goal_markers()
        if demo._mirror_enabled:
            demo._update_mirror_goal_marker()
        demo._apply_pending_target()

        with torch.inference_mode():
            cmd_raw = demo._keyboard.advance()
            demo.cmd_filtered = (
                (1.0 - CMD_FILTER_ALPHA) * demo.cmd_filtered
                + CMD_FILTER_ALPHA * cmd_raw
            )
            obs[:, 9:12] = demo.cmd_filtered.unsqueeze(0)

            cmd_mag = torch.linalg.vector_norm(demo.cmd_filtered).item()
            demo._maybe_switch_mode(cmd_mag)

            action = demo.select_action(obs)
            obs, _, dones, _ = demo.env.step(action)
            demo._handle_env_resets(dones)

        if step % 60 == 0:
            # Distance to arm target(s) for display
            target_world = demo._goal_positions_world()
            arm_groups = []
            if demo.arm_mode in ("left", "both"):
                arm_groups.append(("left",  0, demo.arm_body_ids[0]))
            if demo.arm_mode in ("right", "both"):
                arm_idx = 1 if demo.arm_mode == "both" else 0
                arm_groups.append(("right", arm_idx, demo.arm_body_ids[-1]))

            dist_str = ""
            for label, gi, body_id in arm_groups:
                ee  = demo.robot.data.body_pos_w[0, body_id, :]
                tgt = target_world[gi]
                d   = (tgt - ee).norm().item() * 100
                dist_str += f"  {label}: {d:.1f}cm"

            arm_active = demo._has_active_arm_target() and (demo.mode == "standing")

            print(
                f"[FullDemo] step={step:6d}  mode={demo.mode:8s}  "
                f"|cmd|={cmd_mag:.3f}  arm_active={arm_active}"
                + dist_str
            )

        demo.mode_steps += 1
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
