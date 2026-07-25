# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof asset, ported from ``unitree_rl_lab`` (``source/unitree_rl_lab/unitree_rl_lab/
assets/robots/unitree.py``), 2026-07-21 — see ``29dof_implementation_plan.md`` (repo root)
for why this is used instead of ``isaaclab_assets``'s own ``G1_29DOF_CFG``: this config's
per-joint stiffness/damping match ``deploy.yaml`` (the real, hardware-exported spec) —
confirmed by direct comparison — while ``isaaclab_assets``'s version uses generic,
unverified defaults (e.g. a flat 3000/10 gain for every arm joint).

Only the base classes and the G1 29dof config are ported — ``unitree_rl_lab``'s other
robots (Go2/B2/H1) and the 23dof/MIMIC G1 variants are out of scope for this project.

**Gains reconciled against ``deploy.yaml`` line-by-line, 2026-07-21** (per
``29dof_implementation_plan.md``'s stated tie-breaker rule): found one real discrepancy.
``unitree_rl_lab``'s currently-checked-out actuator config has ``damping=1.0`` for
shoulder/elbow/wrist_roll (N5020-16 group) and wrist_pitch/yaw (W4010-25 group), but
``git log`` on that file shows this was ``10.0`` at the exact commit (``a7c9efa``,
2025-08-06) that produced the G1-29dof velocity policy's ``deploy.yaml`` — a later commit
(``91e175d``, 2025-09-11, "modify g1-23dof cfg") dropped it to ``1.0`` for an unrelated
23dof-config change that silently affected this shared actuator group too, and it was
never revisited for 29dof specifically. Fixed here to ``10.0`` (see the inline comment
at the actual field). Every other value (stiffness, effort/velocity limits, waist_roll/
waist_pitch's own flat 40.0 stiffness, init_state, joint_sdk_names) was confirmed
byte-identical between that commit and the current checkout — nothing else to fix.

**Assets fetched 2026-07-21** — not a full clone of ``unitreerobotics/unitree_model``
(441MB across every robot it hosts); only the 4 files this project actually needs
(``G1/29dof/usd/g1_29dof_rev_1_0/`` — the main USD + its 3 ``configuration/`` sublayers,
~28.4MB total, downloaded directly via HF's ``resolve/main`` URLs, verified as real
binary USD content — ``PXR-USDC`` header — not LFS pointer stubs) into
``~/Elm/Assets/unitree_model/``, mirroring that same subpath. If other robots
(23dof G1, Go2, etc.) are ever needed, fetch their subpaths the same way rather than a
full clone.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from g1_locomotion.assets.robots import unitree_actuators  # noqa: F401 (kept for parity with upstream; not used by the 29dof cfg below, only the friction/delay actuator classes it re-exports)

UNITREE_MODEL_DIR = os.path.expanduser("~/Elm/Assets/unitree_model")


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    """Configuration for Unitree articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )


UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
            ".*_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "N7520-14.3": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            stiffness={
                ".*_hip_.*": 100.0,
                "waist_yaw_joint": 200.0,
            },
            damping={
                ".*_hip_.*": 2.0,
                "waist_yaw_joint": 5.0,
            },
            armature=0.01,
        ),
        "N7520-22.5": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            effort_limit_sim=139,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_roll_.*": 100.0,
                ".*_knee_.*": 150.0,
            },
            damping={
                ".*_hip_roll_.*": 2.0,
                ".*_knee_.*": 4.0,
            },
            armature=0.01,
        ),
        "N5020-16": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim=25,
            velocity_limit_sim=37,
            stiffness=40.0,
            damping={
                # 10.0, not the current unitree_rl_lab checkout's 1.0 — see the module
                # docstring's "gains reconciled against deploy.yaml" note. Verified via
                # git history (unitree_rl_lab commit 91e175d, 2025-09-11, "modify
                # g1-23dof cfg") that 10.0 was the value in place when this exact
                # G1-29dof velocity policy's deploy.yaml was exported (commit a7c9efa,
                # 2025-08-06); the later commit dropped it to 1.0 for a 23dof-config
                # change and never touched it again for 29dof specifically. Using the
                # value the actually-working, hardware-exported policy trained under.
                ".*_shoulder_.*": 10.0,
                ".*_elbow_.*": 10.0,
                ".*_wrist_roll.*": 10.0,
                ".*_ankle_.*": 2.0,
                "waist_.*_joint": 5.0,
            },
            armature=0.01,
        ),
        "W4010-25": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            effort_limit_sim=5,
            velocity_limit_sim=22,
            stiffness=40.0,
            damping=10.0,  # same fix/rationale as N5020-16's damping dict above
            armature=0.01,
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
)
"""Configuration for the Unitree G1 29dof Humanoid robot.

3-DOF waist (yaw/roll/pitch), 7-DOF arms (shoulder pitch/roll/yaw, elbow, wrist
roll/pitch/yaw), 6-DOF legs (hip pitch/roll/yaw, knee, ankle pitch/roll) — matches the
real G1 EDU hardware layout (see ``29dof_pivot_context.md``). ``joint_sdk_names`` is the
real hardware SDK's joint ordering, for whenever a deployment/ONNX-export path needs it
(distinct from whatever order PhysX's ``find_joints()`` returns at runtime, which is
breadth-first and interleaves left/right — never assume the two orderings agree, see
``lessons_learned.md`` #1).
"""
