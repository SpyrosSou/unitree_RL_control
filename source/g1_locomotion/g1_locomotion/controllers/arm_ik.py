# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical arm IK for the G1 29dof — CasADi/Pinocchio solver adapted from Unitree's
``xr_teleoperate`` repo (``teleop/robot_control/robot_arm_ik.py``'s ``G1_29_ArmIK``,
https://github.com/unitreerobotics/xr_teleoperate, Apache License 2.0). See
``ik_arm_integration_plan.md`` (repo root) for the integration plan; this module is
Phase 0's vendored/adapted deliverable.

Adaptations verified against our own assets, 2026-07-27 (see the plan doc's landmines
#7/#8 for the full reasoning):

- Built from **our own** URDF (`g1_29dof.urdf`, vendored into
  ``g1_locomotion/assets/urdf/``) rather than upstream's `g1_body29_hand14.urdf` — the
  arm chain is kinematically identical (verified byte-for-byte at the wrist joints), but
  ours has no hand/finger joints, so nothing needs locking there. Only the kinematic
  tree is used (``pin.buildModelFromUrdf`` — no mesh/geometry loading), so the URDF's own
  mesh references never need to resolve.
- End-effector = the **wrist-yaw joint frame directly**
  (`left_wrist_yaw_joint`/`right_wrist_yaw_joint`) rather than a palm/hand frame — this
  integration only targets the wrist's x/y/z (hands/fingers out of scope). Pinocchio
  already exposes a named frame at every joint, so no custom `OP_FRAME` offset is
  needed (upstream hand-codes an approximate 5cm offset from the wrist for its own
  dex3-gripper palm frame; not applicable here).
- Our URDF's root joint (`floating_base_joint`, world -> pelvis) is an *active* floating
  joint — locked here alongside legs/waist, or the reduced model would silently gain 6
  bogus configuration variables that upstream's locking scheme never anticipates. This
  has no bearing on Isaac Sim or the walking policy: callers must pass wrist targets
  already expressed **relative to the pelvis**, recomputed every control step from the
  pelvis's live pose in sim — once the floating joint is locked at its neutral
  (identity) configuration, this reduced model's own frame poses are inherently in that
  same pelvis-fixed convention, so the pelvis's real, moving pose in the world never
  enters this computation at all.
- Both arms are solved as one 14-DOF problem, matching upstream — the cost decomposes
  additively per arm once the waist is locked, so this is equivalent to solving each arm
  independently; keeping the joint solve avoids diverging from the upstream reference
  design. Callers that only care about one arm should pass that arm's own current FK
  pose (see ``fk_wrist``) as the *other* arm's target, so its error term stays near zero
  and doesn't perturb the solve.

Not carried over from upstream: hand/finger joint locking (none exist in our URDF),
``meshcat`` visualization, human-arm motion scaling (``scale_arms`` — a teleop-specific
feature, unused here).
"""

import os

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from .weighted_moving_filter import WeightedMovingFilter

_URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "urdf", "g1_29dof.urdf")

# Everything except the 14 arm joints (7 per side) is locked at its neutral (zero, or
# identity for the floating joint) configuration — see the module docstring's floating-
# base note for why locking `floating_base_joint` here is correct, not a workaround.
_LOCKED_JOINTS = [
    "floating_base_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

# The reduced model's own joint order (Pinocchio/URDF traversal order — NOT PhysX's
# breadth-first order; translate through these names, never by list position, when
# mapping `sol_q` on/off the simulated robot's joint columns. See landmine #1 in
# `ik_arm_integration_plan.md`).
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class G1ArmIK:
    """Both-arm (14-DOF) numerical IK, solved via CasADi/IPOPT over a reduced Pinocchio
    model of the G1's arms (legs/waist/floating-base locked at neutral). Targets are 4x4
    SE(3) homogeneous transforms for the wrist frames, expressed relative to the pelvis.
    """

    def __init__(self, urdf_path: str = _URDF_PATH):
        full_model = pin.buildModelFromUrdf(urdf_path)
        lock_ids = [full_model.getJointId(name) for name in _LOCKED_JOINTS]
        q_neutral = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        self.data = self.model.createData()

        self.L_hand_id = self.model.getFrameId("left_wrist_yaw_joint")
        self.R_hand_id = self.model.getFrameId("right_wrist_yaw_joint")

        # --- CasADi symbolic setup (mirrors upstream G1_29_ArmIK.__init__) ---
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
            )],
        )
        rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [casadi.vertcat(
                cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
            )],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.var_q_last = self.opti.parameter(self.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        translational_cost = casadi.sumsqr(translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        rotation_cost = casadi.sumsqr(rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        regularization_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit, self.var_q, self.model.upperPositionLimit,
        ))
        # Rotation weight 0.0, not upstream's 1.0 (2026-07-27, see landmine #10 in
        # ik_arm_integration_plan.md): this integration is position-only (x/y/z reach —
        # hands/fingers/wrist orientation are out of scope, see the module docstring),
        # unlike upstream's real orientation-tracking teleop use case. Forcing the wrist
        # toward the target transform's (identity) *rotation* measurably fights the
        # *position* reach for many points — Phase 1's dense sweep found this costing
        # ~2cm of systematic error across large parts of the goal box, enough on its own
        # to fail the accuracy gate. The rotational-error machinery is kept (not deleted)
        # for the future 6-DOF-orientation phase (`policy_status.md` deferred item #2) —
        # only its weight changes here.
        self.opti.minimize(50 * translational_cost + 0.0 * rotation_cost + 0.02 * regularization_cost + 0.1 * smooth_cost)

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 30,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.model.nq)
        self.last_solve_succeeded: bool | None = None  # set by solve_ik; None until first call

    def reset(self, q: np.ndarray | None = None) -> None:
        """Clear the smoothing filter's rolling window and re-seed the warm start.

        ``solve_ik``'s output is a 4-call weighted average of *raw* solver solutions
        (``WeightedMovingFilter``), which is correct and desirable for a continuous
        control loop (consecutive targets close in time/space -> smooth motion) but
        actively wrong across a **discontinuous** target change (a new episode, a
        teleop operator jumping to a far-away point): for up to 3 calls afterward, the
        filter keeps blending in raw solutions from the *previous*, unrelated target,
        silently corrupting the output — this is not a solver error (found 2026-07-27,
        chasing an apparently-inaccurate solve that turned out to be perfectly accurate
        pre-filter). Call this whenever the target changes discontinuously, before the
        first ``solve_ik`` call for the new target.
        """
        self.init_data = np.zeros(self.model.nq) if q is None else np.asarray(q, dtype=float)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.model.nq)

    def solve_ik(
        self,
        left_wrist_tf: np.ndarray,
        right_wrist_tf: np.ndarray,
        current_q: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for joint angles reaching *left_wrist_tf*/*right_wrist_tf* (each a 4x4
        SE(3) homogeneous transform, pelvis frame). *current_q* seeds the warm start
        (falls back to the previous solve's result if omitted).

        Returns ``(sol_q, sol_tauff)`` — both length ``model.nq`` (14), ordered per
        ``ARM_JOINT_NAMES``. ``sol_tauff`` is the feedforward torque via RNEA at zero
        velocity/acceleration (gravity/Coriolis-at-rest compensation only, not a general
        dynamic feedforward). Also sets ``self.last_solve_succeeded`` (``True`` if IPOPT
        converged, ``False`` if this call fell back to its last debug iterate) — check
        it when solve success rate matters (e.g. a dense reachability sweep), since a
        fallback still returns a usable ``sol_q``/``sol_tauff`` rather than raising.
        """
        if current_q is not None:
            self.init_data = current_q
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf_l, left_wrist_tf)
        self.opti.set_value(self.param_tf_r, right_wrist_tf)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.last_solve_succeeded = True
        except Exception:
            sol_q = self.opti.debug.value(self.var_q)
            self.last_solve_succeeded = False

        self.smooth_filter.add_data(sol_q)
        sol_q = self.smooth_filter.filtered_data
        self.init_data = sol_q

        v = np.zeros(self.model.nv)
        sol_tauff = pin.rnea(self.model, self.data, sol_q, v, np.zeros(self.model.nv))
        return sol_q, sol_tauff

    def fk_wrist(self, q: np.ndarray, side: str) -> np.ndarray:
        """Forward-kinematics wrist position (3,) for *side* ("left"/"right") at joint
        configuration *q* (length 14, ``ARM_JOINT_NAMES`` order) — in the same
        pelvis-relative frame ``solve_ik`` expects targets in."""
        pin.framesForwardKinematics(self.model, self.data, q)
        frame_id = self.L_hand_id if side == "left" else self.R_hand_id
        return self.data.oMf[frame_id].translation.copy()
