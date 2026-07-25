# 29dof pivot — context for planning (start here in a new chat)

Written 2026-07-21 as a handoff brief. `lessons_learned.md` covers RL-training gotchas
from the 23dof phase; this file covers the strategic decisions and findings that should
shape the actual 29dof plan. Read both before writing the plan. Full detail behind any
of this lives on the `23_dof` git branch if ever needed — this is the compressed version.

## The hardware reality

The physical robot is the G1 **EDU**, believed 29dof (3-DOF waist — yaw/roll/pitch —
plus 7-DOF arms including wrists; ~99% confirmed by the user, pending final colleague
confirmation). Every checkpoint trained in the 23dof phase (standing, walking, arm) was
built against Isaac Lab's `G1_MINIMAL_CFG` asset — 1-DOF waist, 5-DOF arms, no wrists —
which does not match. All three policies need retraining from scratch against the
correct asset.

**Isaac Lab already ships the correct asset**: `G1_29DOF_CFG` in
`isaaclab_assets/robots/unitree.py` (same file the project already imports
`G1_MINIMAL_CFG` from) — 3-DOF waist, 7-DOF arms with wrists, matches the physical
robot's joint layout. There's also `G1_INSPIRE_FTP_CFG` (29dof body + Inspire 5-finger
hands) if dexterous hands ever become relevant — not needed now, the "7+7 hand DOF" the
user mentioned can be disregarded per their own instruction.

## The Unitree reference repo (`~/Elm/Code/unitree_rl_lab`)

Real, official, actively maintained (`github.com/unitreerobotics/unitree_rl_lab`, Apache
2.0, fully editable, no license/access restriction). Confirmed via its README and its own
`main` branch (the correct branch to use — `git remote show origin` confirms it's the
repo's own HEAD; other branches are narrow compat/experimental forks:
`isaaclab3.0`/`lab21` for different Isaac Lab versions, `devel-newton` for a different
physics backend, `sim2sim` for Mujoco cross-validation, `dev/add_new_robots` WIP).

**What it has:**
- A single, unified stand+walk velocity-tracking policy for G1-29dof (standing is just
  `rel_standing_envs=0.02` — a rare near-zero-velocity edge case of the walking
  distribution, not a dedicated disturbance-robust standing policy the way this project
  built one). Their `feet_gait` reward auto-disables below a velocity-command threshold
  — that's the whole mechanism for how one policy covers both behaviors.
- A reward/termination set worth using as a starting template:
  `joint_deviation_waists` (their torso equivalent) at **-1.0 permanently**, never
  relaxed to 0 the way this project's did; hip deviation at **-1.0** not the original
  -0.1; a hard **`bad_orientation` termination** (`limit_angle=0.8` rad) — ends the
  episode on excess tilt regardless of which joint caused it, structurally different
  from (and likely more robust than) every per-joint reward tweak tried in the 23dof
  phase. Actuator delay/friction modeling already implemented (`CHANGELOG.rst`: "Added
  delay & friction model support") — something this project's own deferred-items list
  had flagged as not-yet-done.
- A real, hardware-validated **deploy harness** (`deploy/` — C++ FSM: Passive/FixStand/
  Velocity states, ONNX runtime inference loop, a full C++ port of
  `isaaclab::ManagerBasedRLEnv` so the exact same obs/action pipeline structure runs on
  hardware as in sim). This is a substantial, currently-nonexistent-in-this-project
  piece of engineering — building an equivalent from scratch would be a real, easily
  underestimated cost.
- **Zero arm/manipulation/IK code anywhere.** Only two task families exist: locomotion
  (above) and `mimic` (motion-imitation dance tasks, e.g. `dance_102`,
  `gangnam_style` — a different technique, unrelated to what this project needs).
- Only an exported `.onnx` inference checkpoint exists (`deploy/robots/g1_29dof/config/
  policy/velocity/v0/exported/policy.onnx`) — **no raw rsl_rl/PyTorch checkpoint**, so
  there is nothing to warm-start or fine-tune from. Using their recipe means training it
  ourselves from scratch, same as adapting our own recipe would.
- One relevant, now-closed GitHub issue (#44, "G1 Robot Unresponsive in Velocity Mode —
  Sim2Real Issue") — real sim2real friction has been hit and resolved by others on this
  exact repo before.

**Important correction made mid-investigation**: the user's own physical-robot testing
(walking, handshake/wave gestures) was NOT running any RL policy — legs used the robot's
stock built-in controller (Unitree's separate, likely-proprietary `ai_sport` service /
`LocoClient` high-level API, FSM states 500/501 — structurally distinct from
`unitree_rl_lab`), and arm gestures were pre-built joint commands from the Unitree
Python SDK plus some custom scripted commands from a different project — not RL, not
IK-based. So the physical testing validates the hardware and Unitree's engineering in
general, but does NOT specifically validate `unitree_rl_lab`'s training recipe — that
recipe is exactly as unproven as anything this project would train itself.

## The decision (already made, not open for re-litigation unless something changes it)

**Use `unitree_rl_lab`'s stand+walk (legs+waist) recipe as the base for retraining**,
build this project's own arm-IK policy on top of it (7-DOF now, was 5-DOF — needs
rebuilding regardless of which base is used, this was never a differentiator between
the two options), keep this project's own validation/testing infrastructure and
methodology (the disturbance-curriculum approach, the diagnostic tooling, the eval
scripts — retarget to the new joint set, don't discard the patterns), and take
`unitree_rl_lab`'s `deploy/` harness as the sim2real vehicle when that phase arrives.
Explicitly reversible: "it's not like our stuff is getting deleted, if the unitree
stand+walk is bad with arms we switch back to ours" (user's own words) — the 23dof
branch and this project's own architecture remain a fallback, not deleted.

Residual known risk, not a reason to change course: `unitree_rl_lab`'s stand+walk has
never faced an active arm-reach disturbance either (same as this project's own standing
task before this phase's work) — expect a real integration/debugging effort getting
that combination to work, not a plug-and-play result.

## Project goal, restated by the user (2026-07-21), for reference

1. Create walk and stand policy/policies
2. Create arm policy
3. Integrate
4. Optional: expand to move arms while walking
5. Deploy on the robot, replacing the existing (stock) walking controller and adding
   real arm control — genuine joint-space RL output, explicitly **not** FK/IK-based at
   deployment (matches this project's existing arm-policy design already — analytic IK
   was only ever a training-time disturbance generator for standing, never deployed).

## Current repo state

`main` branch is cleaned and ready for the 29dof rebuild: all 23dof checkpoints, logs,
and stale scratch/output folders removed (kept safely on the `23_dof` branch + an
external backup at `~/Elm/Backups/g1_locomotion/23_dof/`, see that branch's
`repo_structure.md` for exactly what's where). `testing/` and `validation/*.py` scripts
were deliberately kept as-is — real code/patterns worth adapting, not stale data — the
DOF-specific parts (joint lists, checkpoint paths) will need updating once work starts.

**Not yet done**: the actual 29dof implementation plan itself. That's the next step.
