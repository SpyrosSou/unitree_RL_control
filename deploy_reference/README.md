# Deploy reference (from `unitree_rl_lab`)

Copied 2026-07-21 from `~/Elm/Code/unitree_rl_lab/deploy/` as **inert reference
material** — see `29dof_implementation_plan.md` (repo root, Phase 6). Not wired into
this repo's build or run in any way. Real integration is scoped for later (no physical
robot access for another week, per the user).

This is the real, hardware-validated sim2real vehicle for whenever this project is ready
for it: a C++ FSM (`include/FSM/` — Passive/FixStand/Velocity states, `CtrlFSM.h`) driving
an ONNX Runtime inference loop, plus a C++ port of Isaac Lab's `ManagerBasedRLEnv`
observation/action pipeline (`include/isaaclab/`) so the same obs/action structure that
runs in sim runs on hardware.

## What's here

- `include/` — shared FSM + `isaaclab`-port headers (robot-agnostic).
- `robots/g1_29dof/` — the G1 29dof-specific FSM states, `main.cpp`, `CMakeLists.txt`,
  and `config/` (`config.yaml`, and the velocity-task `deploy.yaml` + exported
  `policy.onnx` — the ground-truth joint order/gain/scale spec referenced throughout
  `29dof_implementation_plan.md`).

## What's deliberately NOT here (and why)

- **`thirdparty/onnxruntime-linux-x64-1.22.0/`** (~22MB of prebuilt binaries) — not
  source code, not worth committing; re-download from the [ONNX Runtime releases
  page](https://github.com/microsoft/onnxruntime/releases) (v1.22.0, linux-x64) when
  actually building this.
- **`config/policy/mimic/`** (dance-motion policies, ~3.2MB of `.onnx`/`.bvh` motion
  data) — unrelated to this project's locomotion/arm goals, per
  `29dof_implementation_plan.md`'s "what gets ported" list. `src/State_Mimic.cpp` /
  `include/State_Mimic.h` are kept (small, and `main.cpp` references the FSM state) even
  though their data dependency isn't.

## Building this (not done yet)

Untouched from upstream — see `~/Elm/Code/unitree_rl_lab/deploy/README.md` (if present)
or the main `unitree_rl_lab` README's deploy section for the real build instructions
(needs `unitree_sdk2`, `Boost`, `yaml-cpp`, Eigen3, CycloneDDS/iceoryx, and the
onnxruntime binaries above). Nothing here has been adapted, tested, or built inside this
repo.
