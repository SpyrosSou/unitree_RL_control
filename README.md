# Unitree RL Control — G1 29dof

Reinforcement-learning-based locomotion and arm control for the Unitree G1 (29-DOF: 3-DOF
waist, 7-DOF arms with wrists) using Isaac Sim, Isaac Lab, and RSL-RL.

This repository contains:

- a unified walking+standing locomotion policy (one policy covers both — standing is
  just the near-zero-velocity edge of the command distribution),
- arm-reaching tasks for left, right, or both arms (pure RL joint-space control, not
  IK — see `policy_status.md` for the current strategic pivot toward IK for precision
  grasping specifically),
- integration demos combining locomotion and arm control on one robot,
- curated deployable checkpoints under `chosen_checkpoints/`.

**Start here if you're new to this repo (human or a fresh Claude chat): read the
"Overview / up to speed" section near the bottom of this file, then `policy_status.md`
for full current status and findings (or `policy_overview.md` for a brief
reward/curriculum/what's-been-tried summary of both current recipes), then
`retrospective.md` for a narrative summary of what was tried this week and why
(including why walking is one combined policy instead of two decoupled ones). If
you're specifically picking up the arm policy's last remaining step, go straight to
`arms_policy_finalisation.md`.**

## What Is In This Repo

Main task families (see each task family's own `__init__.py` for the full gym task
list — these are the primary ones):

- Walking + standing: `G1-Locomotion-Velocity-v0` (base recipe), `G1-Locomotion-Velocity-ArmDisturbance-v0`
  (real deployment recipe — includes the arm-motion-disturbance curriculum while standing)
- Arm reaching: `G1-Arm-Left-v0` (and `-Right-v0`/`-Both-v0`), plus several tuned variants
  registered in `source/g1_locomotion/g1_locomotion/tasks/manager_based/g1_arm/__init__.py`
  (see config-class docstrings in `g1_arm_env.py` for what each variant tests)

Environment/task code lives under `source/g1_locomotion/g1_locomotion/tasks/manager_based/`,
split into `g1_locomotion/` (walking+standing) and `g1_arm/` (arm reaching).

## Environment

Expected local workflow:

```bash
conda activate isaac_g1_control
cd ~/Elm/Code/g1_locomotion
```

Isaac Lab is expected to be installed separately and available in the active Python
environment.

## Curated Checkpoints

This repo ignores large training logs (`logs/`) and intermediate checkpoints, but keeps
a small curated set of deployable models in `chosen_checkpoints/` — see that directory's
own `README.md` for exactly which checkpoint is current and its provenance. As of
2026-07-30: `walking_latest.pt` is the promoted walking+standing checkpoint (the
"standing package" recipe — much stiller when standing, at the cost of more
straight-line heading drift and untrained in-place turning; see
`chosen_checkpoints/README.md`). `arm_left_latest.pt` is still stale — the actual
current-best arm checkpoint (`G1-Arm-Left-Integrated-v0`, which solved precise
reaching via a 2026-07-29 fix) hasn't been promoted there yet: it needs one more
reward-design retrain first (see `arms_policy_finalisation.md`) and isn't yet
compatible with the demo script below.

## Quick Start

### Train

```bash
python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-v0 --num_envs 4096 --headless --max_iterations 8000
python scripts/rsl_rl/train.py --task G1-Arm-Left-v0 --num_envs 4096 --headless --max_iterations 2000
```

### Evaluation (metrics, headless-friendly, writes a `summary.md` per checkpoint)

```bash
python validation/eval_walking.py --checkpoint chosen_checkpoints/walking_latest.pt --arm_disturbance --headless
python validation/eval_arm.py --checkpoint logs/rsl_rl/arms/<run>/model_<iter>.pt --headless
```

### Visual inspection (watch the policy, not just metrics)

See `testing/visual_testing/visual_inspections.md` — a catalog of commands for watching
specific behaviors (drift, disturbance response, transitions) with a one-line note on
what each shows.

### Full Integrated Demo

```bash
python testing/visual_testing/full_demo/g1_full_demo.py \
    --loco_checkpoint chosen_checkpoints/walking_latest.pt \
    --arm_checkpoint logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt \
    --arm left --target 0.3 0.2 1.0
```

See `testing/visual_testing/full_demo/README.md` for controls, keybindings, and the
`--reset_arm_on_walk` flag. `--loco_checkpoint` above now resolves to the 2026-07-30
"standing package" checkpoint (see `chosen_checkpoints/README.md`) automatically.
`--arm_checkpoint` is left on `best_combined` deliberately — the newer, better
`G1-Arm-Left-Integrated-v0` checkpoint is not yet demo-compatible (this script
hand-rolls its own arm control loop rather than reusing `G1ArmEnv`, and doesn't yet
know about that task's integrated-target action pipeline or 46-D observation
layout) — see `chosen_checkpoints/README.md` for what's needed before that changes.

## Repo Layout

- `source/g1_locomotion/` — task registration and environment/task code
- `scripts/` — train/play entry points
- `testing/`
  - `testing/visual_testing/` — everything meant to be *watched*, not just measured:
    `walk/`, `arms/`, `full_demo/` (integrated demo + its own README), and
    `visual_inspections.md` (the command catalog)
  - `testing/general_testing/` — headless-capable diagnostic scripts (`check_*.py`)
- `validation/` — batch eval scripts producing `summary.md`/CSV per checkpoint; each
  script's own docstring/`--help` is the usage reference
- `chosen_checkpoints/` — curated deployable checkpoints (see its own `README.md` for
  provenance)
- `policy_status.md` — **the maintained living status document**: what's working, what's
  been tried and ruled out, current findings, deferred/future items, lessons learned.
  This is the first thing to read for current state.

## Notes On Ignored Files

Ignored by default:

- training logs under `logs/`
- generated outputs under `outputs/`
- local learning notes under `personal_development/`
- intermediate `.pt` checkpoints outside `chosen_checkpoints/`

This keeps the repo small while still shipping usable models.

## Overview / up to speed

*Read this section (and then `policy_status.md`) to get full context on this repo
without needing a long introduction from the user.*

**What this is:** RL control (Isaac Lab + RSL-RL, PPO) for a Unitree G1 humanoid,
29-DOF variant (3-DOF waist, 7-DOF arms with wrists — matches the physical EDU robot).
Three things are being built: a unified walk+stand locomotion policy, an arm-reaching
policy, and integration of the two on one robot. Deployment target is genuine RL
joint-space control, not classical IK — arm-IK is currently only being explored as a
possible *replacement* for the reaching policy specifically (see "current direction"
below), not as a training-time tool.

**Why 29dof specifically:** the physical robot doesn't match Isaac Lab's default
`G1_MINIMAL_CFG` asset (1-DOF waist, 5-DOF arms, no wrists) — everything here was
rebuilt against `UNITREE_G1_29DOF_CFG` starting 2026-07-21, using
`unitreerobotics/unitree_rl_lab`'s stand+walk recipe as the retraining base (this
project's own validation/testing methodology was kept, only the underlying asset/recipe
changed). Older 23-DOF-era work lives on the `23_dof` git branch and an external backup
at `~/Elm/Backups/g1_locomotion/23_dof/` if ever needed — not deleted, just superseded.

**Current status (2026-07-30):**
- **Walking/standing**: working, `chosen_checkpoints/walking_latest.pt` — 0% fall rate
  across every command bucket (including in-distribution turning) and every
  arm-disturbance phase, real verified translation, and — as of the 2026-07-30
  "standing package" promotion — genuinely still while standing (step count ~2,
  lateral drift ~3cm). Traded off to get there: straight-line heading drift while
  actually walking got worse (~112° vs. the prior checkpoint's ~62° over a 20s
  episode), and sustained in-place turning is now essentially untrained/ignored.
  Several earlier drift-fix rounds also made heading drift worse, not better — see
  `policy_status.md` for the full history and what's still open.
- **Arms**: the long-standing ~25-30% plateau is **resolved** — root cause found
  2026-07-29 (the action pipeline capped static holding torque well below what
  gravity needs at the real 40/10 hardware gain) and fixed via an integrated
  action-target parameterization (`G1-Arm-Left-Integrated-v0`). Reaching itself is
  now genuinely solved: 100% reach rate, ~1-3mm typical precision. One more
  reward-design retrain is needed before promotion (the current checkpoint dithers
  right at the 2cm boundary instead of settling deep, due to a termination/bonus
  incentive issue, not a capability gap) — see `arms_policy_finalisation.md` for the
  exact pick-up plan, or `policy_status.md` for the full evidence trail.
- **IK pivot, reassessed**: `unitreerobotics/xr_teleoperate`'s `G1_29_ArmIK` was
  under consideration as a fallback while pure-RL reaching looked stuck (see history
  below) — with reaching now solved, IK is no longer needed as the primary path for
  precise grasping, though it may still be worth keeping as a belt-and-braces option
  for exact SE(3) targets once the arm policy is otherwise finished.

**Working conventions worth knowing before touching anything:**
- Verify before concluding — this project has repeatedly found that plausible-sounding
  explanations ("almost certainly gravity," "should be fine") were wrong when actually
  measured. Prefer a quick real check over an inference chain.
- GPU/Isaac Sim commands need explicit permission *each time*, not just once per
  session — don't launch training/eval/demo scripts without asking first, and don't
  retry immediately after a failure without checking in.
- Only commit/push when explicitly asked.
- `policy_status.md` is where findings get written down as they're confirmed — check it
  first, keep it current as you go.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/g1_locomotion

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set Up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup As Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/g1_locomotion/g1_locomotion/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/g1_locomotion"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
