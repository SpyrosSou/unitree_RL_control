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
for full current status and findings, then `retrospective.md` for a narrative summary
of what was tried this week and why (including why walking is one combined policy
instead of two decoupled ones).**

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
2026-07-27: `walking_latest.pt` is the deployable walking+standing checkpoint;
`arm_left_latest.pt` is stale — the actual current-best arm checkpoint (`best_combined`)
hasn't been promoted there yet, see `chosen_checkpoints/README.md`.

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
`--reset_arm_on_walk` flag.

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

**Current status (2026-07-27):**
- **Walking/standing**: working, `chosen_checkpoints/walking_latest.pt` — 0% fall rate,
  real verified translation, but a known ~24-27° heading drift over a straight 20s walk.
  Three rounds of drift-fix attempts all made heading drift *worse*, not better —
  treated as a real pattern, not bad luck, and currently on hold. A live hypothesis
  (`base_height`'s reward weight forcing near-locked knees, reducing balance-recovery
  margin) is flagged but not yet acted on.
- **Arms**: no single checkpoint is fully trustworthy yet. The training-time success
  metric (up to 99.98% for one reference checkpoint) does NOT reflect true single-shot
  reliability (~30%, confirmed via a dedicated long-hold/single-attempt test) — a real,
  hard-won methodology finding, not just a tuning gap. `best_combined`
  (`logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt`) is the current
  best RL candidate under every eval method tried so far.
- **Current direction**: given RL alone hasn't reliably solved precise single-shot
  reaching, the plan is to use `unitreerobotics/xr_teleoperate`'s `G1_29_ArmIK`
  (Pinocchio+CasADi, official Unitree code, matches this exact robot) for precise
  grasp targets, keeping the RL policy for compliant continuous-gesture motion. This is
  being implemented on a new branch (proposed: `29dof_IK`) — `main` stays exactly as-is
  as the fallback if the IK approach doesn't pan out.

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
