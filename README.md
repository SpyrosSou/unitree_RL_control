# Unitree RL Control

Reinforcement-learning-based locomotion and arm control for the Unitree G1 using
Isaac Sim, Isaac Lab, and RSL-RL.

This repository contains:

- locomotion tasks for walking and standing,
- arm reaching tasks for left, right, or both arms,
- integration demos combining standing, walking, and arm control,
- curated deployable checkpoints under `chosen_checkpoints/`.

## What Is In This Repo

Main task families:

- Walking: `G1-Locomotion-Flat-v0`
- Standing: `G1-Locomotion-Standing-Flat-v0`
- Arm IK: `G1-Arm-IK-Left-v0`, `G1-Arm-IK-Right-v0`, `G1-Arm-IK-Both-v0`

Recent customizations include:

- single walking policy with stronger transition exposure,
- standing policy with a phased arm-motion-disturbance curriculum, teaching it to take a
  corrective step rather than only balancing in place (no push/external-force disturbance —
  that was tried and removed, see `training_regimes.md`),
- arm action filtering and per-step delta limiting for smoother arm motion.

Standing-policy details (current):

- Standing commands are mostly near-zero, with a small corrective-velocity slice.
- Reward shaping is tuned for balance recovery rather than gait generation:
    - `lin_vel_z_l2 = -2.2`
    - `ang_vel_xy_l2 = -0.12`
    - `action_rate_l2 = -0.008`
    - `dof_acc_l2 = -1.5e-7`
    - `feet_air_time = 0.0`
    - `joint_deviation_torso = 0.0` (relaxed so the torso can help with balance recovery)
- Arm disturbance curriculum phases use per-step arm target deltas:
    - Phase 0: `0.00 rad/step`
    - Phase 1: `0.03 rad/step`
    - Phase 2: `0.05 rad/step`
    - Phase 3: `0.10 rad/step`
    - Phase 4: `0.25 rad/step`

Standing logging details:

- Standing training writes `standing_summary.csv` (convergence-at-a-glance) and
  `standing_detailed.csv` (everything, including a stepping-detection signal
  `step_count`/`max_foot_air_time_s`) in each standing run directory — see
  `logging_reference.md` for the full column reference and how to read them.

## Environment

Expected local workflow:

```bash
conda activate isaac_g1_control
cd ~/Elm/Code/g1_locomotion
```

Isaac Lab is expected to be installed separately and available in the active
Python environment.

## Curated Checkpoints

This repo ignores large training logs and intermediate checkpoints, but keeps a
small curated set of deployable models in:

- `chosen_checkpoints/walking_latest.pt`
- `chosen_checkpoints/standing_latest.pt`
- `chosen_checkpoints/arm_left_latest.pt`

These are the default checkpoints referenced by the YAML config files for demos
and tests.

## Quick Start

### Train

```bash
python scripts/rsl_rl/train.py --task G1-Locomotion-Flat-v0 --num_envs 4096 --headless --max_iterations 2500
python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-v0 --num_envs 4096 --headless --max_iterations 1500
python scripts/rsl_rl/train.py --task G1-Arm-IK-Left-v0 --num_envs 4096 --headless --max_iterations 5000
```

### Generic Play

```bash
python scripts/rsl_rl/play.py --task G1-Locomotion-Flat-Play-v0 --checkpoint chosen_checkpoints/walking_latest.pt
python scripts/rsl_rl/play.py --task G1-Locomotion-Standing-Flat-Play-v0 --checkpoint chosen_checkpoints/standing_latest.pt
python scripts/rsl_rl/play.py --task G1-Arm-IK-Left-Play-v0 --checkpoint chosen_checkpoints/arm_left_latest.pt
```

### Full Integrated Demo

```bash
python testing/general_testing/g1_full_demo.py
```

The integrated demo reads default checkpoint paths from:

- `testing/walking_testing/checkpoints.yaml`
- `testing/arm_testing/checkpoints.yaml`
- `testing/general_testing/checkpoints.yaml`

## Repo Layout

- `source/g1_locomotion/` — task registration and environment/task code
- `scripts/` — train/play entry points
- `testing/` — interactive demo/test scripts, one subdirectory per policy family:
  - `testing/walking_testing/` — locomotion demos and switch demos
  - `testing/arm_testing/` — arm evaluation and mirror tests
  - `testing/general_testing/` — combined demos
  - `testing/quickrun_tests.md` — how to run the scripts above
- `chosen_checkpoints/` — curated deployable checkpoints
- `quickrun.md` — command reference (train/play/resume)
- `training_regimes.md` — training setup summary
- `algorithm_explanation.md` — PPO/network/observation-space reference
- `logging_reference.md` — what gets logged during training, where, and what each metric means
- `phase_logs/` — running change log per roadmap phase (see `training_regimes.md`)

## Notes On Ignored Files

Ignored by default:

- training logs under `logs/`
- generated outputs under `outputs/`
- local learning notes under `personal_development/`
- intermediate `.pt` checkpoints outside `chosen_checkpoints/`

This keeps the repo small while still shipping usable models.

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