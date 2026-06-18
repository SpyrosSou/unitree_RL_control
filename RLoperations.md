# RL Operations

All commands assume the conda env is active and you are in the project root:

```bash
conda activate isaac_g1_control
cd ~/Elm/Code/g1_locomotion
```

---

## Environments

| ID | Terrain | Use |
|---|---|---|
| `G1-Locomotion-Flat-v0` | Flat | Training |
| `G1-Locomotion-Flat-Play-v0` | Flat | Evaluation / visualisation |
| `G1-Locomotion-Rough-v0` | Rough | Training |
| `G1-Locomotion-Rough-Play-v0` | Rough | Evaluation / visualisation |
| `G1-Locomotion-Standing-Flat-v0` | Flat | Training (standing-only policy) |
| `G1-Locomotion-Standing-Flat-Play-v0` | Flat | Evaluation / visualisation |

---

## Train

```bash
# Flat terrain — RSL-RL (recommended baseline)
python scripts/rsl_rl/train.py --task G1-Locomotion-Flat-v0 --num_envs 4096 --headless

# Rough terrain — RSL-RL
python scripts/rsl_rl/train.py --task G1-Locomotion-Rough-v0 --num_envs 4096 --headless

# Standing-only policy (isolated logs under standing/)
python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-v0 --num_envs 4096 --headless

# Other frameworks (same pattern)
python scripts/rl_games/train.py --task G1-Locomotion-Flat-v0  --num_envs 4096 --headless
python scripts/skrl/train.py     --task G1-Locomotion-Flat-v0  --num_envs 4096 --headless
python scripts/sb3/train.py      --task G1-Locomotion-Flat-v0  --num_envs 4096 --headless
```

Useful extra flags:
- `--max_iterations 500` — override the iteration count from the config
- `--seed 42` — fix the random seed

## Resume training from a checkpoint

```bash
python scripts/rsl_rl/train.py \
    --task G1-Locomotion-Flat-v0 \
    --num_envs 4096 \
    --headless \
    --resume \
    --checkpoint logs/rsl_rl/g1_locomotion_flat/YYYY-MM-DD_HH-MM-SS/model_150.pt \
    --max_iterations 3000
```

> **Note:** `--resume` loads the weights from the checkpoint but starts a **new** timestamped run folder. Iteration numbering restarts from 0 and runs up to `--max_iterations`. The weights are warm so training picks up where it left off.

---

## Play (with Isaac Sim visualisation window)

Omit `--headless` to open the GUI. The Play env variant uses fewer envs and disables noise/randomisation.

```bash
# Auto-pick the latest checkpoint
python scripts/rsl_rl/play.py --task G1-Locomotion-Flat-Play-v0 --num_envs 16

# Point at a specific checkpoint file
python scripts/rsl_rl/play.py \
    --task G1-Locomotion-Flat-Play-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/g1_locomotion_flat/2026-06-02_14-22-33/model_150.pt

# Load a specific run folder (auto-picks latest .pt inside it)
python scripts/rsl_rl/play.py \
    --task G1-Locomotion-Flat-Play-v0 \
    --num_envs 16 \
    --load_run 2026-06-02_14-22-33

# Use NVIDIA's published pre-trained checkpoint (downloads from Nucleus on first run)
python scripts/rsl_rl/play.py \
    --task G1-Locomotion-Flat-Play-v0 \
    --num_envs 16 \
    --use_pretrained_checkpoint

# Play the standing policy
python scripts/rsl_rl/play.py \
    --task G1-Locomotion-Standing-Flat-Play-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt
```

---

## Checkpoint location

Logs are written relative to wherever you run the script from. Always `cd` to the project root first so they land here:

```
logs/
└── rsl_rl/
    ├── g1_locomotion_flat/
    │   └── YYYY-MM-DD_HH-MM-SS/
    │       ├── model_0.pt
    │       ├── model_50.pt
    │       ├── model_100.pt
    │       ├── ...
    │       ├── model_<final>.pt
    │       └── params/
    │           ├── env.yaml
    │           └── agent.yaml
    └── g1_locomotion_rough/
        └── ...
```

> **Note:** The first training run (before the `experiment_name` cfg fix) used the folder name `g1_flat` instead of `g1_locomotion_flat`. That run's checkpoints are at `logs/rsl_rl/g1_flat/2026-06-02_14-22-33/`.

---

## List registered environments

```bash
python scripts/list_envs.py
```

---

## Sanity-check with dummy agents (no training)

```bash
# Zero-action agent — verifies env loads without error
python scripts/zero_agent.py --task G1-Locomotion-Flat-v0 --num_envs 4 --headless

# Random-action agent
python scripts/random_agent.py --task G1-Locomotion-Flat-v0 --num_envs 4 --headless
```
