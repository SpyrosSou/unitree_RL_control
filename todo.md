# G1 Locomotion — Current Training TODO

> All commands assume:
> ```bash
> conda activate isaac_g1_control
> cd ~/Elm/Code/g1_locomotion
> ```

---

## 1) Standing Policy (with arm disturbances + minimal corrective stepping) — `G1-Locomotion-Standing-Flat-v0`

This task now includes arm-motion disturbances by default during training.
It also includes a small near-zero command slice so the policy can learn
tiny corrective steps instead of only in-place balancing.
You do not need extra flags to enable this.

### Train (new run)

```bash
python scripts/rsl_rl/train.py \
  --task G1-Locomotion-Standing-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 2500
```

### Resume

Option A (resume from latest checkpoint in a specific run):

```bash
python scripts/rsl_rl/train.py \
  --task G1-Locomotion-Standing-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run <YYYY-MM-DD_HH-MM-SS> \
  --max_iterations 2500
```

Option B (resume from an explicit checkpoint path):

```bash
python scripts/rsl_rl/train.py \
  --task G1-Locomotion-Standing-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_<k>.pt \
  --max_iterations 2500
```

### Play / visualize output

```bash
python scripts/rsl_rl/play.py \
  --task G1-Locomotion-Standing-Flat-Play-v0 \
  --num_envs 4 \
  --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_2500.pt
```

Notes:

- Standing play introduces arm disturbances early for visualization.
- For this updated regime, validate at checkpoints around `model_1200`, `model_1800`, and `model_2500`.
- Keep the run that has the lowest fall rate during high-intensity arm phases.
- Each standing training run now writes `standing_metrics.csv` alongside the checkpoints.
- Logs/checkpoints: `logs/rsl_rl/standing/g1_locomotion_flat/<run>/...`

---

## 2) Walking Policy (single policy) — `G1-Locomotion-Flat-v0`

This task now includes:

- faster command resampling,
- reverse/lateral/yaw exposure,
- a partial near-zero/zero command slice.

### Train (new run)

```bash
python scripts/rsl_rl/train.py \
  --task G1-Locomotion-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 2500
```

### Resume

Option A:

```bash
python scripts/rsl_rl/train.py \
  --task G1-Locomotion-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run <YYYY-MM-DD_HH-MM-SS> \
  --max_iterations 2500
```

Option B:

```bash
python scripts/rsl_rl/train.py \
  --task G1-Locomotion-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_<k>.pt \
  --max_iterations 2500
```

### Play / visualize output

```bash
python scripts/rsl_rl/play.py \
  --task G1-Locomotion-Flat-Play-v0 \
  --num_envs 4 \
  --checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_2500.pt
```

Logs/checkpoints: `logs/rsl_rl/legs/g1_locomotion_flat/<run>/...`

---

## 3) Arm Policy (with action filtering + delta cap)

Arm training now includes:

- first-order action filtering (`action_filter_alpha=0.25`),
- per-step action delta cap (`max_action_delta_per_step=0.06 rad`).

These reduce abrupt arm motion.

### Train left arm (recommended first)

```bash
python scripts/rsl_rl/train.py \
  --task G1-Arm-IK-Left-v0 \
  --num_envs 4096 \
  --headless \
  --max_iterations 5000
```

### Resume left arm

Option A:

```bash
python scripts/rsl_rl/train.py \
  --task G1-Arm-IK-Left-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --load_run <YYYY-MM-DD_HH-MM-SS> \
  --max_iterations 5000
```

Option B:

```bash
python scripts/rsl_rl/train.py \
  --task G1-Arm-IK-Left-v0 \
  --num_envs 4096 \
  --headless \
  --resume \
  --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_<k>.pt \
  --max_iterations 5000
```

### Play / visualize left arm output

```bash
python scripts/rsl_rl/play.py \
  --task G1-Arm-IK-Left-Play-v0 \
  --num_envs 4 \
  --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt
```

Optional both-arm training/play:

```bash
python scripts/rsl_rl/train.py --task G1-Arm-IK-Both-v0 --num_envs 4096 --headless --max_iterations 5000
python scripts/rsl_rl/play.py  --task G1-Arm-IK-Both-Play-v0 --num_envs 4 --checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_5000.pt
```

Logs/checkpoints:

- Left: `logs/rsl_rl/arms/g1_arm_ik_left/<run>/...`
- Right: `logs/rsl_rl/arms/g1_arm_ik_right/<run>/...`
- Both: `logs/rsl_rl/arms/g1_arm_ik_both/<run>/...`

---

## 4) Integration After Training

Run integrated test with new checkpoints:

```bash
python general_testing/g1_full_demo.py \
  --standing_checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<stand_run>/model_<s>.pt \
  --walking_checkpoint  logs/rsl_rl/legs/g1_locomotion_flat/<walk_run>/model_<w>.pt \
  --arm_checkpoint      logs/rsl_rl/arms/g1_arm_ik_left/<arm_run>/model_<a>.pt \
  --arm left
```

---

## 5) Useful Helpers

Find latest run folders:

```bash
ls -1t logs/rsl_rl/standing/g1_locomotion_flat | head
ls -1t logs/rsl_rl/legs/g1_locomotion_flat | head
ls -1t logs/rsl_rl/arms/g1_arm_ik_left | head
```

TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl
# open http://localhost:6006
```
