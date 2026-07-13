# Testing Reference

How to run the interactive test scripts in `testing/walking_testing/`, `testing/arm_testing/`, and `testing/general_testing/`.

All commands run from `~/Elm/Code/g1_locomotion` with `conda activate isaac_g1_control`.

---

## Finding your checkpoint

Training runs land in `logs/rsl_rl/` under a timestamped folder:

```
logs/rsl_rl/
  arms/
    g1_arm_ik_left/    2026-06-09_12-00-00/  model_100.pt  model_200.pt … model_5000.pt
    g1_arm_ik_right/   2026-06-09_12-30-00/  ...
    g1_arm_ik_both/    2026-06-09_13-00-00/  ...
  legs/
    g1_locomotion_flat/ 2026-06-03_10-20-23/  model_150.pt … model_3149.pt
    g1_locomotion_rough/ ...
    standing/
        g1_locomotion_flat/ 2026-06-16_10-00-00/ model_0.pt ... model_2500.pt
```

Pick the latest `model_N.pt` in the run you want (higher N = more training):

```bash
# List all checkpoints in a run, newest last
ls -ltr logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_*.pt

# One-liner to grab the latest model in the most recent left-arm run
CKPT=$(ls -t logs/rsl_rl/arms/g1_arm_ik_left/*/model_*.pt | head -1)
echo $CKPT
```

---

## testing/walking_testing/g1_locomotion_demo.py

Interactive flat-terrain locomotion with keyboard control. One G1 robot, flat ground.

### Pre-trained NVIDIA checkpoint (auto-downloaded on first run)

```bash
python testing/walking_testing/g1_locomotion_demo.py
```

### Locally trained checkpoint

```bash
python testing/walking_testing/g1_locomotion_demo.py \
    --checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_1500.pt
```

Replace `<run>` with the timestamped folder, e.g. `2026-06-03_10-20-23`.

### Keyboard controls

| Key | Action |
|-----|--------|
| W | Walk forward |
| S | Stop |
| A | Turn left |
| D | Turn right |
| Q | Strafe left |
| E | Strafe right |
| C | Toggle third-person / free camera |

---

## testing/walking_testing/g1_rough_terrain.py

Same as above but on rough terrain, using the pre-trained NVIDIA rough-terrain checkpoint.
No `--checkpoint` argument — always uses the pre-trained model.

```bash
python testing/walking_testing/g1_rough_terrain.py
```

Keyboard controls are identical to the flat-terrain demo.

---

## testing/walking_testing/g1_standing_demo.py

Interactive demo for the local standing-only policy (`standing`).
By default, this script auto-picks the latest checkpoint from:
`logs/rsl_rl/standing/g1_locomotion_flat/*/model_*.pt`

```bash
python testing/walking_testing/g1_standing_demo.py
```

Or specify one explicitly:

```bash
python testing/walking_testing/g1_standing_demo.py \
    --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt
```

Standing training runs also write `standing_summary.csv` and `standing_detailed.csv` in the same
run folder so you can inspect fall rate, stepping behavior, peak tilt, minimum base height, and
similar episode-level metrics after training — see `../logging_reference.md` for the full column
reference and how to read them.

---

## testing/arm_testing/g1_arm_mirror_test.py

Run a single-arm trained policy on **either arm** without retraining, using a bilateral-symmetry
y-mirror transform. Train once (left or right), deploy on both sides.

When `--trained_arm != --active_arm` the script automatically:
- Negates y-related observation components (ee_pos y, goal y, error y, shoulder/elbow roll & yaw)
- Negates the matching components of the output action before applying it to the robot

> **Accuracy note:** The transform is an approximation. Expect slightly lower precision than a
> natively trained policy, but the arm should reach the correct region.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--trained_arm` | no | `left` | Which arm the checkpoint was trained on |
| `--active_arm` | no | `right` | Which arm to physically move |
| `--checkpoint` | **yes** | — | Path to a single-arm `.pt` file |
| `--targets` | **yes** | — | Flat `x y z` triplets in the **active arm's** frame |
| `--hold_steps` | no | `150` | Steps before prompting for next target |

Target y-coordinate sign:
- `--active_arm left` → y positive (0.08 – 0.40 m)
- `--active_arm right` → y negative (-0.40 – -0.08 m)

### Run left-trained policy on the right arm (mirrored)

```bash
python testing/arm_testing/g1_arm_mirror_test.py \
    --trained_arm left \
    --active_arm right \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \
    --targets 0.3 -0.2 1.0
```

### Run left-trained policy on the left arm (no mirror, sanity check)

```bash
python testing/arm_testing/g1_arm_mirror_test.py \
    --trained_arm left \
    --active_arm left \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0
```

### Run right-trained policy on the left arm (mirrored)

```bash
python testing/arm_testing/g1_arm_mirror_test.py \
    --trained_arm right \
    --active_arm left \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_right/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0
```

### Interactive use

Same prompt behaviour as `g1_arm_reach_test.py`:
- The terminal shows a live overwriting distance line while running.
- When a target is reached or times out, a prompt appears:
  ```
  Next target (x y z)  [Enter = cycle to target 1/1]:
  ```
  Type `0.4 -0.3 1.1` and press Enter to move to a new position,
  or press Enter alone to cycle through the original `--targets` list.

---

Loads a trained arm policy and commands specific (x, y, z) targets. Red spheres appear at each
goal. The terminal prints distance-to-goal; targets cycle automatically when reached (< 2 cm) or
after `--hold_steps` steps.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--arm` | no | `left` | Which arm(s): `left`, `right`, or `both` |
| `--checkpoint` | **yes** | — | Path to a `.pt` checkpoint file |
| `--targets` | **yes** | — | Flat list of `x y z` triplets (robot-local frame) |
| `--hold_steps` | no | `150` | Steps before auto-advancing to next target (~5 s at 30 Hz) |

### Left arm — single target

```bash
python testing/arm_testing/g1_arm_reach_test.py \
    --arm left \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0
```

### Left arm — cycle through multiple targets

```bash
python testing/arm_testing/g1_arm_reach_test.py \
    --arm left \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0  0.4 0.3 1.1  0.2 0.15 0.95
```

### Right arm

```bash
python testing/arm_testing/g1_arm_reach_test.py \
    --arm right \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_right/<run>/model_5000.pt \
    --targets 0.3 -0.2 1.0  0.4 -0.3 1.1
```

Note: right-arm y-coordinates are **negative** (right side of robot).

### Both arms simultaneously

Supply one triplet per target. The y-coordinate is applied to the **left** arm and
**auto-mirrored** (negated) for the right arm — so `0.3 0.2 1.0` means
left=(0.3, 0.2, 1.0) and right=(0.3, −0.2, 1.0).

```bash
python testing/arm_testing/g1_arm_reach_test.py \
    --arm both \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0  0.4 0.3 1.1
```

### Quick-fill with latest checkpoint

```bash
# Grab latest both-arm checkpoint automatically
CKPT=$(ls -t logs/rsl_rl/arms/g1_arm_ik_both/*/model_*.pt | head -1)
python testing/arm_testing/g1_arm_reach_test.py \
    --arm both \
    --checkpoint "$CKPT" \
    --targets 0.3 0.2 1.0
```

### Target coordinate reference (robot-local frame, robot base at origin)

| Axis | Direction | Trained range |
|------|-----------|---------------|
| x | forward | 0.20 – 0.42 m |
| y | lateral — left arm: positive, right arm: negative | ±(0.08 – 0.40) m |
| z | height above ground | 0.9 – 1.15 m |

> Targets outside the trained range will still be attempted but expect degraded accuracy.

---

## testing/general_testing/g1_full_demo.py

Integrated demo that combines:

- standing policy for balance at zero velocity
- walking policy for keyboard-commanded locomotion
- arm IK policy that activates only when an arm target is set

The demo starts in standing mode. When you command walking, locomotion takes over the whole body.
When you stop and the robot settles, the standing policy resumes and the arm policy can continue
tracking the active target.

### Checkpoint resolution order

Each checkpoint path is chosen in this order:

1. explicit CLI override
2. `testing/general_testing/checkpoints.yaml`
3. hardcoded fallback inside the script

Arm mode is chosen in this order:

1. `--arm`
2. `arm_mode` in `testing/general_testing/checkpoints.yaml`
3. default `left`

### Script-specific arguments

| Argument | Required | Default | Description |
| --- | --- | --- | --- |
| `--standing_checkpoint` | no | from YAML or fallback | Standing policy checkpoint |
| `--walking_checkpoint` | no | from YAML or fallback | Walking policy checkpoint |
| `--arm_checkpoint` | no | from YAML or fallback | Arm IK checkpoint matching the active arm mode |
| `--arm` | no | `left` if not set in YAML | Active arm mode: `left`, `right`, or `both` |
| `--target X Y Z` | no | no initial target | Initial robot-local arm target |

### Common inherited launch arguments

The script also accepts standard RSL-RL / Isaac Lab launcher flags. The ones most likely to be
useful at the terminal are:

| Argument | Description |
| --- | --- |
| `--headless` | Run without opening the simulator window |
| `--device DEVICE` | Choose device, e.g. `cpu`, `cuda`, `cuda:0` |
| `--enable_cameras` | Enable camera sensors and related extensions |
| `--livestream {0,1,2}` | Force livestream mode |
| `--verbose` | More simulation log output |
| `--info` | Info-level simulation log output |

It also exposes generic RSL-RL flags such as `--experiment_name`, `--run_name`, `--resume`,
`--load_run`, `--checkpoint`, `--logger`, and `--log_project_name`, although those are usually
not important for normal demo usage.

### Run with defaults from YAML

```bash
python testing/general_testing/g1_full_demo.py
```

With the current `testing/general_testing/checkpoints.yaml`, this means:

- walking checkpoint from `chosen_checkpoints/walking_latest.pt`
- standing checkpoint from `chosen_checkpoints/standing_latest.pt`
- arm mode from `arm_mode: left`
- arm checkpoint from `chosen_checkpoints/arm_left_latest.pt`

### Override checkpoints explicitly

```bash
python testing/general_testing/g1_full_demo.py \
    --standing_checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt \
    --walking_checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_3149.pt \
    --arm_checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_4200.pt \
    --arm left
```

### Start with an initial arm target

```bash
python testing/general_testing/g1_full_demo.py \
    --arm left \
    --target 0.3 0.2 1.0
```

### Both-arm mode

```bash
python testing/general_testing/g1_full_demo.py \
    --arm both \
    --arm_checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_1000.pt \
    --target 0.3 0.2 1.0
```

In `--arm both` mode, the initial target is interpreted as the left-arm target and mirrored to the
right arm by negating `y`. So `0.3 0.2 1.0` becomes:

- left target: `(0.3, 0.2, 1.0)`
- right target: `(0.3, -0.2, 1.0)`

### Keyboard / prompt controls

| Key | Action |
| --- | --- |
| `W` | Walk forward |
| `A` | Turn left |
| `D` | Turn right |
| `Q` | Strafe left |
| `E` | Strafe right |
| `S` | Stop walking |
| `T` | Open a terminal prompt for a new arm target |
| `L` | Select left arm as the active prompt target in `--arm both` mode |
| `R` | Select right arm as the active prompt target in `--arm both` mode |
| `C` | Toggle camera follow (off lets you orbit freely with the mouse) |
| `V` | Reset camera to the default chase view (re-enables follow) |

Pressing `T` blocks the simulation briefly and opens a console prompt for a new target in robot-local
coordinates.

### Target coordinate reference

| Axis | Direction | Typical trained range |
| --- | --- | --- |
| x | forward from robot base | 0.20 - 0.42 m |
| y | left positive, right negative | 0.08 - 0.40 m on left, -0.40 - -0.08 m on right |
| z | height above ground | 0.9 - 1.15 m |

### Useful help command

```bash
python testing/general_testing/g1_full_demo.py --help
```

---

## TensorBoard

Monitor training for both arms and legs in one view:

```bash
tensorboard --logdir logs/rsl_rl/
# open http://localhost:6006
```

Filter by prefix in the TensorBoard UI: `arms/` for arm runs, `legs/` for locomotion runs.
