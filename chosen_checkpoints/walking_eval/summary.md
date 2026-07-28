# G1 29dof Walking Eval

Checkpoint: `chosen_checkpoints/walking_latest.pt`
Arm disturbance forced on: True

## 1. Command-bucket sweep

| Bucket | vx, vy, wz | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) | Foot slip (m/s) | Mean\|heading drift\| (deg) | Mean\|lateral drift\| (m) | Mean step count | Mean max foot air time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| stand_still | 0.0, 0.0, 0.0 | 256 | 0.00% | 0.131 | 0.111 | 0.019 | 45.21 | 0.771 | 42.95 | 0.458 |

## 2. Arm-disturbance phase sweep (standing still, fall rate per phase)

| Phase | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) | Foot slip (m/s) |
|---|---|---|---|---|---|
| 0 | 256 | 0.00% | 0.092 | 0.101 | 0.023 |
| 1 | 256 | 0.00% | 0.130 | 0.111 | 0.019 |
| 2 | 256 | 0.00% | 0.129 | 0.111 | 0.019 |
| 3 | 256 | 0.00% | 0.110 | 0.105 | 0.021 |

## 3. Raw world-frame displacement check

Reads `root_pos_w` directly — independent of section 1's reward-derived tracking error, guarding against misreading a tracking metric as achieved velocity (a real past bug in this project, 2026-07-23/24).

Skipped (`--skip_displacement` passed).

## 4. Knee-angle tracking

Checks a visually-observed hypothesis (round-2 drift checkpoint, 2026-07-27) that near-extended knees during straight-line walking correlate with the asymmetric-step corrections that precede bad heading drift — see `deferred_items_2026-07-21.md` item 8. Knee angle: 0 rad = fully extended, default pose is 0.3 rad (~17 deg), larger = more bent. `min_knee_angle_deg` is the most-extended moment reached per episode. Correlation is Pearson's r between each episode's `min_knee_angle_deg` and its `|heading_drift_deg|` — negative means episodes that stay more extended tend to drift more (supports the hypothesis); near zero means no relationship found in this data.

| Bucket | Mean knee angle (deg) | Mean most-extended angle (deg) | Corr. vs \|heading drift\| |
|---|---|---|---|
| stand_still | 20.10 | 3.01 | — |
