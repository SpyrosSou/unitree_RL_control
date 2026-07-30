# G1 29dof Walking Eval

Checkpoint: `chosen_checkpoints/walking_latest.pt`
Arm disturbance forced on: True

## 1. Command-bucket sweep

| Bucket | vx, vy, wz | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) | Foot slip (m/s) | Mean\|heading drift\| (deg) | Mean\|lateral drift\| (m) | Mean step count | Mean max foot air time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| stand_still | 0.0, 0.0, 0.0 | 256 | 0.00% | 0.049 | 0.076 | 0.011 | 9.38 | 0.076 | 8.23 | 0.433 |
| forward_slow | 0.3, 0.0, 0.0 | 256 | 0.00% | 0.112 | 0.114 | 0.031 | 62.13 | 0.346 | 40.38 | 0.511 |
| forward_medium | 0.6, 0.0, 0.0 | 256 | 0.00% | 0.111 | 0.122 | 0.059 | 52.36 | 0.372 | 42.02 | 0.477 |
| forward_fast | 0.9, 0.0, 0.0 | 256 | 0.00% | 0.130 | 0.140 | 0.091 | 56.88 | 0.625 | 46.16 | 0.425 |
| backward | -0.3, 0.0, 0.0 | 256 | 0.00% | 0.118 | 0.132 | 0.036 | 63.75 | 0.715 | 51.70 | 0.428 |
| strafe_left | 0.0, 0.3, 0.0 | 256 | 0.00% | 0.119 | 0.128 | 0.028 | — | — | 45.11 | 0.434 |
| strafe_right | 0.0, -0.3, 0.0 | 256 | 0.00% | 0.144 | 0.127 | 0.026 | — | — | 43.45 | 0.484 |
| turn_left | 0.0, 0.0, 0.2 | 256 | 0.00% | 0.067 | 0.158 | 0.030 | — | — | 15.47 | 0.452 |
| turn_right | 0.0, 0.0, -0.2 | 256 | 0.00% | 0.080 | 0.139 | 0.034 | — | — | 20.22 | 0.453 |
| forward_turn_combo | 0.5, 0.0, 0.2 | 256 | 0.00% | 0.106 | 0.117 | 0.049 | — | — | 42.38 | 0.487 |

## 2. Arm-disturbance phase sweep (standing still, fall rate per phase)

| Phase | Episodes | Fall rate | Lin. track err (m/s) | Ang. track err (rad/s) | Foot slip (m/s) |
|---|---|---|---|---|---|
| 0 | 257 | 0.78% | 0.062 | 0.090 | 0.019 |
| 1 | 256 | 0.00% | 0.050 | 0.076 | 0.011 |
| 2 | 257 | 0.39% | 0.053 | 0.081 | 0.014 |
| 3 | 256 | 0.00% | 0.054 | 0.082 | 0.015 |

## 3. Raw world-frame displacement check

Reads `root_pos_w` directly — independent of section 1's reward-derived tracking error, guarding against misreading a tracking metric as achieved velocity (a real past bug in this project, 2026-07-23/24).

- Commanded 0.6 m/s for 500 steps (10.0s @ 50Hz)
- Expected displacement if tracking perfectly: 6.00m
- Actual mean displacement across 256 envs: 5.569m (92.8% of expected)
- Range: 4.758m - 6.330m

## 4. Knee-angle tracking

Checks a visually-observed hypothesis (round-2 drift checkpoint, 2026-07-27) that near-extended knees during straight-line walking correlate with the asymmetric-step corrections that precede bad heading drift — see `deferred_items_2026-07-21.md` item 8. Knee angle: 0 rad = fully extended, default pose is 0.3 rad (~17 deg), larger = more bent. `min_knee_angle_deg` is the most-extended moment reached per episode. Correlation is Pearson's r between each episode's `min_knee_angle_deg` and its `|heading_drift_deg|` — negative means episodes that stay more extended tend to drift more (supports the hypothesis); near zero means no relationship found in this data.

| Bucket | Mean knee angle (deg) | Mean most-extended angle (deg) | Corr. vs \|heading drift\| |
|---|---|---|---|
| stand_still | 12.90 | 2.56 | — |
| forward_slow | 24.62 | 0.42 | 0.09 |
| forward_medium | 29.36 | 0.61 | -0.01 |
| forward_fast | 32.57 | 0.55 | 0.17 |
| backward | 20.28 | 2.64 | -0.15 |
| strafe_left | 19.91 | -0.79 | — |
| strafe_right | 22.80 | 1.52 | — |
| turn_left | 17.24 | 2.55 | — |
| turn_right | 13.82 | -0.89 | — |
| forward_turn_combo | 28.27 | 1.50 | — |

## 5. Fall timing breakdown (buckets with any falls)

A raw fall-rate percentage treats a fall at 3s and a fall at 18s as equally bad, but a command held for the full episode is an eval artifact, not something a real user is likely to sustain (e.g. turn_left/turn_right command wz continuously for the whole 20s — see run notes on the dead ang_vel curriculum). This bins WHEN falls happen so 'fails almost immediately' (a real problem) is visible separately from 'only fails after an unrealistically long sustained command' (lower priority).

| Bucket | Total falls | <3s | 3-6s | 6-12s | >12s |
|---|---|---|---|---|---|
| (no falls in any evaluated bucket) | — | —| —| —| —|
