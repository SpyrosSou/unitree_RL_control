# Arm Eval

Checkpoint: `chosen_checkpoints/arm_left_latest.pt`

| Bucket | Episodes | Success rate | Mean reward | Mean dist (cm) | Median dist (cm) | p90 dist (cm) |
|---|---|---|---|---|---|---|
| no_wobble | 597 | 24.29% | -0.687 | 8.02 | 7.31 | 16.57 |
| with_wobble | 594 | 24.24% | -0.691 | 8.35 | 7.56 | 16.94 |

## Joint range utilization

How much of the real hardware range each joint actually used during this eval —
answers "did it learn genuine reaching motion, or just tiny movements near the
default pose that happen to land on nearby goals". Low % isn't automatically bad
(a joint might genuinely not need much range for this goal workspace), but a
joint sitting at ~0% while others are well-exercised is worth a second look.

| Bucket | Joint | HW range (deg) | Achieved range (deg) | % of HW range used |
|---|---|---|---|---|
| no_wobble | left_shoulder_pitch_joint | -160.5, 136.5 | -56.5, 32.7 | 30.0% |
| no_wobble | left_shoulder_roll_joint | -80.0, 118.0 | -11.7, 54.0 | 33.2% |
| no_wobble | left_shoulder_yaw_joint | -135.0, 135.0 | -62.2, 68.2 | 48.3% |
| no_wobble | left_elbow_joint | -51.0, 111.0 | -49.0, 59.9 | 67.2% |
| no_wobble | left_wrist_roll_joint | -101.7, 101.7 | -48.6, 59.8 | 53.3% |
| no_wobble | left_wrist_pitch_joint | -83.2, 83.2 | -83.3, 27.5 | 66.5% |
| no_wobble | left_wrist_yaw_joint | -83.2, 83.2 | -66.9, 57.9 | 75.0% |
| with_wobble | left_shoulder_pitch_joint | -160.5, 136.5 | -56.4, 35.4 | 30.9% |
| with_wobble | left_shoulder_roll_joint | -80.0, 118.0 | -11.7, 57.5 | 35.0% |
| with_wobble | left_shoulder_yaw_joint | -135.0, 135.0 | -124.9, 71.1 | 72.6% |
| with_wobble | left_elbow_joint | -51.0, 111.0 | -46.5, 75.1 | 75.1% |
| with_wobble | left_wrist_roll_joint | -101.7, 101.7 | -70.8, 57.1 | 62.9% |
| with_wobble | left_wrist_pitch_joint | -83.2, 83.2 | -92.5, 24.8 | 70.5% |
| with_wobble | left_wrist_yaw_joint | -83.2, 83.2 | -71.4, 92.5 | 98.4% |

## Reward component breakdown

Per-step mean of each reward term (not per-episode sum) — added 2026-07-26 since this task previously had zero per-term visibility (unlike the walking task's automatic RewardManager logging), making it impossible to check e.g. whether joint_limit or torso_proximity penalties were disproportionately large without guessing. Works retroactively on any checkpoint, not just new training runs.

| Bucket | Component | Mean value |
|---|---|---|
| no_wobble | Curriculum/goal_bounds_frac | +1.0000 |
| no_wobble | Episode_Reward/action_smoothness | -0.0144 |
| no_wobble | Episode_Reward/goal_reached_bonus | +0.3417 |
| no_wobble | Episode_Reward/joint_limit | -0.0046 |
| no_wobble | Episode_Reward/null_space | -0.0701 |
| no_wobble | Episode_Reward/position_dist | -1.6224 |
| no_wobble | Episode_Reward/position_exp_bonus | +0.0000 |
| no_wobble | Episode_Reward/settle | -0.0009 |
| no_wobble | Episode_Reward/torso_proximity | -0.0000 |
| no_wobble | Metrics/frac_envs_reached | +0.0068 |
| no_wobble | Metrics/mean_dist_to_goal_cm | +16.2240 |
| no_wobble | Metrics/mean_joints_at_limit | +0.0046 |
| with_wobble | Curriculum/goal_bounds_frac | +1.0000 |
| with_wobble | Episode_Reward/action_smoothness | -0.0146 |
| with_wobble | Episode_Reward/goal_reached_bonus | +0.3482 |
| with_wobble | Episode_Reward/joint_limit | -0.0033 |
| with_wobble | Episode_Reward/null_space | -0.0700 |
| with_wobble | Episode_Reward/position_dist | -1.6497 |
| with_wobble | Episode_Reward/position_exp_bonus | +0.0000 |
| with_wobble | Episode_Reward/settle | -0.0011 |
| with_wobble | Episode_Reward/torso_proximity | -0.0002 |
| with_wobble | Metrics/frac_envs_reached | +0.0070 |
| with_wobble | Metrics/mean_dist_to_goal_cm | +16.4967 |
| with_wobble | Metrics/mean_joints_at_limit | +0.0033 |
