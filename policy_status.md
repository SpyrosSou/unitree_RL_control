# Policy status — 29dof pivot

Living summary of what's actually validated vs. still open, for the checkpoints in
`chosen_checkpoints/`. Update this when a checkpoint gets promoted or a real gap is
found — don't let it go stale the way `known_issues.md` did on the 23dof branch.

## Walking + standing (`chosen_checkpoints/walking_latest.pt`)

**Status: working, ready for initial real-robot testing.** Promoted 2026-07-25 from
`logs/rsl_rl/walking/arm_disturbance/2026-07-24_15-47-18/model_15998.pt` — warm-started
from a validated base-recipe checkpoint (`walking_2026-07-24_base_only_prev.pt`, kept
alongside for reference), then trained 10,000 more iterations on the real deployment
recipe (`G1-Locomotion-Velocity-ArmDisturbance-v0`).

**Confirmed working, not assumed:**
- Real walking — verified via direct `root_pos_w` displacement measurement (not just
  reward/error metrics, after this project's own earlier mistake reading a track-error
  metric as achieved velocity — see git history 2026-07-23/24). Forward tracking error
  0.10-0.13 m/s at 0.3/0.6 m/s commanded, 0% fall rate.
- Standing under arm disturbance — 0% fall rate across all 4 disturbance phases
  (no/mild/moderate/max), verified with a real, working disturbance signal (found and
  fixed a bug 2026-07-24 where the phase-pin eval flag silently had no effect on an
  env's first episode — see `mdp/events.py`'s `ArmMotionDisturbance.__init__`) and
  confirmed the disturbance itself produces real, scaling joint motion (see
  `testing/general_testing/check_arm_disturbance_magnitude.py`).
- Reward reshape that unblocked this (`action_rate` -0.05→-0.005, `joint_deviation_legs`
  -1.0→-0.1) was validated via a 4-way isolated ablation, not a bundled guess — see
  `g1_locomotion_env_cfg.py`'s `RewardsCfg` comments for the full before/after numbers.

**Known gap — no heading/position drift correction.** Over a single 20s straight-line
episode (0.3-0.6 m/s commanded, ~6-12m of intended travel), the robot ends up ~24-27°
off its starting heading and 0.6-0.9m laterally displaced. Root cause confirmed by
reading every term in `RewardsCfg`: `track_lin_vel_xy` rewards velocity in the robot's
*current* (possibly already-drifted) yaw frame, not world frame, and `track_ang_vel_z`
only penalizes instantaneous angular velocity with no memory of accumulated heading
error. Nothing in the reward has ever looked at absolute heading or lateral position —
this isn't a training failure, the policy is doing exactly what it's rewarded for.
**Fix for next training round**: add a reward term (or termination) that penalizes
absolute heading/lateral deviation from the commanded straight-line path, not just
instantaneous command-tracking.

**Known gap — walking precision regressed somewhat after arm-disturbance training.**
The pure base-recipe checkpoint (no disturbance at all,
`walking_2026-07-24_base_only_prev.pt`) tracked tighter (0.065-0.074 m/s error, 10-11°
drift) than the final arm-disturbance-trained one (0.10-0.13 m/s, 24-27° drift). Some of
this may be the curriculum restarting from scratch on the warm-started run (a resume
across different tasks currently can't recover the source run's `lin_vel_cmd_levels`
progress — see `scripts/rsl_rl/train.py`'s own comment on this), not necessarily the
disturbance training itself. Worth a longer/re-tuned run to see if this closes.

**Not yet done:**
- No real-robot test at all — sim-verified only.
- `eval_full_demo.py` (the actual integration eval, loco+arm together) hasn't been
  re-run against this checkpoint.
- The `debug_vis` remote-asset-hang fix (found 2026-07-24) was applied to
  `eval_walking.py`/`check_real_displacement*.py` but not yet to `eval_full_demo.py` —
  check before relying on that script unattended.

## Arm reaching

**Status: not yet working, actively being iterated on.** See conversation history
2026-07-24/25 for the full ablation trail (`entropy_coef` confirmed unstable —
`Policy/mean_noise_std` exploded 1.0→39-57 within 1500-2000 iterations;
`position_reward_exp_scale` inconclusive and carries a real regression precedent from
the 23dof branch's own history, 55%→11-14% success on its first, more aggressive
attempt there). Currently training `G1-Arm-Left-LockedWrist-v0` (wrist_pitch/wrist_yaw
held at default, 5 controlled DOF instead of 7 — untested but mechanistically
motivated: the 23dof branch's 5-DOF arm hit ~55% baseline success *before any tuning*,
well above where this 7-DOF task has plateaued at, ~28%). `chosen_checkpoints/
arm_left_latest.pt` is stale (predates all of today's fixes) — do not treat it as
current.

Reachability of the goal workspace itself was checked and is *not* the bottleneck
(97% coverage within 2cm — see `validation/arm_reachability/left_summary.md`), unlike
the 23dof branch where goal-bounds reshaping was the dominant fix for a similar plateau.

Next update to this section once the locked-wrist run has a real `eval_arm.py` success
rate, not just a training-reward read.
