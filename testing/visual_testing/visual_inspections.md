# Visual inspections

Commands for watching policy behavior directly (non-headless — omit `--headless` and
the Isaac Sim viewer opens) rather than reading metrics alone. Useful for understanding
failure modes metrics can't fully explain (see `policy_status.md`'s walking-drift
section for why this mattered — the CSV has no time axis, only final per-episode
values), and for recording videos to show others. Add a new bullet whenever a new visual
check is worth keeping around, with a one-line description of what it shows — don't
duplicate a check that's already here.

`conda activate isaac_g1_control && cd ~/Elm/Code/g1_locomotion` before any of these.

## Walking

- **Fixed-speed straight-line walk, small env count**:
  ```
  python validation/eval_walking.py --checkpoint <ckpt> --arm_disturbance --buckets forward_fast --skip_phases --skip_displacement --num_envs 4
  ```
  Watch a pinned fixed-speed rollout — used to visually characterize heading drift
  (swap `forward_fast` for `forward_slow`/`forward_medium`/`backward` etc. for other
  speeds/directions; see `_BUCKETS` in `eval_walking.py` for the full list).

- **Standing under arm disturbance, one phase pinned**:
  ```
  python validation/eval_walking.py --checkpoint <ckpt> --arm_disturbance --buckets stand_still --phases 3 --skip_displacement --num_envs 4
  ```
  Watch balance response with a specific disturbance phase forced from the very first
  step (`--phases 3` = max). **Correction (2026-07-27): the unpinned/"natural cycling"
  version of this check (no `--phases`) does NOT show real disturbance in a short eval
  session** — `ArmMotionDisturbance` samples each env's phase once per episode at
  reset, weighted by elapsed `common_step_counter`; a fresh eval session starts at 0, so
  the very first (and often only) reset always draws phase 0. This looked like "no arm
  movement" visually and was confirmed with real data (see `policy_status.md`) —
  training itself is fine (thousands of resets over a real run reach the higher
  phases), but a short eval/visual check needs `--phases` pinned to see it for real.

- **Stand<->walk transition**:
  ```
  python testing/visual_testing/walk/visualize_stand_walk_transition.py --checkpoint <ckpt> --arm_disturbance --num_envs 4
  ```
  Live-switches the commanded velocity from stand-still to walking and back on the same
  continuous rollout, so the transition moment itself can be watched — not two separate
  clips you have to mentally stitch together.

## Arms

- (nothing catalogued here yet — `testing/visual_testing/arms/g1_arm_reach_test.py`
  exists for interactive arm-target visualization; add the exact command + description
  here once it's actually run this session, same as the walking entries above)
