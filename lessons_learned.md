# Lessons learned (23dof phase) — carry these into the 29dof rebuild

Extracted 2026-07-21 before removing `definitive_next_steps.md`/`known_issues.md` from
`main` (full detail on all of this survives on the `23_dof` branch and in git history —
this is just the subset that isn't specific to the old 1-DOF-waist/no-wrist robot model
and will likely resurface on the 29dof build if forgotten).

1. **PhysX/`find_joints()` orders joints breadth-first from the articulation root, NOT
   in URDF/declaration order.** On the G1 this interleaves left/right
   (`left_shoulder_pitch, right_shoulder_pitch, left_shoulder_roll, ...`). Any code that
   assumes a left-block-then-right-block layout from a hand-written joint name list will
   silently scramble commands between sides. Always derive column/index mappings from
   the actual returned joint ids, never from the order joints were listed in source.
   This exact bug cost about a week on the 23dof arm-reach disturbance.

2. **An unpriced "free" DOF gets exploited, and capping it just moves the exploit to the
   next-cheapest one.** Relaxing a joint-deviation reward to 0 "to let it help with
   balance" is a real trap — it's fine while the disturbance it's absorbing is mild, and
   silently becomes a problem once the disturbance escalates, with no warning until
   someone actually looks. On the 23dof standing task this went torso (60-120°, capped)
   → hip (wide stance) → ankle (driven to the hard mechanical limit) across three
   successive fixes. Price in deviation/posture costs for every joint from day one, not
   reactively per-joint after each one gets discovered. Consider a hard orientation
   termination (not just a soft reward) from the start — the soft-only version is what
   let each of these slip through.

3. **Train against the exact deployment-time actuator gains, not a softer sim-only
   value.** A checkpoint trained with soft gains and deployed against stiff ones (or vice
   versa) can look perfect in its own training distribution and still collapse
   (99-100% falls) the first time it meets the real gain. If sim training uses a
   different gain than deployment for any actuator, that's a real train/deploy mismatch,
   not a minor detail — match it explicitly or curriculum toward it deliberately.

4. **A shared deployment env built from a different lineage than the one being tested
   does not automatically inherit that lineage's config changes.** Twice this phase, an
   eval/demo script that builds its environment from a "sibling" base config (e.g. the
   walking task's env, reused for a standing checkpoint's deployment test) silently
   dropped an action-space change (a clip) that only existed on the tested lineage.
   Symptom was catastrophic-looking numbers that were actually an eval bug, not a real
   regression. Any config change that isn't purely reward/observation-based needs an
   explicit check: does every script that builds this environment from a different base
   actually carry the change over, or does it need to be threaded through manually?

5. **PPO seed variance on this task is large enough that a single run proves nothing.**
   An identical config produced 2.1% vs 100% fall rate on two different seeds once.
   Never trust a single training run's numbers as final; a second seed is the minimum
   bar before treating a result as real.

6. **A wider network measurably hurt mirror-quality on the un-trained (mirrored) side of
   a symmetric task**, even though it looked like a reasonable capacity increase. If the
   29dof arm rebuild still uses a mirror-based approach for the second arm rather than
   training both natively, re-verify this rather than assuming a bigger network is safe.
