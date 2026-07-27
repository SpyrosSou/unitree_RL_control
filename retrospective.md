# 29dof pivot — week retrospective (2026-07-21 to 2026-07-27)

What was actually tried this week and what it showed, in one place. This is a summary
for orientation — `policy_status.md` (repo root) is the living, detailed source of
truth this doc is derived from; when the two disagree, trust `policy_status.md`. Some
early-week runs were deleted before this doc existed to reclaim disk space (see
`policy_status.md`'s "Rejected/superseded experiment logs" section) — their
conclusions survive there even though the raw data doesn't; nothing below is invented
to fill that gap.

## Walking + standing

Started the week stuck in a "never falls, never walks" local optimum. A 4-way reward
ablation (`action_rate` and `joint_deviation_legs` weights loosened) broke the plateau
and became the shared base recipe. From there, the rest of the week was standing
verification (arm-disturbance robustness while stationary) and three successive
attempts at fixing a heading/position drift gap that showed up once real walking
speed was achieved — all three made heading drift worse, not better.

| Run | Iters | Result | Outcome |
|---|---|---|---|
| `ablation_reward_weights` (2026-07-24) | 5999 | Fixed the "never walks" plateau | Base recipe, kept |
| **`walking_latest.pt`** (2026-07-24) | 15998 | 0% fall rate, 0.10-0.13 m/s track err, 24-27° heading / 0.6-0.9m lateral drift | **Promoted, current default** |
| Drift round 1, attempt A (2026-07-25) | 8000 | Drift worse (27-63°/1.6m) + broad regression across unrelated reward terms | Reverted |
| Drift round 1, attempt B (2026-07-25/26) | 6000 | Heading drift much worse (63.6-136.4°) | Reverted |
| Drift round 2 (2026-07-26/27) | 8000 | 0% fall rate; heading drift worse and scales with speed (28.3°→94.5°); lateral drift *better* than baseline forward | Not promoted |

**Current status**: `walking_latest.pt` is the working default (sim-verified, no
real-robot test yet). Drift is a known, open gap — not blocking initial use, actively
being investigated (current lead: `base_height`'s reward weight may be over-rigid,
pushing balance correction toward near-locked knees instead of knee flexion — not yet
confirmed with data).

## Arm reaching

The harder problem this week. Every 7-DOF configuration at the real deployment gain
(40/10 stiffness/damping) plateaus around 25-30% training-time success, while a
200/20-gain reference reaches 99.98% — later found to be a misleading number (see
below). Most of the week was isolating what does and doesn't move that 40/10 plateau.

| Variant | Gain | Iters | Result | Verdict |
|---|---|---|---|---|
| 200/20 reference | 200/20 | 2999 | 99.98% training-metric success | Real, but not representative (see below) |
| 40/10 baseline | 40/10 | 5999 | 27.68% | Reference/fallback |
| `joint_vel_noise` 1.5→0.1 | 40/10 | 5000 | 2.31%/3.44% | Harmful — dropped |
| Privileged critic | 40/10 | 2000 | 29.37%/26.83% | Modestly positive |
| `noise_std_type="log"` | 40/10 | 2000 | 26.61%/26.50% | Modestly positive |
| Rate-limit relaxation | 40/10 | 5000 | Looked better early, 21-24% final | Misleading early signal |
| Wrist-locked (5 DOF) | 40/10 | 17000 | 2.66%/0.94% | Falsified the "fewer DOF = easier" theory |
| Gain 60/1.5 | 60/1.5 | — | 24.43%/23.42%, joints at-limit 86-90% | Underdamped, no improvement |
| Gain 15/1 | 15/1 | — | 5.48%/6.23%, flat from iteration 0 | Too soft, never learned |
| Goal-distance curriculum | 40/10 | 2000 (of 10k planned) | 22.70%/25.42%, still climbing | Extended overnight |
| **`best_combined`** (privileged critic + log_std + `action_fb`) | 40/10 | 2000 | 28.20%/32.85% | **Current best 40/10 candidate, not yet promoted** |

**Critical finding**: built a long-hold eval (one fixed goal, one fresh attempt, 45s)
because the 99.98% number didn't match visual inspection. Real single-shot reliability
of the 200/20 reference is **~30%**, not 99.98% — the standard eval's rapid
goal-resampling inflates the headline number. Root mechanism for the gap wasn't found
despite ruling out several candidates (episode-count weighting, wobble config,
domain-randomization variance, policy stochasticity). Also found and fixed a real
structural gap: the policy never observed its own action-filter's internal state
(added as `action_fb`, baked into the observation for every arm task going forward).

**Current status**: no arm checkpoint has cleared a reliability bar worth promoting to
`chosen_checkpoints/`. Given the gap above, the current plan is a pivot toward
numerical IK for precise reach targets, with RL handling compliant continuous-gesture
motion — see `policy_status.md`'s "Strategic pivot" note.

## Integration (walking + arms together)

Visual testing this week (`g1_full_demo.py`) confirmed: arm target position persists
correctly through a walk command (doesn't reset, a deliberate design choice — see
below), no visible torso/arm creep during a normal demo session, and an informal
stress test where the robot walked backward to maintain balance against an unreachable
arm target before eventually falling — an encouraging but *untrained* generalization,
not a validated capability. Real, reproducible right-arm/torso motion during arm-only
testing (~20° torso lean, elbow settling around -53°) was confirmed via direct
per-joint measurement to be genuine physics/reaction coupling, not a bug — flagged as a
future refinement (gravity compensation or a passive-joint reward term), not a
blocker.

## Why one combined stand+walk policy, not two decoupled ones

Early exploration in this repo included a discrete-switching approach: load a
dedicated standing policy and a dedicated walking policy, and switch between them
online based on commanded velocity magnitude, with a hysteresis threshold and a
minimum dwell time to prevent rapid chattering between the two. That approach was set
aside in favor of the current single policy, trained across the full command range
including zero velocity, for a few reasons:

- **No handoff seam.** A mode switch is exactly the moment control authority passes
  between two separately-trained networks that never experienced each other's
  dynamics mid-transition — precisely where a fall is most likely. A single policy has
  no such boundary; standing is just the zero-command point on the same continuous
  manifold it's trained across.
- **Fewer moving parts to tune.** The switch-based approach needs its own hysteresis
  threshold and dwell-time hyperparameters purely to prevent chattering — extra tuning
  surface with no equivalent in the combined approach.
- **Disturbance training composes for free.** The standing arm-disturbance curriculum
  is defined once, against the single policy's full training distribution. A
  two-policy split would need that curriculum trained and validated twice — once per
  policy — doubling the surface for the kind of "unpriced free DOF" exploit noted in
  `policy_status.md`'s lessons-learned section.
- **One eval/deploy surface instead of two.** Every validation script, checkpoint
  promotion, and deploy config only has to reason about one network's behavior across
  the full velocity range, rather than two networks plus a switching controller as a
  third component with its own failure modes.

This isn't free — one network has to represent both regimes, and the drift gap above
(heading drift growing with commanded speed) is one place that cost shows up. But
splitting into two policies wouldn't fix that by itself; it would relocate the risk
from "learn a harder combined manifold" to "handle the handoff between two
independently-optimized networks," which — based on this repo's own testing — is a
harder failure mode to reason about, not an easier one.
