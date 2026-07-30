# Policy overview — reward, curriculum, what's been tried

Quick-reference companion to `policy_status.md` (full evidence/history lives there).
"Chosen" = last night's runs: arm `G1-Arm-Left-Integrated-v0`
(`logs/rsl_rl/arms/integrated/2026-07-29_20-27-48/model_7999.pt`) and walking
`G1-Locomotion-Velocity-ArmDisturbance-StandingPackage-v0`
(`logs/rsl_rl/walking/arm_disturbance/2026-07-29_23-28-41/model_6999.pt`).

## Arm

### 1. Chosen reward — what it optimizes for

Closing distance to the goal (linear term) plus a flat +50/step bonus for being
within the 2cm threshold — the bonus dominates once close, so the policy is mostly
optimizing "get in the zone and stay counted as in it." Smaller shaping terms:
penalize proximity to joint limits and to the torso (self-collision defense),
deviation from the default pose (a tiebreaker among redundant arm configurations),
jerky/large actions, and — once within 5cm of the goal specifically — joint
velocity (encourages braking into the hold rather than flying through it).

### 2. Training curriculum

None on goal difficulty — goals are sampled across the full workspace box from
iteration 0 (the optional easy→hard `goal_curriculum` exists but is off for this
recipe). The only schedule is a single on/off switch: root-pose "wobble" (a
synthetic tilt/motion signal simulating a walking base) is silent for the first
~1,250 iterations, then switches on at full sampled amplitude for the rest of
training. Reset posture is randomized within 15% of the hardware range every
episode, fixed throughout.

### 3. What's been tried

| Tried | Insight |
|---|---|
| Actuator gain sweep (200/20, 40/10, 60/1.5, 15/1) | Softer/real-hardware gains plateau far below the strong 200/20 reference; too-soft gains (15/1) never learn at all — later found to be a torque-authority ceiling, not a tuning problem. |
| Entropy coefficient sweep (0 / 0.003 / 0.005 / 0.01, adaptive vs. fixed LR schedule) | Zero collapses exploration prematurely; any nonzero value explodes noise variance — no stable middle ground under this reward. |
| Reward `position_reward_exp_scale=3.0` in isolation | No improvement — same noise-collapse pattern as baseline, ruling out reward shape as the driver. |
| Locking wrist_pitch/yaw (7→5 controlled joints) | Success rate collapsed further — fewer DOF hurt rather than helped, so manipulator redundancy wasn't the real bottleneck. |
| Relaxing the action rate-limit/filter | Looked better early in training but ended up worse at convergence — a misleading early signal, precision cost right at the goal. |
| Lowering joint-velocity observation noise | Confirmed harmful (an order of magnitude worse) — the noise was acting as useful implicit regularization. |
| Privileged (noise-free) critic; log-std noise parameterization | Both modestly positive alone (+5pts / +2pts) — combined into the "best_combined" recipe. |
| Goal-distance curriculum (easy→hard workspace) | Still climbing at the iteration budget's cutoff — inconclusive, never finished. |
| Observing the action-filter's own internal state (`action_fb`) | Closed a real POMDP gap (a memoryless policy couldn't sense its own filtered-action momentum) — kept as a permanent default. |
| More iterations alone, twice (2k→8k/10k, two different recipes) | Ruled out iteration count as the lever both times — the plateau was structural, not a training-time problem. |
| No-RL static-hold probe + integrated action-target fix | Root cause found and fixed: the action pipeline capped static holding torque well below what gravity requires; fixing it solved reaching (100% reach rate, mm-scale precision). |

## Walking

### 1. Chosen reward — what it optimizes for

Same Unitree-derived base recipe as every prior checkpoint (track commanded
velocity, stay upright, smooth/low-energy gait, foot clearance and no slip, penalize
arm/leg joint deviation) plus the levers layered in for the arm-integration
priority: a loosened `base_height` penalty (room for balance-correcting knee flex),
a reward for planted feet under near-zero command, and — new last night — commands
under norm 0.1 snap to exact zero and 10x more training time is spent standing
still, on top of a new penalty for drifting from the position held when standing
began. Net effect: explicitly optimizes for a stationary, non-drifting base when not
commanded to walk, at some cost to forward-walking precision and in-place turning.

### 2. Training curriculum

Commanded forward/lateral velocity range starts narrow ((-0.1, 0.1) m/s) and
expands toward the full target range as tracking reward clears a threshold; yaw-rate
range has an equivalent mechanism in code but it was never wired in, so sustained
turning stays untrained throughout. Terrain difficulty ramps on the same
tracking-performance signal. For standing-commanded envs specifically, a scripted
arm-motion disturbance escalates through 4 phases as training progresses, so the
policy is exposed to increasingly large simulated arm motion while it's meant to be
standing still.

### 3. What's been tried

| Tried | Insight |
|---|---|
| 4-way reward-weight ablation vs. the ported Unitree baseline | One change (lighter `action_rate`/`joint_deviation_legs`) broke a "never falls, never walks" local optimum — the other two candidates were no-ops. |
| Heading-drift fix round 1a: two new additive drift-penalty terms | Made drift worse and caused broad regression elsewhere — likely competing new terms plus a stale critic under a warm start. |
| Heading-drift fix round 1b: reward world-frame velocity direction instead of current-yaw-frame | Drift got dramatically worse — this removed an accidental but load-bearing coupling between travel direction and body orientation. |
| Heading-drift fix round 2: one small heading-only penalty, fresh weights | Fall rate stayed perfect, but drift still got worse and scaled badly with speed — possibly still under-trained, not conclusive. |
| Loosening `base_height` (-10 → -4 → -2) | Cut stand-still drift substantially with no fall-rate cost — supports the near-locked-knee/balance-margin hypothesis. |
| Adding `feet_contact_without_cmd` (reward planted feet at zero command) | Improved the drift *outcome* but barely reduced actual stepping frequency — a symptom fix, not the root cause. |
| Adding `joint_mirror` (penalize left/right leg asymmetry) | Genuinely fixed forward-drift and turn_left's fall rate, but reintroduced ~2x the stand-still regression the standing fix had just solved — an unconditional term fighting a standing-specific one. |
| Plain continuation, more iterations, zero reward change | Stand-still drift/stepping regressed even more than the `joint_mirror` experiment did, with a flat/unremarkable training curve — proves "just train longer" isn't safe by default. |
| Standing package (zero-command snap, more standing exposure, position-drift anchor) | Stand-still stepping and drift both improved ~4x with zero falls, at the cost of roughly doubled straight-line heading drift and in-place turning being effectively ignored. |
| Re-checking the eval's turn/strafe commands against what was actually trained | Found they were 3-6x out of distribution — the previously "elevated" turn_left fall rate was almost entirely an eval artifact, not a real policy weakness. |
