# Algorithm Explanation

A reference for the reinforcement learning algorithm, network architecture, and
observation/action spaces used in this project. This document covers what stays
constant across policy iterations (the algorithm, the architecture, the general
reward-shaping rationale). For the current, specific per-policy configuration
(reward weights, command ranges, curricula, iteration counts) — which changes as
the policies are tuned — see `training_regimes.md` instead. Splitting it this way
avoids the two documents drifting out of sync with each other.

---

## Table of Contents

1. [Big Picture: How the Policies Are Used](#1-big-picture-how-the-policies-are-used)
2. [Policy Families at a Glance](#2-policy-families-at-a-glance)
3. [Proximal Policy Optimization (PPO)](#3-proximal-policy-optimization-ppo)
4. [Network Architecture — Actor-Critic](#4-network-architecture--actor-critic)
5. [Observation & Action Spaces](#5-observation--action-spaces)
6. [Reward Shaping Rationale](#6-reward-shaping-rationale)
7. [Shared Hyperparameters & Simulation Settings](#7-shared-hyperparameters--simulation-settings)

---

## 1. Big Picture: How the Policies Are Used

This repo contains three policy families:
- arm IK policies that move one or both arms to a 3-D target
- walking policies that track velocity commands
- standing policies that keep the robot upright with minimal motion

Each policy is trained by trial and error. The simulator runs many copies of the
robot in parallel, each robot produces actions, the environment returns a reward,
and PPO updates the network so future actions score better. Over time, the policy
learns a control habit that matches the task it was trained for.

At a high level the workflow is:
1. Define the task and its reward signal.
2. Run many simulated environments in parallel.
3. Collect actions, observations, rewards, and terminations.
4. Update the neural network with PPO.
5. Save checkpoints and evaluate them in the play/demo scripts.

The policy is not programmed with hard-coded motion — it learns behavior from the
reward structure, the command ranges, and the physical limits of the robot.

---

## 2. Policy Families at a Glance

The three families solve different problems and therefore use different inputs,
outputs, and training settings:

- **Arm IK** policies learn to move the arm end-effector to a Cartesian target.
  Training is deliberately isolated from locomotion — the robot's base is fixed
  to the world (`fix_root_link=True`), so the learner focuses on shoulder/elbow
  control rather than balancing a free-floating body. It uses a fixed-base
  `DirectRLEnv` with a delta-joint-position action space.
- **Walking** policies learn to track a velocity command while staying balanced,
  using the full body (`ManagerBasedRLEnv`, all joints in the action space).
- **Standing** policies learn to keep the robot upright and stable under zero or
  near-zero commands, with a curriculum of arm-motion disturbances so the policy
  learns corrective stepping — see `training_regimes.md` for the current
  curriculum details.

Because the goals are different, a strong walking policy is not automatically a
strong standing policy, and a single arm policy is not directly interchangeable
with a locomotion policy. The task definition determines what the network sees,
what it is allowed to output, and what the reward encourages.

---

## 3. Proximal Policy Optimization (PPO)

Given a mini-batch of transitions $(s, a, r, s', \hat{V})$:

**Policy (actor) loss** — clipped surrogate objective:

$$\mathcal{L}_{\pi} = -\mathbb{E}\left[\min\!\left( r_t(\theta)\hat{A}_t,\;
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right)\right]$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_\text{old}}(a_t | s_t)$
is the probability ratio and $\hat{A}_t$ is the Generalized Advantage Estimate (GAE).

**Value (critic) loss** — clipped MSE:

$$\mathcal{L}_V = \frac{1}{2}\mathbb{E}\!\left[\max\!\left(
(V_\theta(s_t) - R_t)^2,\;
(\text{clip}(V_\theta(s_t), \hat{V}_t \pm \epsilon) - R_t)^2
\right)\right]$$

**Entropy bonus** — encourages exploration:

$$\mathcal{L}_H = -\mathcal{H}[\pi_\theta(·|s_t)]$$

**Total loss:**

$$\mathcal{L} = \mathcal{L}_{\pi} + c_V \mathcal{L}_V + c_H \mathcal{L}_H$$

### Generalized Advantage Estimation (GAE)

$$\hat{A}_t = \sum_{k=0}^{T} (\gamma\lambda)^k \delta_{t+k}, \qquad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- $\gamma$ (`gamma`) — discount factor (how far ahead rewards matter)
- $\lambda$ (`lam`) — GAE lambda (bias-variance tradeoff; higher = lower bias)

### Adaptive learning rate

All PPO runners here use `schedule: adaptive`. After each update, the empirical KL
divergence between old and new policy is computed:

$$\text{KL} \approx \frac{1}{2}\mathbb{E}\!\left[(\log r_t)^2\right]$$

- If $\text{KL} > 2 \times \texttt{desired\_kl}$ → learning rate is halved
- If $\text{KL} < 0.5 \times \texttt{desired\_kl}$ → learning rate is doubled

This keeps updates small enough to be stable without needing manual LR tuning.
See `training_regimes.md` / `params/agent.yaml` in a run's log folder for the
exact `desired_kl`, `entropy_coef`, etc. currently in use per task.

---

## 4. Network Architecture — Actor-Critic

All locomotion and standing policies share one architecture shape (widths differ
between flat and rough terrain — see `training_regimes.md`):

```
Observation (N-D)  →  [Linear H1 + ELU]  →  [Linear H2 + ELU]  →  [Linear H3 + ELU]
                                                                        ↓
                                                           Actor head:  Linear → N_actions (mean)
                                                           + trainable log-std (scalar, init=1.0)
                                                           Critic head: Linear → 1 (value)
```

- **Activation:** ELU (Exponential Linear Unit) — avoids the "dead ReLU" problem,
  produces negative outputs which helps center activations around zero.
- **Output noise:** diagonal Gaussian, `init_noise_std = 1.0`, learned during training.

### Policy output interpretation

The actor outputs a **mean** action vector. During training, actions are sampled
from `N(mean, std)` to explore. During inference (play), the mean is used directly.
For the locomotion/standing action term, `use_default_offset=True` means the
output is an **offset from the robot's default pose**, not an absolute joint angle.

---

## 5. Observation & Action Spaces

### Locomotion / standing policies (`ManagerBasedRLEnv`)

| Observation group | Contents |
|---|---|
| Base angular velocity | `robot.data.root_ang_vel_b` |
| Projected gravity | `robot.data.projected_gravity_b` |
| Velocity command | `[v_x, v_y, ω_z]` (from keyboard / curriculum), lives at **obs[:, 9:12]** |
| Joint positions | `joint_pos - default_joint_pos`, all joints |
| Joint velocities | `joint_vel`, all joints |
| Previous actions | action from last step |

> The demos inject `obs[:, 9:12] = vel_cmd` every step to override the
> environment's sampled command with the keyboard-driven one.

**Action space:** joint position offsets from default pose, all joints (`torso_joint`
included), `JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)`.

**Note on sim2real fidelity:** `base_lin_vel` normally appears in this observation
group too. A real G1 has no ground-truth base linear velocity sensor — it would need
to be estimated (IMU + kinematics, or a learned estimator), and that estimate is
noisier/biased relative to sim. This is a known gap, not yet addressed — see the
roadmap plan for options being considered (heavier noise on that term, or an
asymmetric actor/critic split).

### Arm policies (`DirectRLEnv`)

| Component | Dims |
|---|---|
| Arm joint positions (5 joints) | 5 |
| Arm joint velocities (first 3) | 3 |
| End-effector world position | 3 |
| Goal world position | 3 |
| Position error (goal − ee_pos) | 3 |
| **Total (single arm)** | **17** |
| **Total (both arms)** | **34** |

**Action space (single arm):** 5-D delta joint positions, applied as
`target = current + action * action_scale`, filtered with a first-order lag and
clamped to a max per-step delta before being clamped again to soft joint limits.

**Note:** unlike the locomotion/standing observation, the arm policy's observation
has no base-orientation or base-velocity term at all — it has never been trained
with, or made aware of, a moving/tilting base. This is a known gap the roadmap
plan addresses before the arm policy is asked to work while standing or walking.

### Bilateral symmetry (left/right arm)

The left and right arms are bilaterally symmetric. Given a left-arm policy, the
right arm can be controlled by:
1. Negating the y-components of `ee_pos`, `goal`, `error` in the observation.
2. Negating the roll and yaw joint angles/velocities.
3. Negating the roll and yaw delta commands in the action.

This is currently done as a runtime mirror transform (`testing/arm_testing/g1_arm_mirror_test.py`).
Isaac Lab also ships a native mechanism for this (`RslRlSymmetryCfg` — data
augmentation and/or a mirror loss baked into the PPO objective itself), which the
roadmap plan flags as a likely better fit than the runtime transform.

---

## 6. Reward Shaping Rationale

This section explains *why* certain reward terms exist — the actual current
weights are in `training_regimes.md`.

### Why exponential tracking rewards?

`track_lin_vel_xy_exp` uses $e^{-\alpha \|\text{error}\|^2}$ (a shaped Gaussian).
This gives a smooth, dense reward signal even when the robot is far from the
target velocity, unlike a simple L2 penalty that only meaningfully differs near
the target.

### The `feet_air_time` reward

This is what produces a gait in the walking policy. Without it, the policy could
maximise reward by standing still whenever that's compatible with the commanded
velocity (zero command → zero tracking error, zero torque cost). The bonus forces
periodic foot lift-off, which can only happen by taking steps.

In the standing policy this term is set to zero — it isn't needed to *produce* a
gait (standing's default behavior isn't a gait), and it's the arm-motion
disturbance curriculum, not this reward, that's meant to create situations where
a corrective step becomes necessary.

### `joint_deviation_*` terms

During locomotion training, joints not essential to the core task (hips, arms,
fingers, torso) are included in the action space but penalized for deviating from
their default pose — this prevents them from drifting to arbitrary/wasteful
configurations that destabilize the body or waste energy. The torso term in
particular is one to watch closely once arm-motion disturbances are introduced:
it needs to be loose enough to let the torso act as a balance-compensation DOF
rather than fighting that behavior. See `training_regimes.md` for the current
weight and the roadmap plan for the reasoning.

---

## 7. Shared Hyperparameters & Simulation Settings

These are stable across the current tasks; task-specific overrides (e.g. arm's
`entropy_coef=0.0` vs. the locomotion/standing default) are in `training_regimes.md`.

### PPO (shared defaults, locomotion/standing)

| Parameter | Value |
|---|---|
| `learning_rate` | 0.001 (initial; adapted each update) |
| `schedule` | adaptive |
| `desired_kl` | 0.01 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `clip_param` | 0.2 |
| `value_loss_coef` | 1.0 |
| `gamma` | 0.99 |
| `lam` | 0.95 |
| `max_grad_norm` | 1.0 |
| `use_clipped_value_loss` | true |

### Simulation settings

| Parameter | Value |
|---|---|
| Physics dt | 0.005 s (200 Hz) |
| Control decimation | 4 (locomotion/standing) / 2 (arm IK) |
| Effective control rate | 50 Hz (locomotion/standing) / 30 Hz (arm IK) |
| Number of environments (training) | 4096 (typical) |
| Number of environments (play) | 1–50 |
