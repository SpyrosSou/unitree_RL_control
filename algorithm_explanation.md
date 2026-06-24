# Algorithm Explanation

A detailed reference for the reinforcement learning algorithms, network architectures,
and training configurations used in this project.

---

## Table of Contents

1. [Big Picture: How the Policies Are Used](#1-big-picture-how-the-policies-are-used)
2. [Policy Families at a Glance](#2-policy-families-at-a-glance)
3. [Training Approach by Policy Family](#3-training-approach-by-policy-family)
4. [Proximal Policy Optimization (PPO)](#4-proximal-policy-optimization-ppo)
5. [Network Architecture — Actor-Critic](#5-network-architecture--actor-critic)
6. [Observation & Action Spaces](#6-observation--action-spaces)
7. [Walking Policy](#7-walking-policy)
8. [Standing Policy](#8-standing-policy)
9. [Transition-Focused Variants](#9-transition-focused-variants)
10. [Arm IK Policy](#10-arm-ik-policy)
11. [Reward Shaping Reference](#11-reward-shaping-reference)
12. [Exact Hyperparameter Tables](#12-exact-hyperparameter-tables)

---

## 1. Big Picture: How the Policies Are Used

This repo contains three policy families:
- arm IK policies that move one or both arms to a 3-D target
- walking policies that track velocity commands
- standing policies that keep the robot upright with zero motion

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

That means the policy is not programmed with hard-coded motion. Instead, it learns
behavior from the reward structure, the command ranges, and the physical limits of
the robot.

---

## 2. Policy Families at a Glance

The three families solve different problems and therefore use different inputs,
outputs, and training settings:

- Arm IK policies learn to move the arm end-effector to a Cartesian target.
- Walking policies learn to track a velocity command while staying balanced.
- Standing policies learn to keep the robot still and upright when command input
  is zero or near zero.

Because the goals are different, a strong walking policy is not automatically a
strong standing policy, and a single arm policy is not directly interchangeable
with a locomotion policy. The task definition determines what the network sees,
what it is allowed to output, and what the reward encourages.

---

## 3. Training Approach by Policy Family

### Arm training

Arm training is deliberately isolated from locomotion. The arm environment fixes
the robot base to the world, so the learner focuses on shoulder and elbow control
instead of balancing a free-floating body. That makes the reward signal easier to
interpret and prevents the arm task from being polluted by leg dynamics.

The arm policy uses a fixed-base `DirectRLEnv` with a position-servo style action
space. In practice, that means the policy learns small delta joint commands that
move the hand toward the target. The legs stay locked in a neutral posture while
the arm learns reachability, smoothness, and joint-limit avoidance.

### Walking training

Walking training uses the full body and all joints. The policy learns to convert a
velocity command into a gait while keeping the torso stable. For the standard
walking policy, the command distribution is mostly about forward motion, turning,
and lateral correction.
The standing policy is trained to hold the robot upright under zero or near-zero
commands, with a small micro-command slice so the policy can learn minimal
corrective stepping instead of only static balance.
better at starting, stopping, and switching direction instead of only cruising at
steady speed.

### Standing training

| `lin_vel_z_l2` weight | **−2.2** | Strongly penalise vertical bounce |
| `ang_vel_xy_l2` weight | **−0.12** | Penalise roll/pitch wobble without forbidding recovery |
| `action_rate_l2` weight | **−0.008** | Smooth actions for quiet standing |
| `dof_acc_l2` weight | **−1.5×10⁻⁷** | Smoother joint acceleration |

The transition-focused standing variant adds a small amount of micro-command
exposure so it can absorb the residual motion that appears when the robot decelerates
from walking into a stop.
A walking policy trained with `rel_standing_envs = 0` (no zero-command envs)
has little incentive to remain perfectly still — it optimises for forward motion.
When you release the joystick, the walking policy continues to drift, sway, or
oscillate. A dedicated standing policy, by contrast, is trained to suppress
those motions while still allowing small corrective steps when arm disturbance
is high.
lets each policy specialize:
- arm policy: reach targets cleanly
- walking policy: move and balance
- `max_iterations = 2500` in the current repo snapshot
- Validate intermediate checkpoints around `model_1200`, `model_1800`, and `model_2500`
- Standing runs now write `standing_metrics.csv` in the log folder
- standing policy: hold still and stabilize

The integrated demo can then combine those specialized policies at runtime.

---

## 4. Proximal Policy Optimization (PPO)

- `resampling_time_range = (0.8, 2.0)` s  — commands change more frequently, forcing the policy to handle stop/start/reversal
- `rel_standing_envs = 0.25` — teaches the walking policy to brake cleanly rather than stumbling
- Wider command ranges: `lin_vel_x = (−1, 1)`, `lin_vel_y = (−0.8, 0.8)`, `ang_vel_z = (−1, 1)` — includes backward walking and tight reversals
- Accurately predict value (critic / value head)

### The PPO update step

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

- $\gamma = 0.99$ — discount factor (how far ahead rewards matter)
- $\lambda = 0.95$ — GAE lambda (bias-variance tradeoff; higher = lower bias)

### Adaptive learning rate

The project uses `schedule: adaptive`. After each update, the empirical KL divergence
between old and new policy is computed:
$$\text{KL} \approx \frac{1}{2}\mathbb{E}\!\left[(\log r_t)^2\right]$$
- If $\text{KL} > 2 \times \texttt{desired\_kl}$ → learning rate is halved
- If $\text{KL} < 0.5 \times \texttt{desired\_kl}$ → learning rate is doubled
- $\texttt{desired\_kl} = 0.01$ — target per-update KL budget

This keeps updates small enough to be stable without needing manual LR tuning.

---

## 5. Network Architecture — Actor-Critic

All locomotion and standing policies share one architecture (flat terrain):

```
Observation (N-D)  →  [Linear 256 + ELU]  →  [Linear 128 + ELU]  →  [Linear 128 + ELU]
                                                                        ↓
                                                           Actor head:  Linear → N_actions (mean)
                                                           + trainable log-std (scalar, init=1.0)
                                                           Critic head: Linear → 1 (value)
```

- **Hidden dims (flat):** `[256, 128, 128]`
- **Hidden dims (rough):** `[512, 256, 128]` — wider because height-scan adds ~187 obs dims
- **Activation:** ELU (Exponential Linear Unit) — smooth gradient, no dying-neuron issue
- **Output noise:** diagonal Gaussian, `init_noise_std = 1.0`, learned during training
- **Normalization:** `actor_obs_normalization: false` — Isaac Lab normalizes obs separately

### Why ELU?

ELU(x) = x if x > 0, else α(e^x − 1). It avoids the "dead ReLU" problem
and produces negative outputs, which helps center activations around zero and
can speed up convergence.

### Policy output interpretation

The actor outputs a **mean** action vector. During training, actions are sampled
from `N(mean, std)` to explore. During inference (play), the mean is used directly.
`use_default_offset=True` in the action config means the output is an **offset
from the robot's default pose** (not an absolute joint angle).

---

## 6. Observation & Action Spaces

### Locomotion / standing policies (ManagerBasedRLEnv)

| Observation group | Contents | Dims |
|---|---|---|
| Base angular velocity | `robot.data.root_ang_vel_b` | 3 |
| Projected gravity | `robot.data.projected_gravity_b` | 3 |
| Velocity command | `[v_x, v_y, ω_z]` (from keyboard / curriculum) | 3 |
| Joint positions | `joint_pos - default_joint_pos` for all joints | 23 |
| Joint velocities | `joint_vel` for all joints | 23 |
| Previous actions | action from last step | 23 |
| **Total (flat)** | | **~78** |
| Height scan | 187 grid points (rough env only) | 187 |
| **Total (rough)** | | **~265** |

> Velocity command lives at **obs[:, 9:12]** — this is why the demos inject
> `obs[:, 9:12] = vel_cmd` to override the sampled command.

**Action space:** 23-D, joint position offsets from default pose, `scale=0.5` rad.
`JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)`

### Arm policies (DirectRLEnv)

| Component | Dims |
|---|---|
| Arm joint positions (5 joints) | 5 |
| Arm joint velocities (first 3) | 3 |
| End-effector world position | 3 |
| Goal world position | 3 |
| Position error (goal − ee_pos) | 3 |
| **Total (single arm)** | **17** |
| **Total (both arms)** | **34** |

**Action space (single arm):** 5-D delta joint positions, `action_scale=0.5`.
Applied as: `target = current_joint_pos + action * 0.5`, clamped to soft limits.

---

## 7. Walking Policy

**Task:** `G1-Locomotion-Flat-v0`  
**Runner cfg:** `G1LocomotionFlatPPORunnerCfg`  
**Log namespace:** `legs/g1_locomotion_flat`

### Command curriculum

During training the command $[v_x, v_y, \omega_z]$ is resampled every
`resampling_time_range` seconds from uniform distributions:

| Component | Range |
|---|---|
| $v_x$ (forward) | 0.0 – 1.0 m/s |
| $v_y$ (lateral) | −0.5 – 0.5 m/s |
| $\omega_z$ (yaw rate) | −1.0 – 1.0 rad/s |

`rel_standing_envs = 0.0` — **all** environments receive non-zero commands.

### Key reward terms (flat)

| Term | Weight | Description |
|---|---|---|
| `track_lin_vel_xy_exp` | 1.0 | Exponential reward for tracking commanded $v_x, v_y$ |
| `track_ang_vel_z_exp` | 1.0 | Tracking commanded $\omega_z$ |
| `lin_vel_z_l2` | −0.2 | Penalty on vertical body velocity (anti-bounce) |
| `ang_vel_xy_l2` | −0.05 | Penalty on roll/pitch body angular velocity |
| `dof_torques_l2` | −2×10⁻⁶ | Torque minimization (hip/knee joints only) |
| `dof_acc_l2` | −1×10⁻⁷ | Joint acceleration smoothness |
| `action_rate_l2` | −0.005 | Action change rate (jitter penalty) |
| `feet_air_time` | +0.75 | Bonus when feet are in the air — encourages gait cycles |
| `feet_air_time threshold` | 0.4 s | Minimum flight time to earn the bonus |
| `joint_deviation_arms` | −0.1 | Keep arm joints near default — prevents flailing |
| `joint_deviation_hip` | −0.1 | Keep hip yaw/roll near default |

### Training schedule

- `num_steps_per_env = 24` — steps per rollout per environment
- `max_iterations = 3000` — policy updates (our local training ran 3149)
- Effective batch size: 4096 envs × 24 steps = **98,304 transitions per rollout**
- Mini-batches: 4 → **24,576 per mini-batch**
- `num_learning_epochs = 5` → 5 passes over each rollout

---

## 8. Standing Policy

**Task:** `G1-Locomotion-Standing-Flat-v0`  
**Runner cfg:** `G1LocomotionStandingFlatPPORunnerCfg`  
**Log namespace:** `standing/g1_locomotion_flat`

The standing policy is trained to hold the robot upright under zero or near-zero
commands, with a small micro-command slice so the policy can learn minimal
corrective stepping instead of only static balance.

### Key reward shaping changes vs. walking

| Term | Value | Reason |
|---|---|---|
| `lin_vel_z_l2` weight | **−2.2** | Strongly penalise vertical bounce |
| `ang_vel_xy_l2` weight | **−0.12** | Penalise roll/pitch wobble without forbidding recovery |
| `action_rate_l2` weight | **−0.008** | Smooth actions for quiet standing |
| `dof_acc_l2` weight | **−1.5×10⁻⁷** | Smoother joint acceleration |
| `feet_air_time` weight | **0.0** | No gait cycle needed — feet stay on ground |

### Why train a separate standing policy?

A walking policy trained with `rel_standing_envs = 0` (no zero-command envs)
has little incentive to remain perfectly still — it optimises for forward motion.
When you release the joystick, the walking policy continues to drift, sway, or
oscillate. A dedicated standing policy, by contrast, is trained to suppress
those motions while still allowing small corrective steps when arm disturbance
is high.

### Training schedule

- `max_iterations = 2500` in the current repo snapshot
- Validate intermediate checkpoints around `model_1200`, `model_1800`, and `model_2500`
- Standing runs now write `standing_metrics.csv` in the log folder



## 9. Transition-Focused Variants

These tasks improve the quality of mode switches in the stand/walk switch demo.

### Walking transition (`G1-Locomotion-Flat-Transition-v0`)

Changes vs. standard walking:
- `resampling_time_range = (0.8, 2.0)` s — commands change more frequently, forcing the policy to handle stop/start/reversal
- `rel_standing_envs = 0.25` — teaches the walking policy to brake cleanly rather than stumbling
- Wider command ranges: `lin_vel_x = (−1, 1)`, `lin_vel_y = (−0.8, 0.8)`, `ang_vel_z = (−1, 1)` — includes backward walking and tight reversals

---

## 10. Arm IK Policy

**Framework:** `DirectRLEnv` (bypasses the manager-based event/reward system)  
**Tasks:** `G1-Arm-IK-Left-v0`, `G1-Arm-IK-Right-v0`, `G1-Arm-IK-Both-v0`  
**Log namespace:** `arms/g1_arm_ik_{left,right,both}`

### Key differences from locomotion training

| Property | Locomotion | Arm IK |
|---|---|---|
| Base link | Free (floating) | **Fixed to world** (`fix_root_link=True`) |
| Episode length | 1,000,000 s (no reset in play) | **10 s** per episode |
| Action | Joint position offsets (23-D, all joints) | Delta positions for **5 arm joints** |
| Actuator | Default (torque-based) | `stiffness=200, damping=20` (position servo) |
| Goal | Track velocity command | Reach a 3-D Cartesian target |

### Observation layout (17-D per arm)

```
[0:5]   joint_pos  — 5 arm joint angles (rad)
[5:8]   joint_vel  — velocity of first 3 arm joints (rad/s)
[8:11]  ee_pos     — end-effector world position (m)
[11:14] goal       — target world position (m)
[14:17] error      — goal − ee_pos (m)
```

### Action

5-D delta joint positions: `target = current + action * 0.5`  
Clamped to soft joint position limits before being sent to the actuator.

### Reward

| Term | Formula | Scale |
|---|---|---|
| Distance penalty | $-\|p_{ee} - p_{goal}\|$ | ×10 |
| Goal reached bonus | +1 if dist < 2 cm | ×50 |
| Action smoothness | $-\|\mathbf{a}\|$ | ×0.01 |
| Joint limit penalty | count joints near limits | ×1.0 |

### Goal workspace

| Arm | x (forward) | y (lateral) | z (height) |
|---|---|---|---|
| Left | 0.1 – 0.5 m | 0.05 – 0.45 m | 0.9 – 1.2 m |
| Right | 0.1 – 0.5 m | −0.45 – −0.05 m | 0.9 – 1.2 m |

### Bilateral symmetry (mirror test)

The left and right arms have bilateral symmetry. Given a left-arm policy, the
right arm can be controlled by:
1. Negating the y-components of ee_pos, goal, error in the observation
2. Negating the roll and yaw joint angles/velocities (indices 1, 2, 4, 6, 7 in obs)
3. Negating the roll and yaw delta commands in the action (indices 1, 2, 4)

This avoids training a separate right-arm policy at the cost of slight accuracy
loss near the workspace boundaries.

---

## 11. Reward Shaping Reference

### Why exponential tracking rewards?

`track_lin_vel_xy_exp` uses $e^{-\alpha \|\text{error}\|^2}$ (a shaped Gaussian).
This gives a smooth, dense reward signal even when the robot is far from the
target velocity, unlike a simple L2 penalty that goes to zero only at the target.

### The feet_air_time reward

This is crucial for producing a gait. Without it, the policy quickly discovers
that it can maximise reward by **standing still** (zero command → zero tracking
error, zero torque cost). The `feet_air_time` bonus forces the policy to lift
its feet periodically, which can only happen if it is taking steps.

In the standing policy this term is set to **0.0** — standing still with feet on
the ground is exactly what we want.

### joint_deviation_arms (−0.1)

During locomotion training the arm joints are included in the 23-D action space.
Without a penalty they may swing to arbitrary configurations. The deviation reward
keeps arms near the default T-pose, preventing flailing that would destabilise the
torso and waste energy.

---

## 12. Exact Hyperparameter Tables

### PPO (shared across all locomotion tasks)

| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | 0.001 | Initial; adapted each update |
| `schedule` | adaptive | Halve/double based on KL |
| `desired_kl` | 0.01 | Target KL per update |
| `num_learning_epochs` | 5 | Passes over each rollout |
| `num_mini_batches` | 4 | Mini-batches per epoch |
| `clip_param` | 0.2 | PPO ε (probability ratio clip) |
| `value_loss_coef` | 1.0 | Weight of critic loss |
| `entropy_coef` | 0.008 | Weight of entropy bonus |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `use_clipped_value_loss` | true | Clip value loss similarly to policy loss |

### Runner (per-task differences)

| Task | `num_steps_per_env` | `max_iterations` | `save_interval` |
|---|---|---|---|
| Walking flat | 24 | 3000 (ran 3149) | 50 |
| Standing flat | 24 | 1500 | 50 |
| Walking transition | 24 | 2500 (recommended) | 50 |
| Standing transition | 24 | 1500 (recommended) | 50 |
| Arm IK (single) | 24 | 5000 (recommended) | 100 |
| Arm IK (both) | 24 | 5000 (recommended) | 100 |

### Network dimensions

| Task | Actor hidden | Critic hidden | Activation |
|---|---|---|---|
| Flat locomotion / standing | [256, 128, 128] | [256, 128, 128] | ELU |
| Rough locomotion | [512, 256, 128] | [512, 256, 128] | ELU |
| Arm IK | [256, 128, 128] | [256, 128, 128] | ELU |

### Simulation settings

| Parameter | Value |
|---|---|
| Physics dt | 0.005 s (200 Hz) |
| Control decimation | 4 (locomotion) / 2 (arm IK) |
| Effective control rate | 50 Hz (locomotion) / 30 Hz (arm IK) |
| Number of environments (training) | 4096 |
| Number of environments (play) | 1–50 |
