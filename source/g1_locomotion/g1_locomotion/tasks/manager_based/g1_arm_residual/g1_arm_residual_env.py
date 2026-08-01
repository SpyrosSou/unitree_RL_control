# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof arm — residual RL on top of a numerical IK baseline.

Phase 4 of ``ik_arm_integration_plan.md`` (the "residuals" part). Built 2026-07-28,
directly following Phase 2's conclusion: with the base genuinely fixed
(``fix_root_link=True``, no walking policy) and the frame-conversion bug fixed
(``eval_arm_ik_standing.py``'s yaw-only leak fix), the IK solver (``G1ArmIK``, this
repo's vendored ``xr_teleoperate`` port) reaches kinematic accuracy matching or beating
Phase 1's own static-sweep ceiling (mean ~0.8-2.9cm) essentially everywhere in the
trimmed goal box. What's left is a real, systematic, ~10-20cm gap between that
kinematic solution and where the arm's PD controller + static gravity feedforward
actually settles it in sim — height-correlated, narrow-range (std ~3.5cm on a 15-target
sample), and NOT explained by any remaining bug (target delivery, gains, and torque
headroom were all directly verified correct on 2026-07-28). That gap is exactly what
this project's architecture always planned to hand to an RL residual layer (see
``personal_development/learned_ik_vs_residual_rl.md`` and
``personal_development/residual_rl.md`` for the underlying theory/design writeup) — not
a bug to keep patching in the IK/frame layer.

## What "residual" means here, concretely

Every control step, for the CURRENT (fixed-for-the-episode) goal:

1. A numerical IK solve (this repo's ``G1ArmIK``) already produced a joint-space
   baseline target ``q_ik`` and a gravity/Coriolis-at-rest feedforward torque
   ``tau_ik`` — computed ONCE per episode at reset (see "Why the baseline is solved
   once, not every step" below), not re-solved every control step.
2. The RL policy observes the usual arm-reaching features (joint state, goal, EE
   error, action feedback) PLUS ``q_ik - current_joint_pos`` (how far the baseline
   still is from where the arm currently sits) and outputs a small, BOUNDED per-joint
   correction (``cfg.residual_action_scale``, default 0.15 rad — enough to cover the
   ~10-20cm/few-degree-per-joint gap Phase 2 measured, not enough to re-invent full
   reaching).
3. The commanded joint target is ``q_ik + residual``, and ``tau_ik`` is applied as
   feedforward effort, same as every eval script this session used.

This is the textbook residual-RL formulation (Johannink et al., "Residual
Reinforcement Learning for Robot Control"): ``a_total = a_baseline + f_theta(s)``, a
STATE-DEPENDENT, bounded correction on top of a fixed analytical answer — not an
accumulating delta the policy has to build up over many steps the way the pure-RL arm
task (``g1_arm_env.py``) has to, since that task has no baseline to lean on at all.

## Why the baseline is solved once per episode, not every control step

``G1ArmIK.solve_ik`` is a single-instance, CPU-bound CasADi/IPOPT solve (~1-2ms per
call per Phase 0/1's own measurement) — there is no batched/GPU version. Calling it
once per env per control step (as the walking-integrated eval scripts do, where the
target genuinely moves every step because the pelvis is swaying) would cost
``num_envs`` solves every single step — completely impractical for RL training
(thousands of steps, potentially thousands of parallel envs).

The fixed-base task doesn't need that: goals are sampled once at reset and held fixed
for the whole episode (see ``g1_arm_env.py``'s ``_sample_goal_positions``/
``_reset_idx``), and the base never moves (genuinely bolted,
``fix_root_link=True``) — so the correct pelvis-relative IK target is ALSO fixed for
the whole episode. Solving once per reset and caching the result in a per-env buffer
(``self.ik_baseline_q``/``self.ik_baseline_tauff``) is both correct and cheap: at
steady state only a rolling handful of envs reset on any given step (num_envs /
episode_length_steps, e.g. ~1024/300 ≈ 3-4 solves/step at this file's defaults), not
num_envs solves/step. The one real cost is the FIRST reset (all envs at once, at env
construction) — a one-time ~1-2 second startup delay at 1024 envs, not a per-step cost.

## Scope decisions (deliberate, not oversights)

- **Single arm only** (``cfg.arm`` must be "left" or "right", never "both") — matches
  every eval this session ran, and doubling the IK-solve loop for a second arm is
  unnecessary complexity before the single-arm mechanism is even validated.
- **Goal box defaults to the trimmed region** (``goal_bounds_x_override=(0.20, 0.31)``,
  matching every walking/fixed-base eval this session used) — training the residual on
  the far-reach half's genuinely-hard-to-solve corners would just teach it to do
  nothing useful there (no residual can fix a point outside the kinematic workspace;
  that's a solver/reachability question, not a tracking-gap question). Expand this
  once the near-half residual is confirmed working.
- **Reuses ``g1_arm_env.py``'s ``G1ArmEnv``/``G1ArmLeftEnvCfg`` via subclassing**, not a
  from-scratch rewrite — the scene setup, reward function (already rewards ACHIEVED,
  physically-simulated EE position error against the goal, exactly what the residual
  needs to improve), termination, and root-wobble domain randomization are all reused
  unchanged. Only ``__init__`` (add the IK solver + baseline buffers), ``_reset_idx``
  (solve the baseline once per newly-sampled goal), ``_apply_action`` (command
  ``q_ik + residual`` instead of ``current_pose + delta``), and ``_get_observations``
  (append the baseline-error feature) are overridden.
- **Symmetry data augmentation disabled** for this task's PPO cfg (see
  ``agents/rsl_rl_ppo_cfg.py``) — ``mdp/symmetry.py``'s ``mirror_arm_obs`` hard-codes
  the base task's 32/39/64-dim observation layouts and doesn't know about this task's
  extra 7-dim baseline-error block, the same reason ``G1ArmLeftLockedWristPPORunnerCfg``
  disabled it for its own modified (28-dim) layout.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from g1_locomotion.controllers.arm_ik import G1ArmIK

from ..g1_arm.g1_arm_env import (
    _LEFT_ARM_JOINTS,
    _RIGHT_ARM_JOINTS,
    G1ArmEnv,
    G1ArmLeftEnvCfg,
    G1ArmRightEnvCfg,
)


def _se3(p: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, 3] = p
    return tf


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------


# Shared field values below are duplicated across the left/right cfgs rather than
# factored into a mixin: every other cfg variant in g1_arm_env.py uses plain single
# inheritance, and mixing a second, field-only base into a @configclass dataclass
# hierarchy risks MRO/field-ordering surprises for no real benefit here (six fields,
# two classes) -- matching the codebase's own proven pattern is worth the small
# duplication.


@configclass
class G1ArmResidualLeftEnvCfg(G1ArmLeftEnvCfg):
    """See module docstring for the reasoning behind each field below."""

    arm: str = "left"

    # 39 (base G1ArmEnv observation) + 7 (ik_baseline_error) — see _get_observations.
    observation_space: int = 46
    action_space: int = 7

    # Fewer parallel envs than the pure-RL task's 4096 default: this task's IK-solve
    # loop at reset is CPU-single-instance (no GPU batching), so fewer envs keeps both
    # the one-time startup solve batch and the steady-state per-step reset-solve count
    # manageable. See module docstring's "why the baseline is solved once" section.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0)

    # Small, bounded per-joint correction on top of the IK baseline -- sized to cover
    # the ~10-20cm (roughly a few degrees per joint) achieved-vs-kinematic gap Phase 2
    # measured on a genuinely fixed base, not to re-invent full reaching. This is the
    # ONLY bound on the residual's magnitude (unlike the pure-RL task, this env does
    # NOT also apply max_action_delta_per_step to the residual itself -- that field
    # was a per-step RATE limit relative to a moving "current pose" baseline in the
    # original task; here the baseline is fixed for the whole episode and the residual
    # is a direct, state-dependent correction each step, not something that needs to
    # accumulate/travel, so a single clear scale is the right bound, not two
    # overlapping ones).
    residual_action_scale: float = 0.15

    # Apply the IK solver's own RNEA-at-rest feedforward torque (sol_tauff) as
    # set_joint_effort_target, additive on top of the position-target PD term -- the
    # SAME fix `ik_arm_integration_plan.md` Phase 2 found necessary (gravity sag was
    # stalling the rate-limiter itself, not just drooping a few cm). Kept switchable
    # for a clean A/B if needed, default on.
    apply_ik_tauff: bool = True

    # Train on the reliably-reachable region only (matches every walking/fixed-base
    # eval this session ran, --x_max 0.31) -- see module docstring's scope-decisions
    # section for why the far-reach half doesn't belong in this task at all.
    goal_bounds_x_override: tuple[float, float] | None = (0.20, 0.31)

    # 2026-07-28: bumped 20x from the inherited default (0.05). Found via direct
    # TensorBoard inspection across a full 6000-iteration run: reach_rate_no_hold hit
    # 99.5%+ while success_rate (reach+HOLD for goal_hold_steps consecutive steps)
    # actually DROPPED (21%->13%) as training progressed. Episode_Reward/settle stayed
    # essentially flat the entire run (~-0.09) while Episode_Reward/goal_reached_bonus
    # climbed steadily (~22->39) and Train/mean_episode_length stayed near the 300-step
    # cap (not dropping toward early-success termination) -- three independent signals
    # agreeing that goal_reached_bonus (paid out EVERY step reached=True, no stillness
    # or consecutiveness required) was ~400x larger in typical per-step contribution
    # than the settle penalty, making "oscillate rapidly through the 2cm zone" reward-
    # optimal over "settle and hold", and MORE training only sharpened that exploit
    # rather than fixing it -- a reward-shaping imbalance, not a code bug (the settle
    # mechanism itself was confirmed correctly wired and contributing, just far too
    # weakly to compete). This is a deliberately isolated, single-variable change so
    # the next eval can attribute any improvement to this specific fix.
    #
    # 2026-07-28 FOLLOW-UP: confirmed via TensorBoard that this scale bump alone did
    # NOT fix it -- Episode_Reward/settle jumped immediately (arithmetic, from the 20x
    # multiplier) but then stayed FLAT (~-1.7) for a further ~1900 iterations post-
    # resume, no directional improvement, while goal_reached_bonus KEPT climbing
    # (38.7->39.9) -- i.e. the bigger penalty changed the NUMBER, not the underlying
    # velocity behavior. AT THE TIME this was believed "harmless, complementary" once
    # goal_reached_max_vel became the primary mechanism -- THAT ASSUMPTION WAS WRONG,
    # see the 2026-07-29 note below. REVERTED to the pre-bump default (0.05).
    #
    # 2026-07-29: this 20x bump is the actual cause of the 4k-iteration run stalling
    # completely (Metrics/frac_envs_reached ~0%, goal_reached_bonus ~0.0 for the
    # ENTIRE run past iter ~200, min_dist_to_goal_cm frozen at 4.96-5.4cm for every
    # goal regardless of position -- r<0.03 correlation with goal x/y/z, the
    # signature of a reward-shaped equilibrium, not an undertrained policy). Root
    # cause: this penalty was only "harmless" back when goal_reached_bonus paid out
    # unconditionally and dwarfed it (50 vs ~1). Once goal_reached_bonus ALSO required
    # low velocity (the fix directly below), that dominant signal vanished, and this
    # 20x-boosted penalty became the loudest thing near the goal -- and it fires the
    # instant an env enters settle_proximity_m (5cm) while still noisy, which is
    # ALWAYS true this early in training (Policy/mean_noise_std was still ~0.52 at
    # iteration 2800, and that raw noise is injected straight into the commanded
    # joint target every step via _apply_action's `residual = filtered_actions *
    # residual_action_scale`, so real physical joint velocity rarely drops near zero
    # while PPO is still exploring). Net effect: crossing into the 5cm ring cost more
    # than staying just outside it, so the policy learned to hover at the boundary
    # instead of pushing through toward the (also newly-gated, see below)
    # goal_reached_bonus. Reverted to the original 0.05 -- this term's job (discourage
    # oscillating through the zone) is now handled directly and correctly by
    # goal_reached_max_vel gating the bonus itself; it doesn't need to also be a large
    # independent penalty.
    settle_velocity_penalty_scale: float = 0.05

    # 2026-07-28 (the actual, structural fix): goal_reached_bonus now ALSO requires
    # joint velocity under this threshold (rad/s, L2 norm over the 7 controlled
    # joints) -- see G1ArmResidualEnv._get_rewards' override docstring for the full
    # reasoning (the settle_velocity_penalty_scale bump above tried to out-WEIGH the
    # oscillate-for-reward exploit with a bigger separate penalty and failed; this
    # instead removes the exploit directly by coupling the bonus itself to stillness).
    # A first estimate, not yet validated/tuned against real data -- same "flag it,
    # don't pretend it's final" treatment settle_velocity_penalty_scale itself got
    # when it was first introduced for the base arm task.
    #
    # 2026-07-29: 0.5 was too strict relative to actual on-policy exploration noise --
    # Policy/mean_noise_std was still ~0.52 (raw action units) at iteration 2800, and
    # that noise is injected straight into the commanded joint target every step (see
    # settle_velocity_penalty_scale's 2026-07-29 note for the full mechanism), so real
    # physical joint velocity rarely dropped anywhere near 0.5 rad/s while PPO was
    # still exploring -- goal_reached_bonus fired ~0 times in the entire 2800-iteration
    # run as a direct result, leaving the policy with no gradient toward genuine
    # precision at all. Loosened 3x to tolerate typical exploration-driven jitter at
    # current noise levels while still excluding genuinely fast/uncontrolled motion --
    # a first estimate, same "flag it" caveat as before. As Policy/mean_noise_std
    # anneals over training, the *effective* precision needed to also satisfy
    # goal_hold_steps (15 CONSECUTIVE qualifying steps) should tighten on its own,
    # without needing to hand-tune a shrinking threshold.
    goal_reached_max_vel: float = 1.5

    # 2026-07-29: fixes a SECOND, deeper exploit found via clean (post eval-path-bug-fix)
    # per-checkpoint data -- success_rate collapsed from 95.8% (2500 iters) to 12.9%
    # (5000 iters) even though reach_rate and the per-episode best/final distance all
    # stayed roughly constant, and TensorBoard showed the reward STILL climbing the
    # whole time (not flat, ruling out "just needs more training" again). Root cause:
    # goal_reached_bonus pays the SAME amount whether it's the 1st or the 14th
    # consecutive qualifying step -- nothing in the reward gives a consecutive streak
    # any more value than the same number of qualifying steps scattered with gaps, so
    # PPO has no gradient pressure distinguishing "settle once, hold" from "hover near
    # the boundary, flickering in and out" -- confirmed directly in the per-step
    # dist-to-goal-over-time snapshots (metrics_wrappers.py's ArmMetricsCsvWrapper):
    # both the failing AND many nominally-"successful" long episodes oscillate in the
    # 1-3cm band for the ENTIRE episode rather than converging and staying. This scales
    # goal_reached_bonus by (1 + hold_counter * this), so a longer, unbroken streak is
    # worth strictly more than the same steps scattered -- breaking a streak now has a
    # real, escalating opportunity cost, not just the one missed step's flat bonus.
    # First estimate (not yet tuned): at hold_counter=14 (just before goal_hold_steps=15
    # would fire success/termination), this gives a ~3.1x multiplier.
    #
    # 2026-07-31: with terminate_on_success=False (below), episodes always run to
    # timeout, so an uncapped streak multiplier would grow without bound over a long
    # hold (300-step hold -> 46x -> ~2300/step by episode end) and distort the value
    # function. _get_rewards now caps the streak used for scaling at goal_hold_steps
    # (max multiplier 1 + 15*0.15 = 3.25x) -- keeps the escalating incentive while a
    # hold is being established, flat once it's genuinely held.
    hold_streak_bonus_scale: float = 0.15

    # 2026-07-31: THE fix for the success_rate collapse with more training (95.8% at
    # 2500 iters -> 12.9% at 5000 on run 2026-07-28_22-17-21). With termination on
    # success, completing the 15-consecutive-step hold ends the episode and forfeits
    # the remaining per-step goal_reached_bonus stream (~50-160/step x the ~200+
    # remaining steps) -- so the reward-optimal policy dithers at the 2cm boundary,
    # flickering in and out, and never completes the hold; more training = better
    # optimization of that objective = lower success rate. Independently found and
    # proven on main (arms_policy_finalisation.md, 2026-07-30: the Integrated run's
    # 2k->8k training improved every real metric while legacy success FELL 10.8%->
    # 6.5%; their NoTerm retrain is the same fix). Neither the velocity gate nor the
    # streak bonus removes this cliff -- the streak bonus actually sharpens the
    # exploit (a repeatable 14-step-streak/break/rebuild cycle out-earns terminating).
    # With False, staying in-zone strictly dominates leaving and there is nothing
    # left to gain from dithering. NOTE: Train/mean_episode_length stops being a
    # progress signal (always ~max); watch Episode_Reward/goal_reached_bonus and
    # Metrics/frac_envs_reached instead. Eval note: the metrics wrapper's legacy
    # `success` CSV column reads bool(terminated) and is 0% BY CONSTRUCTION under
    # this cfg -- judge checkpoints by the settled/tail-settle metrics
    # eval_arm_residual.py now reports (ported from main's eval_arm.py).
    terminate_on_success: bool = False

    # 2026-08-01: tiered joint-limit-proximity penalty, replacing the base task's
    # single 5%-margin/scale=1.0 "at limit" flag (the base's own
    # `_joint_limit_margin_fraction` is hardcoded 5% in G1ArmEnv.__init__, not a cfg
    # field, and this task's _get_rewards already fully overrides the base's
    # joint_limit_term computation — see that method's own docstring for why).
    # Motivated by a real, visually-confirmed observation (2026-08-01, checkpoint
    # 2026-07-31_07-52-15/model_5999.pt): the existing term contributed essentially
    # nothing in eval (-0.03 to -0.04 vs. a +154 goal_reached_bonus — 4000x smaller),
    # so nothing was actually discouraging the residual from resolving the arm's
    # redundancy (7 DOF for a 3-DOF position-only goal) via extreme, awkward-looking
    # joint configurations that still land on the same end-effector point. `at_limit`
    # (the outermost 5% band) stayed a low ~3.5% dwell-time average, but several
    # joints' ACHIEVED range in eval touched 94-111% of their soft hw range — the
    # existing single hard cutoff gave no anticipatory pressure before that band.
    # Three concentric margin bands (20%/10%/5% of each joint's span, ordered
    # widest-to-narrowest) with ESCALATING, CUMULATIVE scales approximate a soft
    # barrier that gets steeper the closer a joint sits to its real limit: a joint
    # within the outer 20% pays only the mildest penalty; one within the innermost
    # 5% (a subset of both outer bands) pays all three added together. The 5%-band
    # scale is kept at the base task's own default (1.0) specifically so behavior at
    # the tightest, most safety-relevant band doesn't regress from what's already
    # validated; the 10%/20% bands are new, lighter, first-pass estimates (not yet
    # tuned against real data — same "flag it, don't pretend it's final" treatment
    # every other first-estimate reward weight in this file gets). Deliberately NOT
    # touching null_space_penalty_scale (0.05, inherited unchanged) or anything in
    # the termination/streak/action-interface fixes above -- this is an isolated,
    # single-mechanism addition so its effect (if any) on Settled/Tail-settle can be
    # attributed cleanly, per this branch's own established practice.
    joint_limit_margin_fraction_20: float = 0.20
    joint_limit_margin_fraction_10: float = 0.10
    joint_limit_margin_fraction_05: float = 0.05
    joint_limit_penalty_scale_20: float = 0.1
    joint_limit_penalty_scale_10: float = 0.3
    joint_limit_penalty_scale_05: float = 1.0  # matches the base task's existing joint_limit_penalty_scale default


@configclass
class G1ArmResidualRightEnvCfg(G1ArmRightEnvCfg):
    """Mirror of G1ArmResidualLeftEnvCfg — see that class / the module docstring."""

    arm: str = "right"
    observation_space: int = 46
    action_space: int = 7
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0)
    residual_action_scale: float = 0.15
    apply_ik_tauff: bool = True
    goal_bounds_x_override: tuple[float, float] | None = (0.20, 0.31)
    settle_velocity_penalty_scale: float = 0.05  # see G1ArmResidualLeftEnvCfg's own comment
    goal_reached_max_vel: float = 1.5  # see G1ArmResidualLeftEnvCfg's own comment
    hold_streak_bonus_scale: float = 0.15  # see G1ArmResidualLeftEnvCfg's own comment
    terminate_on_success: bool = False  # see G1ArmResidualLeftEnvCfg's own comment
    joint_limit_margin_fraction_20: float = 0.20  # see G1ArmResidualLeftEnvCfg's own comment
    joint_limit_margin_fraction_10: float = 0.10  # see G1ArmResidualLeftEnvCfg's own comment
    joint_limit_margin_fraction_05: float = 0.05  # see G1ArmResidualLeftEnvCfg's own comment
    joint_limit_penalty_scale_20: float = 0.1  # see G1ArmResidualLeftEnvCfg's own comment
    joint_limit_penalty_scale_10: float = 0.3  # see G1ArmResidualLeftEnvCfg's own comment
    joint_limit_penalty_scale_05: float = 1.0  # see G1ArmResidualLeftEnvCfg's own comment


@configclass
class G1ArmResidualLeftEnvCfg_PLAY(G1ArmResidualLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 20.0


@configclass
class G1ArmResidualRightEnvCfg_PLAY(G1ArmResidualRightEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 20.0


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


class G1ArmResidualEnv(G1ArmEnv):
    """Residual RL on top of a per-episode-cached numerical IK baseline.

    See module docstring for the full design. Supports a single controlled arm only
    (``cfg.arm`` "left" or "right") — asserted in ``__init__``.
    """

    cfg: G1ArmResidualLeftEnvCfg | G1ArmResidualRightEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        assert self.n_arms == 1, (
            "G1ArmResidualEnv supports a single controlled arm only (cfg.arm='left' "
            "or 'right'), not 'both' -- see module docstring's scope decisions."
        )
        self._side = cfg.arm
        self._other_side = "right" if self._side == "left" else "left"

        # One shared, single-instance CPU solver -- see module docstring for why this
        # is solved once per episode (at reset) rather than every control step.
        self._arm_ik = G1ArmIK()

        all14_ids, _ = self.robot.find_joints(list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS))
        self._all14_joint_ids = torch.tensor(all14_ids, dtype=torch.long, device=self.device)
        self._default_q14 = (
            self.robot.data.default_joint_pos[0, self._all14_joint_ids].detach().cpu().numpy()
        )
        # Fixed rest-pose FK for the OTHER, uncontrolled arm (it never moves in this
        # task) -- passed to solve_ik as ITS target so its own cost term stays ~0 and
        # doesn't perturb the controlled arm's solve. Computed once; this arm's default
        # pose never changes.
        self._other_arm_default_fk = self._arm_ik.fk_wrist(self._default_q14, self._other_side)

        # Per-env cached IK baseline -- solved once per episode at reset (see
        # _solve_ik_baseline_for_envs), read every step in _apply_action. Initialized
        # to default pose / zero torque; overwritten before the first real step since
        # DirectRLEnv resets every env at construction.
        arm = self._arm_groups[0]
        self.ik_baseline_q = self.robot.data.default_joint_pos[:, arm["joint_tensor"]].clone()
        self.ik_baseline_tauff = torch.zeros_like(self.ik_baseline_q)

        # 2026-07-31: persistent commanded-target state for _apply_action's slew-rate
        # limiter (re-initialized to the arm's actual post-reset pose in _reset_idx).
        # Rate-limiting the COMMANDED-TARGET TRAJECTORY (target_t = target_{t-1} +
        # clamped delta) bounds how fast the PD setpoint can move -- the legitimate
        # smoothness/deployment concern behind the 2026-07-29 rate-limit fix --
        # WITHOUT the static-torque ceiling that fix accidentally introduced by
        # re-anchoring to the MEASURED position every step (|target - current| <=
        # 0.06 rad caps sustained PD torque at kp*0.06 = 2.4 Nm; main's
        # check_arm_static_torque_ceiling.py probe confirmed goal-box postures need
        # 4.5-5.7 Nm, and model_2000's own eval shows the residual closing an 8.6cm
        # baseline gap only by commanding well beyond it). Same integrated-target
        # principle as main's integrate_action_targets fix.
        self._cmd_targets = self.robot.data.default_joint_pos[:, arm["joint_tensor"]].clone()

    # ------------------------------------------------------------------
    # IK baseline (solved once per episode, at reset)
    # ------------------------------------------------------------------

    def _solve_ik_baseline_for_envs(self, env_ids: torch.Tensor):
        """Solve the fixed-for-the-episode IK baseline for each newly-reset env's
        freshly-sampled goal. CPU/single-instance, looped per env -- see module
        docstring for why this is cheap in aggregate despite being unbatched."""
        arm = self._arm_groups[0]
        side_slice = slice(0, 7) if self._side == "left" else slice(7, 14)

        for env_id in env_ids.tolist():
            goal_world = self.goal_positions[env_id, 0, :]
            root_pos = self.robot.data.root_pos_w[env_id]
            # Base is genuinely fixed (fix_root_link=True) at identity orientation --
            # "pelvis-relative" reduces to a pure translation here, matching
            # eval_arm_ik_fixed_base.py's own simplification (no rotation needed
            # since the root never actually rotates).
            target_pelvis = (goal_world - root_pos).detach().cpu().numpy()

            # Always re-seed from the same neutral default warm start, not the
            # previous env's leftover solver state (G1ArmIK's internal warm
            # start/smoothing filter is a SINGLE shared instance across all envs in
            # this loop -- reset() before every call keeps each env's solve
            # independent, matching Phase 0/1's own proven-reliable pattern of
            # solving fresh from a neutral pose rather than benefiting from
            # continuity that would otherwise leak between unrelated envs' goals).
            self._arm_ik.reset(self._default_q14)
            if self._side == "left":
                left_tf, right_tf = _se3(target_pelvis), _se3(self._other_arm_default_fk)
            else:
                left_tf, right_tf = _se3(self._other_arm_default_fk), _se3(target_pelvis)
            sol_q, sol_tauff = self._arm_ik.solve_ik(left_tf, right_tf, current_q=self._default_q14)

            self.ik_baseline_q[env_id] = torch.tensor(
                sol_q[side_slice], dtype=torch.float32, device=self.device
            )
            self.ik_baseline_tauff[env_id] = torch.tensor(
                sol_tauff[side_slice], dtype=torch.float32, device=self.device
            )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)  # samples new goals into self.goal_positions, resets joint state, etc.
        env_ids_tensor = (
            torch.tensor(env_ids, dtype=torch.long, device=self.device)
            if not isinstance(env_ids, torch.Tensor) else env_ids
        )
        self._solve_ik_baseline_for_envs(env_ids_tensor)
        # Re-seed the slew-limiter's persistent commanded target at the arm's actual
        # (randomized) post-reset pose, so the first commanded step is continuous from
        # wherever the episode starts -- see __init__'s _cmd_targets comment.
        arm = self._arm_groups[0]
        self._cmd_targets[env_ids_tensor] = self.robot.data.joint_pos[env_ids_tensor][:, arm["joint_tensor"]]

    # ------------------------------------------------------------------
    # Action: q_ik + bounded residual (not current_pose + delta)
    # ------------------------------------------------------------------

    def _apply_action(self):
        """2026-07-29: was missing max_action_delta_per_step entirely -- the residual
        could jump the commanded target by its FULL range (+-residual_action_scale,
        i.e. up to a 0.3 rad swing) in a single control step, only softened by
        action_filter_alpha's EMA (a low-pass, not a hard bound). Confirmed via clean
        per-checkpoint eval data (2026-07-29): every FAILED episode runs the full
        episode timeout with dist_to_goal rhythmically oscillating between <0.3cm and
        2-3.5cm on a multi-second cycle that never damps out. A rate limit bounds the
        velocity the PD controller is ever asked to chase.

        2026-07-31 (before this version was ever trained): the first rate-limit
        implementation clamped the target relative to the arm's CURRENT MEASURED
        position (G1ArmEnv._apply_action's legacy pattern) -- which re-introduces the
        static-torque ceiling main probe-confirmed breaks arm holding
        (check_arm_static_torque_ceiling.py: |target - current| <= 0.06 rad caps
        sustained PD torque at kp*0.06 = 2.4 Nm vs the 4.5-5.7 Nm goal-box postures
        need; the target would trail the sagging arm downward with no equilibrium
        until residual gravity <= 2.4 Nm + tau_ff). It would also have capped the
        residual's whole correction authority at 2.4 Nm above tau_ff -- model_2000's
        eval shows the residual closing an 8.6cm baseline gap to 1.15cm precisely by
        commanding well beyond the baseline (elbow at 102% of hw range), which that
        clamp forbids. This version slew-limits the COMMANDED-TARGET TRAJECTORY
        instead (self._cmd_targets, persistent state re-seeded at reset): same 0.06
        rad/step bound on how fast the setpoint moves (the actual smoothness concern),
        zero cap on the static bias it can accumulate -- the same integrated-target
        principle main's integrate_action_targets fix deploys with.
        """
        self._apply_root_wobble()

        alpha = float(self.cfg.action_filter_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        self.filtered_actions = alpha * self.actions + (1.0 - alpha) * self.filtered_actions

        residual = self.filtered_actions * self.cfg.residual_action_scale
        desired_targets = self.ik_baseline_q + residual
        delta = desired_targets - self._cmd_targets
        max_delta = float(self.cfg.max_action_delta_per_step)
        if max_delta > 0.0:
            delta = delta.clamp(min=-max_delta, max=max_delta)
        targets = self._cmd_targets + delta
        targets = targets.clamp(self._arm_hw_limits[:, 0], self._arm_hw_limits[:, 1])
        self._cmd_targets = targets
        self.robot.set_joint_position_target(targets, joint_ids=self.arm_joint_indices_tensor)
        if self.cfg.apply_ik_tauff:
            self.robot.set_joint_effort_target(self.ik_baseline_tauff, joint_ids=self.arm_joint_indices_tensor)
        self.previous_actions[:] = self.filtered_actions

        # lock_wrist_pitch_yaw (if enabled): same explicit-hold behavior as the base
        # task -- see G1ArmEnv._apply_action's identical comment.
        if self._locked_wrist_joint_indices is not None:
            self.robot.set_joint_position_target(
                self._locked_wrist_default_pos, joint_ids=self._locked_wrist_joint_indices
            )

    # ------------------------------------------------------------------
    # Observations: base 39-D + ik_baseline_error(7)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        arm = self._arm_groups[0]
        current = self.robot.data.joint_pos[:, arm["joint_tensor"]]
        # How far the (fixed-for-the-episode) IK baseline still is from where the arm
        # currently sits -- same "give the network the precomputed gap directly"
        # philosophy as the base task's own `error` (goal - ee_pos) feature, not
        # something the policy should have to re-derive from joint_pos alone.
        baseline_error = self.ik_baseline_q - current
        obs["policy"] = torch.cat([obs["policy"], baseline_error], dim=-1)
        obs["critic"] = torch.cat([obs["critic"], baseline_error], dim=-1)
        return obs

    # ------------------------------------------------------------------
    # Rewards: full copy of G1ArmEnv._get_rewards with THREE changes -- see below
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """Copy of ``G1ArmEnv._get_rewards`` with three behavioral changes:
        (1) ``goal_reached_bonus`` requires low joint velocity too, not just
        proximity, (2) that bonus is scaled by how long the current consecutive
        hold streak is, so a sustained hold is worth strictly more than the same
        number of qualifying steps scattered with gaps, and (3, added 2026-08-01)
        ``joint_limit_term`` uses a tiered/cumulative 20%-10%-5% margin penalty
        instead of the base task's single 5%-margin flag — see
        ``G1ArmResidualLeftEnvCfg``'s own comment for the motivation. Full method
        duplicated (not a small patch) because these changes need state computed
        earlier/differently than the base version does, and Python has no clean way
        to patch a few lines out of a parent method — if ``G1ArmEnv._get_rewards``
        changes later, this needs to be re-synced by hand.

        2026-07-28: found via direct TensorBoard inspection (not guessed) that a
        20x bump to ``settle_velocity_penalty_scale`` (see that field's own comment)
        did NOT fix a real problem: ``reach_rate_no_hold`` hit 99.5%+ while
        ``success_rate`` (reach + HOLD for ``goal_hold_steps`` consecutive steps)
        actually DROPPED (21%->13%) with more training, because
        ``goal_reached_bonus`` — paid out every single step ``reached=True``, with
        no stillness or consecutiveness requirement — was roughly 400x larger in
        typical per-step contribution than the settle penalty, making "oscillate
        rapidly through the 2cm zone" reward-optimal over genuinely settling. The
        scale bump changed the settle reward's NUMBER (jumped immediately, matching
        the multiplier) but not the underlying velocity behavior (flat for ~1900
        further iterations) — i.e. bigger-same-lever wasn't enough. This instead
        removes the exploit directly: touching the zone while still moving fast no
        longer pays the bonus at all, only genuinely being close AND still does.

        2026-07-29: fix (1) alone was NOT sufficient either -- clean, per-checkpoint
        eval data (after fixing an unrelated eval-output-path bug that had been
        silently mixing different checkpoints' episodes together) showed
        ``success_rate`` collapse from 95.8% (2500 iters) to 12.9% (5000 iters) even
        though ``reach_rate_no_hold`` and the per-episode best/final distance all
        stayed roughly constant, and TensorBoard showed the reward STILL climbing
        the whole time (ruling out "just needs more training" again). Confirmed via
        the per-step dist-to-goal-over-time snapshots
        (``metrics_wrappers.py``'s ``ArmMetricsCsvWrapper``): both failing AND many
        nominally-"successful" long episodes oscillate in the 1-3cm band for the
        ENTIRE episode rather than converging and staying -- present even at 2500
        iterations (a minority of harder episodes), just far more prevalent by 5000.
        Root cause: ``goal_reached_bonus`` paid the SAME amount whether it's the 1st
        or the 14th consecutive qualifying step, so nothing gave a genuine streak any
        more value than the same steps scattered with gaps -- fix (2) scales the
        bonus by the current streak length, so breaking one now has a real,
        escalating opportunity cost.

        The ``success``/``_hold_counter`` definition itself is UNCHANGED (still
        pure proximity, matching the existing, already-meaningful ``goal_hold_steps``
        criterion, and keeping it comparable to every number reported so far) — only
        the reward terms that were creating the exploits are changed.
        """
        total = torch.zeros(self.num_envs, device=self.device)
        all_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        position_dist_term = torch.zeros(self.num_envs, device=self.device)
        position_exp_term = torch.zeros(self.num_envs, device=self.device)
        goal_bonus_term = torch.zeros(self.num_envs, device=self.device)
        torso_proximity_term = torch.zeros(self.num_envs, device=self.device)
        null_space_term = torch.zeros(self.num_envs, device=self.device)
        settle_term = torch.zeros(self.num_envs, device=self.device)
        mean_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

        torso_pos = self.robot.data.body_pos_w[:, self._torso_body_idx, :]
        margin = self.cfg.torso_proximity_margin_m

        for i, arm in enumerate(self._arm_groups):
            ee_pos = self.robot.data.body_pos_w[:, arm["ee_idx"], :]
            dist = torch.norm(self.goal_positions[:, i, :] - ee_pos, dim=-1)
            mean_dist_to_goal += dist
            position_dist_term += -dist * self.cfg.position_reward_scale
            position_exp_term += torch.exp(-dist / self.cfg.position_reward_exp_sigma) * self.cfg.position_reward_exp_scale

            reached = dist < self.cfg.goal_threshold
            all_reached &= reached  # success/hold_counter definition: proximity only, UNCHANGED

            jt = arm["joint_tensor"]
            joint_vel_norm = torch.norm(self.robot.data.joint_vel[:, jt], dim=-1)

            # THE fix: goal_reached_bonus now ALSO requires low velocity -- see this
            # method's own docstring for why.
            still_enough = joint_vel_norm < self.cfg.goal_reached_max_vel
            goal_bonus_term += (reached & still_enough).float() * self.cfg.goal_reached_bonus

            torso_dist = torch.norm(ee_pos - torso_pos, dim=-1)
            torso_proximity_term += -torch.clamp(margin - torso_dist, min=0.0) * self.cfg.torso_proximity_penalty_scale

            ref_pose = self.robot.data.default_joint_pos[:, jt]
            pose_deviation = torch.norm(
                (self.robot.data.joint_pos[:, jt] - ref_pose) * arm["null_space_weight"], dim=-1
            )
            null_space_term += -pose_deviation * self.cfg.null_space_penalty_scale

            within_settle_zone = (dist < self.cfg.settle_proximity_m).float()
            settle_term += -joint_vel_norm * within_settle_zone * self.cfg.settle_velocity_penalty_scale

        mean_dist_to_goal /= self.n_arms

        # THE second fix: scale goal_bonus_term by how long the CURRENT streak would
        # become if this step counts (prospective hold_counter, computed here but not
        # yet committed to self._hold_counter -- that still happens at the end of this
        # method, unchanged). A single arm only (asserted in __init__), so applying
        # this to the already-accumulated total is equivalent to doing it per-arm
        # inside the loop above. Zero whenever not currently reached (all_reached=False
        # resets the prospective counter to 0 too), growing with each additional
        # consecutive qualifying step -- see this method's docstring / the cfg field's
        # own comment for the full reasoning.
        prospective_hold_counter = torch.where(
            all_reached, self._hold_counter + 1, torch.zeros_like(self._hold_counter)
        )
        # 2026-07-31: cap the streak used for SCALING at goal_hold_steps -- with
        # terminate_on_success=False episodes run to timeout, and an uncapped
        # multiplier would grow ~46x over a full-episode hold (see the cfg field's
        # 2026-07-31 note). The committed self._hold_counter itself stays uncapped
        # (success definition unchanged).
        streak_for_scaling = prospective_hold_counter.clamp(max=self.cfg.goal_hold_steps)
        goal_bonus_term = goal_bonus_term * (1.0 + streak_for_scaling.float() * self.cfg.hold_streak_bonus_scale)

        action_smoothness_term = -torch.norm(self.previous_actions, dim=-1) * self.cfg.action_smoothness_scale

        # 2026-08-01: tiered joint-limit-proximity penalty (three concentric margin
        # bands, cumulative) -- see G1ArmResidualLeftEnvCfg's own comment for the full
        # motivation/design. Replaces the base task's single-margin/single-scale
        # at_limit_mask computation entirely (this method already fully overrides
        # G1ArmEnv._get_rewards, so nothing upstream depends on the old computation).
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_indices_tensor]
        limits = self._arm_hw_limits
        span = limits[:, 1] - limits[:, 0]

        def _at_margin(frac: float) -> torch.Tensor:
            margin = frac * span
            return (joint_pos < limits[:, 0] + margin) | (joint_pos > limits[:, 1] - margin)

        at_limit_20 = _at_margin(self.cfg.joint_limit_margin_fraction_20)
        at_limit_10 = _at_margin(self.cfg.joint_limit_margin_fraction_10)
        at_limit_05 = _at_margin(self.cfg.joint_limit_margin_fraction_05)
        joint_limit_term = -(
            at_limit_20.float().sum(-1) * self.cfg.joint_limit_penalty_scale_20
            + at_limit_10.float().sum(-1) * self.cfg.joint_limit_penalty_scale_10
            + at_limit_05.float().sum(-1) * self.cfg.joint_limit_penalty_scale_05
        )
        # Metrics/mean_joints_at_limit keeps its old meaning (tightest 5% band only —
        # matches the base task's diagnostic convention) even though the reward now
        # also considers the two looser bands.
        at_limit = at_limit_05.float().sum(-1)

        total = (
            position_dist_term + position_exp_term + goal_bonus_term + torso_proximity_term
            + null_space_term + action_smoothness_term + joint_limit_term + settle_term
        )

        self.extras["log"] = {
            "Episode_Reward/position_dist": position_dist_term.mean(),
            "Episode_Reward/position_exp_bonus": position_exp_term.mean(),
            "Episode_Reward/goal_reached_bonus": goal_bonus_term.mean(),
            "Episode_Reward/torso_proximity": torso_proximity_term.mean(),
            "Episode_Reward/null_space": null_space_term.mean(),
            "Episode_Reward/action_smoothness": action_smoothness_term.mean(),
            "Episode_Reward/joint_limit": joint_limit_term.mean(),
            "Episode_Reward/settle": settle_term.mean(),
            "Metrics/mean_dist_to_goal_cm": mean_dist_to_goal.mean() * 100.0,
            "Metrics/mean_joints_at_limit": at_limit.mean(),
            "Metrics/frac_envs_reached": all_reached.float().mean(),
            "Curriculum/goal_bounds_frac": torch.tensor(
                getattr(self, "_last_goal_curriculum_frac", 1.0), device=self.device
            ),
        }

        # Same value already computed above for the streak-scaled bonus -- committing
        # it here now that this step's reward is fully finalized.
        self._hold_counter = prospective_hold_counter
        self.successes = self._hold_counter >= self.cfg.goal_hold_steps
        return total
