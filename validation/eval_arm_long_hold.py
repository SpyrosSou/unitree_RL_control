# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Long-hold verification for the 200/20-gain reference arm checkpoint (or any legacy
32-D checkpoint).

goal_hold_steps=15 (0.5s at 30Hz) only proves the arm can touch-and-briefly-hold a
target long enough to trigger early termination; it says nothing about whether that
hold is *sustained*. This runs one fixed-goal episode per env, 45s long
(G1ArmLeftEnvCfg_LongHold200 — early-termination-on-success disabled), and reports:

  - what fraction of envs ever satisfy the hold criterion (env.successes) at all
  - of those, what fraction of the time *after* first satisfying it they continue to
    satisfy it (vs. drift out and re-enter repeatedly)
  - the longest continuous in-threshold streak (env._hold_counter) actually achieved,
    in seconds — goal_hold_steps=15 is the minimum (~0.5s); this checks how far past
    that minimum the real behavior goes

2026-07-26: built to answer an open question from this session's gain-tuning work —
does the 99.98%-success 200/20-gain reference represent genuine long-term stability, or
just barely clearing the 0.5s bar used as the training/eval success criterion?

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_arm_long_hold.py \\
        --checkpoint logs/rsl_rl/arms/left/2026-07-22_06-20-55/model_2999.pt \\
        --headless

Output:
    Prints the stats above and writes them to <checkpoint_dir>/long_hold_summary.txt.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Long-hold verification for a legacy 32-D arm-policy checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained arm checkpoint (.pt).")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs.")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible rollouts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import os

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
import torch
from g1_locomotion.tasks.manager_based.g1_arm.agents.rsl_rl_ppo_cfg import G1ArmLeftPPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import G1ArmEnv, G1ArmLeftEnvCfg_LongHold200

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def main():
    env_cfg = G1ArmLeftEnvCfg_LongHold200()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    print(f"[LongHold] Checkpoint: {args_cli.checkpoint}")
    print("[LongHold] Building simulation...")
    base_env = G1ArmEnv(env_cfg)
    device = base_env.device

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        wrapped_env = RslRlVecEnvWrapper(base_env)

        agent_cfg = G1ArmLeftPPORunnerCfg()
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)

        num_steps = int(base_env.max_episode_length)
        n = args_cli.num_envs
        print(
            f"[LongHold] num_envs={n}  episode_length_s={env_cfg.episode_length_s}  "
            f"num_steps={num_steps}"
        )

        first_reach_step = torch.full((n,), -1, dtype=torch.long, device=device)
        max_hold_streak = torch.zeros((n,), dtype=torch.long, device=device)
        steps_in_hold_after_first_reach = torch.zeros((n,), dtype=torch.long, device=device)

        # Lenient metric (2026-07-26): "success" = held for goal_hold_steps=15 CONSECUTIVE
        # steps is a training-convenience gate, not a real application requirement — the
        # user is fine with "around 2cm," not specifically a clean unbroken 15-step block.
        # hold_counter >= 1 is exactly "within goal_threshold (2cm) THIS step" (it resets
        # to 0 the instant the arm drifts out — see _get_rewards' torch.where), so this
        # gives a per-step in/out-of-tolerance signal without requiring any consecutive-
        # run length at all. Tracks: (a) did it ever get within 2cm even once, (b) what
        # fraction of the ENTIRE window was spent within 2cm, (c) of the time after first
        # ever touching 2cm, what fraction was spent within 2cm (does it broadly hover
        # near the goal once it gets close, even if it dips out and back repeatedly).
        first_touch_step = torch.full((n,), -1, dtype=torch.long, device=device)
        steps_within_threshold_total = torch.zeros((n,), dtype=torch.long, device=device)
        steps_within_threshold_after_first_touch = torch.zeros((n,), dtype=torch.long, device=device)

        obs, _ = wrapped_env.reset()
        for step in range(num_steps):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

            successes = base_env.successes
            hold_counter = base_env._hold_counter
            within_threshold = hold_counter >= 1

            max_hold_streak = torch.maximum(max_hold_streak, hold_counter)

            newly_reached = (first_reach_step < 0) & successes
            first_reach_step[newly_reached] = step

            already_reached = first_reach_step >= 0
            steps_in_hold_after_first_reach += (successes & already_reached).long()

            steps_within_threshold_total += within_threshold.long()
            newly_touched = (first_touch_step < 0) & within_threshold
            first_touch_step[newly_touched] = step
            already_touched = first_touch_step >= 0
            steps_within_threshold_after_first_touch += (within_threshold & already_touched).long()

    control_dt = env_cfg.episode_length_s / num_steps
    ever_reached = first_reach_step >= 0
    n_reached = int(ever_reached.sum().item())

    lines = [
        f"Checkpoint: {args_cli.checkpoint}",
        f"num_envs={n}  episode_length_s={env_cfg.episode_length_s}  num_steps={num_steps}",
        f"envs that ever satisfied hold criterion: {n_reached}/{n} ({n_reached / n:.1%})",
    ]

    if n_reached > 0:
        steps_since_first_reach = (num_steps - first_reach_step[ever_reached]).float()
        sustain_frac = steps_in_hold_after_first_reach[ever_reached].float() / steps_since_first_reach
        max_streak_s = max_hold_streak[ever_reached].float() * control_dt

        lines += [
            f"mean fraction of post-first-reach time spent in-threshold: {sustain_frac.mean().item():.1%}",
            f"median fraction: {sustain_frac.median().item():.1%}",
            f"mean longest continuous hold streak: {max_streak_s.mean().item():.2f}s "
            f"(goal_hold_steps=15 minimum -> {15 * control_dt:.2f}s)",
            f"p10 longest continuous hold streak: {max_streak_s.quantile(0.10).item():.2f}s",
            f"frac of reached envs with max streak > 5s: {(max_streak_s > 5.0).float().mean().item():.1%}",
            f"frac of reached envs with max streak > 20s: {(max_streak_s > 20.0).float().mean().item():.1%}",
        ]
    else:
        lines.append("No env ever reached the hold criterion in this window — check checkpoint/env cfg match.")

    # Lenient metric: "success" = within goal_threshold at all, no consecutive-run
    # requirement — see the tracking comment above for why this is the more
    # application-relevant definition than the strict 15-consecutive-step gate.
    ever_touched = first_touch_step >= 0
    n_touched = int(ever_touched.sum().item())
    overall_frac_within_threshold = steps_within_threshold_total.float() / num_steps

    lines += [
        "",
        "--- Lenient metric: within 2cm at all (no consecutive-hold requirement) ---",
        f"envs that ever came within {env_cfg.goal_threshold * 100:.0f}cm at least once: "
        f"{n_touched}/{n} ({n_touched / n:.1%})",
        f"mean fraction of the ENTIRE {env_cfg.episode_length_s:.0f}s window spent within threshold "
        f"(all envs): {overall_frac_within_threshold.mean().item():.1%}",
        f"median fraction of the entire window spent within threshold (all envs): "
        f"{overall_frac_within_threshold.median().item():.1%}",
    ]
    if n_touched > 0:
        steps_since_first_touch = (num_steps - first_touch_step[ever_touched]).float()
        touch_sustain_frac = steps_within_threshold_after_first_touch[ever_touched].float() / steps_since_first_touch
        lines += [
            f"of envs that ever touched threshold, mean fraction of post-first-touch time "
            f"still within threshold: {touch_sustain_frac.mean().item():.1%}",
            f"median: {touch_sustain_frac.median().item():.1%}",
        ]

    for line in lines:
        print(f"[LongHold] {line}")

    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    out_path = os.path.join(checkpoint_dir, "long_hold_summary.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[LongHold] Summary written to: {out_path}")

    base_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
