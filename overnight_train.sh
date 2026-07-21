#!/usr/bin/env bash
# Step 2c of the 2026-07-20/21 recovery (definitive_next_steps.md) — TorsoClip +
# strengthened orientation/hip rewards, ONE seed, ONE variant (2026-07-21).
#
# Context: TorsoClip (+/-30deg) and TorsoLock (0deg) both retrained and evaluated
# 2026-07-20/21. Clip clearly beat Lock (policy-mode integration fall rate much lower,
# and — decisively — a live g1_full_demo.py visual check of Lock showed a wide, unstable
# stance with obvious tilt even standing still). Lock is DROPPED as a direction as of
# today. But Clip's own idle pose isn't good either: gate-eval (arms frozen, 0% falls)
# tilt went 8.4deg (GainMatch, unclipped) -> 37.2deg (Clip) even though torso itself is
# now correctly bounded (60.3deg -> 32.2deg). joint_diagnostics.csv shows why: hip_roll
# frac_of_soft_limit_used jumped 10.9%/27.1% -> 60.7%/10.9% (GainMatch -> Clip,
# left/right) — a live visual check confirmed legs spread far wider than GainMatch's
# already-reasonable pose. Root cause: two reward terms that ARE already active
# (flat_orientation_l2 -1.0, joint_deviation_hip -0.1) are inherited unchanged from the
# WALKING-tuned G1RoughEnvCfg and were never retuned for standing — they were only ever
# "good enough" while torso rotation acted as a free, unpenalized escape hatch. Capping
# that hatch (correctly) didn't remove the underlying compensation need, it just moved it
# onto the next-cheapest lever (hip abduction). See
# G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipOrientHipEnvCfg's docstring
# in g1_locomotion_env_cfg.py for the full writeup.
#
# Fix tested here: strengthen both terms for standing specifically —
# flat_orientation_l2 -1.0 -> -3.0, joint_deviation_hip -0.1 -> -0.5. Starting points, not
# tuned values (same spirit as base_height_l2's own -10.0 starting point) — check reward
# curves after training. Built on TorsoClip (not Lock) — keeping some torso authority
# while ALSO pricing in tilt/hip-deviation properly is the hypothesis under test.
#
# ONE seed, ONE variant this time (not a multi-config sweep like the Clip-vs-Lock night):
# a single, well-motivated next step the user explicitly asked to run right now, not a
# comparison requiring several arms. Revisit with a second seed once this direction looks
# right, per the project's proven seed-variance history (2.1% vs 100% fall rate on an
# identical config, different seed).
#
# Train -> GATE (native, arms frozen; pass = fall_rate ~0) -> native eval (own
# disturbance) -> TWO integration evals, BOTH passing --torso_clip_deg 30 (the
# eval_full_demo.py fix added 2026-07-21 — that script builds its env from the walking
# lineage and never inherits the standing lineage's torso clip on its own; omitting the
# flag would silently test this checkpoint under an unclipped action space it was never
# trained on, exactly the bug that made last night's first integration numbers useless).
#
# Deliberately does NOT use `set -e` — log FAILED lines instead of aborting the queue.
# Deliberately no shutdown/poweroff at the end.
#
# Usage:
#   cd ~/Elm/Code/g1_locomotion
#   nohup ./overnight_train.sh &
#   (or run in tmux/screen and detach)

set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
STANDING_ITERS=6000
WALKING_CKPT="chosen_checkpoints/walking_latest.pt"
ARM_CKPT="chosen_checkpoints/arm_left_latest.pt"
SEED=42
TORSO_CLIP_DEG=30

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight_$(date +%Y-%m-%d_%H-%M-%S).log"

# NOTE: no `set -u` (only pipefail) — conda activate sources Isaac Sim's
# setup_conda_env.sh, which references $ZSH_VERSION with no default and isn't
# nounset-safe.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

exec > >(tee -a "$LOG_FILE") 2>&1

run_step() {
    local desc="$1"; shift
    echo ""
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $desc"
    echo "=============================================================="
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED (exit $status): $desc — continuing with the rest of the queue"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $desc"
    fi
}

# $1 = experiment_name (e.g. standing/g1_locomotion_flat) -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

RUN_NAME="ikreach_height_intent_gainmatch_torsoclip30_orienthip"
TASK="G1-Locomotion-Standing-Flat-IKReach-Height-Intent-GainMatch-TorsoClip-OrientHip-v0"
ENV_CFG_KEY="ikreachintentgainmatchtorsocliporienthip"

echo "Step 2c (TorsoClip + strengthened orientation/hip rewards, 1 seed, $STANDING_ITERS iters) starting $(date). Full log: $LOG_FILE"

run_step "Train: standing $RUN_NAME (seed=$SEED, $STANDING_ITERS iters, task=$TASK)" \
    python scripts/rsl_rl/train.py --task "$TASK" --headless \
        --max_iterations "$STANDING_ITERS" --seed "$SEED" --run_name "$RUN_NAME"

RUN_DIR=$(latest_run_dir "standing/g1_locomotion_flat")
CKPT=$(latest_checkpoint "$RUN_DIR")

if [ -z "$CKPT" ]; then
    echo "[WARN] No fresh checkpoint found under $RUN_DIR for $RUN_NAME — aborting eval steps."
else
    # GATE FIRST (~90s): native env with arms frozen at default. no_reach_prob=0.15 means
    # this should pass by construction (fall_rate ~0); if it doesn't, everything below is
    # pre-explained and this checkpoint is a discard regardless of what it shows.
    run_step "Gate (native, arms frozen): $RUN_NAME" \
        python validation/eval_standing_ikreach.py --checkpoint "$CKPT" --env_cfg "$ENV_CFG_KEY" --freeze_arms --headless
    run_step "Eval (native): $RUN_NAME" \
        python validation/eval_standing_ikreach.py --checkpoint "$CKPT" --env_cfg "$ENV_CFG_KEY" --headless
    run_step "Eval (integration, event mode — self-consistency check): $RUN_NAME" \
        python validation/integration_validation/eval_full_demo.py \
            --standing_checkpoint "$CKPT" --walking_checkpoint "$WALKING_CKPT" \
            --arm_checkpoint "$ARM_CKPT" --arm_driver event --torso_clip_deg "$TORSO_CLIP_DEG" --num_envs 32 --headless
    run_step "Eval (integration, POLICY mode — the decisive real-deployment test): $RUN_NAME" \
        python validation/integration_validation/eval_full_demo.py \
            --standing_checkpoint "$CKPT" --walking_checkpoint "$WALKING_CKPT" \
            --arm_checkpoint "$ARM_CKPT" --arm_driver policy --torso_clip_deg "$TORSO_CLIP_DEG" --num_envs 32 --headless

    echo "$RUN_NAME checkpoint: $CKPT"
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Step 2c (TorsoClip + OrientHip) complete."
echo "Full log: $LOG_FILE"
echo "Native eval output:      <checkpoint_dir>/ikreach_eval/<timestamp>/summary.csv"
echo "  + joint_diagnostics.csv in the same folder — THE NUMBERS THAT MATTER THIS TIME:"
echo "  mean_max_tilt_deg (gate/idle) should drop from Clip's 37.2deg toward GainMatch's"
echo "  8.4deg or better, and hip_roll frac_of_soft_limit_used should drop from Clip's"
echo "  60.7%/10.9% toward GainMatch's 10.9%/27.1% or better — without torso_joint"
echo "  creeping back up past ~32deg (it shouldn't, the clip is unchanged and hard)."
echo "Integration eval output: validation/integration_validation/<timestamp>/*/*_summary.csv"
echo "  (both runs used --torso_clip_deg 30 — verify mean_max_abs_torso_deg reads ~30-32,"
echo "  NOT 150, before trusting anything else in these files)."
echo "COMPARE against TorsoClip's own numbers (gate tilt 37.2deg/torso 32.2deg; native"
echo "26.3% falls; policy-mode 0%/2.1%/24.7%/22.7% left/left_edge/right/right_edge) —"
echo "the fall-rate numbers should NOT regress just because the reward changed; if they"
echo "do, -3.0/-0.5 may be fighting legitimate disturbance-recovery motion too hard."
echo "If this looks like a clear improvement: do a live g1_full_demo.py visual check"
echo "(--torso_clip_deg 30) before promoting to chosen_checkpoints/standing_latest.pt"
echo "(ask first) — the whole reason this run exists is a visual complaint, so a visual"
echo "check is the real pass/fail bar, not just the numbers."
echo "This is ONE seed — a promising direction should get a second seed before being"
echo "trusted as final, per the project's proven seed-variance history."
echo "=============================================================="
