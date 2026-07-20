#!/usr/bin/env bash
# Two-branch posture experiment + transfer-collapse bisect (2026-07-15 evening, replaces
# the single consolidated_torso queue drafted earlier today):
#
#   1. consolidated_torso  — consolidated + joint_deviation_torso -0.05: posture *penalty*.
#      "Is the rotated-torso bracing posture load-bearing or an unpenalized habit?"
#   2. consolidated_intent — consolidated + the 10-D arm-intent observation: posture
#      *information*. The standing policy sees the arm's commanded targets and can brace
#      per-target instead of committing to one fixed twist.
#      Both are single-variable branches off the same consolidated baseline (left reach
#      falls 78% -> 2.1%, but torso rotation cost most of the inner reach workspace and
#      the right side still falls ~55% — plan.md §3).
#   3. Transfer-collapse bisect (cheap, ~2min each): the leg_symmetry checkpoint scores 0%
#      falls natively but 100% in the integration env even standing still (same collapse
#      as dwell/torso before it). The two confirmed env differences are (a) arms actively
#      reaching natively vs statically held in integration, (b) push_robot on natively vs
#      off in integration. --freeze_arms / --no_push toggle these inside the native env,
#      one at a time, with the consolidated checkpoint as the does-transfer control.
#
# Reading the results:
#   - torso branch:  falls stay ~2% AND mean_max_abs_torso_deg (new column) drops AND left
#     reach success recovers toward ~90%+ concluded => rotation was a habit, penalty wins.
#     Falls return => rotation was load-bearing; intent branch (or nothing) is the answer.
#   - intent branch: same success criteria, plus watch the right-side buckets — intent is
#     the lever expected to fix the one-sided-solution pattern, so right reach falls
#     dropping below ~20% would be its distinctive signature.
#   - bisect: symmetry ckpt SURVIVES native+freeze_arms => static arms aren't the killer,
#     suspect shifts to the env base / command internals. Symmetry ckpt COLLAPSES with
#     freeze_arms (while consolidated control survives) => these checkpoints genuinely
#     cannot stand still without arm motion — a training-distribution hole, fixable.
#
# Deliberately does NOT use `set -e`: one failed step shouldn't silently kill the rest of
# an unattended queue. Each step's pass/fail is logged instead; check the log for FAILED
# lines afterward.
#
# Deliberately no shutdown/poweroff command anywhere in this script — leave the machine
# running when it's done.
#
# Usage:
#   cd ~/Elm/Code/g1_locomotion
#   nohup ./overnight_train.sh &
#   (or just run it in a tmux/screen session and detach)

set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
STANDING_ITERS=6000
WALKING_CKPT="chosen_checkpoints/walking_latest.pt"
ARM_CKPT="chosen_checkpoints/arm_left_latest.pt"

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight_$(date +%Y-%m-%d_%H-%M-%S).log"

# NOTE: no `set -u` here (only pipefail) — conda activate sources Isaac Sim's
# setup_conda_env.sh, which references $ZSH_VERSION with no default and isn't
# nounset-safe. `set -u` made that fatal on the first run before anything even
# started training. The rest of this script doesn't rely on unset-variable
# checking for correctness, so it's simplest to just leave it off entirely.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Mirror everything to a log file too, so it survives even if the terminal is closed.
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

echo "Intent branch starting $(date). Full log: $LOG_FILE"

# ---------------------------------------------------------------------------
# ALREADY COMPLETED in the 2026-07-15_08-37-40 run (do not re-run):
#   - Bisect A/B/control — DECISIVE: symmetry ckpt falls 100% in its OWN native env with
#     arms frozen at default (identical with/without push; consolidated control 0%).
#     The transfer collapse is a training-distribution hole ("standing with motionless
#     arms" is OOD for the analytic-IK-era checkpoints), not an env-config bug.
#   - consolidated_torso train + native eval (0% falls, height 0.727 — but
#     mean |torso| ~48.5deg: the -0.05 penalty did NOT remove the rotation).
#   - Its integration eval crashed on an obs_groups bug in the new checkpoint-inferred
#     loader (fixed 2026-07-15 afternoon) — run it manually, see plan.md.
#
# Remaining queue below: the consolidated_intent branch only. Updated 2026-07-15 after
# the torso branch's integration collapse (100% falls even standing still — torso-penalty
# lever now considered dead after two independent failures):
#   - The intent config gained no_reach_prob=0.15 (15% of episodes hold both arms at
#     default — closes the static-arm distribution hole the bisect proved lethal).
#   - A freeze_arms GATE eval runs before the normal evals: pass = fall_rate ~0.
# ---------------------------------------------------------------------------
run_step "Train: standing consolidated_intent ($STANDING_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-Consolidated-Intent-v0 --headless \
        --max_iterations "$STANDING_ITERS" --run_name consolidated_intent
INTENT_RUN_DIR=$(latest_run_dir "standing/g1_locomotion_flat")
INTENT_CKPT=$(latest_checkpoint "$INTENT_RUN_DIR")

if [ -n "$INTENT_CKPT" ]; then
    # GATE FIRST (90s): native env with arms frozen at default — the condition that
    # kills collapse-group checkpoints in ~2.5s (see the 2026-07-15 bisect). With
    # no_reach_prob=0.15 in the intent config this should pass by construction
    # (fall_rate ~0 in its summary.csv); if it doesn't, the integration numbers
    # below are pre-explained and the checkpoint is a discard.
    run_step "Gate (native, arms frozen): standing consolidated_intent" \
        python validation/eval_standing_ikreach.py --checkpoint "$INTENT_CKPT" --env_cfg intent --freeze_arms --headless
    run_step "Eval (native): standing consolidated_intent" \
        python validation/eval_standing_ikreach.py --checkpoint "$INTENT_CKPT" --env_cfg intent --headless
    run_step "Eval (integration): standing consolidated_intent" \
        python validation/integration_validation/eval_full_demo.py \
            --standing_checkpoint "$INTENT_CKPT" --walking_checkpoint "$WALKING_CKPT" \
            --arm_checkpoint "$ARM_CKPT" --num_envs 32 --headless
else
    echo "[WARN] No fresh checkpoint found under $INTENT_RUN_DIR for consolidated_intent — skipping both evals."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Posture two-branch + transfer bisect complete."
echo "Full log: $LOG_FILE"
echo "consolidated_intent checkpoint: $INTENT_CKPT"
echo "Native eval output:      <checkpoint_dir>/ikreach_eval/<timestamp>/summary.csv"
echo "Integration eval output: validation/integration_validation/<timestamp>/*/*_summary.csv"
echo "Baselines to compare against: consolidated integration run 2026-07-15_05-13-58"
echo "(left falls 2.1%, left concluded success ~16%, right falls 54.7%) and height_reward"
echo "re-baseline 2026-07-15_01-51-10 (left concluded success ~96%). Watch the new"
echo "mean_max_abs_torso_deg column in the *_stability summaries."
echo "=============================================================="
