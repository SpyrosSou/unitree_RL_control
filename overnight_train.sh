#!/usr/bin/env bash
# Reusable overnight/unattended training template. Edit the TASK QUEUE section below
# for whatever's next (check policy_status.md's "Deferred / future items" for
# candidates), then run it. Consolidated 2026-07-27 from a series of one-off
# `overnight_train_N.sh` / `overnight_arms_walking.sh` / `goal_curriculum_run.sh`
# scripts (each already run to completion, findings folded into policy_status.md) —
# keep exactly this one file going forward instead of a new numbered copy each time.
#
# Log-and-continue on a step failure (not `set -e`) so one broken step doesn't waste
# the rest of the window.
set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
SEED=42

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight_$(date +%Y-%m-%d_%H-%M-%S).log"

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

# $1 = experiment_name (matches --experiment_name / the logs/rsl_rl/<name>/ path) -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

echo "Overnight run starting $(date). Full log: $LOG_FILE"

# ==========================================================================================
# TASK QUEUE — 2026-07-28 overnight run
# ==========================================================================================
# CORRECTED 2026-07-29: RSL-RL's OnPolicyRunner.learn() computes
# tot_iter = current_learning_iteration (from the resumed checkpoint) + num_learning_iterations
# (confirmed directly from on_policy_runner.py source) — i.e. --max_iterations is ADDITIVE
# on a --resume, NOT an absolute target. The first run of this script used the target
# totals (10000/26000) directly as --max_iterations, which actually produced totals of
# ~12000 (1999+10000) and would have produced ~47000 (20996+26000) — caught after the arm
# step confirmed the 12000 number, stopped before the walking step ever started. Flag
# values below are corrected to actually land at the intended totals.
#
# 1) Arm: continue best_combined (currently 2000 iters, 28.20%/32.85% success, plateaued
#    at that iteration count per policy_status.md) to a TOTAL of ~10000 (flag: 8001 more,
#    1999+8001=10000) — confirms or kills the "just needs more iterations" theory once and
#    for all (arm_left_latest already suggested no, at ~8000 iters/same result, but that
#    was a different, older recipe — this tests the actual BestCombined recipe directly).
# 2) Walking: continue the FROZEN "chosen" checkpoint (LooseHeightNoStep, model_20996 —
#    good stepping/drift, poor turn_left — NOT the joint_mirror variant, which traded
#    that win away for better turn_left/drift, see policy_status.md's 2026-07-28 tradeoff
#    note) another 5000 iterations (flag: 5000 more, 20996+5000=25996, same recipe, no new
#    reward changes). Deliberately NOT training joint_mirror further tonight — that line
#    of work is paused pending the command_norm-gating fix (see policy_status.md), not
#    abandoned.
# Each training step is followed by its own eval so results are ready to read in the
# morning, not just raw checkpoints.

run_step "Train: arm best_combined -> ~10000 iters total (+8001)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-BestCombined-v0 \
        --resume --checkpoint logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt \
        --resume_new_dir --max_iterations 8001 --headless --seed "$SEED"

ARM_RUN_DIR=$(latest_run_dir "arms/best_combined")
ARM_CKPT=$(latest_checkpoint "$ARM_RUN_DIR")
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm best_combined @ ~10000 iters" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --log_std --headless
else
    echo "[SKIP] arm eval — no checkpoint found in $ARM_RUN_DIR."
fi

run_step "Train: walking LooseHeightNoStep -> ~26000 iters total (+5000)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-LooseHeightNoStep-v0 \
        --resume --checkpoint logs/rsl_rl/walking/arm_disturbance/2026-07-28_11-41-33/model_20996.pt \
        --resume_new_dir --max_iterations 5000 --headless --seed "$SEED"

WALK_RUN_DIR=$(latest_run_dir "walking/arm_disturbance")
WALK_CKPT=$(latest_checkpoint "$WALK_RUN_DIR")
if [ -n "$WALK_CKPT" ]; then
    run_step "Eval: walking LooseHeightNoStep @ ~26000 iters" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT" \
            --arm_disturbance --skip_displacement --headless
else
    echo "[SKIP] walking eval — no checkpoint found in $WALK_RUN_DIR."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run complete."
echo "Full log: $LOG_FILE"
echo "=============================================================="
