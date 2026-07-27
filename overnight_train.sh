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
# TASK QUEUE — edit this section per run, delete the example below
# ==========================================================================================

# Example:
# run_step "Train: <description>" \
#     python scripts/rsl_rl/train.py --task <TASK_ID> \
#         --headless --max_iterations <N> --seed "$SEED"
#
# RUN_DIR=$(latest_run_dir "<experiment_name>")
# CKPT=$(latest_checkpoint "$RUN_DIR")
# if [ -n "$CKPT" ]; then
#     run_step "Eval: <description>" \
#         python validation/eval_arm.py --checkpoint "$CKPT" --headless
# else
#     echo "[SKIP] eval — no checkpoint found."
# fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run complete."
echo "Full log: $LOG_FILE"
echo "=============================================================="
