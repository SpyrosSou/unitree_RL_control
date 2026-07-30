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
# TASK QUEUE — 2026-07-29 overnight run (~9h window; see timing note per step)
# ==========================================================================================
# 1) Arm: G1-Arm-Left-Integrated-v0, FRESH weights, 8000 iters. First-ever training of
#    the 2026-07-29 static-torque-ceiling fix (integrated action targets + env-local
#    ee/goal obs + target_fb, 46-D — see policy_status.md's 2026-07-29 "ROOT CAUSE
#    FOUND" entry and check_arm_static_torque_ceiling.py). Fresh run, so
#    --max_iterations is absolute (the ADDITIVE-on-resume trap corrected last night
#    doesn't apply). Benchmark to beat: best_combined 28.20%/32.85% at 2000 iters /
#    30.15%/28.10% at ~10000. Early tell in tensorboard: Train/mean_episode_length
#    finally dropping well below ~290 (the 200/20 reference converges to ~60).
#    Est. ~2h50m (best_combined did 8001 iters in that last night).
# 2) Eval the arm run TWICE: once at model_2000.pt (direct apples-to-apples vs
#    best_combined's own 2000-iter result) and once at the final checkpoint.
#    eval_arm.py writes to <checkpoint_dir>/arm_eval/ — both checkpoints share a run
#    dir, so the mid-run eval is renamed to arm_eval_iter2000/ before the final one
#    runs, or the second would overwrite the first.
# 3) Walking: G1-Locomotion-Velocity-ArmDisturbance-StandingPackage-v0, FRESH weights
#    (deliberate — changed reward composition + the stale-critic lessons in
#    policy_status.md), 7000 iters. The 2026-07-29 in-distribution eval of
#    walking_latest (turn/strafe buckets fixed to the trained envelope) showed 0.00%
#    falls in EVERY bucket incl. turn_left — the old 31.79% turn_left number was an
#    out-of-distribution artifact — so the standing package is the right next target,
#    not turning. TIMING NOTE: at last night's walking pace (~5000 iters in ~5.5h),
#    7000 ≈ 7.5-8h — this step will likely still be running (or its eval pending) at
#    the 9h mark. Safe to stop in the morning (stop_overnight.sh): checkpoints save
#    every 100 iters, and the eval can be run manually on whatever the latest is.
# Each training step is followed by its own eval so results are ready to read in the
# morning, not just raw checkpoints.

run_step "Train: arm Integrated (fresh) -> 8000 iters" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-Integrated-v0 \
        --max_iterations 8000 --headless --seed "$SEED"

ARM_RUN_DIR=$(latest_run_dir "arms/integrated")
ARM_MID_CKPT=""
[ -n "$ARM_RUN_DIR" ] && [ -f "$ARM_RUN_DIR/model_2000.pt" ] && ARM_MID_CKPT="$ARM_RUN_DIR/model_2000.pt"
if [ -n "$ARM_MID_CKPT" ]; then
    run_step "Eval: arm Integrated @ 2000 iters (vs best_combined's 28.20%/32.85%)" \
        python validation/eval_arm.py --checkpoint "$ARM_MID_CKPT" --integrated --headless
    if [ -d "$ARM_RUN_DIR/arm_eval" ]; then
        mv "$ARM_RUN_DIR/arm_eval" "$ARM_RUN_DIR/arm_eval_iter2000"
        echo "[INFO] mid-run eval stashed at ${ARM_RUN_DIR}arm_eval_iter2000/"
    fi
else
    echo "[SKIP] arm mid-run eval — no model_2000.pt in ${ARM_RUN_DIR:-<no run dir>}."
fi

ARM_CKPT=$(latest_checkpoint "$ARM_RUN_DIR")
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm Integrated @ final ($(basename "$ARM_CKPT"))" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --integrated --headless
else
    echo "[SKIP] arm final eval — no checkpoint found in ${ARM_RUN_DIR:-<no run dir>}."
fi

run_step "Train: walking StandingPackage (fresh) -> 7000 iters" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-StandingPackage-v0 \
        --max_iterations 7000 --headless --seed "$SEED"

WALK_RUN_DIR=$(latest_run_dir "walking/arm_disturbance")
WALK_CKPT=$(latest_checkpoint "$WALK_RUN_DIR")
if [ -n "$WALK_CKPT" ]; then
    run_step "Eval: walking StandingPackage @ final ($(basename "$WALK_CKPT"))" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT" \
            --arm_disturbance --skip_displacement --headless
else
    echo "[SKIP] walking eval — no checkpoint found in ${WALK_RUN_DIR:-<no run dir>}."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run complete."
echo "Full log: $LOG_FILE"
echo "=============================================================="
