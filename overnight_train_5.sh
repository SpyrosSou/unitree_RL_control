#!/usr/bin/env bash
# 2026-07-24 overnight run (6pm-9am, 15h window): the "real" walking+arm-disturbance
# recipe, warm-started from the validated base-recipe walking checkpoint, and the
# locked-wrist arm variant (see g1_arm_env.py's G1ArmLeftLockedWristEnvCfg / this
# session's conversation for the full reasoning on why: exp_scale and entropy_coef
# were both dropped as either confirmed-broken or carrying a real regression
# precedent from the 23dof branch's own history; wrist-locking is the one
# structural, non-reward change going into tonight's arm run).
#
# Iteration counts sized for the actual time window, not a round number:
#   walking: 10000 iters @ ~2.2s/iter (3072 envs)  ~= 6.1h
#   arm:     20000 iters @ ~1.2s/iter (4096 envs)  ~= 6.7h
#   + evals + ~10% safety margin, comfortably inside 15h.
# Walking runs FIRST (finishes around the ~6h mark) so a midnight check-in sees a
# complete, evaluated result rather than a run in progress; arm (the riskier bet)
# runs through the rest of the night regardless of whether anyone checks in.
#
# Log-and-continue on a step failure (not `set -e`) so one broken step doesn't waste
# the rest of the night.
set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
SEED=42
WALK_NUM_ENVS=3072
WALK_ITERS=10000
ARM_ITERS=20000
WARM_START_CKPT="$PROJECT_ROOT/chosen_checkpoints/walking_latest.pt"

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight5_$(date +%Y-%m-%d_%H-%M-%S).log"

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

# $1 = experiment_name -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

echo "Overnight run 5 starting $(date). Full log: $LOG_FILE"
echo "Plan: walking+disturbance ($WALK_ITERS iters, warm-started) -> eval -> arm locked-wrist ($ARM_ITERS iters, fresh) -> eval"

# --- [1/4] Walking + arm disturbance, warm-started from the validated walking checkpoint
if [ ! -f "$WARM_START_CKPT" ]; then
    echo "[WARN] Warm-start checkpoint not found at $WARM_START_CKPT — falling back to fresh weights."
    run_step "Train: walking+disturbance, FRESH weights (seed=$SEED, $WALK_ITERS iters)" \
        python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-v0 \
            --headless --max_iterations "$WALK_ITERS" --num_envs "$WALK_NUM_ENVS" --seed "$SEED"
else
    # Cross-task warm start: --checkpoint as a direct file path bypasses --load_run's
    # same-experiment_name directory scanning entirely (see train.py's resume_path
    # resolution — os.path.isfile(agent_cfg.load_checkpoint) short-circuits straight to
    # that path). --resume_new_dir so this starts a fresh timestamped dir under
    # walking/arm_disturbance instead of writing into the source run's own folder.
    # rsl_rl's --max_iterations is ADDITIVE under --resume (confirmed in on_policy_
    # runner.py this session) — the source checkpoint is iter 5999, so this yields
    # ~5999+10000 = ~16000 as the final checkpoint number, matching the 10000 NEW
    # iterations actually being budgeted here, not an absolute-16000 confusion.
    run_step "Train: walking+disturbance, warm-started from walking_latest.pt (+$WALK_ITERS new iters)" \
        python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-v0 \
            --headless --resume --resume_new_dir --checkpoint "$WARM_START_CKPT" \
            --max_iterations "$WALK_ITERS" --num_envs "$WALK_NUM_ENVS" --seed "$SEED"
fi

WALK_RUN=$(latest_run_dir "walking/arm_disturbance")
WALK_CKPT=$(latest_checkpoint "$WALK_RUN")

# --- [2/4] Walking eval: per-phase stand_still (phase-pin bug now fixed) + forward buckets
if [ -n "$WALK_CKPT" ]; then
    for PHASE in 0 1 2 3; do
        run_step "Eval: walking+disturbance stand_still, phase $PHASE pinned" \
            python validation/eval_walking.py --checkpoint "$WALK_CKPT" --arm_disturbance \
                --buckets stand_still --pin_disturbance_phase "$PHASE" --headless
    done
    run_step "Eval: walking+disturbance forward_slow/forward_medium" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT" --arm_disturbance \
            --buckets forward_slow forward_medium --headless
else
    echo "[SKIP] walking eval — no checkpoint found."
fi

# --- [3/4] Arm, locked-wrist variant, fresh weights (new action/obs shape — cannot resume
# from any prior arm checkpoint) -----------------------------------------------------
run_step "Train: arm locked-wrist, fresh weights (seed=$SEED, $ARM_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-LockedWrist-v0 \
        --headless --max_iterations "$ARM_ITERS" --seed "$SEED"

ARM_RUN=$(latest_run_dir "arms/left_locked_wrist")
ARM_CKPT=$(latest_checkpoint "$ARM_RUN")

# --- [4/4] Arm eval -------------------------------------------------------------------
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm locked-wrist" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --headless
else
    echo "[SKIP] arm eval — no checkpoint found."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run 5 complete."
echo "Full log: $LOG_FILE"
echo "Walking+disturbance checkpoint: ${WALK_CKPT:-<none>}"
echo "Walking eval outputs: <walk_run>/command_eval_phase{0,1,2,3}/summary.md (standing),"
echo "                      <walk_run>/command_eval/summary.md (forward buckets)"
echo "Arm locked-wrist checkpoint: ${ARM_CKPT:-<none>}"
echo "Arm eval output: <arm_run>/arm_eval/summary.md"
echo "=============================================================="
