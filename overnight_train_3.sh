#!/usr/bin/env bash
# 2026-07-24 overnight run: short arm sanity check with the entropy_coef/
# position_reward_exp_scale fixes (see g1_arm_env.py and agents/rsl_rl_ppo_cfg.py's own
# comments for the full rationale — Train/mean_reward had plateaued after ~600/6000
# iterations on the previous run while exploration noise kept shrinking), then continue
# walking to its full ~6000-iteration budget. Arm is deliberately left at the short
# 2000-iteration checkpoint, NOT auto-continued — that result needs review before
# committing to a longer arm run, not an automatic continuation.
#
# Order: arm short check (fresh, 2000 iters) -> arm eval (on that same 2000-iter
# checkpoint) -> walking continuation (resume the reward_weights ablation checkpoint,
# +4000 iters to ~6000 total) -> walking eval.
#
# Log-and-continue on a step failure (not `set -e`) so one broken step doesn't waste
# the rest of the run.
set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
SEED=42
NUM_ENVS=3072

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight3_$(date +%Y-%m-%d_%H-%M-%S).log"

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

echo "Overnight run 3 starting $(date). Full log: $LOG_FILE"

# --- [1/3] Arm: short sanity check, fresh weights, entropy_coef + exp_scale fixes ----
# Deliberately NOT continued further in this script — needs review before committing
# to a longer run, same as the walking ablations did.
run_step "Train: arm left, fresh weights, short sanity check (seed=$SEED, 2000 iters)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-v0 \
        --headless --max_iterations 2000 --seed "$SEED"

ARM_RUN=$(latest_run_dir "arms/left")
ARM_CKPT=$(latest_checkpoint "$ARM_RUN")
if [ -z "$ARM_CKPT" ]; then
    echo "[WARN] No arm checkpoint found under $ARM_RUN — arm eval will be skipped."
fi

# --- [2/3] Arm eval (on the 2000-iteration checkpoint) ------------------------------
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm (post entropy_coef/exp_scale fix, 2000 iters)" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --headless
else
    echo "[SKIP] arm eval — no checkpoint."
fi

# --- [3/3] Walking: continue the reward_weights run to ~6000 total ----------------
WALK_RUN="logs/rsl_rl/walking/ablation_reward_weights/2026-07-24_03-13-16/"
WALK_CKPT="model_1999.pt"
if [ -f "${WALK_RUN}${WALK_CKPT}" ]; then
    run_step "Train: walking reward_weights, continue to ~6000 total (resume, +4000 iters)" \
        python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-Ablation-RewardWeights-v0 \
            --headless --resume --load_run "2026-07-24_03-13-16" --checkpoint "$WALK_CKPT" \
            --max_iterations 4000 --num_envs "$NUM_ENVS" --seed "$SEED"
    WALK_CKPT_FULL=$(latest_checkpoint "$WALK_RUN")
else
    echo "[WARN] Expected walking checkpoint ${WALK_RUN}${WALK_CKPT} not found — skipping walking continuation."
    WALK_CKPT_FULL=""
fi

# --- Walking eval (part of [3/3]) --------------------------------------------------
if [ -n "$WALK_CKPT_FULL" ]; then
    run_step "Eval: walking (stand_still + forward_slow + forward_medium)" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT_FULL" \
            --buckets stand_still forward_slow forward_medium --headless
else
    echo "[SKIP] walking eval — no checkpoint."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run 3 complete."
echo "Full log: $LOG_FILE"
echo "Arm checkpoint: ${ARM_CKPT:-<none>}"
echo "Arm eval output: <arm_run_dir>/arm_eval/summary.md"
echo "Walking checkpoint: ${WALK_CKPT_FULL:-<none>}"
echo "Walking eval output: ${WALK_RUN}command_eval/summary.md"
echo "=============================================================="
