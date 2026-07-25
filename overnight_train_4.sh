#!/usr/bin/env bash
# 2026-07-24 overnight run: resume walking from 4800 to ~6000 total, eval that final
# checkpoint, then run the 2 arm ablations (entropy_coef alone, position_reward_exp_scale
# alone — see G1ArmLeftAblation* classes in g1_arm_env.py/agents/rsl_rl_ppo_cfg.py) at
# 1500 iters each.
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
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight4_$(date +%Y-%m-%d_%H-%M-%S).log"

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

echo "Overnight run 4 starting $(date). Full log: $LOG_FILE"

WALK_RUN="logs/rsl_rl/walking/ablation_reward_weights/2026-07-24_03-13-16/"
WALK_RUN_NAME="2026-07-24_03-13-16"

# --- [1/5] (skipped — no fresh eval needed on 4800 before resuming) ----------------
if [ -d "${WALK_RUN}command_eval" ] && [ ! -d "${WALK_RUN}command_eval_4800" ]; then
    mv "${WALK_RUN}command_eval" "${WALK_RUN}command_eval_4800"
    echo "[INFO] Preserved existing eval output at ${WALK_RUN}command_eval_4800"
fi

# --- [2/5] Walking: continue to ~6000 total -----------------------------------------
run_step "Train: walking reward_weights, continue to ~6000 total (resume from 4800, +1200 iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-Ablation-RewardWeights-v0 \
        --headless --resume --load_run "$WALK_RUN_NAME" --checkpoint "model_4800.pt" \
        --max_iterations 1200 --num_envs "$NUM_ENVS" --seed "$SEED"

WALK_CKPT_FULL=$(latest_checkpoint "$WALK_RUN")

# --- [3/5] Eval the final ~6000-iteration walking checkpoint ------------------------
if [ -n "$WALK_CKPT_FULL" ]; then
    run_step "Eval: walking @ ~6000 iters (stand_still + forward_slow + forward_medium)" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT_FULL" \
            --buckets stand_still forward_slow forward_medium --headless
    if [ -d "${WALK_RUN}command_eval" ]; then
        mv "${WALK_RUN}command_eval" "${WALK_RUN}command_eval_final"
    fi
else
    echo "[SKIP] final walking eval — no checkpoint found."
fi

# --- [4/5] Arm ablation 1/2: entropy_coef alone -------------------------------------
run_step "Train: arm ablation — entropy_coef=0.01 alone (seed=$SEED, 1500 iters)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-Ablation-EntropyCoef-v0 \
        --headless --max_iterations 1500 --seed "$SEED"

# --- [5/5] Arm ablation 2/2: position_reward_exp_scale alone ------------------------
run_step "Train: arm ablation — position_reward_exp_scale=3.0 alone (seed=$SEED, 1500 iters)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-Ablation-ExpScale-v0 \
        --headless --max_iterations 1500 --seed "$SEED"

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run 4 complete."
echo "Full log: $LOG_FILE"
echo "Walking checkpoint (4800): ${WALK_RUN}model_4800.pt"
echo "Walking eval (4800) output: ${WALK_RUN}command_eval_4800/summary.md"
echo "Walking checkpoint (~6000): ${WALK_CKPT_FULL:-<none>}"
echo "Walking eval (~6000) output: ${WALK_RUN}command_eval_final/summary.md"
echo "Arm ablation checkpoints: logs/rsl_rl/arms/ablation_entropy_coef/<newest>, logs/rsl_rl/arms/ablation_exp_scale/<newest>"
echo "=============================================================="
