#!/usr/bin/env bash
# 2026-07-23/24 overnight run: arm training (priority), then a clean, consistent
# re-run of all 4 walking-fix ablations at 2000 iterations each (see
# g1_locomotion_env_cfg.py's Ablation*EnvCfg classes), each followed by a real
# fall-rate/tracking-error eval (not just training-time reward curves) so there's
# comparable, standardized data for all 4 (plus the existing control run) to decide
# tomorrow's walking approach from.
#
# Arms are FIRST and ablations are last on purpose — if this doesn't finish overnight,
# it's the ablations that get cut short, not the arm training.
#
# All 4 ablations are trained FRESH (not resumed from the earlier 1k partial runs) —
# comparing a 1k run against 2k runs isn't apples-to-apples, and term_penalty's
# "inconclusive" 1k result is exactly the kind of thing more iterations might resolve.
#
# Log-and-continue on a step failure (not `set -e`) so one broken step doesn't waste
# the rest of the night.
set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
ARM_ITERS=6000
ABLATION_ITERS=2000
SEED=42
NUM_ENVS=3072

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight2_$(date +%Y-%m-%d_%H-%M-%S).log"

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

# $1 = experiment_name (e.g. arms/left) -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

echo "Overnight run 2 starting $(date). Full log: $LOG_FILE"

# --- [1/N] Arm training (left, fresh weights) -------------------------------------
run_step "Train: arm left, fresh weights (seed=$SEED, $ARM_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-v0 \
        --headless --max_iterations "$ARM_ITERS" --seed "$SEED"

ARM_RUN=$(latest_run_dir "arms/left")
ARM_CKPT=$(latest_checkpoint "$ARM_RUN")
if [ -z "$ARM_CKPT" ]; then
    echo "[WARN] No arm checkpoint found under $ARM_RUN — arm eval will be skipped."
else
    echo "Arm checkpoint: $ARM_CKPT"
fi

# --- [2/N] Arm eval -----------------------------------------------------------------
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --headless
else
    echo "[SKIP] arm eval — no checkpoint."
fi

# --- [3/N] Walking ablations, fresh at 2000 iterations each -------------------------
declare -A ABLATION_TASKS=(
    [term_penalty]="G1-Locomotion-Velocity-Ablation-TermPenalty-v0"
    [curriculum]="G1-Locomotion-Velocity-Ablation-Curriculum-v0"
    [reward_weights]="G1-Locomotion-Velocity-Ablation-RewardWeights-v0"
    [all]="G1-Locomotion-Velocity-Ablation-All-v0"
)

for NAME in term_penalty curriculum reward_weights all; do
    TASK="${ABLATION_TASKS[$NAME]}"
    run_step "Train: ablation '$NAME' ($TASK), fresh weights (seed=$SEED, $ABLATION_ITERS iters)" \
        python scripts/rsl_rl/train.py --task "$TASK" \
            --headless --max_iterations "$ABLATION_ITERS" --num_envs "$NUM_ENVS" --seed "$SEED"

    RUN_DIR=$(latest_run_dir "walking/ablation_$NAME")
    CKPT=$(latest_checkpoint "$RUN_DIR")
    if [ -z "$CKPT" ]; then
        echo "[WARN] No checkpoint found under $RUN_DIR — eval for '$NAME' will be skipped."
        continue
    fi
    echo "Ablation '$NAME' checkpoint: $CKPT"

    run_step "Eval: ablation '$NAME' (stand_still + forward_slow + forward_medium)" \
        python validation/eval_walking.py --checkpoint "$CKPT" \
            --buckets stand_still forward_slow forward_medium --headless
done

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run 2 complete."
echo "Full log: $LOG_FILE"
echo "Arm checkpoint: ${ARM_CKPT:-<none>}"
echo "Arm eval output: <arm_run_dir>/arm_eval/summary.md"
echo ""
echo "Ablation eval outputs (fall rate + track err for stand_still/forward_slow/forward_medium):"
echo "  logs/rsl_rl/walking/base/2026-07-22_08-05-36/command_eval/summary.md   (CONTROL, 0 changes, already exists)"
echo "  logs/rsl_rl/walking/ablation_term_penalty/<newest>/command_eval/summary.md"
echo "  logs/rsl_rl/walking/ablation_curriculum/<newest>/command_eval/summary.md"
echo "  logs/rsl_rl/walking/ablation_reward_weights/<newest>/command_eval/summary.md"
echo "  logs/rsl_rl/walking/ablation_all/<newest>/command_eval/summary.md"
echo "=============================================================="
