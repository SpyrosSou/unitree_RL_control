#!/usr/bin/env bash
# Second overnight queue (2026-07-15, night): launch this WHILE overnight_train.sh
# (the consolidated_intent branch) is still running — it waits for that script to finish
# before touching the GPU, then runs two more trainings so the night isn't wasted:
#
#   1. consolidated_noreach — consolidated + no_reach_prob=0.15, WITHOUT the intent
#      observation. The CONTROL for the intent run, which deliberately bundles two
#      changes (intent obs + idle slice): comparing intent vs this separates "the policy
#      learned from seeing the arm's intent" from "the policy just stopped being fragile
#      about motionless arms."
#   2. consolidated_seed7 — plain consolidated retrained with a different seed. Every
#      experiment so far is a single seed; the collapse-group/transfer-group split (and
#      consolidated's own 2.1%/54.7% numbers) may partly be seed luck. This measures that
#      directly: if seed 7 lands materially elsewhere (or fails the freeze_arms gate its
#      sibling passes), run-to-run variance is large and single-run comparisons need more
#      caution across the board.
#
# Each training is followed by the freeze_arms GATE (pass = fall_rate ~0 in its
# summary.csv; fail = collapse-group member, integration numbers pre-explained), then the
# normal native + integration evals.
#
# Same conventions as overnight_train.sh: no `set -e` (log FAILED lines instead), no
# shutdown at the end, everything tee'd to phase_logs/.
#
# Usage (while overnight_train.sh is still running is fine — that's the point):
#   cd ~/Elm/Code/g1_locomotion
#   nohup ./overnight_train_2.sh &

set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
STANDING_ITERS=6000
WALKING_CKPT="chosen_checkpoints/walking_latest.pt"
ARM_CKPT="chosen_checkpoints/arm_left_latest.pt"

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight2_$(date +%Y-%m-%d_%H-%M-%S).log"

# NOTE: no `set -u` (see overnight_train.sh — Isaac's setup_conda_env.sh isn't nounset-safe).
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

latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

echo "Second overnight queue starting $(date). Full log: $LOG_FILE"

# ---------------------------------------------------------------------------
# 0. Wait for overnight_train.sh (the intent branch) to finish — one GPU, one job.
#    pgrep -f "overnight_train.sh" does NOT match this script's own cmdline
#    (overnight_train_2.sh) — the "_2" breaks the substring.
# ---------------------------------------------------------------------------
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for overnight_train.sh to finish (checking every 2 min)..."
while pgrep -f "overnight_train.sh" > /dev/null; do
    sleep 120
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] overnight_train.sh no longer running — starting queue."

# ---------------------------------------------------------------------------
# 1. consolidated_noreach (control for the intent run's idle slice)
# ---------------------------------------------------------------------------
run_step "Train: standing consolidated_noreach ($STANDING_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-Consolidated-NoReach-v0 --headless \
        --max_iterations "$STANDING_ITERS" --run_name consolidated_noreach
NOREACH_RUN_DIR=$(latest_run_dir "standing/g1_locomotion_flat")
NOREACH_CKPT=$(latest_checkpoint "$NOREACH_RUN_DIR")

if [ -n "$NOREACH_CKPT" ]; then
    run_step "Gate (native, arms frozen): consolidated_noreach" \
        python validation/eval_standing_ikreach.py --checkpoint "$NOREACH_CKPT" --env_cfg noreach --freeze_arms --headless
    run_step "Eval (native): consolidated_noreach" \
        python validation/eval_standing_ikreach.py --checkpoint "$NOREACH_CKPT" --env_cfg noreach --headless
    run_step "Eval (integration): consolidated_noreach" \
        python validation/integration_validation/eval_full_demo.py \
            --standing_checkpoint "$NOREACH_CKPT" --walking_checkpoint "$WALKING_CKPT" \
            --arm_checkpoint "$ARM_CKPT" --num_envs 32 --headless
else
    echo "[WARN] No checkpoint found under $NOREACH_RUN_DIR for consolidated_noreach — skipping evals."
fi

# ---------------------------------------------------------------------------
# 2. consolidated, second seed (variance probe)
# ---------------------------------------------------------------------------
run_step "Train: standing consolidated_seed7 ($STANDING_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-Consolidated-v0 --headless \
        --max_iterations "$STANDING_ITERS" --seed 7 --run_name consolidated_seed7
SEED7_RUN_DIR=$(latest_run_dir "standing/g1_locomotion_flat")
SEED7_CKPT=$(latest_checkpoint "$SEED7_RUN_DIR")

if [ -n "$SEED7_CKPT" ] && [ "$SEED7_CKPT" != "$NOREACH_CKPT" ]; then
    run_step "Gate (native, arms frozen): consolidated_seed7" \
        python validation/eval_standing_ikreach.py --checkpoint "$SEED7_CKPT" --env_cfg consolidated --freeze_arms --headless
    run_step "Eval (native): consolidated_seed7" \
        python validation/eval_standing_ikreach.py --checkpoint "$SEED7_CKPT" --env_cfg consolidated --headless
    run_step "Eval (integration): consolidated_seed7" \
        python validation/integration_validation/eval_full_demo.py \
            --standing_checkpoint "$SEED7_CKPT" --walking_checkpoint "$WALKING_CKPT" \
            --arm_checkpoint "$ARM_CKPT" --num_envs 32 --headless
else
    echo "[WARN] No fresh checkpoint found under $SEED7_RUN_DIR for consolidated_seed7 — skipping evals."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Second overnight queue complete."
echo "Full log: $LOG_FILE"
echo "consolidated_noreach checkpoint: $NOREACH_CKPT"
echo "consolidated_seed7 checkpoint:   $SEED7_CKPT"
echo "Tomorrow's comparison set (all under the same fixed-eval convention):"
echo "  consolidated (baseline, integ. 2026-07-15_05-13-58) vs consolidated_intent vs"
echo "  consolidated_noreach vs consolidated_seed7 — gates first, then integration"
echo "  fall rates, right-side buckets (intent's signature), mean_max_abs_torso_deg,"
echo "  and left-reach success_rate_concluded."
echo "=============================================================="
