#!/usr/bin/env bash
# 2026-07-22 unattended run for the ~2hr window while the user is away.
# Deliberately does NOT auto-launch the big arm-disturbance retrain at the end — that
# decision should be made with real eval numbers in hand, not blind. Every resume below
# computes its own remaining-iteration delta from the checkpoint's own saved `iter`
# value (never a hardcoded absolute), because rsl_rl's --max_iterations under --resume
# is ADDITIVE to the resumed iteration, not an absolute target (see 2026-07-21/22's
# overnight-run incident — this script exists specifically to not repeat that mistake).
set -o pipefail

cd "$HOME/Elm/Code/g1_locomotion"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate isaac_g1_control

mkdir -p phase_logs
LOG_FILE="phase_logs/afternoon_$(date +%Y-%m-%d_%H-%M-%S).log"
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

latest_ckpt() { ls -v "$1"/model_*.pt 2>/dev/null | tail -1; }
ckpt_iter() {
    python3 -c "
import torch, sys
print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('iter', 0)))
" "$1"
}

echo "Afternoon run starting $(date). Full log: $LOG_FILE"

# --- [1/4] Finish the plain-recipe walking diagnostic to iteration 3000 total ---
WALK_RUN="logs/rsl_rl/walking/base/2026-07-22_08-05-36"
WALK_CKPT=$(latest_ckpt "$WALK_RUN")
WALK_ITER=$(ckpt_iter "$WALK_CKPT")
WALK_TARGET=3000
WALK_DELTA=$((WALK_TARGET - WALK_ITER))
echo "[INFO] Plain walking diagnostic: at iter $WALK_ITER (from $WALK_CKPT), target $WALK_TARGET, delta $WALK_DELTA"
if [ "$WALK_DELTA" -gt 0 ]; then
    run_step "Resume plain walking diagnostic (+$WALK_DELTA iters, to $WALK_TARGET total)" \
        python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-v0 --headless \
            --resume --checkpoint "$WALK_CKPT" --max_iterations "$WALK_DELTA" --num_envs 3072 --seed 42
else
    echo "[INFO] Already at/past target — skipping."
fi
WALK_CKPT=$(latest_ckpt "$WALK_RUN")
echo "[INFO] Final plain walking checkpoint: $WALK_CKPT"

# --- [2/4] Isolated eval: stand_still + forward buckets, no arm disturbance (this task never had it) ---
if [ -n "$WALK_CKPT" ]; then
    run_step "Eval: plain walking diagnostic (stand_still/forward_slow/forward_medium)" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT" \
            --buckets stand_still forward_slow forward_medium --headless
else
    echo "[SKIP] walking eval — no checkpoint."
fi

# --- [3/4] Resume arm training to iteration 3000 total ---
ARM_RUN="logs/rsl_rl/arms/left/2026-07-22_06-20-55"
ARM_CKPT=$(latest_ckpt "$ARM_RUN")
ARM_ITER=$(ckpt_iter "$ARM_CKPT")
ARM_TARGET=3000
ARM_DELTA=$((ARM_TARGET - ARM_ITER))
echo "[INFO] Arm training: at iter $ARM_ITER (from $ARM_CKPT), target $ARM_TARGET, delta $ARM_DELTA"
if [ "$ARM_DELTA" -gt 0 ]; then
    run_step "Resume arm training (+$ARM_DELTA iters, to $ARM_TARGET total)" \
        python scripts/rsl_rl/train.py --task G1-Arm-Left-v0 --headless \
            --resume --checkpoint "$ARM_CKPT" --max_iterations "$ARM_DELTA" --seed 42
else
    echo "[INFO] Already at/past target — skipping."
fi
ARM_CKPT=$(latest_ckpt "$ARM_RUN")
echo "[INFO] Final arm checkpoint: $ARM_CKPT"

# --- [4/4] Arm eval ---
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --headless
else
    echo "[SKIP] arm eval — no checkpoint."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Afternoon run complete."
echo "Full log: $LOG_FILE"
echo "Walking checkpoint: ${WALK_CKPT:-<none>}  ->  <run_dir>/command_eval/summary.md"
echo "Arm checkpoint    : ${ARM_CKPT:-<none>}  ->  <run_dir>/arm_eval/summary.md (or similar)"
echo "Deliberately NOT auto-starting the arm-disturbance retrain — review stand_still"
echo "fall rate above first, decide with real numbers in hand."
echo "=============================================================="
