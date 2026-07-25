#!/usr/bin/env bash
# 2026-07-22 overnight run: walking (arm-disturbance variant, picking up today's
# phase-mixing + lin_vel_cmd_levels curriculum fixes) and arm (corrected 40/10
# hardware gain), each followed immediately by its own eval suite.
#
# Both trainings are FRESH starts, not resumes — deliberately, so neither confounds
# "did today's fix help" with "just more training," and so there's no additive-
# resume iteration-count risk (rsl_rl's --max_iterations under --resume is ADDITIVE
# to the resumed checkpoint's iteration, not an absolute target — see this project's
# own 2026-07-21/22 overnight-run incident for why that distinction matters).
#
# Includes the integration demo (eval_full_demo.py) as the final step, once both
# fresh checkpoints exist — --active_arm_gain overridden to 40/10 there (not the
# script's own stale 200/20 default) to match tonight's corrected arm gain.
#
# Log-and-continue on a step failure (not `set -e`) so one broken step doesn't waste
# the rest of the night.
set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
WALK_ITERS=6000
ARM_ITERS=6000
SEED=42
NUM_ENVS=3072  # not the 4096 default — this GPU OOM'd at 4096 earlier this session

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight_$(date +%Y-%m-%d_%H-%M-%S).log"

# NOTE: no `set -u` (only pipefail) — conda activate sources Isaac Sim's
# setup_conda_env.sh, which references $ZSH_VERSION with no default and isn't
# nounset-safe.
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

# $1 = experiment_name (e.g. walking/arm_disturbance) -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

echo "Overnight run starting $(date). Full log: $LOG_FILE"

# --- [1/5] Walking (arm-disturbance variant, fresh weights) ---------------------
# Picks up two same-day fixes: mdp.ArmMotionDisturbance's phase-mixing (each standing
# env independently samples its difficulty tier at reset, weighted toward the
# frontier but with real mass on earlier tiers, instead of one global deterministic
# phase shared by everyone) and mdp.lin_vel_cmd_levels's episode-length-normalization
# fix (was dividing accumulated reward by the fixed 20s max regardless of how long
# each env's episode actually ran, structurally blocking promotion whenever fall
# rate was high — see mdp/curriculums.py's own comment for the full diagnosis).
run_step "Train: walking arm-disturbance variant, fresh weights (seed=$SEED, $WALK_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-ArmDisturbance-v0 \
        --headless --max_iterations "$WALK_ITERS" --num_envs "$NUM_ENVS" --seed "$SEED"

WALK_RUN=$(latest_run_dir "walking/arm_disturbance")
WALK_CKPT=$(latest_checkpoint "$WALK_RUN")
if [ -z "$WALK_CKPT" ]; then
    echo "[WARN] No walking checkpoint found under $WALK_RUN — walking eval steps will be skipped."
else
    echo "Walking checkpoint: $WALK_CKPT"
fi

# --- [2/5] Walking eval: per-phase standing breakdown + forward-speed check -----
# Per-phase (not blended) stand_still fall rate is the informative read — a single
# blended number hides which tier is actually the problem (see 2026-07-22's
# diagnosis: the eval's own default phase cycling blends whatever mix of phases
# happened to occur into one number, dominated by whichever phase produced the most
# short/failing episodes).
if [ -n "$WALK_CKPT" ]; then
    for PHASE in 0 1 2 3; do
        run_step "Eval: walking stand_still, disturbance phase $PHASE pinned" \
            python validation/eval_walking.py --checkpoint "$WALK_CKPT" --arm_disturbance \
                --buckets stand_still --pin_disturbance_phase "$PHASE" --headless
    done
    run_step "Eval: walking forward_slow/forward_medium (regression check)" \
        python validation/eval_walking.py --checkpoint "$WALK_CKPT" --arm_disturbance \
            --buckets forward_slow forward_medium --headless
else
    echo "[SKIP] walking eval — no checkpoint."
fi

# --- [3/5] Arm (left, fresh weights, corrected 40/10 gain) ----------------------
# Fresh start, not a resume of model_2999.pt — that checkpoint trained entirely at
# the old, wrong 200/20 gain (see g1_arm_env.py's own fix comment, 2026-07-22).
# Resuming it into the corrected gain would confound "did the gain fix help" with
# "just more training," the same trap a resume would set for the walking curriculum
# fix above.
run_step "Train: arm left, fresh weights, corrected gain (seed=$SEED, $ARM_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-v0 \
        --headless --max_iterations "$ARM_ITERS" --seed "$SEED"

ARM_RUN=$(latest_run_dir "arms/left")
ARM_CKPT=$(latest_checkpoint "$ARM_RUN")
if [ -z "$ARM_CKPT" ]; then
    echo "[WARN] No arm checkpoint found under $ARM_RUN — arm eval will be skipped."
else
    echo "Arm checkpoint: $ARM_CKPT"
fi

# --- [4/5] Arm eval --------------------------------------------------------------
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm (corrected gain)" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --headless
else
    echo "[SKIP] arm eval — no checkpoint."
fi

# --- [5/5] Integration eval: walking + arm combined ------------------------------
# --active_arm_gain 40 10 explicit here for clarity — as of 2026-07-23 this now matches
# eval_full_demo.py's own default (also fixed to 40/10, was stale at 200/20), so this
# flag is redundant but kept to make the intent explicit at the call site.
if [ -n "$WALK_CKPT" ] && [ -n "$ARM_CKPT" ]; then
    run_step "Eval (integration): walking + arm combined" \
        python validation/integration_validation/eval_full_demo.py \
            --loco_checkpoint "$WALK_CKPT" --arm_checkpoint "$ARM_CKPT" \
            --active_arm_gain 40 10 --headless
else
    echo "[SKIP] integration eval — missing walking and/or arm checkpoint."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run complete."
echo "Full log: $LOG_FILE"
echo "Walking checkpoint: ${WALK_CKPT:-<none>}"
echo "Arm checkpoint    : ${ARM_CKPT:-<none>}"
echo "Walking eval output: <run_dir>/command_eval_phase{0,1,2,3}/summary.md (standing),"
echo "                     <run_dir>/command_eval/summary.md (forward buckets)"
echo "Arm eval output    : <run_dir>/arm_eval/summary.md"
echo "Integration eval output: validation/integration_validation/<timestamp>/*_summary.csv"
echo ""
echo "Compare walking per-phase stand_still fall rates against today's two reference runs:"
echo "  deterministic phase (2026-07-22_11-57-02): 0% / 33% / 66% / 84% (phases 0/1/2/3)"
echo "  phase-mixing only   (2026-07-22_16-27-25): 0% / 0%  / 7%  / 21% (phases 0/1/2/3)"
echo "This run adds the lin_vel_cmd_levels fix on top of phase-mixing — the number to"
echo "watch most closely is forward_medium (was 63.64% deterministic-only, 90.53% with"
echo "phase-mixing alone) — that regression is what this run's curriculum fix targets."
echo "=============================================================="
