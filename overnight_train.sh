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
# TASK QUEUE — 2026-07-30 overnight run (~9h window; arm-only, ~4-5h expected)
# ==========================================================================================
# Arm: G1-Arm-Left-IntegratedNoTerm-v0, FRESH weights, 12000 iters. Fixes the hold-
# quality problem found in the first Integrated run (logs/rsl_rl/arms/integrated/
# 2026-07-29_20-27-48, model_7999.pt): reaching itself was already solved there (100%
# reach rate, ~1-3mm precision), but terminate_on_success made completing the
# 15-consecutive-step hold reward-NEGATIVE (ends the per-step goal_reached_bonus
# stream, respawns a harder goal), so the policy learned to dither at the 2cm boundary
# instead of settling deep and still — legacy success rate read only 6.5-9.4% despite
# the real capability being there. This run disables terminate_on_success entirely
# (see G1ArmLeftIntegratedNoTermEnvCfg's docstring in g1_arm_env.py) — episodes always
# run the full length, so the bonus keeps paying for every step spent in-zone and
# there's nothing left to gain from leaving. Same 46-D obs/integrated-target
# pipeline/gain/reward as the first Integrated run otherwise — an incentive-only
# change, not a capability change. See arms_policy_finalisation.md step 1 (option (a))
# and policy_status.md's 2026-07-30 arm entries for the full reasoning.
#
# Fresh run (no --resume), so --max_iterations is absolute. Est. ~4-4.5h at last
# night's Integrated pace (8001 iters in ~2h50m).
#
# NOTE: Train/mean_episode_length is NOT a useful progress signal for this run (it's
# always ~max, since nothing terminates early) — watch Episode_Reward/goal_reached_
# bonus (ceiling ~50/step) and Metrics/frac_envs_reached in tensorboard instead.
#
# Eval afterward MUST use --integrated_no_term (not --integrated) — it selects the
# matching NoTerm PLAY env (terminate_on_success=False there too) so the eval's
# tail-settle-rate metric always sees a full trailing window per episode, and reports
# the new Settled-<2cm/<3cm + Tail-settle columns as the real judge of this run
# (legacy "success rate" will read a structural, expected 0% here — see
# arms_policy_finalisation.md, do not read that as a regression).

run_step "Train: arm IntegratedNoTerm (fresh) -> 12000 iters" \
    python scripts/rsl_rl/train.py --task G1-Arm-Left-IntegratedNoTerm-v0 \
        --max_iterations 12000 --headless --seed "$SEED"

ARM_RUN_DIR=$(latest_run_dir "arms/integrated_no_term")
ARM_CKPT=$(latest_checkpoint "$ARM_RUN_DIR")
if [ -n "$ARM_CKPT" ]; then
    run_step "Eval: arm IntegratedNoTerm @ final ($(basename "$ARM_CKPT"))" \
        python validation/eval_arm.py --checkpoint "$ARM_CKPT" --integrated_no_term --headless
else
    echo "[SKIP] arm eval — no checkpoint found in ${ARM_RUN_DIR:-<no run dir>}."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight run complete."
echo "Full log: $LOG_FILE"
echo "=============================================================="
