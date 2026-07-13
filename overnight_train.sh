#!/usr/bin/env bash
# Standing baseline follow-up (2026-07-13, replaces the previous torso/policy-driven
# queue from last night): 1 training step + both evals.
#
# Last night's two experiments are both dropped from this queue:
#   - torso_retighten (joint_deviation_torso 0.0 -> -0.05): measurably didn't help —
#     integration fall rates were flat or slightly worse across every bucket.
#   - policy_driven_disturbance (real arm-IK policy instead of analytic IK): fixed
#     standing_still outright (94% -> 0% fail) but standing_arm_left_reach got much worse
#     (73% -> 96.5% fail). Root cause found afterward: nothing in the whole reward stack
#     penalizes hip-pitch/knee deviation from default, so standing had been free to sink
#     into a deep, stable squat (~0.4-0.5m root height vs. ~0.75m spawn) as a genuinely
#     free way to buy stability — worse under the harder, policy-driven disturbance. The
#     squat is almost certainly what broke arm-reach too: the arm-IK policy is fixed-root
#     trained at nominal standing height, so a habitually crouched torso puts the arm in a
#     geometry it never learned to reach from.
#
# This queue instead builds directly on the LAST GOOD baseline (dwell_phase_fix,
# 2026-07-12 — analytic IK + dwell-until-timeout goal holding, i.e. it already trains
# "reach and hold", not just continuous motion) and adds exactly one new, mechanistically
# targeted thing: mdp.base_height_l2 (stock Isaac Lab), a direct L2 penalty on root height
# against target_height=0.75 (G1's own spawn height) — see
# G1LocomotionStandingFlatIKReachHeightEnvCfg's docstring in g1_locomotion_env_cfg.py for
# the full mechanism. Single-variable change against a baseline that was already
# reasonably balanced, not another stack of guesses.
#
# Deliberately does NOT use `set -e`: one failed step shouldn't silently kill the rest of
# an unattended queue. Each step's pass/fail is logged instead; check the log for FAILED
# lines afterward.
#
# Deliberately no shutdown/poweroff command anywhere in this script — leave the machine
# running when it's done.
#
# Usage:
#   cd ~/Elm/Code/g1_locomotion
#   nohup ./overnight_train.sh &
#   (or just run it in a tmux/screen session and detach)

set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
STANDING_ITERS=6000
WALKING_CKPT="chosen_checkpoints/walking_latest.pt"
ARM_CKPT="chosen_checkpoints/arm_left_latest.pt"

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/overnight_$(date +%Y-%m-%d_%H-%M-%S).log"

# NOTE: no `set -u` here (only pipefail) — conda activate sources Isaac Sim's
# setup_conda_env.sh, which references $ZSH_VERSION with no default and isn't
# nounset-safe. `set -u` made that fatal on the first run before anything even
# started training. The rest of this script doesn't rely on unset-variable
# checking for correctness, so it's simplest to just leave it off entirely.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Mirror everything to a log file too, so it survives even if the terminal is closed.
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

# $1 = experiment_name (e.g. standing/g1_locomotion_flat) -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

echo "Standing baseline follow-up (height reward) starting $(date). Full log: $LOG_FILE"

# ---------------------------------------------------------------------------
# 1. Height reward, on top of the dwell_phase_fix baseline
# ---------------------------------------------------------------------------
run_step "Train: standing height_reward ($STANDING_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-IKReach-Height-v0 --headless \
        --max_iterations "$STANDING_ITERS" --run_name height_reward
HEIGHT_RUN_DIR=$(latest_run_dir "standing/g1_locomotion_flat")
HEIGHT_CKPT=$(latest_checkpoint "$HEIGHT_RUN_DIR")

if [ -n "$HEIGHT_CKPT" ]; then
    run_step "Eval (native): standing height_reward" \
        python validation/eval_standing_ikreach.py --checkpoint "$HEIGHT_CKPT" --env_cfg height --headless
    run_step "Eval (integration): standing height_reward" \
        python validation/integration_validation/eval_full_demo.py \
            --standing_checkpoint "$HEIGHT_CKPT" --walking_checkpoint "$WALKING_CKPT" \
            --arm_checkpoint "$ARM_CKPT" --num_envs 32 --headless
else
    echo "[WARN] No checkpoint found under $HEIGHT_RUN_DIR for height_reward — skipping both evals."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Standing baseline follow-up complete."
echo "Full log: $LOG_FILE"
echo "height_reward checkpoint: $HEIGHT_CKPT"
echo "Native eval output:      validation/eval_standing_ikreach/<timestamp>/summary.csv"
echo "Integration eval output: validation/integration_validation/<timestamp>/*/*_summary.csv"
echo "=============================================================="
