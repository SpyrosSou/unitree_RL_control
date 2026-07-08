#!/usr/bin/env bash
# Overnight sweep (2026-07-07): 3 isolated arm-policy experiments (targeting the
# confirmed ~55% success plateau, see known_issues.md) + 1 standing retrain with the
# new phase-5 (33 rad/s) curriculum. Runs sequentially (single GPU) — each training is
# immediately followed by its matching eval script so results are ready by morning
# instead of needing to be run by hand one at a time.
#
# Deliberately does NOT use `set -e`: one failed step (e.g. an OOM on one run) should
# not silently kill the rest of an unattended overnight queue. Each step's pass/fail is
# logged instead; check the log for FAILED lines in the morning.
#
# Usage:
#   cd ~/Elm/Code/g1_locomotion
#   nohup ./overnight_train.sh &
#   (or just run it in a tmux/screen session and detach)

set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
ARM_ITERS=5000
STANDING_ITERS=8000

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

# $1 = experiment_name (e.g. arms/g1_arm_ik_left_reward_shape) -> newest run dir, or empty
latest_run_dir() {
    ls -td "logs/rsl_rl/$1"/*/ 2>/dev/null | head -1
}

# $1 = run dir -> highest-iteration checkpoint in it, or empty
latest_checkpoint() {
    [ -n "$1" ] && ls -v "$1"/model_*.pt 2>/dev/null | tail -1
}

train_and_eval_arm() {
    # $1 = gym task id, $2 = experiment_name, $3 = run_name, $4... = extra eval_arm.py args
    local task="$1" exp_name="$2" run_name="$3"; shift 3
    run_step "Train: $run_name" \
        python scripts/rsl_rl/train.py --task "$task" --headless \
            --max_iterations "$ARM_ITERS" --run_name "$run_name"
    local run_dir; run_dir=$(latest_run_dir "$exp_name")
    local ckpt; ckpt=$(latest_checkpoint "$run_dir")
    if [ -n "$ckpt" ]; then
        run_step "Eval: $run_name" \
            python validation/eval_arm.py --checkpoint "$ckpt" --headless "$@"
    else
        echo "[WARN] No checkpoint found under $run_dir for '$run_name' — skipping eval."
    fi
}

echo "Overnight sweep starting $(date). Full log: $LOG_FILE"

# ---------------------------------------------------------------------------
# Arm experiments (3 isolated changes vs. the G1-Arm-IK-Left-v0 baseline, each its own
# gym task + experiment_name — see g1_arm_env.py / agents/rsl_rl_ppo_cfg.py). Full
# 5000-iteration budget each (same as the confirmed baseline run, for a clean
# apples-to-apples comparison) since there's time overnight.
# ---------------------------------------------------------------------------
train_and_eval_arm "G1-Arm-IK-Left-RewardShape-v0" \
    "arms/g1_arm_ik_left_reward_shape" "reward_shape"

train_and_eval_arm "G1-Arm-IK-Left-GoalCurriculum-v0" \
    "arms/g1_arm_ik_left_goal_curriculum" "goal_curriculum"

# WideNet's checkpoint has a different actor/critic layer shape than the baseline
# ([512,256,128] vs [256,128,64]) — eval_arm.py must be told this via --hidden_dims or
# runner.load() fails on a shape mismatch (see eval_arm.py's --hidden_dims docstring).
train_and_eval_arm "G1-Arm-IK-Left-WideNet-v0" \
    "arms/g1_arm_ik_left_wide_net" "wide_net" \
    --hidden_dims 512 256 128

train_and_eval_arm "G1-Arm-IK-Left-Entropy-v0" \
    "arms/g1_arm_ik_left_entropy" "entropy"

# ---------------------------------------------------------------------------
# Standing — full retrain with the new phase-5 (33 rad/s hardware-limit) curriculum
# phase, added 2026-07-07 but never trained until now.
#
# Iteration count: StandingArmTrajectoryDisturbance's phase boundaries are in raw
# env-steps; at num_steps_per_env=24, phase 5 begins at env-step 120,000 -> iteration
# 5000. Phase 4 (the previous hardest phase) got 40,000 steps (~1667 iterations) of
# training time before the curriculum moved on. 8000 total gives phase 5 roughly
# 3000 iterations of exposure — comfortably more than phase 4 got, appropriate given
# phase 5 is both the newest and the hardest (never visually or numerically validated
# before this run, see known_issues.md).
# ---------------------------------------------------------------------------
run_step "Train: standing (6-phase curriculum, $STANDING_ITERS iters)" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-v0 --headless \
        --max_iterations "$STANDING_ITERS" --run_name phase5_curriculum
STANDING_RUN_DIR=$(latest_run_dir "standing/g1_locomotion_flat")
STANDING_CKPT=$(latest_checkpoint "$STANDING_RUN_DIR")
if [ -n "$STANDING_CKPT" ]; then
    # No --phases override: eval_standing.py defaults to sweeping every phase the
    # curriculum currently defines, which now includes the new phase 5 automatically.
    run_step "Eval: standing (all phases incl. new phase 5)" \
        python validation/eval_standing.py --checkpoint "$STANDING_CKPT" --headless
else
    echo "[WARN] No checkpoint found under $STANDING_RUN_DIR for standing — skipping eval."
fi

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Overnight sweep complete."
echo "Full log: $LOG_FILE"
echo "Arm eval summaries: logs/rsl_rl/arms/g1_arm_ik_left_{reward_shape,goal_curriculum,wide_net,entropy}/*/arm_eval/summary.md"
echo "Standing eval summary: $STANDING_RUN_DIR/disturbance_eval/summary.md"
echo "=============================================================="
