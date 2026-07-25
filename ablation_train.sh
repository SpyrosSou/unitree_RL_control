#!/usr/bin/env bash
# 2026-07-23 walking-fix ablation: 4 sequential 1k-iteration runs, each testing one
# candidate fix (or all three) in isolation against the unmodified Unitree baseline —
# see the Ablation*EnvCfg classes in g1_locomotion_env_cfg.py for what each changes.
# Compare against logs/rsl_rl/walking/base/2026-07-22_08-05-36 (the existing,
# already-trained unmodified-baseline run) as the "0 changes" control — no need to
# retrain that one.
set -o pipefail

CONDA_ENV="isaac_g1_control"
PROJECT_ROOT="$HOME/Elm/Code/g1_locomotion"
ITERS=1000
SEED=42
NUM_ENVS=3072

cd "$PROJECT_ROOT"
mkdir -p phase_logs
LOG_FILE="$PROJECT_ROOT/phase_logs/ablation_$(date +%Y-%m-%d_%H-%M-%S).log"

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

echo "Ablation run starting $(date). Full log: $LOG_FILE"

run_step "Ablation 1/4: termination penalty only" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-Ablation-TermPenalty-v0 \
        --headless --max_iterations "$ITERS" --num_envs "$NUM_ENVS" --seed "$SEED"

run_step "Ablation 2/4: curriculum loosening only" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-Ablation-Curriculum-v0 \
        --headless --max_iterations "$ITERS" --num_envs "$NUM_ENVS" --seed "$SEED"

run_step "Ablation 3/4: lighter action_rate/joint_deviation_legs only" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-Ablation-RewardWeights-v0 \
        --headless --max_iterations "$ITERS" --num_envs "$NUM_ENVS" --seed "$SEED"

run_step "Ablation 4/4: all three combined" \
    python scripts/rsl_rl/train.py --task G1-Locomotion-Velocity-Ablation-All-v0 \
        --headless --max_iterations "$ITERS" --num_envs "$NUM_ENVS" --seed "$SEED"

echo ""
echo "=============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ablation run complete."
echo "Full log: $LOG_FILE"
echo "Compare Episode_Reward/track_lin_vel_xy, Episode_Reward/action_rate, and"
echo "Episode_Termination/bad_orientation across:"
echo "  logs/rsl_rl/walking/base/2026-07-22_08-05-36              (0 changes, control)"
echo "  logs/rsl_rl/walking/ablation_term_penalty/<newest>        (termination penalty)"
echo "  logs/rsl_rl/walking/ablation_curriculum/<newest>          (curriculum loosening)"
echo "  logs/rsl_rl/walking/ablation_reward_weights/<newest>      (lighter action_rate/joint_dev)"
echo "  logs/rsl_rl/walking/ablation_all/<newest>                 (all three combined)"
echo "=============================================================="
