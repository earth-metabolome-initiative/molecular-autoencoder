#!/bin/bash
#SBATCH --job-name=mauto-smoke-1gpu
#SBATCH --account=ac_scscollab
#SBATCH --partition=es0
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:GRTX2080TI:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/%u/molecular-autoencoder/logs/smoke_%j.out
#SBATCH --error=/global/scratch/users/%u/molecular-autoencoder/logs/smoke_%j.err
#
# Smoke-test the multi-GPU plumbing on es0 with a single 2080 Ti. Runs
# train for one epoch capped at --max-train-batches 5, just enough to
# confirm the binary launches, loads the manifest, hits the GPU, and
# writes a checkpoint. Prebuild on the login node first via setup_env.sh.

set -euo pipefail

REPO_DIR="${MAUTO_REPO_DIR:-$HOME/molecular-autoencoder}"
SCRATCH_ROOT="${MAUTO_SCRATCH_ROOT:-/global/scratch/users/$USER/molecular-autoencoder}"
CACHE_DIR="${MAUTO_CACHE_DIR:-$SCRATCH_ROOT/cache}"
CHECKPOINT_DIR="${MAUTO_CHECKPOINT_DIR:-$SCRATCH_ROOT/checkpoints}"
TARGET_DIR="${CARGO_TARGET_DIR:-$SCRATCH_ROOT/target}"
GCC_MODULE="${MAUTO_GCC_MODULE:-gcc/11.4.0}"
CUDA_MODULE="${MAUTO_CUDA_MODULE:-cuda/12.2.1}"

clean_rust_compiler_environment() {
    unset CC CXX AR CFLAGS CXXFLAGS LDFLAGS
    unset CC_x86_64_unknown_linux_gnu CXX_x86_64_unknown_linux_gnu AR_x86_64_unknown_linux_gnu
    unset CFLAGS_x86_64_unknown_linux_gnu CXXFLAGS_x86_64_unknown_linux_gnu
}

load_user_cargo_environment() {
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        . "$HOME/.cargo/env"
    fi
}

clean_rust_compiler_environment
module purge
module load "$GCC_MODULE" "$CUDA_MODULE"
load_user_cargo_environment

cd "$REPO_DIR"

BINARY="$TARGET_DIR/release/train"
if [ ! -x "$BINARY" ]; then
    echo "ERROR: $BINARY missing. Run bash slurm/lrc/setup_env.sh on the login node first."
    exit 1
fi

export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

echo "Host:         $(hostname)"
echo "Start:        $(date -Iseconds)"
echo "Job ID:       ${SLURM_JOB_ID:-<none>}"
echo "CPUs:         ${SLURM_CPUS_ON_NODE:-?}"
echo "GPUs (gres):  ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "Binary:       $BINARY"
echo "Cache:        $CACHE_DIR"
echo "Checkpoints:  $CHECKPOINT_DIR"

"$BINARY" \
    "$CACHE_DIR" \
    "$CHECKPOINT_DIR" \
    --epochs 1 \
    --max-train-batches 5 \
    --max-valid-batches 2 \
    --batch-size 4096 \
    "$@"

echo "Done:         $(date -Iseconds)"
