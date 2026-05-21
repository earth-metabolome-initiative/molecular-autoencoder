#!/bin/bash
#SBATCH --job-name=mauto-ddp-4gpu
#SBATCH --account=ac_scscollab
#SBATCH --partition=es0
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:GRTX2080TI:4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/global/scratch/users/%u/molecular-autoencoder/logs/ddp_%j.out
#SBATCH --error=/global/scratch/users/%u/molecular-autoencoder/logs/ddp_%j.err
#
# Full DDP training run on a 4x 2080 Ti es0 node. burn-collective handles
# per-step gradient all-reduce across the four peer threads (one per GPU)
# inside a single process. Build the binary on the login node first via
# setup_env.sh; this script reuses it.

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

# cudarc 0.19.4 dlopens libnccl.so.2 from LD_LIBRARY_PATH and needs
# symbols (ncclAlltoAll) added in NCCL 2.19+. setup_env.sh installs the
# nvidia-nccl-cu12 manylinux2014 wheel under $HOME by default;
# MAUTO_NCCL_LIB_DIR can point this script at a different libnccl
# install if the user maintains one elsewhere.
NCCL_LIB_DIR="${MAUTO_NCCL_LIB_DIR:-$HOME/local/nccl-cu12/nvidia/nccl/lib}"
module purge
module load "$GCC_MODULE" "$CUDA_MODULE"
load_user_cargo_environment

if [ ! -f "$NCCL_LIB_DIR/libnccl.so.2" ]; then
    echo "ERROR: libnccl.so.2 not found at $NCCL_LIB_DIR"
    echo "Run 'bash slurm/lrc/setup_env.sh' on the login node first, or"
    echo "set MAUTO_NCCL_LIB_DIR to point at an existing libnccl.so.2 directory."
    exit 1
fi
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"
echo "NCCL libdir:  $NCCL_LIB_DIR"

# cubecl-cuda reads CUDA_PATH specifically; the Lmod cuda module only
# sets CUDA_HOME / CUDA_ROOT. Derive CUDA_PATH from nvcc's location so it
# tracks whichever cuda module is loaded.
if [ -z "${CUDA_PATH:-}" ]; then
    nvcc_path="$(command -v nvcc)"
    if [ -z "$nvcc_path" ]; then
        echo "ERROR: nvcc not in PATH after 'module load $CUDA_MODULE'."
        exit 1
    fi
    CUDA_PATH="$(dirname "$(dirname "$nvcc_path")")"
    export CUDA_PATH
fi
echo "CUDA_PATH:    $CUDA_PATH"

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
nvidia-smi topo -m | sed 's/^/  /'
echo "Binary:       $BINARY"
echo "Cache:        $CACHE_DIR"
echo "Checkpoints:  $CHECKPOINT_DIR"

"$BINARY" \
    "$CACHE_DIR" \
    "$CHECKPOINT_DIR" \
    --cuda-devices 0,1,2,3 \
    --epochs 20 \
    "$@"

echo "Done:         $(date -Iseconds)"
