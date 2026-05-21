#!/bin/bash
# One-time Lawrencium (lrc) login-node setup for molecular-autoencoder.
#
# Run this on lrc-login.lbl.gov before submitting any sbatch in this dir.
# It installs rustup if needed (the cluster's rust/1.70.0 module is too old
# for edition 2024), clones / updates the repo, wires the scratch
# directories into $HOME via symlinks, loads gcc + cuda modules, and runs
# `cargo build --release --features cuda-fusion,datasets --locked`.
#
# Compute-node jobs reuse the prebuilt binary; they do not rebuild.

set -euo pipefail

REPO_URL="${MAUTO_REPO_URL:-https://github.com/earth-metabolome-initiative/molecular-autoencoder.git}"
REPO_DIR="${MAUTO_REPO_DIR:-$HOME/molecular-autoencoder}"
SCRATCH_ROOT="${MAUTO_SCRATCH_ROOT:-/global/scratch/users/$USER/molecular-autoencoder}"
CACHE_DIR="$SCRATCH_ROOT/cache"
CHECKPOINT_DIR="$SCRATCH_ROOT/checkpoints"
LOGS_DIR="$SCRATCH_ROOT/logs"
TARGET_DIR="$SCRATCH_ROOT/target"
MIN_RUST_VERSION="1.85.0"
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

install_rustup() {
    local installer
    installer="$(mktemp)"
    echo "Installing Rust with rustup under $HOME/.cargo"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "$installer"
    sh "$installer" -y --profile minimal --default-toolchain stable
    rm -f "$installer"
    load_user_cargo_environment
}

cargo_version() {
    local output version
    output="$(cargo --version 2> /dev/null)" || return 1
    version="${output#cargo }"
    printf '%s\n' "${version%% *}"
}

version_ge() {
    local candidate="$1"
    local required="$2"
    local cmajor cminor cpatch rmajor rminor rpatch
    IFS=. read -r cmajor cminor cpatch <<< "$candidate"
    IFS=. read -r rmajor rminor rpatch <<< "$required"
    cpatch="${cpatch%%[^0-9]*}"
    rpatch="${rpatch%%[^0-9]*}"
    cmajor="${cmajor:-0}"; cminor="${cminor:-0}"; cpatch="${cpatch:-0}"
    rmajor="${rmajor:-0}"; rminor="${rminor:-0}"; rpatch="${rpatch:-0}"
    if [ "$cmajor" -ne "$rmajor" ]; then [ "$cmajor" -gt "$rmajor" ]; return; fi
    if [ "$cminor" -ne "$rminor" ]; then [ "$cminor" -gt "$rminor" ]; return; fi
    [ "$cpatch" -ge "$rpatch" ]
}

cargo_is_recent_enough() {
    local version
    command -v cargo > /dev/null 2>&1 || return 1
    version="$(cargo_version)" || return 1
    version_ge "$version" "$MIN_RUST_VERSION"
}

ensure_cargo() {
    load_user_cargo_environment
    if cargo_is_recent_enough; then return; fi

    # The cluster's rust/1.70.0 module is far too old for edition 2024.
    # Skip the module entirely and install rustup stable.
    if command -v cargo > /dev/null 2>&1; then
        echo "Found $(cargo --version), but this project requires Rust $MIN_RUST_VERSION+."
    fi

    if ! command -v rustup > /dev/null 2>&1; then
        install_rustup
    fi
    rustup toolchain install stable --profile minimal
    rustup default stable
    load_user_cargo_environment

    if ! cargo_is_recent_enough; then
        echo "ERROR: cargo is still older than $MIN_RUST_VERSION after rustup setup."
        exit 1
    fi
}

load_cluster_modules() {
    if ! command -v module > /dev/null 2>&1; then
        echo "ERROR: 'module' command not found. Are you on a Lawrencium host?"
        exit 1
    fi
    module purge
    module load "$GCC_MODULE" "$CUDA_MODULE"
    echo "Loaded modules:"
    module list 2>&1 | sed 's/^/  /'
    echo "nvcc: $(command -v nvcc) -> $(nvcc --version | tail -1)"
}

link_scratch_directory() {
    local link_path="$1"
    local target_path="$2"
    if [ -L "$link_path" ]; then
        echo "$link_path already points to $(readlink "$link_path")"
    elif [ -e "$link_path" ]; then
        echo "ERROR: $link_path exists and is not a symlink. Move it before rerunning."
        exit 1
    else
        ln -s "$target_path" "$link_path"
        echo "Created $link_path -> $target_path"
    fi
}

clean_rust_compiler_environment
load_cluster_modules
ensure_cargo

if [ -d "$REPO_DIR/.git" ]; then
    echo "Updating $REPO_DIR"
    git -C "$REPO_DIR" pull --ff-only
elif [ -e "$REPO_DIR" ]; then
    echo "ERROR: $REPO_DIR exists but is not a git checkout."
    exit 1
else
    echo "Cloning $REPO_URL into $REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR"
fi

mkdir -p "$CACHE_DIR" "$CHECKPOINT_DIR" "$LOGS_DIR" "$TARGET_DIR"
link_scratch_directory "$REPO_DIR/cache" "$CACHE_DIR"
link_scratch_directory "$REPO_DIR/checkpoints" "$CHECKPOINT_DIR"
link_scratch_directory "$REPO_DIR/logs" "$LOGS_DIR"

cd "$REPO_DIR"

# Build target also lives on scratch so the 30 GB $HOME quota stays clean.
export CARGO_TARGET_DIR="$TARGET_DIR"

# x86-64-v3 (Haswell+ AVX2) is the safe baseline across Lawrencium
# partitions. target-cpu=native would SIGILL on older compute nodes whose
# microarchitecture is older than the login node, as the spectral-similarities
# slurm scripts already learned the hard way.
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=x86-64-v3}"

echo "Toolchain: $(rustc --version)"
echo "Cargo:     $(cargo --version)"
echo "Target:    $CARGO_TARGET_DIR"
echo "RUSTFLAGS: $RUSTFLAGS"

cargo build --release --features cuda-fusion,datasets --locked

"$CARGO_TARGET_DIR/release/train" --help > /dev/null
echo "train --help: OK"

cat <<EOM

Setup complete.
  Repo:        $REPO_DIR
  Cache:       $CACHE_DIR  (linked as $REPO_DIR/cache)
  Checkpoints: $CHECKPOINT_DIR  (linked as $REPO_DIR/checkpoints)
  Logs:        $LOGS_DIR  (linked as $REPO_DIR/logs)
  Binary:      $CARGO_TARGET_DIR/release/train

Next steps:
  bash slurm/lrc/submit.sh smoke              # 1x 2080 Ti, --max-train-batches 5
  bash slurm/lrc/submit.sh ddp                # 4x 2080 Ti, full DDP run
  bash slurm/lrc/submit.sh smoke --dry-run    # print the sbatch command without submitting
EOM
