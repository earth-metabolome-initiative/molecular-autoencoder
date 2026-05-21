#!/bin/bash
# Submit a molecular-autoencoder job on Lawrencium (lrc) es0.
#
# Usage:
#   bash slurm/lrc/submit.sh <smoke|ddp> [--dry-run] [-- TRAIN_ARGS...]
#
# Extra args after `--` are forwarded to the train binary inside the
# sbatch script (e.g. `--datasets pubchem --force-preprocess`).

set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  bash slurm/lrc/submit.sh <smoke|ddp> [--dry-run] [-- TRAIN_ARGS...]

Targets:
  smoke   1x GRTX2080TI, --max-train-batches 5, 30 min walltime
  ddp     4x GRTX2080TI exclusive, 12 hour walltime, full DDP run

Options:
  --dry-run    Print the sbatch command without submitting.

Anything after `--` is forwarded to the train binary, e.g.
  bash slurm/lrc/submit.sh smoke -- --datasets pubchem
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

TARGET="$1"
shift

DRY_RUN=false
TRAIN_ARGS=()
PASSTHROUGH=false
for arg in "$@"; do
    if [ "$PASSTHROUGH" = true ]; then
        TRAIN_ARGS+=("$arg")
        continue
    fi
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --) PASSTHROUGH=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg"; usage; exit 1 ;;
    esac
done

case "$TARGET" in
    smoke) JOB_SCRIPT="slurm/lrc/smoke_1gpu.sh" ;;
    ddp)   JOB_SCRIPT="slurm/lrc/ddp_4gpu.sh" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown target: $TARGET"; usage; exit 1 ;;
esac

REPO_DIR="${MAUTO_REPO_DIR:-$HOME/molecular-autoencoder}"
SCRATCH_ROOT="${MAUTO_SCRATCH_ROOT:-/global/scratch/users/$USER/molecular-autoencoder}"
LOGS_DIR="$SCRATCH_ROOT/logs"

mkdir -p "$LOGS_DIR"

if [ ! -f "$REPO_DIR/$JOB_SCRIPT" ]; then
    echo "ERROR: $REPO_DIR/$JOB_SCRIPT not found. Did you run setup_env.sh?"
    exit 1
fi

CMD=(sbatch "$REPO_DIR/$JOB_SCRIPT")
if [ "${#TRAIN_ARGS[@]}" -gt 0 ]; then
    CMD+=("${TRAIN_ARGS[@]}")
fi

if [ "$DRY_RUN" = true ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

"${CMD[@]}"
