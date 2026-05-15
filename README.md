# molecular-autoencoder

[![CI](https://github.com/earth-metabolome-initiative/molecular-autoencoder/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/molecular-autoencoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/molecular-autoencoder/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/molecular-autoencoder)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

Burn components for a molecular autoencoder trained from counted ECFP targets
and molecule-derived descriptor side tasks over PubChem (~123M SMILES) and
ZINC20 (~1G SMILES). The crate provides
deterministic preprocessing, sparse shard IO, Burn batch types, model code,
losses, metrics, and CUDA counted-Tanimoto geometry support.

## Architecture

![Molecular autoencoder architecture](https://raw.githubusercontent.com/earth-metabolome-initiative/molecular-autoencoder/main/docs/model-architecture.svg)

## Training

This command resolves PubChem (~123M SMILES) and ZINC20 (~1G SMILES)
through `smiles-parser`, creates cached numeric shards if needed, and trains the
CUDA model:

The default training architecture uses a 512-d latent space with
4096,2048,1024 encoder hidden widths and a mirrored decoder. The default
side-loss weights are 0.05 for descriptor regression and 0.10 for latent
Tanimoto geometry. The Tanimoto geometry loss uses gap-weighted sampled
softmax cross-entropy over counted-Tanimoto candidate sets, with a default
latent temperature of 0.10.

The command below is tuned for the current training workstation: an NVIDIA
GeForce RTX 5090 with 32607 MiB VRAM, CUDA 12.9, and WSL CUDA libraries under
`/usr/lib/wsl/lib`. It uses `RUSTFLAGS="-C target-cpu=native"` and a 32768
batch size; lower the batch size on smaller GPUs.

```bash
CUDA_PATH=/usr/local/cuda-12.9 \
PATH=/usr/local/cuda-12.9/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
CUDARC_CUDA_VERSION=12090 \
RUSTFLAGS="-C target-cpu=native" \
cargo run --release --no-default-features --features std,cuda-fusion,train,tui,datasets \
  --bin train -- shards/pubchem-zinc20 runs/cuda-ae-pubchem-zinc20-large-100e \
  --datasets pubchem,zinc20 \
  --rows-per-shard 10000000 --epochs 100 --batch-size 32768 --loader-workers 20 \
  --metric-every 50 --descriptor-weight 0.05 --tanimoto-ranking-weight 0.10 \
  --preprocess-threads 64 --cuda-device 0
```

Use `--datasets pubchem` or `--datasets zinc20` to preprocess one source. For
a partial ZINC20 pass, add `--zinc20-chunks FIRST-LAST`. Omit `--datasets` to
train from an existing cached manifest without re-running preprocessing.

Use a fresh checkpoint directory when changing architecture or loss defaults;
`--resume` reuses the existing `model-config.json`. Add `--resume` to continue
from `runs/cuda-ae-pubchem-zinc20-large-100e/model.mpk`,
`runs/cuda-ae-pubchem-zinc20-large-100e/optimizer.mpk`, and
`runs/cuda-ae-pubchem-zinc20-large-100e/state.json`.
