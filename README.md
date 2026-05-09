# molecular-autoencoder

[![CI](https://github.com/earth-metabolome-initiative/molecular-autoencoder/actions/workflows/ci.yml/badge.svg)](https://github.com/earth-metabolome-initiative/molecular-autoencoder/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/earth-metabolome-initiative/molecular-autoencoder/branch/main/graph/badge.svg)](https://codecov.io/gh/earth-metabolome-initiative/molecular-autoencoder)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust 2024](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

Burn components for a PubChem-scale molecular autoencoder trained from counted
ECFP targets and molecule-derived descriptor side tasks. The crate provides
deterministic preprocessing, sparse shard IO, Burn batch types, model code,
losses, metrics, and CUDA counted-Tanimoto ranking support.

## Architecture

![Molecular autoencoder architecture](https://raw.githubusercontent.com/earth-metabolome-initiative/molecular-autoencoder/main/docs/model-architecture.svg)

## Training

This command resolves PubChem SMILES through `smiles-parser`, creates cached
numeric shards if needed, and trains the CUDA model:

```bash
CUDA_PATH=/usr/local/cuda-12.9 \
PATH=/usr/local/cuda-12.9/bin:$PATH \
LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
CUDARC_CUDA_VERSION=12090 \
RUSTFLAGS="-C target-cpu=native" \
cargo run --release --no-default-features --features std,cuda-fusion,train,tui,datasets \
  --example train_cached_shards -- pubchem shards/pubchem runs/cuda-ae \
  --rows-per-shard 10000000 --epochs 10 --batch-size 24576 --loader-workers 6 \
  --cuda-device 0
```

Add `--resume` to continue from `runs/cuda-ae/model.mpk`,
`runs/cuda-ae/optimizer.mpk`, and `runs/cuda-ae/state.json`.
