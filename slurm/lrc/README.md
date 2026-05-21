# Lawrencium (lrc) SLURM scripts

Multi-GPU smoke tests and full DDP runs on the LBL Lawrencium / Einsteinium `es0` partition (4x NVIDIA RTX 2080 Ti per node, fairshare-only, infinite walltime).

This directory mirrors the layout of [`spectral-similarities-by-peaks/slurm/lrc`](https://github.com/earth-metabolome-initiative/spectral-similarities-by-peaks/tree/main/slurm/lrc), adapted for a CUDA + DDP workload. The model itself runs single-node multi-GPU via `burn-collective` (see `src/bin/train/ddp.rs`).

## One-time login-node setup

SSH into `lrc-login.lbl.gov` (PIN+OTP). To avoid retyping the OTP every command, generate a 12-hour cert:

```
ssh-keygen -t ed25519  # if you do not already have one
request_cert.sh -p lrc # interactive: enter username, PIN+OTP
```

Then on the login node:

```bash
git clone https://github.com/earth-metabolome-initiative/molecular-autoencoder.git ~/molecular-autoencoder
cd ~/molecular-autoencoder
bash slurm/lrc/setup_env.sh
```

`setup_env.sh` is idempotent. It:

- Loads `gcc/11.4.0` and `cuda/12.2.1` Lmod modules.
- Installs `rustup` under `$HOME/.cargo` (the cluster's `rust/1.70.0` module is too old for edition 2024).
- Creates the scratch tree under `/global/scratch/users/$USER/molecular-autoencoder/{cache,checkpoints,logs,target}` and symlinks the first three back into the repo.
- Builds the release binary with `CARGO_TARGET_DIR` on scratch and `RUSTFLAGS=-C target-cpu=x86-64-v3` for cross-partition portability. (`target-cpu=native` SIGILLs on older lr4 nodes per the spectral-sim experience; safer to stick with the v3 baseline.) Defaults to the `cuda-no-fusion,datasets` feature combo because `cuda-fusion`'s first-batch JIT took 75+ minutes on the es0 2080 Tis with re-fires on every new tensor shape (backward, optimizer step, validation). Override with `MAUTO_CARGO_FEATURES=cuda-fusion,datasets` if you have time to wait out the fusion warmup.
- Sanity-checks `train --help`.

Re-run `setup_env.sh` after `git pull` to rebuild.

### Override defaults

| env var | default | purpose |
| --- | --- | --- |
| `MAUTO_REPO_DIR` | `$HOME/molecular-autoencoder` | repo checkout location |
| `MAUTO_SCRATCH_ROOT` | `/global/scratch/users/$USER/molecular-autoencoder` | scratch tree root |
| `MAUTO_GCC_MODULE` | `gcc/11.4.0` | Lmod gcc module |
| `MAUTO_CUDA_MODULE` | `cuda/12.2.1` | Lmod cuda module |
| `CARGO_TARGET_DIR` | `$SCRATCH_ROOT/target` | cargo build target dir |
| `RUSTFLAGS` | `-C target-cpu=x86-64-v3` | rustc codegen flags |

## Submitting jobs

```bash
bash slurm/lrc/submit.sh smoke              # 1 GPU, 30 min, 5 batches
bash slurm/lrc/submit.sh ddp                # 4 GPUs exclusive, 12h
bash slurm/lrc/submit.sh smoke --dry-run    # print the sbatch line, do not submit
bash slurm/lrc/submit.sh smoke -- --datasets pubchem --force-preprocess   # forward extra args to `train`
```

### `smoke_1gpu.sh`

- `--gres=gpu:GRTX2080TI:1`, `--cpus-per-task=8`, `--mem=24G`, 30 minute walltime.
- Polite request that does not block all four GPUs on the node.
- Runs `train --epochs 1 --max-train-batches 5 --max-valid-batches 2 --batch-size 4096`.
- Use case: prove the binary launches, reads the manifest, hits the GPU, writes a checkpoint.

### `ddp_4gpu.sh`

- `--gres=gpu:GRTX2080TI:4`, `--cpus-per-task=8`, `--mem=80G`, 12 hour walltime.
- es0 nodes have only `CPUTot=8` total cores despite carrying 4 GPUs, so 8 is the per-node ceiling. The partition runs `OverSubscribe=FORCE:1`, so `--exclusive` would conflict.
- The sbatch leaves `--batch-size` unset so the binary default (4096) applies and the caller can override via `-- --batch-size N`. 2080 Ti has 11 GB VRAM, so 4096 is the safe starting point; bisect up (8192) if VRAM permits.
- DDP needs NCCL at runtime (cudarc `dlopen`s `libnccl.so.2`, requires `ncclAlltoAll` from NCCL 2.19+). The Lmod `cuda/12.2.1` module ships no NCCL, and `nvhpc/23.11` only ships NCCL 2.18.5 (too old). `setup_env.sh` pip-downloads the `nvidia-nccl-cu12` manylinux2014 wheel (currently 2.21+) once and unpacks it to `$HOME/local/nccl-cu12/nvidia/nccl/lib/`. `ddp_4gpu.sh` then prepends that to `LD_LIBRARY_PATH`. Override with `MAUTO_NCCL_LIB_DIR` if you keep an NCCL install elsewhere.
- Runs `train --cuda-devices 0,1,2,3 --epochs 20 --batch-size 16384`.
- Use case: validate `burn-collective` all-reduce on real hardware.

#### GPU topology on es0

Measured on `n0028.es1`:

```
GPU0 -- NV2 -- GPU1     (NVLink pair on NUMA 0)
  \              /
   NODE (PCIe across host bridges, same NUMA node)
  /              \
GPU2 -- NV2 -- GPU3     (NVLink pair on NUMA 1)
```

burn-collective's all-reduce gets ~50 GB/s NVLink bandwidth inside each pair (GPU0-1 and GPU2-3) and drops to PCIe NODE between pairs. With 4-way DDP the cross-pair step is the bottleneck, not the per-pair NVLink. Throughput numbers should be interpreted against that; correctness is unaffected.

The Mellanox `mlx5_0` InfiniBand NIC visible in the same `nvidia-smi topo -m` is the hook for the future multi-node milestone (NCCL/burn-collective over IB).

Both jobs assume the shard cache already exists at `$MAUTO_CACHE_DIR`. The first time, append `-- --datasets pubchem` (or `--datasets zinc20 --zinc20-chunks 1-1` for a smaller first slice) to trigger preprocessing inside the same job. Subsequent runs reuse the manifest automatically.

## Logs and monitoring

- stdout / stderr: `/global/scratch/users/$USER/molecular-autoencoder/logs/{smoke,ddp}_<jobid>.{out,err}`
- Queue status: `squeue -u $USER` (or `squeue -u $USER -O JobID,Partition,Name,State,TimeUsed,TimeLimit,ReasonList`)
- Live tail: `tail -f $MAUTO_SCRATCH_ROOT/logs/<jobtype>_<jobid>.out`
- Cancel: `scancel <jobid>`
- After completion, sacct: `sacct -j <jobid> -o JobID,JobName,State,Elapsed,MaxRSS,AveCPU,ReqGRES`

## Troubleshooting

- `srun: job <id> queued and waiting for resources` is normal on es0 (only two nodes; you wait when both are busy).
- If `setup_env.sh` ever picks up the `rust/1.70.0` module, it will reject and fall back to rustup; if you see that path, ensure `module purge` happens before any `rustup` call.
- `nvcc: command not found` inside an sbatch usually means the `module load` failed silently. Submit with `--dry-run`, log into the compute node interactively, and rerun the module loads to see the actual error.
- Out-of-memory at batch size 16384 on the 2080 Ti (11 GB): reduce `--batch-size` to 8192 and retry.

## Sources

- LBL Einsteinium GPU docs: https://scienceit-docs.lbl.gov/hpc/systems/einsteinium/
- LBL CUDA module guide: https://scienceit-docs.lbl.gov/hpc/software/libraries/cuda/
- LBL MFA / cert workflow: https://scienceit-docs.lbl.gov/hpc/accounts/loggingin/
