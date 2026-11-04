# 23-job benchmark results

All runs execute the 23-job benchmark (`3 steps` per job across `13` Qwen3-0.6B LoRA runs, `6` Gemma-4-E2B LoRA runs, `2` Qwen3-0.6B time-sliced full fine-tunes, `1` Gemma-4-E2B time-sliced full fine-tune, and `1` Gemma-4-E2B exclusive full fine-tune) on one `8 x H100 80 GB` node in the `perf` cluster. Every reported run completed `23/23` jobs with `0` job errors and `0` Redis errors unless noted.

## Summary across image waves

| Wave | Image | Runs | Makespans (s) | Median (s) | Mean (s) | Best (s) |
|---|---|---|---|---|---|---|
| Baseline (`726a48d`, paired in E23) | `main1` | 5 | `152.8, 161.4, 150.1, 170.2, 151.2` | `152.8` | `157.1` | `150.1` |
| Baseline (`726a48d`, full day B0 to E23) | `main1` | 34 | `148.0` to `186.2` | `159.2` | `160.4` | `148.0` |
| E23 (Liger kernels + 128-token CUDA pad + GPU adapter init) | `hc-liger1` | 5 | `149.3, 152.2, 150.5, 153.1, 160.0` | `152.2` | `153.0` | `149.3` |
| E24 (E23 + parallel host slab pin + size-balanced shards + drop `self.lm`) | `hc-stack2` | 2 | `185.3, 142.5` | `163.9` | `163.9` | `142.5` |
| E25a (E24 + in-place move on all single-GPU trainers, reverted) | `hc-stack3` | 2 | `177.5, 179.3` | `178.4` | `178.4` | `177.5` |
| E25b (E24 + in-place move restricted to `w.kind.lora`) | `hc-stack4` | 3 | `150.0, 140.7, 150.6` | `150.0` | `147.1` | `140.7` |

## Per-group completion times (`report.md`)

Each table shows `first step done (median s) | last step done (median / max s) | step time (median s)` per job group.

### Baseline `main1` (`btn-044953`, makespan `160.0 s`, Jain `0.917`)

| Group | Jobs | First step done (s) | Last step done (med / max s) | Step time (med s) |
|---|---|---|---|---|
| `gemma-fft-x` | 1 | 131 | 158 / 158 | 14 |
| `gemma-fft-ts` | 1 | 125 | 160 / 160 | 17 |
| `qwen-fft-ts` | 2 | 44 | 70 / 84 | 9 |
| `gemma-lora` | 6 | 103 | 132 / 138 | 13 |
| `qwen-lora` | 13 | 96 | 145 / 153 | 21 |

### E23 `hc-liger1` (`btn-044521`, makespan `150.5 s`, Jain `0.913`)

| Group | Jobs | First step done (s) | Last step done (med / max s) | Step time (med s) |
|---|---|---|---|---|
| `gemma-fft-x` | 1 | 118 | 146 / 146 | 14 |
| `gemma-fft-ts` | 1 | 106 | 151 / 151 | 23 |
| `qwen-fft-ts` | 2 | 38 | 65 / 72 | 10 |
| `gemma-lora` | 6 | 94 | 128 / 129 | 15 |
| `qwen-lora` | 13 | 94 | 146 / 149 | 24 |

### E24 `hc-stack2` (`btn-050351`, makespan `142.5 s`, Jain `0.917`)

| Group | Jobs | First step done (s) | Last step done (med / max s) | Step time (med s) |
|---|---|---|---|---|
| `gemma-fft-x` | 1 | 83 | 116 / 116 | 17 |
| `gemma-fft-ts` | 1 | 99 | 130 / 130 | 15 |
| `qwen-fft-ts` | 2 | 43 | 67 / 83 | 5 |
| `gemma-lora` | 6 | 98 | 126 / 127 | 14 |
| `qwen-lora` | 13 | 87 | 138 / 143 | 21 |

### E25b `hc-stack4` run 1 (`btn-051847`, makespan `150.0 s`, Jain `0.898`)

| Group | Jobs | First step done (s) | Last step done (med / max s) | Step time (med s) |
|---|---|---|---|---|
| `gemma-fft-x` | 1 | 109 | 141 / 141 | 16 |
| `gemma-fft-ts` | 1 | 102 | 145 / 145 | 22 |
| `qwen-fft-ts` | 2 | 45 | 59 / 60 | 7 |
| `gemma-lora` | 6 | 77 | 106 / 107 | 14 |
| `qwen-lora` | 13 | 88 | 148 / 150 | 20 |

### E25b `hc-stack4` run 2 (`btn-052156`, makespan `140.7 s`, Jain `0.907`)

| Group | Jobs | First step done (s) | Last step done (med / max s) | Step time (med s) |
|---|---|---|---|---|
| `gemma-fft-x` | 1 | 104 | 137 / 137 | 17 |
| `gemma-fft-ts` | 1 | 106 | 141 / 141 | 17 |
| `qwen-fft-ts` | 2 | 42 | 55 / 56 | 6 |
| `gemma-lora` | 6 | 75 | 103 / 104 | 13 |
| `qwen-lora` | 13 | 88 | 130 / 139 | 19 |

### E25b `hc-stack4` run 3 (`btn-052516`, makespan `150.6 s`, Jain `0.918`)

| Group | Jobs | First step done (s) | Last step done (med / max s) | Step time (med s) |
|---|---|---|---|---|
| `gemma-fft-x` | 1 | 107 | 138 / 138 | 15 |
| `gemma-fft-ts` | 1 | 107 | 145 / 145 | 19 |
| `qwen-fft-ts` | 2 | 69 | 82 / 108 | 6 |
| `gemma-lora` | 6 | 73 | 102 / 103 | 14 |
| `qwen-lora` | 13 | 106 | 148 / 151 | 20 |

## Where the 140.7 s goes in `btn-052156` (and why `btn-052516` took 150.6 s)

### Client view summed over 3 steps (`btn-052156` vs `btn-052516`, medians in seconds)

| Group | `btn-052156` save | `btn-052156` train | `btn-052156` sample | `btn-052156` end | `btn-052516` save | `btn-052516` train | `btn-052516` sample | `btn-052516` end |
|---|---|---|---|---|---|---|---|---|
| `gemma-fft-ts` | 10.7 | 40.9 | 19.8 | 140.7 | 15.7 | 52.3 | 19.4 | 144.9 |
| `gemma-fft-x` | 10.7 | 38.8 | 18.7 | 137.1 | 10.4 | 49.9 | 18.3 | 138.3 |
| `gemma-lora` | 2.8 | 24.9 | 30.1 | 103.6 | 2.6 | 24.1 | 31.2 | 103.1 |
| `qwen-fft-ts` | 0.7 | 11.0 | 14.7 | 56.1 | 0.8 | 10.8 | 19.3 | 108.2 |
| `qwen-lora` | 0.5 | 25.3 | 62.5 | 138.5 | 0.9 | 56.5 | 47.4 | 150.6 |

### Per-worker GPU accounting (`btn-052156`, `140.7 s`)

| Worker | Role and group | Moves | `move_s` | `wait_s` | `hold_s` | `busy_s` | Top commands (`seconds`, `count`) |
|---|---|---|---|---|---|---|---|
| `w121068` | `trai-gemma-fft-xx1` | 0 | 0.0 | 0.0 | 122.9 | 85.9 | `optim_step` (`24.6 s`, 3), `save_weights_for_sampler` (`20.3 s`, 5), `save_weights` (`19.4 s`, 1) |
| `w121059` | `trai-gemma-fftx1` | 0 | 0.0 | 0.0 | 127.3 | 88.4 | `optim_step` (`27.1 s`, 3), `save_weights` (`23.8 s`, 1), `save_weights_for_sampler` (`16.0 s`, 5) |
| `w121071` | `trai-qwen-lorax13` | 2 | 0.9 | 0.0 | 99.1 | 78.9 | `forward_backward` (`49.0 s`, 39), `save_weights` (`9.1 s`, 13), `create_model` (`8.4 s`, 13) |
| `w121050` | `trai-gemma-lorax6` | 2 | 7.0 | 3.7 | 69.8 | 54.7 | `forward_backward` (`40.0 s`, 18), `create_model` (`8.0 s`, 6), `save_weights_for_sampler` (`3.3 s`, 30) |
| `w121056` | `trai-qwen-fftx1` | 0 | 0.0 | 2.4 | 39.5 | 13.5 | `forward_backward` (`7.7 s`, 3), `create_model` (`3.5 s`, 1), `save_weights_for_sampler` (`1.1 s`, 5) |
| `w121078` | `trai-qwen-fftx1` | 0 | 0.0 | 0.0 | 38.4 | 13.8 | `forward_backward` (`8.1 s`, 3), `create_model` (`3.6 s`, 1), `save_weights_for_sampler` (`1.1 s`, 5) |
| `w121144` | `samp-qwen-lorax52` | 0 | 0.0 | 0.0 | 108.6 | 90.0 | 52 sampling sessions across 13 LoRA runs |
| `w121198` | `samp-gemma-lorax24` | 0 | 0.0 | 0.2 | 67.1 | 49.8 | 24 sampling sessions across 6 LoRA runs |
| `w121227` | `samp-gemma-fft-xx4` | 0 | 0.0 | 0.0 | 97.1 | 20.6 | vLLM engine boot (~`49 s`) + 4 sampling sessions |
| `w121232` | `samp-gemma-fftx4` | 0 | 0.0 | 0.0 | 100.5 | 21.5 | vLLM engine boot (~`49 s`) + 4 sampling sessions |
| `w121132` | `samp-qwen-fftx8` | 0 | 0.0 | 0.0 | 27.7 | 19.7 | 8 sampling sessions across 2 full-FT runs |

### Why `btn-052516` took `150.6 s` vs `btn-052156`'s `140.7 s`

In `btn-052516`, at `t = 46.07 s`, both Gemma-4-E2B full-FT trainers (`w121659` and `w121663`) finished step 1 `forward_backward` and entered `optim_step` #1, allocating `2 x 36.3 GiB = 72.6 GiB` of pinned host buffers for offloaded AdamW moments. With `/ckpt-cache` holding `1.44 TB` of Linux page cache and `/sys/kernel/mm/transparent_hugepage/defrag` set to `[madvise]`, calling `m.madvise(mmap.MADV_HUGEPAGE)` stalled 64 threads in synchronous kernel THP compaction (`322,492` `compact_stall` events in `/proc/vmstat`):

1. `w121659` and `w121663` spent `31.05 s` and `30.90 s` in `optim_step` #1 (`t = 46.07 s` to `t = 77.14 s`), compared to `18.18 s` and `20.76 s` in `btn-052156` and `5.57 s` to `7.54 s` in `E24` when `MemFree` was high.
2. During that same window (`t = 48.26 s` to `t = 74.33 s`), `w121667` (`trai-qwen-fftx1` on GPU 2) stalled `14.79 s` in `save_weights` (`7.2 GiB`) and `11.25 s` in `park_on_host` waiting on host pinning and `cudaHostRegister`.
3. `w121677` (`trai-qwen-lorax13`), which had moved in place onto GPU 2 in `0.55 s` at `t = 30.22 s`, requested GPU 2 at `t = 53.54 s` and waited `20.8 s` until `w121667` finished parking at `t = 74.33 s`, pushing `qwen-lora` completion from `138.5 s` (`btn-052156`) to `150.6 s` (`btn-052516`).

Removing `m.madvise(mmap.MADV_HUGEPAGE)` in `pinned_bytes` pins `2 x 36 GiB` concurrently on GPUs 6 and 7 in `4.23 s` to `4.30 s` with `0` `compact_stall` events.

## Full hillclimb decision log (`decision.tsv`)

| ID | Hypothesis | Change | Before | After | Delta | Verdict | Note |
|---|---|---|---|---|---|---|---|
| B0 | Baseline on `origin/main` (`726a48d`) | None | - | `162.8 s` med (`n=9`) | `0.0 s` | baseline | Range `148.2` to `185.6 s` (`1` run `22/23`, `8` runs `23/23`) |
| E1 | Raising `TOKEN_BUDGET` from `4096` to `16384` reduces passes | `TOKEN_BUDGET=16384` | `1.47 s` | OOM on Gemma LoRA | regression | reverted | Gemma LoRA OOMs at `16384` without grad checkpointing |
| S1 | Deleting delta checkpoints simplifies `Store` | Drop chain/sparse delta upload (`-146` LOC) | `160.1 s` | `167.9 s` | `+7.8 s` | reverted | Slower `3/3` paired rounds |
| E2 | Compare Nov 1 image `snap2` (`88fb643`) on today's cluster | A/B `snap2` vs `main` (`n=5` paired) | `157.6 s` | `142.3 s` | `-15.3 s` | baseline-ref | Proves gap is in code, not cluster hardware |
| E4 | Background GCS uploads contend with trainers | `TINYRL_NO_UPLOAD=1` | `160.1 s` | `171.2 s` | `+11.1 s` | reverted | Upload contention refuted |
| E5/E8 | Immediate session teardown (`#56`) triggers exit/park storms | Disable immediate `DELETE_MODEL` / `UNPLACED` (`n=5`) | `168.4 s` | `151.2 s` | `-17.2 s` | finding | Isolates `#56` teardown cost; keep `#56` semantics and fix its root causes |
| E6 | Node daemon `terminate()` stalls GPU handoff | Delay `proc.terminate()` by `120 s` | `160.1 s` | `160.5 s` | `+0.4 s` | reverted | Trainers exit on their own via `release_gpu_and_worker` |
| E7/E7b | Sharing GPUs across LoRA samplers saves `2` GPUs | Allow LoRA samplers `0.42` GPU budget (`n=7`) | `163.8 s` | `154.0 s` | `-9.8 s` | reverted | Unsafe for large LoRA base models |
| E9 | Exiting without `park_on_host` speeds up teardown | Skip `park_on_host` on process exit | `158.0 s` | `174.1 s` | `+16.1 s` | reverted | Next holder waits for full process exit anyway |
| E10 | `torch.compile` on `chunk_logprobs` output head | `@torch.compile` on `chunk_logprobs` | `158.0 s` | `161.2 s` | `+3.2 s` | reverted | Per-process Inductor compile cost exceeds savings |
| E11 | Keeping emptied trainers warm avoids cold restarts | Keep `place.emptied` workers alive | `158.0 s` | `165.3 s` | `+7.3 s` | reverted | Idle processes hold host memory |
| E13 | `delete_adapter` on LoRA teardown is slow | No-op `lm.delete_adapter(run)` | `173.0 s` | `180.1 s` | `+7.1 s` | reverted | Adapter deletion is `< 0.1 s` |
| E14/E15 | Memory-estimated sampler placement | Place samplers by `estimator.estimate` | `158.0 s` | `164.5 s` | `+6.5 s` | reverted | Slower `3/3` |
| E16 | Move trainers in place across GPUs (`v1`) | `torch.cuda.set_device` on main thread only | `162.4 s` | `286.8 s` | `+124.4 s` | reverted | `set_device` is thread-local; `to_thread` stayed on old GPU |
| M2 | Build PEFT adapters directly on GPU | `with torch.device(self.lm.device): add_adapter` | `0.417 s` | `0.280 s` | `-32.9%` | kept | Included in Commit 1 |
| E17 | Move trainers in place (`v2`, 1-thread CUDA executor) | Pin default executor to `set_device(gpu)` (`n=4`) | `162.4 s` | `154.9 s` | `-7.5 s` | candidate | `3/4` faster; Gemma LoRA `105-120 s` vs `131-143 s` (`4/4`) |
| E18 | Split 13 Qwen LoRA runs across 2 trainers | `RUNS_PER_TRAINER=7` (`n=4`) | `165.1 s` | `180.0 s` | `+14.9 s` | reverted | 8 GPUs are already saturated |
| E19 | Full-FT CUDA IPC weight transfer prototype | Hold unpacked `9.3 GiB` bf16 copy for IPC | `162.4 s` | `219.6 s` | `+57.2 s` | reverted | `2` Gemma full-FT OOMs from extra `9.3 GiB` GPU buffer |
| T1/T2 | First-class spans + Kineto kernel traces (`PR #72`) | `tinyrl/spans.py` + `tools/trace.py` | - | `201.3 s` (profiled) | - | shipped | Shipped in PR #72 |
| E20 | `PYTORCH_CUDA_ALLOC_CONF=pinned_num_register_threads:32` | Env var on trainers (`n=3`) | `158.2 s` | `170.4 s` | `+12.2 s` | reverted | Only parallelizes `cudaHostRegister`, not single-threaded page faults |
| M6 | Parallel `memset` + single `cudaHostRegister` (`pinned_bytes`) | `mmap` + 32-thread `memset` + `cudaHostRegister` | `4.50-5.40 s` | `0.32 s` (`16 GiB`) | `-93.5%` | kept | Included in Commit 2 |
| E21 | `OMP_NUM_THREADS=16` on trainers | Env var on trainers | `158.0 s` | `159.1 s` | `+1.1 s` | reverted | Node loadavg is only `11-21` on `208` vCPUs |
| M7/E22 | Pad CUDA sequence widths to multiples of 128 (`pad1`) | `width = -(-max(lengths) // 128) * 128` (`n=4`) | `162.8 s` | `159.1 s` | `-3.7 s` | kept | Eliminates cuDNN SDPA plan compilation spikes (`-9%` micro, `3/4` cluster) |
| M8 | Defer `loss.item()` and `.cpu()` syncs (`fb-nosync`) | Sync once after pass loop | `1.283 s` | `1.282 s` | `-0.001 s` | reverted | Qwen LoRA runs `1` pass per step |
| M9 | Size-balanced safetensors shards + `24 GiB` pin slab | Greedy LPT shard bin-packing + `MAX_PIN_BYTES=24 GiB` | `8.75 s` | `4.33 s` (`54 GiB`) | `-50.5%` | kept | Max shard drops `19.13 GiB -> 8.75 GiB`; included in Commit 2 |
| M10 | `liger-kernel` Triton ops + fused linear cross-entropy | Patch RMSNorm/RoPE/MLP + `LigerFusedLinearCrossEntropyLoss` | `1.411 s` | `0.829 s` (`Qwen`) | `-41.2%` | kept | `max|dlp| <= 0.001`; included in Commit 1 |
| E23 | Cluster A/B of Commit 1 (`hc-liger1` vs `main1`, `n=5` paired) | Commit 1 on cluster | `152.8 s` med (`157.1 s` mean) | `152.2 s` med (`153.0 s` mean) | `-4.1 s` mean | kept | `qwen-lora` `fb` `73.5 s -> 59.1 s` (`5/5`), `gemma-lora` `fb` `45.7 s -> 41.5 s` (`5/5`) |
| E24 | Stack Commit 2 on E23 (`hc-stack2`, `n=2`) | Commit 1 + Commit 2 | `152.2 s` | `142.5 s` (`btn-050351`) | `-9.7 s` | kept | `gemma-fft-x` busy `98.8 s -> 65.7 s`, finish `149.3 s -> 116.4 s` |
| E25a | Stack Commit 3 on all single-GPU trainers (`hc-stack3`) | `moves_itself` on all `len(gpus) == 1` trainers | `142.5 s` | `177.5 s` (`btn-050938`) | `+35.0 s` | reverted | `CUDA_VISIBLE_DEVICES=0..7` on full-FT stalled `36 GiB` `cudaHostRegister` (`51.4 s`) |
| E25b | Restrict `moves_itself` to `w.kind.lora` (`hc-stack4`, `n=3`) | `w.kind.lora` in `moves_itself` | `157.1 s` mean (`main1`) | `147.1 s` mean (`140.7 s` best) | `-10.0 s` mean | kept | `qwen-lora` 2 moves in `0.9 s`, `gemma-lora` finish `103.6 s`, `23/23` OK |
