# Qwen3.8-27B on four V100 32GB (SM70, no NVLink)

Measured results for running **Qwen3.8-27B** (27.8B dense, 262K native context) at production
concurrency on a 2017-class box: **4x Tesla V100 32GB PCIe** on a Threadripper 2990WX /
ASRock X399 Taichi, no NVLink, mixed-width PCIe (x8 on two cards, x16 on the other two,
two NUMA islands).

The engine is this fork with its SM70 patches; upstream vLLM requires sm_75+ and will not
run on Volta. Full write-up: [Four 2017 V100s, One Fork, and a 27B Model That Refused to Fit](https://www.clankwrangler.com/blog/20).

## Setup

| Item | Value |
| --- | --- |
| Model | Qwen3.8-27B, online 128x128 block-FP8 weights via the SM70 packed kernel (QPN8) |
| KV cache | fp8_e4m3, 2,048-token blocks |
| Tensor parallel | 4 (all cards); the 4B co-tenant engine runs TP2 on the two x16 cards |
| Window / sessions | 240k window, 6 sessions in production; 9 full-window sessions fit at the 262k window |
| KV pool | 1,470,967 tokens (production); 2,380,815 tokens at the full window |
| Runtime | torch 2.10.0+cu128, CUDA 12.8, CPython 3.12 |
| Required env | `NCCL_P2P_LEVEL=SYS` — without it, half of every TP4 step is lost to a slower all-reduce ring |

## Headline results (8-bit QPN8 weights, fp8 KV)

| Metric | Value |
| --- | --- |
| Single-stream decode | **46.2 tok/s** (BF16 reference: 31.8) |
| 6-session aggregate decode | **153.6 tok/s** (26.3 each) |
| 8-session aggregate decode | **200.2 tok/s** (26.4 each; BF16: 144.8) |
| Weights per card | 7.34 GiB (BF16: 14.28) |
| Cold prefill, 8.2k prompt, TP4 | 1,583 tok/s after the `NCCL_P2P_LEVEL=SYS` fix (913 before) |
| Long-context prefill, TP4 | 1,672 tok/s @ 32k (TTFT 19.6 s), 1,355 @ 131k (95.8 s), 1,211 @ 200k (165 s) |
| Prefill at 200k, TP2 x16 pair | 926 tok/s (-23.5% vs TP4) — the all-reduce cost is per step, not per rank |
| TTFT with prefix cache | 7.93 s first call, 0.73 s repeat (12,288 of 13,295 tokens cached) |
| Startup | 8 min 20 s cold (empty compile caches), ~2.5 min warm |

SXM2/NVLink reference (maintainers' box, same model/workload): 97.7 tok/s decode, 6,950 tok/s
cold prefill — the PCIe fabric is worth about 2x decode and 4.4x prefill on this model.

## Results

Structured measurements, config-tagged per file:

- [`results/decode-sweep.json`](results/decode-sweep.json) — decode concurrency sweep (TP4 C1-C8, TP2 C1-C2), BF16 and int8 W8A16 references, per-frame KV capacity
- [`results/prefill-and-env.json`](results/prefill-and-env.json) — TP sweep, the NCCL fix, long-context prefill, prefix caching, startup, production layout

## Reproduce

The benchmark clients in this directory point at the OpenAI-compatible endpoint
(`http://127.0.0.1:18080/v1/chat/completions` by default; pass a URL as the first argument):

- `bench-decode.py` — streaming vs non-streaming decode (the two disagree when speculative
  decoding is on; report the engine's own numbers)
- `bench-prefill.py` — prompt-length/TTFT curve
- `bench-concurrency.py` — N concurrent sessions, aggregate and per-session decode
- `bench-long-context.py` — deep-context decode stability
- `bench-kv-restore.py` / `bench-kv-quality.py` — KV offload archive restore and quality probes

The engine is launched by the operator's `serve-v100.sh` (env-tunable: `TP`, `SPEC`, `KV`,
`UTIL`, `SEQS`, `TOKENS`, `BLOCK`, `LEN`). A/B runs override env and run the script directly;
restart the service for anything permanent. Expect minutes, not seconds, for engine restarts
— the compile caches must be pinned or every restart is a cold 8-minute build.

Measurement hygiene that cost time on this box:

- Poll `/health` only after confirming no previous engine answers it.
- Kill the engine by its process group (`ps -eo pgid,args | grep -a '[v]llm.entrypoints'`),
  never by a `pkill -f` pattern that can match the invoking shell.
- Pin `CUDA_VISIBLE_DEVICES` numerically with `CUDA_DEVICE_ORDER=PCI_BUS_ID` — vLLM rejects
  GPU UUIDs.
- Report decode from the engine's own counters, not stream-chunk intervals.
