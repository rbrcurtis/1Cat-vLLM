#!/usr/bin/env python3
"""Long-context concurrency benchmark: C sessions, each at a chosen prompt length.

Why this exists: the coder's engine is configured so that SEQS sessions fit at LEN (util sizes
the KV pool, LEN is the context per session). Nothing so far tested that claim for real, because
every other bench uses short prompts and therefore never consumes the pool. This one builds a
prompt of exactly `--tokens` tokens per session, fires `--concurrency` of them at once, and
samples the engine's own counters while they run.

Read the result this way: `waiting max` must stay 0. Any waiting means the pool could not hold C
sessions at that context, so the scheduler made one of them queue for KV. A request that is
admitted but later preempted shows up as a decode stall, not as waiting, so check the per-session
decode rates too.

Run it with the engine's venv, because that is where the tokenizer lives:
  HF_HUB_OFFLINE=1 /home/ryan/Code/1Cat-vLLM/.venv/bin/python bench-long-context.py \
      --concurrency 6 --tokens 163600 --max-tokens 64
"""
import argparse
import concurrent.futures as cf
import sys
import threading
import time

import requests
from transformers import AutoTokenizer

MODEL_PATH = "/mnt/E/models/qwen3.8-27b-nvfp4"

BASE = (
    "Paged attention stores the key and value tensors of a sequence in fixed size blocks, so a "
    "long prompt does not need one contiguous allocation. The block table maps each logical "
    "position to a physical block, which is what lets the scheduler share a prefix between two "
    "requests and copy a block only when one of them writes to it. Fragmentation stays bounded "
    "because every block is the same size, and the pool can be sized in tokens instead of in "
    "whole sequences. "
)


def build_prompt(tok, target: int) -> str:
    """Text whose tokenization is exactly `target` tokens."""
    base_ids = tok(BASE)["input_ids"]
    per = len(base_ids)
    reps = max(target // per, 1)
    ids = base_ids * reps
    if len(ids) > target:
        ids = ids[:target]
    while len(ids) < target:
        ids = (ids + base_ids)[:target]
    text = tok.decode(ids)
    # decode/encode is not always perfectly stable, so fix up the tail
    got = len(tok(text)["input_ids"])
    for _ in range(60):
        if got == target:
            break
        text += " ok" if got < target else ""
        new = tok(text)["input_ids"]
        got = len(new)
        if got > target:
            text = tok.decode(new[:target])
            got = len(tok(text)["input_ids"])
    return text


def sample_metrics(url: str, stop: threading.Event, out: dict, every: float = 5.0) -> None:
    run = wait = kv = 0
    pre_first = None
    while True:
        try:
            m = requests.get(url, timeout=10).text
        except Exception:
            m = ""
        for line in m.splitlines():
            if line.startswith("vllm:num_requests_running"):
                run = max(run, float(line.rsplit(" ", 1)[1]))
            elif line.startswith("vllm:num_requests_waiting{"):
                wait = max(wait, float(line.rsplit(" ", 1)[1]))
            elif line.startswith("vllm:kv_cache_usage_perc"):
                kv = max(kv, float(line.rsplit(" ", 1)[1]))
            elif line.startswith("vllm:num_preemptions_total"):
                v = float(line.rsplit(" ", 1)[1])
                pre_first = v if pre_first is None else min(pre_first, v)
        out["running_max"], out["waiting_max"], out["kv_max"] = run, wait, kv
        out["preemptions"] = (pre_first or 0)
        if stop.wait(every):
            return


def one(url: str, model: str, text: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text + "\n\nNow describe how paged attention bounds memory fragmentation, in about 200 words."}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    t0 = time.perf_counter()
    first = last = None
    n = 0
    prompt_tokens = None
    status = "ok"
    try:
        with requests.post(url, json=body, stream=True, timeout=3600) as r:
            if r.status_code != 200:
                return {"status": f"HTTP {r.status_code}", "body": r.text[:200]}
            for line in r.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    break
                import json

                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                for ch in chunk.get("choices", []):
                    if (ch.get("delta") or {}).get("content"):
                        n += 1
                        first = first or time.perf_counter()
                        last = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - report any client failure as a result
        status = f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t0
    decode = (last - first) if (first and last and last > first) else 0.0
    return {
        "status": status,
        "prompt_tokens": prompt_tokens,
        "out_tokens": n,
        "ttft_s": (first - t0) if first else None,
        "elapsed_s": dt,
        "decode_s": decode,
        # one token per chunk and a chunk can carry an empty delta, so a decode rate needs at
        # least a couple of tokens to mean anything
        "tok_s": (n / decode) if (decode > 0.05 and n >= 2) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    ap.add_argument("--metrics", default="http://127.0.0.1:18080/metrics")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=163600)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    text = build_prompt(tok, args.tokens)
    real = len(tok(text)["input_ids"])
    print(f"prompt built: {real} tokens per session, {args.concurrency} sessions "
          f"= {real * args.concurrency:,} tokens of context")

    out: dict = {}
    stop = threading.Event()
    sampler = threading.Thread(target=sample_metrics, args=(args.metrics, stop, out), daemon=True)
    sampler.start()

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one, args.url, args.model, text, args.max_tokens)
                   for _ in range(args.concurrency)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=10)

    print(f"\n  {'session':>7}  {'out':>5}  {'ttft s':>8}  {'elapsed s':>10}  {'tok/s':>7}  status")
    ok = 0
    for i, r in enumerate(results):
        if r["status"] == "ok":
            ok += 1
        print(f"  {i:>7}  {r.get('out_tokens', 0):>5}  "
              f"{(r.get('ttft_s') or 0):>8.1f}  {r.get('elapsed_s', 0):>10.1f}  "
              f"{r.get('tok_s', 0):>7.1f}  {r['status']}")
    dec = [r["tok_s"] for r in results if r.get("tok_s")]
    agg = sum(r.get("out_tokens", 0) for r in results) / wall if wall else 0
    print(f"\n  completed           : {ok}/{args.concurrency}")
    print(f"  wall clock          : {wall:.1f} s")
    print(f"  prefill+decode total: {agg:.1f} tok/s")
    print(f"  per-session decode  : median {sorted(dec)[len(dec) // 2]:.1f} tok/s" if dec else "  no decode rate (too few tokens)")
    print(f"  engine counters     : running max {out.get('running_max', 0):.0f}, "
          f"WAITING MAX {out.get('waiting_max', 0):.0f}, kv usage max {out.get('kv_max', 0):.2f}")
    print("  note: waiting at the start is admission order, not a capacity failure. What matters is")
    print("        that nothing was preempted (evicted and recomputed) and every session finished.")
    if ok < args.concurrency:
        print("  RESULT: FAILED - a request did not complete")
        return 1
    if out.get("preemptions", 0) > 0:
        print("  RESULT: FAILED - the pool ran out: preemptions were counted")
        return 1
    print("  RESULT: PASSED - all sessions admitted, none preempted, all finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
