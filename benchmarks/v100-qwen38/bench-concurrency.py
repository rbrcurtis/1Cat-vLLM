#!/usr/bin/env python3
"""Concurrency benchmark: several agent sessions generating at once.

Why this exists: the old llama.cpp coder held ONE contiguous 262144-token slot, so coder
requests queued behind each other. The vLLM engine keeps 8.16 x 262144 tokens of paged KV and
admits up to --max-num-seqs (8) at once, which is the point of the change.

Read this carefully: the two phases have different bottlenecks. Prefill runs at ~970 tok/s
AGGREGATE no matter how many sessions ask for it, so concurrent long prompts pay TTFT in
series. Decode is latency-bound (each round is a fixed ~20 ms of kernel + PCIe all-reduce
work), so it should stay roughly FLAT in aggregate as sessions are added, with each session's
share dropping. Measuring "tokens / wall" mixes prefill into decode and hides all of that, so
each session's decode rate is measured from its first token to its last.

Usage: bench-concurrency.py [N] [URL]      (default N=8, engine on 127.0.0.1:18080)
"""
import concurrent.futures as cf
import statistics
import sys
import time

import requests

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
URL = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18080/v1/chat/completions"
PROMPT = "List 40 short tips for writing clear technical documentation, one per line."
MAX_TOKENS = 400


def one(idx: int) -> tuple[float, int, float]:
    """(ttft_s, tokens, decode_s) for one streaming session."""
    body = {"model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": f"{PROMPT} (variant {idx})"}],
            "max_tokens": MAX_TOKENS, "temperature": 0.7, "stream": True}
    t0 = time.perf_counter()
    first = None
    last = None
    n = 0
    with requests.post(URL, json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith(b"data: ") and b'"content"' in line:
                now = time.perf_counter()
                first = first or now
                last = now
                n += 1
    if n < 2 or first is None or last is None:
        return (first - t0 if first else 0.0), n, 0.0
    return first - t0, n, last - first


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=N) as pool:
    results = list(pool.map(one, range(N)))
wall = time.perf_counter() - t0

ttfts = [t for t, _, _ in results]
decodes = [d for _, _, d in results]
tokens = [n for _, n, _ in results]
per_session = [n / d for n, d in zip(tokens, decodes) if d > 0]
# Aggregate decode: every session ran concurrently, so the busiest session's window is the
# fair denominator for "how many tokens/s came out while all of them were generating".
agg = sum(tokens) / max(decodes) if decodes and max(decodes) > 0 else 0.0

print(f"  {N} concurrent sessions, {MAX_TOKENS} max tokens each")
print(f"  TTFT: median {statistics.median(ttfts):5.2f}s  max {max(ttfts):5.2f}s")
print(f"  decode per session: median {statistics.median(per_session) if per_session else 0:5.2f} tok/s")
print(f"  AGGREGATE decode while all were generating: {agg:6.2f} tok/s"
      f"   (tokens={sum(tokens)}, window={max(decodes) if decodes else 0:.1f}s)")
print(f"  whole run wall: {wall:.1f}s")
