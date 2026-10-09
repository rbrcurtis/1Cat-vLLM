#!/usr/bin/env python3
"""Decode benchmark for the 1Cat-vLLM engine: streaming vs non-streaming.

Why both modes: they do NOT agree. Measured 2026-09-23 with the DFlash2 drafter enabled,
non-streaming gave 64.2 tok/s while streaming gave 26.7 tok/s with a flat ~38 ms gap between
tokens (one token per verification round). With speculation off both modes agree near 50 tok/s.
Report decode from the engine's own numbers, not from stream chunk intervals: the fork's own
acceptance protocol says "client stream intervals are transport evidence, not instrumented
GPU-round latency".

Usage:  bench-decode.py [URL]     (default http://127.0.0.1:18080/v1/chat/completions)
        Point it at the gateway (http://127.0.0.1:11434/v1/chat/completions) to measure the
        proxied path instead.
"""
import statistics
import sys
import time

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080/v1/chat/completions"
PROMPT = ("Write a detailed 400-word explanation of how paged attention manages KV cache "
          "memory, including fragmentation, block tables and copy-on-write.")


def run(stream: bool, max_tokens: int = 300) -> float:
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    t0 = time.perf_counter()
    if not stream:
        r = requests.post(URL, json=body, timeout=900)
        dt = time.perf_counter() - t0
        n = r.json()["usage"]["completion_tokens"]
        print(f"  non-stream: {n:4d} tok in {dt:6.2f}s = {n/dt:6.2f} tok/s ({1000*dt/n:5.1f} ms/tok)")
        return n / dt
    stamps = []
    with requests.post(URL, json=body, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith(b"data: ") and b'"content"' in line:
                stamps.append(time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    n = len(stamps)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    rate = (n - 1) / (dt - stamps[0]) if n > 1 else 0.0
    med = statistics.median(gaps) * 1000 if gaps else 0.0
    bursts = sum(1 for g in gaps if g < 0.005)
    print(f"  stream    : {n:4d} tok in {dt:6.2f}s = {rate:6.2f} tok/s; gap median={med:.1f}ms "
          f"bursts<5ms={bursts}/{len(gaps)}")
    return rate


if __name__ == "__main__":
    ns = run(False)
    st = run(True)
    run(True)
    print(f"  SUMMARY non-stream={ns:.1f} tok/s  stream={st:.1f} tok/s")
