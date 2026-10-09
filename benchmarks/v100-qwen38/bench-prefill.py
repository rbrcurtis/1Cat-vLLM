#!/usr/bin/env python3
"""Prefill / TTFT curve for the 1Cat-vLLM engine.

Measured 2026-09-23 (speculation off, block 2048 / mamba grid 8192):
  4.4K prompt -> 863 tok/s, 17K -> 859, 35K -> 1556, 69K -> 1516.
So TTFT is ~5s at 4K but ~46s at 69K, which is what a coding agent pasting a large file pays.
The 4K point is slower than the 35K point because the very first request also warms kernels.

Usage:  bench-prefill.py [URL]
"""
import sys
import time

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080/v1/chat/completions"
FILLER = ("The transformer architecture processes tokens in parallel and uses attention "
          "to relate them. Paged attention stores key and value blocks in fixed-size pages. ")

for target in (4000, 16000, 32000, 64000):
    prompt = (f"Summarize the following text in one sentence.\n\n{FILLER * (target // 25 + 1)}")
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8, "temperature": 0, "stream": False}
    t0 = time.perf_counter()
    r = requests.post(URL, json=body, timeout=1800)
    dt = time.perf_counter() - t0
    pt = r.json().get("usage", {}).get("prompt_tokens", 0)
    print(f"  prompt_tokens={pt:6d}  TTFT~{dt:7.2f}s  prefill={pt/dt:7.1f} tok/s")
