#!/usr/bin/env python3
"""Archive round-trip: write a session to the disk KV archive, then restore it.

A restore is only possible when a prompt's blocks are already in the archive and the
connector can hold the whole matched prefix in the CPU tier at once (two KV groups for
this model, so a 160K window needs 160 blocks = 10 GiB; KV_CPU_GB defaults to 16).

    # 1. write: a prompt no run has sent before
    bench-kv-restore.py --seed 7 --tokens 20000
    # 2. cold the engine (GPU prefix cache and CPU tier both empty)
    sudo systemctl restart coder@day.service
    # 3. restore: the same prompt must come back from disk
    bench-kv-restore.py --seed 7 --tokens 20000

The seed makes the prompt unique and reproducible: the same seed rebuilds the same token
stream, so step 3 asks for the blocks step 1 stored. Expect the first run to take about a
minute of prefill per 25K tokens, and the second run a few seconds.

The printed CPU_to_GPU delta is the proof: it counts bytes the connector moved from the
CPU tier into the GPU, which only happens when blocks come back from the archive. It
stays 0 when a run recomputes.

PASS: the restore run reports CPU_to_GPU > 0, and its elapsed time is far below the write
run's.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request

WORDS = [
    "harbor", "lantern", "quartz", "meadow", "cinder", "willow", "tundra",
    "fathom", "ember", "gantry", "obsidian", "radish", "kestrel", "vellum",
    "zephyr", "basalt", "nimbus", "cobalt", "juniper", "sable",
]


def cpu_to_gpu_bytes(url: str) -> float:
    """Bytes the connector has moved from the CPU tier to the GPU, 0 if never."""
    metrics_url = url.rsplit("/v1/", 1)[0] + "/metrics"
    try:
        with urllib.request.urlopen(metrics_url, timeout=20) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("vllm:kv_offload_total_bytes_total") and (
                    'transfer_type="CPU_to_GPU"' in line
                ):
                    return float(line.rsplit(" ", 1)[1])
    except (urllib.error.URLError, OSError):
        return 0.0
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1, help="makes the prompt unique")
    p.add_argument("--tokens", type=int, default=20000)
    p.add_argument(
        "--url", default="http://127.0.0.1:18080/v1/chat/completions"
    )
    a = p.parse_args()

    rng = random.Random(a.seed)
    # roughly 1.3 tokens per word
    words = [rng.choice(WORDS) for _ in range(int(a.tokens / 1.3))]
    prompt = (
        f"seed={a.seed} " + " ".join(words) + "\n\nReply with the single word: done"
    )

    before = cpu_to_gpu_bytes(a.url)
    req = urllib.request.Request(
        a.url,
        data=json.dumps(
            {
                "model": "qwen3.8-27b",
                "max_tokens": 4,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    elapsed = time.time() - t0
    restored = cpu_to_gpu_bytes(a.url) - before

    print(f"  seed            : {a.seed}")
    print(f"  prompt tokens   : {out['usage']['prompt_tokens']}")
    print(f"  elapsed         : {elapsed:.1f} s")
    print(f"  CPU_to_GPU      : {restored / 1e6:.1f} MB")
    if restored > 0:
        print("  RESULT: restored from the disk archive")
    else:
        print("  RESULT: computed the prompt (nothing came back from disk)")


if __name__ == "__main__":
    main()
