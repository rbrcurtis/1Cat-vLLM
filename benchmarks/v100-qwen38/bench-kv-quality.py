#!/usr/bin/env python3
"""Retrieval and identifier recall at a chosen context length, for one engine config.

Why this exists: a memory (see "KV cache quantization - quality-first reassessment") says
average perplexity hides exactly the failures that matter to an agent, and that identifier,
instruction and retrieval errors show up long before an average moves. Perplexity also needs
logits, which the serving API does not expose. So this probe asks the model for things it can
only answer by reading, scores each answer as an exact string, and reports which items failed.

What it plants in the filler, at fixed fractional depths:
  - 8 identifiers (TAG_xxxxxxxx) that look like symbols in a code base, so a token-level slip
    changes the answer.
  - 2 needles with unusual values (a port number, a release string).

The same prompt is sent for every run, so configs are compared on byte-identical input. Run the
control first: at a short length the score must be 100%, otherwise the probe or the scorer is
wrong and the long result means nothing.

Run it with the engine's venv, because that is where the tokenizer lives:
  HF_HUB_OFFLINE=1 /home/ryan/Code/1Cat-vLLM/.venv/bin/python bench-kv-quality.py \
      --tokens 200000 --model-path /mnt/E/models/qwen3.8-27b-fp8 --out /tmp/kvq-4bitnc.json
"""
import argparse
import json
import random
import re
import sys
import time

import requests
from transformers import AutoTokenizer

FILLER = (
    "The scheduler keeps a block table per sequence. Each entry maps a logical position to a "
    "physical block, so a long prompt never needs one contiguous allocation. Copy-on-write keeps "
    "shared prefixes cheap: two requests that start from the same file read the same blocks until "
    "one of them writes. Fragmentation stays bounded because every block is the same size. "
    "A worker returns a lease, the pool releases the block, and the table is rebuilt from the "
    "checkpoint. The retry path is idempotent: the same request id must produce the same answer. "
)
NEEDLE_DEPTHS = (0.10, 0.62)
IDENTIFIER_DEPTHS = (0.05, 0.16, 0.28, 0.40, 0.53, 0.66, 0.78, 0.91)
NEEDLE_KEYS = ("deploy_port", "release")
IDENTIFIER_KEYS = tuple(f"tag_{i}" for i in range(1, 9))
INSTRUCTION = (
    "\n\nThe document above mentions some deployment facts and some build tags. Answer with "
    "exactly {n} lines, no preamble, one per key, in this order: {keys}. Use the form "
    "key=value. Copy each value exactly as it appears. If a value is not in the document, write "
    "UNKNOWN for that key. If the same key appears more than once, use the last one."
)


def build_prompt(tok, target: int, needles: dict, identifiers: dict):
    """Filler of about `target` tokens with the items planted at fixed fractional depths."""
    ids = tok(FILLER)["input_ids"]
    total = max(target // len(ids), 40) * len(ids)
    out = []
    marks = {}
    for depth, key in zip(NEEDLE_DEPTHS, NEEDLE_KEYS):
        marks[round(depth * total)] = f"\n{key} = {needles[key]}\n"
    for depth, key in zip(IDENTIFIER_DEPTHS, IDENTIFIER_KEYS):
        marks[round(depth * total)] = f"\n{key} = {identifiers[key]}\n"
    at = 0
    while at < total:
        out.extend(ids)
        at += len(ids)
        for pos in sorted(p for p in marks if at >= p):
            out.extend(tok(marks.pop(pos))["input_ids"])
    keys = list(NEEDLE_KEYS) + list(IDENTIFIER_KEYS)
    out.extend(tok(INSTRUCTION.format(n=len(keys), keys=", ".join(keys)))["input_ids"])
    return tok.decode(out)


def score(text: str, expected: dict):
    """Exact match per key. Values are compared without surrounding whitespace."""
    got = {}
    for line in text.splitlines():
        m = re.match(r"\s*([a-z_0-9]+)\s*=\s*(.+?)\s*$", line)
        if m:
            got[m.group(1)] = m.group(2).strip().strip("`\"'")
    results = {}
    for key, want in expected.items():
        results[key] = got.get(key) == want
    return results, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--model-path", default="/mnt/E/models/qwen3.8-27b-fp8",
                    help="tokenizer source; use the checkpoint the engine serves")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default=None, help="write the full answer and score to this JSON")
    ap.add_argument("--dump-prompt", default=None, help="write the prompt to this file")
    args = ap.parse_args()

    rng = random.Random(20260926)
    needles = {"deploy_port": str(rng.randrange(40000, 49000)),
               "release": f"{rng.randrange(3, 9)}.{rng.randrange(0, 9)}.{rng.randrange(0, 9)}-alpha.{rng.randrange(2, 9)}"}
    identifiers = {f"tag_{i}": "TAG_" + "".join(rng.choice("0123456789abcdef") for _ in range(8))
                   for i in range(1, 9)}
    expected = {**needles, **identifiers}

    tok = AutoTokenizer.from_pretrained(args.model_path)
    prompt = build_prompt(tok, args.tokens, needles, identifiers)
    ntok = len(tok(prompt)["input_ids"])
    if args.dump_prompt:
        with open(args.dump_prompt, "w") as f:
            f.write(prompt)
    print(f"prompt: {ntok} tokens, {len(expected)} items to recall")
    print("expected: " + json.dumps(expected))

    t0 = time.time()
    r = requests.post(
        f"{args.url}/v1/chat/completions",
        json={"model": args.model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": args.max_tokens, "temperature": 0.0},
        timeout=3600,
    )
    elapsed = time.time() - t0
    if r.status_code != 200:
        print(f"FAIL: HTTP {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    body = r.json()
    text = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")

    results, got = score(text, expected)
    passed = sum(results.values())
    print(f"answer in {elapsed:.1f}s | prompt_tokens={usage.get('prompt_tokens')} "
          f"cached={cached} output_tokens={usage.get('completion_tokens')}")
    print(f"SCORE: {passed}/{len(expected)} = {100.0 * passed / len(expected):.1f}%")
    for key, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'} {key}: want={expected[key]} got={got.get(key, '<missing>')}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"prompt_tokens": ntok, "elapsed_s": elapsed, "usage": usage,
                       "expected": expected, "got": got, "results": results,
                       "answer": text}, f, indent=2)
        print(f"wrote {args.out}")
    sys.exit(0 if passed == len(expected) else 2)


if __name__ == "__main__":
    main()
