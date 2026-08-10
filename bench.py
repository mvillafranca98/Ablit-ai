"""Compare models on the same prompts, headless.

    uv run python bench.py                          # benchmark config.TEXT_MODEL
    uv run python bench.py <hf-repo> [<hf-repo>...]  # benchmark specific models

Loads one model at a time and frees it before the next, so this stays within a
16 GB budget even when comparing several.
"""

from __future__ import annotations

import sys
import time

from src.ablit_ai import config

PROMPTS = [
    (
        "iso8601",
        "Write a Python function parse_duration(s) that converts an ISO-8601 "
        "duration like 'P3DT4H5M' into total seconds. Handle the T separator "
        "and multi-digit values. Return only the function.",
    ),
    (
        "debug",
        "This Python is wrong. Explain the bug in one sentence, then give the "
        "fix:\n\n```python\ndef dedupe(items):\n    for i in items:\n        "
        "if items.count(i) > 1:\n            items.remove(i)\n    return items\n```",
    ),
    (
        "sql",
        "Write a PostgreSQL query returning the top 5 customers by total order "
        "value in the last 90 days. Tables: customers(id, name), "
        "orders(id, customer_id, total_cents, created_at).",
    ),
]


def run(repo: str, max_tokens: int = 700) -> None:
    import gc

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from src.ablit_ai.chat import _apply_template

    print(f"\n{'=' * 70}\n{repo}\n{'=' * 70}")

    t0 = time.time()
    model, tokenizer = load(repo)
    print(f"load: {time.time() - t0:.1f}s\n")

    sampler = make_sampler(temp=0.2, top_p=0.95)
    speeds = []

    for name, prompt in PROMPTS:
        messages = [
            {"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Thinking off, matching the app default -- otherwise models whose
        # template defaults to reasoning spend the token budget thinking and
        # the comparison measures verbosity rather than capability.
        text = _apply_template(tokenizer, messages, thinking=False)

        out, response = "", None
        for response in stream_generate(
            model, tokenizer, text, max_tokens=max_tokens, sampler=sampler
        ):
            out += response.text

        speeds.append(response.generation_tps)
        print(f"--- {name} · {response.generation_tps:.1f} tok/s · "
              f"{response.peak_memory:.1f} GB peak ---")
        print(out.strip()[:900])
        print()

    print(f">>> {repo}: mean {sum(speeds) / len(speeds):.1f} tok/s")

    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


if __name__ == "__main__":
    for repo in sys.argv[1:] or [config.TEXT_MODEL]:
        run(repo)
