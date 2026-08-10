"""Generate code from a model, then actually run it against test cases.

Eyeballing model output tells you whether it looks plausible. Executing it tells
you whether it works. These are very different things for small models.

    uv run python verify.py <hf-repo> [<hf-repo>...]
"""

from __future__ import annotations

import gc
import re
import sys

TASK = (
    "Write a Python function parse_duration(s) that converts an ISO-8601 "
    "duration like 'P3DT4H5M' into total seconds. Handle the T separator, "
    "multi-digit values, and the fact that M means months before T and "
    "minutes after T. Assume 1 year = 365 days and 1 month = 30 days. "
    "Return ONLY the function in a single Python code block, no explanation."
)

CASES = [
    ("PT1H", 3600),
    ("PT30M", 1800),
    ("P1D", 86_400),
    ("P3DT4H5M", 273_900),
    ("P1DT2H3M4S", 93_784),
    ("PT45S", 45),
    ("P2W", 1_209_600),
]


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len)
    return text if "def parse_duration" in text else None


def check(code: str) -> tuple[int, list[str]]:
    namespace: dict = {}
    try:
        exec(code, namespace)
    except Exception as exc:
        return 0, [f"code does not even import/compile: {type(exc).__name__}: {exc}"]

    fn = namespace.get("parse_duration")
    if not callable(fn):
        return 0, ["no callable parse_duration defined"]

    passed, failures = 0, []
    for text, expected in CASES:
        try:
            got = fn(text)
        except Exception as exc:
            failures.append(f"{text}: raised {type(exc).__name__}: {exc}")
            continue
        if got == expected:
            passed += 1
        else:
            failures.append(f"{text}: got {got!r}, expected {expected}")
    return passed, failures


def run(repo: str, trials: int = 3) -> None:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    from src.ablit_ai.chat import _apply_template

    print(f"\n{'=' * 70}\n{repo}\n{'=' * 70}")
    model, tokenizer = load(repo)

    messages = [
        {"role": "system", "content": "You are a precise coding assistant."},
        {"role": "user", "content": TASK},
    ]
    prompt = _apply_template(tokenizer, messages, thinking=False)

    scores, speeds = [], []
    for trial in range(trials):
        # Vary temperature slightly per trial -- one greedy sample tells you
        # about one sample, not about the model.
        sampler = make_sampler(temp=0.2 + 0.1 * trial, top_p=0.95)

        out, response = "", None
        for response in stream_generate(
            model, tokenizer, prompt, max_tokens=1200, sampler=sampler
        ):
            out += response.text
        speeds.append(response.generation_tps)

        code = extract_code(out)
        if code is None:
            scores.append(0)
            print(f"  trial {trial + 1}: no code block found")
            continue

        passed, failures = check(code)
        scores.append(passed)
        print(f"  trial {trial + 1}: {passed}/{len(CASES)} pass "
              f"({response.generation_tokens} tok @ {response.generation_tps:.1f} tok/s)")
        for failure in failures[:3]:
            print(f"      ✗ {failure}")

    print(f"\n>>> {repo.split('/')[-1]}: "
          f"best {max(scores)}/{len(CASES)}, "
          f"mean {sum(scores) / len(scores):.1f}/{len(CASES)}, "
          f"{sum(speeds) / len(speeds):.1f} tok/s")

    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        run(repo)
