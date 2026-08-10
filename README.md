# ablit-AI

A local GUI for testing abliterated (refusal-ablated) models on Apple Silicon.
Runs entirely on-device via [MLX](https://github.com/ml-explore/mlx) — after the
initial model download there are no network calls.

Built for a **base M4 Mac mini, 16 GB unified memory**. Model choices below are
sized for that budget.

## Run it

```bash
uv run python app.py
```

Opens at <http://127.0.0.1:7860>.

First launch downloads ~5 GB of weights and takes a minute to load. Use
**Preload text model** in Settings to pay that cost before your first prompt.

## What it does

- **Streaming chat** with live tokens/sec, prompt-processing speed, and peak memory
- **File attachments** — code, text, Markdown, CSV, JSON, PDF get extracted and
  fenced into the prompt with their filename
- **Images** — routed to a vision model automatically (see below)
- **Thinking mode** toggle, collapsed into a foldable block in the transcript
- Sampling controls (temperature, top-p, max tokens) and an editable system prompt

## The one-model-at-a-time thing

The text model is **text-only — it cannot see images.** That is an architectural
limit, not a setting. Vision requires a separate vision-language model.

16 GB cannot hold both at once alongside the KV cache, so `ModelManager` keeps
exactly one resident and swaps when you attach an image. Swapping costs a few
seconds of reload. Running out of wired memory costs a hard stall, so the trade
is worth it.

| Slot | Default | Size |
|---|---|---|
| Text | `huihui-ai/Huihui-Qwen3.5-9B-abliterated-mlx-4bit` | ~5.2 GB |
| Vision | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~5.4 GB |

Swap either without touching code — the 14B is already downloaded:

```bash
ABLIT_TEXT_MODEL=mlx-community/Qwen2.5-Coder-14B-Instruct-abliterated-4bit uv run python app.py
```

Don't try `Qwen3-Coder-30B-A3B` — 4-bit weights are ~17 GB and will not fit.

## Measured on this machine

`verify.py` generates an ISO-8601 duration parser and *executes* it against 7
cases, 3 trials per model:

| Model | Best | Mean | Speed |
|---|---|---|---|
| Qwen2.5-Coder-14B-abliterated | 6/7 | 2.0/7 | 12.2 tok/s |
| Qwen3.5-9B-abliterated | 0/7 | 0.0/7 | 20.5 tok/s |

The 14B has a meaningfully higher ceiling; neither is *reliable*. Run-to-run
variance dominates, and temperature does not fix it — at temp 0.0 (deterministic)
the 14B fails this task consistently, while a 0.1 sample passed 6/7. Treat any
single good answer as luck, not capability.

## Give the GPU more memory

macOS wires down ~10.6 GB of 16 GB for the GPU by default. Raising it helps the
larger models. Resets on reboot:

```bash
sudo sysctl iogpu.wired_limit_mb=12288
```

## Expected speed

On a base M4 (120 GB/s memory bandwidth):

- 9B @ 4-bit — ~20–25 tok/s
- 14B @ 4-bit — ~10–12 tok/s

Attachments are capped (24k chars per file, 60k total) because every pasted token
grows the KV cache. Tune in `src/ablit_ai/config.py`.

## Layout

```
app.py                      Gradio UI and event wiring
src/ablit_ai/config.py      model ids, limits, defaults
src/ablit_ai/models.py      ModelManager — single-resident load/swap
src/ablit_ai/attachments.py file → text extraction
src/ablit_ai/chat.py        prompt assembly + streaming for both model kinds
```
