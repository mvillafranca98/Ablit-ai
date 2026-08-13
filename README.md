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

## Video analysis

Turn a screen recording into a teardown document — what the product does, how
the workflow works, and a structural description of the UI.

```bash
uv run python analyze_video.py demo.mov
```

Several recordings of the same platform merge into one document:

```bash
uv run python analyze_video.py training/*.mov --frames 40 --language es
```

Options: `--frames N` (max keyframes **per video**, default 12), `--out`,
`--language`, `--keep-frames`, `--frame-tokens N`.

`--frames` is a cap, not a target — deduplication decides the real count, so
setting it high just means "capture every distinct screen". A static 5-minute
intro yields 4 screens at either 12 or 40; a dense 11-minute walkthrough yields
12 at `--frames 12` but 23 at `--frames 40`. Under-sampling silently loses
screens, so prefer a high cap and pay the time.

Budget roughly **50s per unique screen**, not per minute of video.

### Crop first if the app doesn't fill the frame

Recordings of a video call, or of a window on a bigger desktop, waste most of
the frame. Check one frame before committing an hour of compute:

```bash
ffmpeg -ss 120 -i recording.mov -frames:v 1 /tmp/check.png
```

If the app is a region rather than the whole frame, crop to it:

```bash
uv run python analyze_video.py rec.mov --crop 1240:552:14:283 --frames 40
```

This matters twice over. Cropping spends the downscale budget on the UI instead
of on desktop chrome, and it stops the model documenting the surrounding
desktop — a Google Meet call, a dock, browser tabs — as though it were part of
the product. Reported geometry is measured against the cropped region, so
coordinates stay usable.

**Do not compress the source to speed things up.** Runtime is dominated by
vision-model inference per frame, not by file size or decoding, so compression
saves nothing on the slow part while destroying the small on-screen text that is
the most valuable signal. If anything, record at a *higher* bitrate.

Videos dropped into the chat GUI are refused with a pointer to this command —
they are never silently skipped, because a model handed no content will invent a
convincing document about a platform it has never seen.

Requires ffmpeg:

```bash
brew install ffmpeg
```

### How it works

Strictly sequential — 16 GB cannot hold these three models at once, so each is
freed before the next loads:

```
ffprobe          resolution, duration, fps
ffmpeg           16 kHz mono audio
whisper          timestamped transcript          ~1.6 GB, freed
ffmpeg           keyframes, perceptually deduped
Qwen3-VL         per-frame layout description    ~5.4 GB, freed
abliterated LLM  synthesis into Markdown         ~5.2 GB
```

Keyframes are over-sampled (scene cuts *and* even intervals), then collapsed by
average-hash similarity before the vision stage. Extracting a frame is cheap;
describing one costs ~45s. On a 26s test video this cut 8 candidate frames to 3
unique screens with no loss.

**Neither ffmpeg nor Whisper has anything to abliterate.** ffmpeg is a codec
toolchain with no content policy; Whisper is speech-to-text and doesn't follow
instructions, so it has no refusal behaviour. Only the final synthesis step uses
the abliterated model.

### Accuracy

Measured on a 1280x720 synthetic demo with known ground truth:

- **Structure and text: reliable.** Correctly recovered the sidebar/header/main
  split, all four table columns, six row labels, three data-source options, and
  most on-screen values.
- **Geometry: approximate.** Reported the sidebar as ~256px when it was 220px.
- **Colors: rough.** Hex estimates land in the right family, not on the value.
- **OCR is not perfect.** It read a metric as `12,830` where the video showed
  `12,930`.

Treat dimensions as proportional guidance, not a spec. The report header says so
too.

## Layout

```
app.py                      Gradio UI and event wiring
analyze_video.py            video → teardown document CLI
bench.py / verify.py        model comparison; verify.py executes the code
src/ablit_ai/config.py      model ids, limits, defaults
src/ablit_ai/models.py      ModelManager — single-resident load/swap
src/ablit_ai/attachments.py file → text extraction
src/ablit_ai/chat.py        prompt assembly + streaming for both model kinds
src/ablit_ai/video.py       ffmpeg probe / audio / keyframe extraction + dedupe
src/ablit_ai/transcribe.py  Whisper via MLX
```
