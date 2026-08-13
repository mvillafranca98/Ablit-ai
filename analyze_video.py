"""Analyse screen recordings into a document you could build an integration from.

    uv run python analyze_video.py demo.mov
    uv run python analyze_video.py training/*.mov --out platform.md

Pipeline, strictly sequential because 16 GB cannot hold these together:

    ffprobe   ->  resolution / duration
    ffmpeg    ->  16 kHz mono audio
    whisper   ->  timestamped transcript        (~1.6 GB, freed after)
    ffmpeg    ->  keyframes, perceptually deduped
    Qwen3-VL  ->  per-frame layout description  (~5.4 GB, freed after)
    abliterated LLM -> per-video synthesis, then a combined document

With several videos the synthesis is hierarchical: each video is summarised on
its own, then the summaries are merged. Feeding four full transcripts plus fifty
frame descriptions into one prompt would need a KV cache larger than the machine.

Neither ffmpeg nor Whisper has any content restriction to work around. Only the
synthesis steps use the abliterated model.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.ablit_ai import chat, config, transcribe, video
from src.ablit_ai.models import MANAGER

FRAME_SYSTEM = (
    "You describe user interfaces with the precision of someone writing a spec "
    "for a developer who must automate the screen without ever seeing it."
)

SYNTHESIS_SYSTEM = (
    "You are a senior integration engineer documenting a third-party platform so "
    "your team can drive it programmatically. Be concrete and specific. Never "
    "invent features, endpoints, or field names you have no evidence for."
)


@dataclass
class VideoResult:
    info: video.VideoInfo
    transcript: str = ""
    frame_notes: list[str] = field(default_factory=list)
    report: str = ""
    resolution: str = ""

    def __post_init__(self) -> None:
        self.resolution = self.resolution or self.info.resolution


def effective_resolution(info: video.VideoInfo, crop: str | None) -> str:
    """Geometry must be described against what the model actually sees.

    With a crop the source resolution is meaningless -- reporting positions
    against 1724x1080 when the model is looking at a 1240x552 region would put
    every coordinate in the document out by the crop offset.
    """
    if not crop:
        return info.resolution
    parts = crop.split(":")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}x{parts[1]}"
    return info.resolution


def frame_prompt(resolution: str, timestamp: float, index: int, total: int) -> str:
    return (
        f"Frame {index} of {total}, captured at {timestamp:.1f}s from a screen "
        f"recording. The visible area is {resolution}.\n\n"
        "Describe this screen as a spec for someone automating it:\n"
        "1. LAYOUT: overall structure (sidebar / header / main / modal). For each "
        "region give approximate position and size, as a percentage of the frame "
        f"AND as approximate pixels at {resolution}.\n"
        "2. INTERACTIVE ELEMENTS: every button, input, dropdown, tab, checkbox, "
        "table, link — with its visible label, position, and approximate size. "
        "Labels matter more than anything else here: transcribe them exactly.\n"
        "3. DATA: any table columns, field names, IDs, URLs, or values visible. "
        "Transcribe verbatim.\n"
        "4. STATE: what state this screen is in (empty, loaded, error, modal open, "
        "row selected) and anything indicating what action just happened.\n"
        "5. TYPOGRAPHY & COLOR: relative sizes, weights, and the palette.\n\n"
        "Report only what is visible. If something is cut off or illegible, say so "
        "rather than guessing."
    )


def per_video_prompt(result: VideoResult) -> str:
    frames_block = "\n\n".join(result.frame_notes)
    spoken = result.transcript.strip() or "(no speech in this video)"
    return (
        f"Below is everything extracted from '{result.info.path.name}', a "
        f"{result.info.duration:.0f}s screen recording at {result.resolution}.\n\n"
        f"=== NARRATION (timestamped) ===\n{spoken}\n\n"
        f"=== PER-FRAME UI OBSERVATIONS ===\n{frames_block}\n\n"
        "=== TASK ===\n"
        "Write a Markdown section documenting THIS recording, with:\n\n"
        "### What is demonstrated\n"
        "The purpose of this segment in one paragraph.\n\n"
        "### Workflow\n"
        "The exact steps performed, in order, with timestamps. For each step name "
        "the UI element clicked or typed into, and what happened as a result.\n\n"
        "### Screens and elements\n"
        "Each distinct screen, its layout regions with approximate dimensions at "
        f"{result.resolution}, and its interactive elements with exact labels.\n\n"
        "### Data observed\n"
        "Field names, table columns, IDs, URLs, and values seen on screen.\n\n"
        "Base every claim on the evidence above. Where evidence is thin, write "
        "'unclear from the recording'."
    )


def combine_prompt(results: list[VideoResult]) -> str:
    sections = "\n\n".join(
        f"=== {r.info.path.name} ({r.info.duration:.0f}s) ===\n{r.report}"
        for r in results
    )
    resolution = results[0].resolution
    return (
        f"You have per-recording notes from {len(results)} training videos covering "
        f"one platform, totalling {sum(r.info.duration for r in results) / 60:.0f} "
        f"minutes at {resolution}.\n\n"
        f"{sections}\n\n"
        "=== TASK ===\n"
        "Merge these into ONE coherent Markdown document about the platform. The "
        "reader must be able to build an external tool that drives this platform, "
        "so precision about actions, inputs and identifiers matters more than prose.\n\n"
        "Use exactly these sections:\n\n"
        "## What the platform does\n"
        "## Core concepts and data model\n"
        "Entities, their fields, and how they relate — inferred from what was shown.\n\n"
        "## How it works: end-to-end workflows\n"
        "Each complete workflow as numbered steps, naming the screen, the element "
        "and its exact label, the input required, and the resulting state.\n\n"
        "## Interface reference\n"
        f"Persistent layout (nav, header, main) with approximate dimensions at "
        f"{resolution}, then each screen with its interactive elements and labels.\n\n"
        "## Integration notes\n"
        "For someone automating this externally: the actions available, the inputs "
        "each needs, identifiers and labels usable as selectors, the order "
        "operations must happen in, and any validation or state constraints "
        "observed. Flag anything that looks like it would block automation.\n\n"
        "## Open questions\n"
        "What a developer would still need to confirm before building against it.\n\n"
        "Deduplicate across recordings — the same screen may appear in several. "
        "Where recordings conflict, say so explicitly rather than picking one."
    )


def describe_frames(
    resolution: str, frames: list[tuple[float, Path]], max_tokens: int
) -> list[str]:
    notes = []
    for index, (timestamp, image_path) in enumerate(frames, start=1):
        started = time.time()
        text = ""
        for partial, _ in chat.stream_vision_reply(
            images=[str(image_path)],
            system_prompt=FRAME_SYSTEM,
            user_text=frame_prompt(resolution, timestamp, index, len(frames)),
            max_tokens=max_tokens,
            temperature=0.2,
        ):
            text = partial
        notes.append(f"--- Frame {index} @ {timestamp:.1f}s ---\n{text.strip()}")
        print(f"      frame {index}/{len(frames)} @ {timestamp:6.1f}s "
              f"({time.time() - started:.0f}s)", flush=True)
    return notes


def synthesise(prompt: str, max_tokens: int) -> tuple[str, str]:
    text, status = "", ""
    for partial, current in chat.stream_text_reply(
        history=[],
        system_prompt=SYNTHESIS_SYSTEM,
        user_text=prompt,
        max_tokens=max_tokens,
        temperature=0.3,
        thinking=False,
    ):
        text, status = partial, current
    return text, status


def analyse_one(
    path: str, workdir: Path, frames_per_video: int, frame_tokens: int,
    language: str | None, crop: str | None,
) -> VideoResult:
    info = video.probe(path)
    result = VideoResult(info=info, resolution=effective_resolution(info, crop))
    print(f"\n  {info.path.name}: {info.resolution}, {info.duration:.0f}s, "
          f"audio={'yes' if info.has_audio else 'no'}"
          + (f", crop -> {result.resolution}" if crop else ""))

    if info.has_audio:
        print("    transcribing…", flush=True)
        wav = video.extract_audio(info.path, workdir / f"{info.path.stem}.wav")
        MANAGER.unload()  # make room for Whisper
        transcript = transcribe.transcribe(wav, language=language)
        result.transcript = transcript.as_timestamped()
        print(f"    {len(transcript.text.split())} words, lang={transcript.language}")

    frames = video.extract_keyframes(
        info, workdir / f"{info.path.stem}_frames",
        max_frames=frames_per_video, crop=crop,
    )
    print(f"    {len(frames)} unique screens after dedupe")
    if not frames:
        return result

    result.frame_notes = describe_frames(result.resolution, frames, frame_tokens)

    print("    synthesising this recording…", flush=True)
    result.report, status = synthesise(per_video_prompt(result), 2400)
    print(f"    {status}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="one or more video files")
    parser.add_argument("--frames", type=int, default=12,
                        help="max keyframes per video (default 12)")
    parser.add_argument("--out", default=None, help="output markdown path")
    parser.add_argument("--language", default=None, help="force language, e.g. en, es")
    parser.add_argument("--frame-tokens", type=int, default=700,
                        help="max tokens per frame description")
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep extracted frames beside the report")
    parser.add_argument("--crop", default=None, metavar="W:H:X:Y",
                        help="crop each frame before analysis, e.g. 1240:552:14:283. "
                             "Use when the app fills only part of the frame (a shared "
                             "screen in a call, a window on a larger desktop).")
    args = parser.parse_args()

    paths = sorted(args.videos)
    default_name = (
        f"{Path(paths[0]).with_suffix('')}.analysis.md" if len(paths) == 1
        else str(Path(paths[0]).parent / "platform-analysis.md")
    )
    out_path = Path(args.out or default_name)
    workdir = Path(tempfile.mkdtemp(prefix="ablit-video-"))
    started = time.time()

    print(f"Analysing {len(paths)} recording(s) → {out_path}")

    results: list[VideoResult] = []
    for position, path in enumerate(paths, start=1):
        print(f"\n[{position}/{len(paths)}]", end="")
        try:
            results.append(
                analyse_one(path, workdir, args.frames, args.frame_tokens,
                            args.language, args.crop)
            )
        except (video.FFmpegMissing, FileNotFoundError, ValueError) as exc:
            print(f"  skipped: {exc}", file=sys.stderr)

    if not results:
        print("error: nothing could be analysed", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    if len(results) == 1:
        document = results[0].report
    else:
        print(f"\nMerging {len(results)} recordings into one document…", flush=True)
        document, status = synthesise(combine_prompt(results), 4000)
        print(f"  {status}")

    total_seconds = sum(r.info.duration for r in results)
    total_frames = sum(len(r.frame_notes) for r in results)
    header = (
        f"# Platform analysis\n\n"
        f"{len(results)} recording(s) · {total_seconds / 60:.0f} min · "
        f"{results[0].resolution} · {total_frames} unique screens analysed\n\n"
        f"> Geometry below is **estimated by a vision model**, not measured. Treat "
        f"dimensions as proportional guidance, not a pixel spec. Element labels and "
        f"transcribed text are more reliable than coordinates.\n\n---\n\n"
    )

    appendix = ["\n\n---\n\n# Appendix: per-recording detail\n"]
    for result in results:
        appendix.append(f"\n## {result.info.path.name}\n\n{result.report}\n")
        if result.transcript:
            appendix.append(
                f"\n<details><summary>Transcript — {result.info.path.name}"
                f"</summary>\n\n```\n{result.transcript}\n```\n</details>\n"
            )

    out_path.write_text(header + document.strip() + "".join(appendix), encoding="utf-8")

    if args.keep_frames:
        dest = out_path.parent / f"{out_path.stem}_frames"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        for frame_dir in workdir.glob("*_frames"):
            shutil.copytree(frame_dir, dest / frame_dir.name)
        print(f"  frames kept in {dest}")

    shutil.rmtree(workdir, ignore_errors=True)
    print(f"\ndone in {(time.time() - started) / 60:.1f} min → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
