"""ffmpeg/ffprobe wrappers: probe metadata, pull audio, pull keyframes.

ffmpeg has no content policy of any kind -- it is a codec and filter toolchain.
Nothing here needs an "uncensored" anything.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class FFmpegMissing(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FFmpegMissing(
            f"{tool} not found on PATH. Install it with:  brew install ffmpeg"
        )
    return path


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def probe(path: str | Path) -> VideoInfo:
    ffprobe = _require("ffprobe")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such video: {path}")

    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)

    video_stream = next(
        (s for s in data["streams"] if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise ValueError(f"{path.name} has no video stream")

    has_audio = any(s.get("codec_type") == "audio" for s in data["streams"])

    # avg_frame_rate arrives as "30000/1001"; guard against a 0 denominator.
    fps = 0.0
    rate = video_stream.get("avg_frame_rate", "0/0")
    if "/" in rate:
        num, _, den = rate.partition("/")
        if float(den or 0):
            fps = float(num) / float(den)

    return VideoInfo(
        path=path,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        duration=float(data["format"].get("duration", 0.0)),
        fps=fps,
        has_audio=has_audio,
    )


def extract_audio(video: Path, out_wav: Path) -> Path:
    """Whisper wants 16 kHz mono PCM; give it exactly that."""
    ffmpeg = _require("ffmpeg")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)],
        check=True, capture_output=True,
    )
    return out_wav


def _scene_timestamps(video: Path, threshold: float) -> list[float]:
    """Timestamps where the frame changes substantially.

    For a screen recording these land on navigation and state changes, which is
    exactly what you want to look at -- far better than sampling on a timer.
    """
    ffmpeg = _require("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-i", str(video), "-vf",
         f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]


def _even_timestamps(duration: float, count: int) -> list[float]:
    if duration <= 0 or count <= 0:
        return [0.0]
    # Offset by half a step so we never grab a black first/last frame.
    step = duration / count
    return [round(step * (i + 0.5), 3) for i in range(count)]


def _thin(values: list, limit: int) -> list:
    if len(values) <= limit:
        return values
    stride = len(values) / limit
    return [values[int(i * stride)] for i in range(limit)]


def _average_hash(path: Path, size: int = 16) -> int:
    """Cheap perceptual hash: downscale to greyscale, threshold on the mean."""
    from PIL import Image

    with Image.open(path) as img:
        small = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        pixels = list(small.getdata())

    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value >= mean:
            bits |= 1 << index
    return bits


def _too_similar(a: int, b: int, max_distance: int) -> bool:
    return bin(a ^ b).count("1") <= max_distance


def extract_keyframes(
    info: VideoInfo,
    out_dir: Path,
    max_frames: int = 12,
    scene_threshold: float = 0.30,
    max_width: int = 1280,
    dedupe_distance: int = 12,
    crop: str | None = None,
) -> list[tuple[float, Path]]:
    """Return [(timestamp_seconds, png_path)] in chronological order.

    Frames are downscaled to `max_width` to keep vision-token count and memory
    sane; the caller still reports geometry against the true resolution.

    `dedupe_distance` is the Hamming radius on a 256-bit average hash below
    which two frames count as the same screen. Raise it to be more aggressive
    about collapsing near-identical frames, lower it to keep subtle state
    changes (a dropdown opening, a row highlighting).

    `crop` is an ffmpeg crop spec, "W:H:X:Y". Use it when the app under study
    fills only part of the frame -- a shared screen inside a video call, a
    window on a larger desktop. Cropping first means the downscale budget is
    spent on the UI instead of on window chrome, and the model stops describing
    the surrounding desktop as though it were part of the product.
    """
    ffmpeg = _require("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extracting a frame is cheap; describing one with a vision model is not.
    # So over-sample here, then throw away duplicates before the expensive step.
    candidates = set(_scene_timestamps(info.path, scene_threshold))
    candidates.update(_even_timestamps(info.duration, max_frames * 3))
    timestamps = _thin(sorted(candidates), max_frames * 4)

    # Crop before scaling, so max_width applies to the region we care about.
    filters = ([f"crop={crop}"] if crop else []) + [f"scale='min({max_width},iw)':-2"]
    scale = ",".join(filters)
    kept: list[tuple[float, Path]] = []
    hashes: list[int] = []

    for index, ts in enumerate(timestamps):
        dest = out_dir / f"frame_{index:03d}_{ts:.2f}s.png"
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{ts}",
             "-i", str(info.path), "-frames:v", "1", "-vf", scale, str(dest)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
            continue

        try:
            digest = _average_hash(dest)
        except Exception:
            kept.append((ts, dest))  # can't hash it, keep it rather than lose it
            continue

        if any(_too_similar(digest, seen, dedupe_distance) for seen in hashes):
            dest.unlink(missing_ok=True)
            continue

        hashes.append(digest)
        kept.append((ts, dest))

    return _thin(kept, max_frames)
