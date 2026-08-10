"""One model resident at a time.

A 16 GB Mac cannot hold a 9B text model and an 8B vision model simultaneously
once you account for the KV cache and whatever else macOS wants. So this keeps
exactly one loaded and swaps when the other is needed. Swapping costs a few
seconds of reload; running out of wired memory costs a hard stall or a crash.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import mlx.core as mx

from . import config


@dataclass
class LoadedModel:
    kind: str  # "text" | "vision"
    repo: str
    model: object
    # Tokenizer for text models, processor for vision models.
    processor: object
    # Vision models need their raw config dict for the chat template.
    raw_config: dict | None = None


def _free_memory() -> None:
    gc.collect()
    # mx.clear_cache() releases MLX's buffer pool back to the OS. Guarded
    # because the name has moved around across MLX releases.
    for name in ("clear_cache", "reset_peak_memory"):
        fn = getattr(mx, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


class ModelManager:
    """Thread-safe holder for the single resident model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None

    @property
    def status(self) -> str:
        if self._current is None:
            return "No model loaded"
        gb = mx.get_active_memory() / 1e9
        return f"{self._current.kind}: {self._current.repo} ({gb:.1f} GB resident)"

    def unload(self) -> None:
        with self._lock:
            self._current = None
            _free_memory()

    def ensure(self, kind: str) -> LoadedModel:
        """Return a loaded model of `kind`, swapping the other one out if needed."""
        with self._lock:
            if self._current is not None and self._current.kind == kind:
                return self._current

            # Drop the incumbent before allocating the replacement, or we
            # briefly need room for both.
            if self._current is not None:
                self._current = None
                _free_memory()

            if kind == "text":
                from mlx_lm import load as load_lm

                model, tokenizer = load_lm(config.TEXT_MODEL)
                self._current = LoadedModel("text", config.TEXT_MODEL, model, tokenizer)
            elif kind == "vision":
                from mlx_vlm import load as load_vlm
                from mlx_vlm.utils import load_config as load_vlm_config

                model, processor = load_vlm(config.VISION_MODEL)
                raw_config = load_vlm_config(config.VISION_MODEL)
                self._current = LoadedModel(
                    "vision", config.VISION_MODEL, model, processor, raw_config
                )
            else:
                raise ValueError(f"unknown model kind: {kind!r}")

            return self._current


MANAGER = ModelManager()
