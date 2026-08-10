"""Turn uploaded files into something a language model can actually consume.

Images are passed through as paths (the vision model handles them natively).
Everything else is flattened to text and fenced with its filename.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Attachments:
    """The result of processing whatever the user dragged into the box."""

    images: list[str] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    def as_prompt_preamble(self) -> str:
        """Text to prepend to the user's message, or "" if there's nothing."""
        if not self.text_blocks:
            return ""
        joined = "\n\n".join(self.text_blocks)
        return f"The user attached the following file(s):\n\n{joined}\n\n---\n\n"


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - pypdf is a declared dependency
        return "[pypdf not installed; cannot read PDF]"

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"--- page {i + 1} ---\n{text}")
    if not pages:
        return "[no extractable text -- this PDF is probably scanned images]"
    return "\n\n".join(pages)


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8, returning None if it's binary."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        return None
    except OSError as exc:
        return f"[could not read: {exc}]"


def process(paths: list[str]) -> Attachments:
    result = Attachments()
    budget = config.MAX_TOTAL_ATTACHMENT_CHARS

    for raw in paths or []:
        path = Path(raw)
        if not path.exists():
            result.notes.append(f"{path.name}: file disappeared before reading")
            continue

        suffix = path.suffix.lower()

        if suffix in config.IMAGE_EXTENSIONS:
            result.images.append(str(path))
            continue

        if suffix == ".pdf":
            content = _read_pdf(path)
        elif suffix in config.TEXT_EXTENSIONS or suffix == "":
            content = _read_text(path)
            if content is None:
                result.notes.append(f"{path.name}: looks binary, skipped")
                continue
        else:
            content = _read_text(path)
            if content is None:
                result.notes.append(
                    f"{path.name}: unsupported binary type ({suffix or 'no extension'}), skipped"
                )
                continue

        if len(content) > config.MAX_CHARS_PER_FILE:
            content = content[: config.MAX_CHARS_PER_FILE]
            result.notes.append(
                f"{path.name}: truncated to {config.MAX_CHARS_PER_FILE:,} chars"
            )

        if len(content) > budget:
            content = content[:budget]
            result.notes.append(f"{path.name}: hit the total attachment budget")
        budget -= len(content)

        lang = suffix.lstrip(".") if suffix else ""
        result.text_blocks.append(f"`{path.name}`:\n```{lang}\n{content}\n```")

        if budget <= 0:
            result.notes.append("Attachment budget exhausted; later files ignored.")
            break

    return result


def describe(paths: list[str]) -> str:
    """Short human-readable summary for the status line."""
    if not paths:
        return ""
    names = [os.path.basename(p) for p in paths]
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:3])} +{len(names) - 3} more"
