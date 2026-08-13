"""Model choices and tunables.

Everything here can be overridden with an environment variable so you can
A/B a different checkpoint without touching code:

    ABLIT_TEXT_MODEL=mlx-community/Qwen2.5-Coder-14B-Instruct-abliterated-4bit uv run python app.py
"""

import os

# Text model. ~5.2 GB at 4-bit and ~1.7x faster than the 14B, leaving room for
# a longer KV cache and a less painful swap to the vision model. The 14B has a
# higher ceiling on hard tasks but costs ~3 GB more resident:
#   mlx-community/Qwen2.5-Coder-14B-Instruct-abliterated-4bit
TEXT_MODEL = os.environ.get(
    "ABLIT_TEXT_MODEL", "huihui-ai/Huihui-Qwen3.5-9B-abliterated-mlx-4bit"
)

# Vision model, loaded only when you attach an image. The text model above is
# text-only and physically cannot see pictures -- different architecture.
VISION_MODEL = os.environ.get(
    "ABLIT_VISION_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise coding assistant. Prefer complete, runnable code over "
    "snippets. State assumptions explicitly instead of guessing silently."
)

# Generation defaults.
MAX_TOKENS = 2048
# Lowered from 0.7 for reproducibility when comparing runs -- note that in
# testing this did NOT measurably improve correctness (see verify.py).
TEMPERATURE = 0.3
TOP_P = 0.95

# Attachment limits. A 9B with a 16 GB budget has a small effective context --
# the KV cache grows with every token you paste in, so cap it hard.
MAX_CHARS_PER_FILE = 24_000
MAX_TOTAL_ATTACHMENT_CHARS = 60_000

# Extensions we treat as readable text even though they aren't .txt.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".jsonl",
    ".html", ".htm", ".css", ".scss", ".sass",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".rs", ".go", ".java", ".kt", ".swift",
    ".rb", ".php", ".sh", ".bash", ".zsh", ".fish", ".sql",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".svg", ".graphql", ".proto", ".dockerfile", ".gitignore",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

# Video can't be answered inline -- it goes through analyze_video.py, which
# needs Whisper and the vision model loaded in sequence over several minutes.
# Detected here so the UI can say so instead of silently dropping the file.
VIDEO_EXTENSIONS = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".wmv",
}
