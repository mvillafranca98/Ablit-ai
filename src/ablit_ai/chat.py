"""Prompt assembly and streaming generation for both model kinds."""

from __future__ import annotations

from collections.abc import Iterator

from . import config
from .models import MANAGER


def content_text(content: object) -> str:
    """Flatten one Gradio message's content down to plain text.

    Gradio 6 hands history back as a list of typed parts —
    ``[{"text": "...", "type": "text"}]`` — not as a string. Assuming a string
    here silently drops every prior turn, and the model then answers each
    message as if the conversation had just started.

    Attachment messages arrive as ``{"path": ...}`` and carry no text for the
    model to re-read, so they flatten to "" and get skipped.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
        return "\n".join(parts)
    if isinstance(content, dict) and content.get("text"):
        return str(content["text"])
    return ""


def build_messages(
    history: list[dict],
    system_prompt: str,
    user_text: str,
    max_history_chars: int = 24_000,
) -> list[dict]:
    """Assemble an OpenAI-style message list from Gradio chat history.

    Older turns are dropped once the replayed history exceeds
    `max_history_chars`. The KV cache grows with every replayed token, and on a
    16 GB machine an unbounded conversation will eventually stall the model.
    """
    turns: list[dict] = []
    for turn in history:
        role = turn.get("role")
        text = content_text(turn.get("content")).strip()
        if role in ("user", "assistant") and text:
            turns.append({"role": role, "content": text})

    # Keep the most recent turns that fit the budget, walking backwards.
    kept: list[dict] = []
    budget = max_history_chars
    for turn in reversed(turns):
        cost = len(turn["content"])
        if cost > budget:
            break
        budget -= cost
        kept.append(turn)
    kept.reverse()

    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(kept)
    messages.append({"role": "user", "content": user_text})
    return messages


def _apply_template(tokenizer, messages: list[dict], thinking: bool) -> str:
    """Render the chat template, tolerating models without a thinking switch."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except (TypeError, ValueError):
        # Model's template doesn't accept enable_thinking.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def stream_text_reply(
    history: list[dict],
    system_prompt: str,
    user_text: str,
    max_tokens: int = config.MAX_TOKENS,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    thinking: bool = False,
) -> Iterator[tuple[str, str]]:
    """Yield (accumulated_text, status_line) as the model generates."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    loaded = MANAGER.ensure("text")
    tokenizer = loaded.processor

    messages = build_messages(history, system_prompt, user_text)
    prompt = _apply_template(tokenizer, messages, thinking)
    sampler = make_sampler(temp=temperature, top_p=top_p)

    accumulated = ""
    response = None
    for response in stream_generate(
        loaded.model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        accumulated += response.text
        yield accumulated, f"generating… {response.generation_tps:.1f} tok/s"

    if response is not None:
        yield accumulated, (
            f"{response.generation_tokens} tokens · "
            f"{response.generation_tps:.1f} tok/s generation · "
            f"{response.prompt_tps:.0f} tok/s prompt · "
            f"{response.peak_memory:.1f} GB peak"
        )
    else:
        yield accumulated, "no output"


def stream_vision_reply(
    images: list[str],
    system_prompt: str,
    user_text: str,
    max_tokens: int = config.MAX_TOKENS,
    temperature: float = config.TEMPERATURE,
) -> Iterator[tuple[str, str]]:
    """Same contract as stream_text_reply, but for image inputs.

    Vision history is deliberately not replayed -- re-encoding prior images on
    every turn is the fastest way to blow the memory budget on this machine.
    """
    from mlx_vlm import apply_chat_template
    from mlx_vlm import stream_generate as vlm_stream_generate

    loaded = MANAGER.ensure("vision")

    question = user_text.strip() or "Describe this image in detail."
    if system_prompt.strip():
        question = f"{system_prompt.strip()}\n\n{question}"

    prompt = apply_chat_template(
        loaded.processor, loaded.raw_config, question, num_images=len(images)
    )

    accumulated = ""
    response = None
    for response in vlm_stream_generate(
        loaded.model,
        loaded.processor,
        prompt,
        image=images,
        max_tokens=max_tokens,
        temperature=temperature,
    ):
        accumulated += response.text
        yield accumulated, "generating…"

    if response is not None:
        yield accumulated, (
            f"{response.generation_tokens} tokens · "
            f"{response.generation_tps:.1f} tok/s · "
            f"{response.peak_memory:.1f} GB peak"
        )
    else:
        yield accumulated, "no output"
