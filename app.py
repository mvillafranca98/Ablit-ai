"""Gradio front-end for a local abliterated model on Apple Silicon.

    uv run python app.py

Then open http://127.0.0.1:7860 — everything runs on-device, no network calls
after the initial model download.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from src.ablit_ai import attachments, chat, config
from src.ablit_ai.models import MANAGER

CSS = """
.status-line { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
               font-size: 12px; opacity: 0.75; }
"""


def respond(
    user_input: dict,
    history: list[dict],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: bool,
):
    history = list(history or [])
    text = (user_input or {}).get("text") or ""
    files = (user_input or {}).get("files") or []

    if not text.strip() and not files:
        yield history, "Nothing to send.", gr.update()
        return

    att = attachments.process(files)
    # Pass the whole transcript through; build_messages normalises Gradio's
    # part-list content and drops anything with no text of its own.
    prior = list(history)

    # Video can't be answered inline. Say so loudly -- silently dropping the
    # file leaves the model with no content, and it will happily invent a
    # plausible-looking document about a platform it has never seen.
    if att.has_videos:
        quoted = " ".join(f'"{p}"' for p in att.videos)
        names = "\n".join(f"- `{Path(p).name}`" for p in att.videos)
        history.append({
            "role": "user",
            "content": f"{text}\n\n📎 {attachments.describe(files)}".strip(),
        })
        history.append({
            "role": "assistant",
            "content": (
                f"**I can't analyse video in this chat box — and I won't guess.**\n\n"
                f"You attached {len(att.videos)} video file(s):\n{names}\n\n"
                "Video needs Whisper for the audio and the vision model for the "
                "frames, loaded in sequence over several minutes. That runs in "
                "`analyze_video.py`, not here.\n\n"
                "Run this in your terminal:\n\n"
                f"```bash\nuv run python analyze_video.py {quoted} --frames 12\n```\n\n"
                "It writes a Markdown document covering the workflows, interface "
                "layout with approximate dimensions, and integration notes.\n\n"
                "*Anything I produced here without reading the video would be "
                "fabricated.*"
            ),
        })
        yield history, f"{len(att.videos)} video(s) — use analyze_video.py", gr.update(
            value=None, interactive=True
        )
        return

    # Show attachments in the transcript before the text turn.
    for image_path in att.images:
        history.append({"role": "user", "content": {"path": image_path}})

    shown = text
    if att.text_blocks:
        shown = f"{text}\n\n📎 {attachments.describe(files)}".strip()
    history.append({"role": "user", "content": shown})
    history.append({"role": "assistant", "content": ""})

    notes = " · ".join(att.notes)
    loading = "Loading vision model…" if att.has_images else "Loading model…"
    yield history, f"{loading} {notes}".strip(), gr.update(
        value=None, interactive=False
    )

    try:
        if att.has_images:
            stream = chat.stream_vision_reply(
                images=att.images,
                system_prompt=system_prompt,
                user_text=text,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
        else:
            stream = chat.stream_text_reply(
                history=prior,
                system_prompt=system_prompt,
                user_text=att.as_prompt_preamble() + text,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                thinking=thinking,
            )

        final_status = ""
        for partial, status in stream:
            history[-1]["content"] = partial
            final_status = f"{status} · {notes}".strip(" ·")
            yield history, final_status, gr.update()
    except Exception as exc:  # surface the error in the UI, don't kill the app
        history[-1]["content"] = f"**Error**\n\n```\n{type(exc).__name__}: {exc}\n```"
        yield history, "failed", gr.update(interactive=True)
        return

    # Keep the closing stats on screen -- they're the point of a test harness.
    yield history, final_status, gr.update(interactive=True)


def preload():
    MANAGER.ensure("text")
    return MANAGER.status


def unload():
    MANAGER.unload()
    return MANAGER.status


with gr.Blocks(title="ablit-AI", fill_height=True) as demo:
    gr.Markdown(
        f"### ablit-AI — local model tester\n"
        f"`{config.TEXT_MODEL}` · vision falls back to `{config.VISION_MODEL}` "
        f"when you attach an image (one model resident at a time)."
    )

    chatbot = gr.Chatbot(
        height=520,
        label="Conversation",
        # Collapses <think>…</think> into a foldable block instead of dumping
        # raw reasoning into the answer.
        reasoning_tags=[("<think>", "</think>")],
    )
    status = gr.Markdown("", elem_classes="status-line")

    box = gr.MultimodalTextbox(
        placeholder="Ask something, or drop in code files / a screenshot…",
        file_count="multiple",
        show_label=False,
        autofocus=True,
    )

    with gr.Accordion("Settings", open=False):
        system_prompt = gr.Textbox(
            value=config.DEFAULT_SYSTEM_PROMPT,
            label="System prompt",
            lines=3,
        )
        with gr.Row():
            max_tokens = gr.Slider(128, 8192, config.MAX_TOKENS, step=128, label="Max tokens")
            temperature = gr.Slider(0.0, 1.5, config.TEMPERATURE, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, config.TOP_P, step=0.05, label="Top-p")
        thinking = gr.Checkbox(
            value=False,
            label="Enable thinking mode (slower, shows reasoning if the model supports it)",
        )
        with gr.Row():
            load_btn = gr.Button("Preload text model")
            unload_btn = gr.Button("Unload (free RAM)")
            clear_btn = gr.Button("Clear chat")
        model_status = gr.Markdown(MANAGER.status, elem_classes="status-line")

    inputs = [box, chatbot, system_prompt, max_tokens, temperature, top_p, thinking]
    outputs = [chatbot, status, box]

    box.submit(respond, inputs, outputs)
    load_btn.click(preload, None, model_status)
    unload_btn.click(unload, None, model_status)
    clear_btn.click(lambda: ([], ""), None, [chatbot, status])


if __name__ == "__main__":
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        css=CSS,
    )
