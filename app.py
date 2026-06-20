from __future__ import annotations

import logging
import threading
from pathlib import Path

import gradio as gr
import gradio_client.utils as _gradio_client_utils

from src.history import HistoryEntry, HistoryStore
from src.service import transcribe_file
from src.utils import MediaDecodeError, ProcessingError, load_yaml_file, setup_logging


def _install_gradio_client_bool_schema_patch() -> None:
    """Work around a gradio_client 1.3.0 crash on boolean JSON-schema nodes.

    Loading the GUI's main page builds the API info, which walks every
    component's JSON schema via gradio_client's ``_json_schema_to_python_type``.
    pydantic 2.13 emits ``additionalProperties: true`` (a *bool*) for open dict
    types; the walker then runs ``"const" in True`` and raises
    ``TypeError: argument of type 'bool' is not iterable``. gradio surfaces that
    as HTTP 500 on ``/``, and because its launch-time ``url_ok`` check then sees
    a 500 it aborts with the misleading "localhost is not accessible" error — so
    the app never actually serves. We can't upgrade gradio (gradio>=5 needs
    Python>=3.10; this project is pinned to 3.9), so we make the walker treat any
    non-dict schema node as ``Any``. Cosmetic only: it affects the generated
    type-hint strings in the API docs, never the GUI's behavior. Idempotent and
    self-disabling if a future gradio_client drops the private function.
    """
    try:
        original = _gradio_client_utils._json_schema_to_python_type
    except AttributeError:  # pragma: no cover - future gradio_client without this internal
        logging.getLogger(__name__).warning(
            "gradio_client._json_schema_to_python_type not found; "
            "skipping the boolean-schema compatibility patch."
        )
        return
    if getattr(original, "_bool_schema_safe", False):
        return

    def _json_schema_to_python_type(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return original(schema, defs)

    _json_schema_to_python_type._bool_schema_safe = True
    _gradio_client_utils._json_schema_to_python_type = _json_schema_to_python_type


# Must run before any Blocks app builds its API info (i.e. before build_ui/launch).
_install_gradio_client_bool_schema_patch()

CONFIG_PATH = "configurations/general_config.yaml"
MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
LANGUAGE_CHOICES = ["Auto-detect", "en", "ru", "es", "fr", "de", "ar", "it", "pt", "zh", "ja"]

_cfg = load_yaml_file(CONFIG_PATH)
_transcripts_dir = Path(_cfg["paths"]["transcripts"])
_default_model = load_yaml_file(_cfg["files"]["params"])["transcription_model"]
_history = HistoryStore(_transcripts_dir / ".history.json")

logger = setup_logging(
    logs_dir=_cfg["paths"]["logs"],
    level=_cfg["logging"]["level"],
    file_name=_cfg["logging"]["file_name"],
    log_format=_cfg["logging"]["format"],
)

_HISTORY_COLUMNS = ["id", "file", "model", "language", "duration (s)", "words", "date", "status"]


def _history_rows(records=None) -> list[list]:
    records = _history.list() if records is None else records
    rows = []
    for r in records:
        rows.append([
            r.get("id", ""),
            r.get("source_filename", ""),
            r.get("model", ""),
            r.get("language", ""),
            round(r.get("duration_seconds") or 0.0, 1),
            r.get("word_count", 0),
            (r.get("created_at", "") or "")[:19].replace("T", " "),
            r.get("status", ""),
        ])
    return rows


def _make_monotonic_progress(progress_fn, desc: str = "Transcribing…"):
    """Wrap a Gradio progress callable so it is safe to drive from asr_engine's
    background ticker thread AND the main thread's finally(1.0): serialize the
    two with a lock and never move the bar backwards (drop out-of-order updates).
    """
    lock = threading.Lock()
    state = {"last": -1.0}

    def _cb(fraction: float) -> None:
        with lock:
            if fraction >= state["last"]:
                state["last"] = fraction
                progress_fn(fraction, desc=desc)

    return _cb


def _run_transcription(file_path, model, language, progress=gr.Progress()):
    if not file_path:
        raise gr.Error("Please upload a media file first.")
    lang = None if language == "Auto-detect" else language
    progress(0.0, desc="Starting…")
    safe_progress = _make_monotonic_progress(progress)
    try:
        result = transcribe_file(
            file_path,
            model=model,
            language=lang,
            sanitize_output=True,
            config_path=CONFIG_PATH,
            logger=logger,
            progress_callback=safe_progress,
        )
    except MediaDecodeError as exc:
        raise gr.Error(str(exc))
    except ProcessingError as exc:
        raise gr.Error(f"Processing failed: {exc}")

    if result.status == "failed":
        raise gr.Error(result.message or "Transcription failed.")

    doc = result.document
    _history.add(
        HistoryEntry(
            id=doc.id,
            source_filename=Path(file_path).name,
            model=result.model,
            language=result.language,
            options=result.options,
            duration_seconds=doc.duration_seconds,
            word_count=len((doc.full_text or "").split()),
            created_at=doc.created_at.isoformat(),
            status=result.status,
            outputs={"txt": result.txt_path, "clean": result.clean_path,
                     "srt": result.srt_path, "vtt": result.vtt_path},
            message=result.message,
        )
    )
    text = doc.full_text if doc else ""
    return text, result.txt_path, gr.update(value=_history_rows())


def _load_selected(history_table, evt: gr.SelectData):
    # evt.index = [row, col]; first column is the id
    if evt is None or not history_table:
        return "", None
    row = evt.index[0]
    entry_id = history_table[row][0]
    rec = _history.get(entry_id)
    if not rec:
        return "", None
    txt = (rec.get("outputs") or {}).get("txt")
    content = Path(txt).read_text(encoding="utf-8") if txt and Path(txt).exists() else "(file missing)"
    return content, txt


def _delete_selected(entry_id):
    if not entry_id:
        raise gr.Error("Select a row to delete (its id appears in the box).")
    _history.delete(entry_id)
    return gr.update(value=_history_rows())


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Whisper Transcription") as demo:
        gr.Markdown("# 🎙️ Local Whisper Transcription\nLocal, private, zero-cost.")

        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(
                        label="Upload audio/video",
                        file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".m4a"],
                        type="filepath",
                    )
                    model_in = gr.Dropdown(
                        MODEL_CHOICES, value=_default_model, label="Model",
                        info="Bigger = more accurate but slower on CPU. large-v3 downloads ~3 GB on first use.",
                    )
                    lang_in = gr.Dropdown(LANGUAGE_CHOICES, value="Auto-detect", label="Language")
                    run_btn = gr.Button("Transcribe", variant="primary")
                with gr.Column(scale=2):
                    out_text = gr.Textbox(label="Transcript", lines=18, show_copy_button=True)
                    out_file = gr.File(label="Download .txt")

        with gr.Tab("History"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                del_id = gr.Textbox(label="Delete by id", scale=2)
                del_btn = gr.Button("Delete", variant="stop")
            history_table = gr.Dataframe(
                headers=_HISTORY_COLUMNS, value=_history_rows(), interactive=False, wrap=True,
            )
            hist_text = gr.Textbox(label="Selected transcript", lines=14, show_copy_button=True)
            hist_file = gr.File(label="Download selected .txt")

        run_btn.click(
            _run_transcription,
            inputs=[file_in, model_in, lang_in],
            outputs=[out_text, out_file, history_table],
        )
        refresh_btn.click(lambda: gr.update(value=_history_rows()), outputs=history_table)
        history_table.select(_load_selected, inputs=[history_table], outputs=[hist_text, hist_file])
        del_btn.click(_delete_selected, inputs=[del_id], outputs=[history_table])

    return demo


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", inbrowser=True)


if __name__ == "__main__":
    main()
