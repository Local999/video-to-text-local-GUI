import logging
from pathlib import Path
from unittest.mock import MagicMock

from src.transcription.asr_engine import transcribe_audio


def _fake_model(segments):
    model = MagicMock()
    model.transcribe.return_value = {
        "segments": segments,
        "text": " ".join(s["text"] for s in segments),
    }
    return model


def test_transcribe_disables_condition_on_previous_text():
    """Regression guard for the trailing-hallucination bug.

    With Whisper's default ``condition_on_previous_text=True`` the large-v3
    decoder ran away into a repeated/garbled loop, emitting ~18s of phantom
    segments past the true end of audio. The fix passes
    ``condition_on_previous_text=False``; this test pins that contract.
    """
    model = _fake_model([{"text": "hello world", "start": 0.0, "end": 1.0}])

    transcribe_audio(
        model=model,
        audio_path=Path("/tmp/does_not_exist.mp3"),
        language="en",
        progress_update_interval_seconds=0.25,
        logger=logging.getLogger("test_asr"),
    )

    model.transcribe.assert_called_once()
    _, kwargs = model.transcribe.call_args
    assert kwargs.get("condition_on_previous_text") is False, (
        "condition_on_previous_text must be False to prevent runaway "
        "trailing hallucination"
    )
    assert kwargs.get("language") == "en"
    assert kwargs.get("fp16") is False


def test_transcribe_preserves_segment_data():
    """Segment text/timestamps from Whisper survive into the document."""
    model = _fake_model(
        [
            {"text": " first ", "start": 0.0, "end": 1.5},
            {"text": "second", "start": 1.5, "end": 3.0},
        ]
    )

    doc = transcribe_audio(
        model=model,
        audio_path=Path("/tmp/does_not_exist.mp3"),
        language="en",
        progress_update_interval_seconds=0.25,
        logger=logging.getLogger("test_asr"),
    )

    assert [s.text for s in doc.segments] == ["first", "second"]
    assert doc.segments[0].start_time == 0.0
    assert doc.segments[-1].end_time == 3.0
