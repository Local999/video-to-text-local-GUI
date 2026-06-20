from pathlib import Path

import src.service as service
from src.models import PipelineState, TranscriptDocument, TranscriptSegment


def _stub_doc():
    return TranscriptDocument(
        segments=[
            TranscriptSegment(text="Hello world.", start_time=0.0, end_time=1.5),
            TranscriptSegment(text="Second line.", start_time=1.5, end_time=3.0),
        ],
        language="en",
        pipeline_state=PipelineState.TRANSCRIBED,
    )


class TestModelCache:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def test_same_model_loaded_once(self, monkeypatch):
        loads = []

        def fake_load(name, device=None):
            loads.append(name)
            return f"model:{name}"

        monkeypatch.setattr(service.whisper, "load_model", fake_load)
        a = service.get_whisper_model("base", "cpu")
        b = service.get_whisper_model("base", "cpu")
        assert a is b
        assert loads == ["base"]  # loaded exactly once

    def test_model_change_evicts_and_reloads(self, monkeypatch):
        loads = []
        collected = []
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: (loads.append(name) or f"model:{name}"))
        monkeypatch.setattr(service.gc, "collect", lambda: collected.append(True))

        service.get_whisper_model("base", "cpu")
        service.get_whisper_model("large-v3", "cpu")
        assert loads == ["base", "large-v3"]
        assert collected  # eviction ran gc.collect at least once


class TestTranscribeFile:
    def setup_method(self):
        service.reset_caches()

    def teardown_method(self):
        service.reset_caches()

    def _patch(self, monkeypatch):
        # Model load is a no-op object; transcription returns a stub document.
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: object())
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cpu")

        def fake_transcribe(model, audio_path, language, progress_update_interval_seconds, logger, progress_callback=None):
            if progress_callback:
                progress_callback(1.0)
            doc = _stub_doc()
            doc.source_file = str(audio_path)
            return doc

        monkeypatch.setattr("src.pipeline.steps_transcription.transcribe_audio", fake_transcribe)

    def test_transcribes_audio_and_writes_txt(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        # point outputs at tmp by overriding config
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)  # skip real decode probe

        result = service.transcribe_file(
            src_audio, model="base", language="en", config_path=cfg, sanitize_output=True
        )
        assert result.status == "success"
        assert Path(result.txt_path).exists()
        assert "Hello world." in Path(result.txt_path).read_text(encoding="utf-8")
        assert result.model == "base"

    def test_does_not_mutate_params_yaml(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        params_file = tmp_path / "params.yaml"
        before = params_file.read_text(encoding="utf-8")
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)
        service.transcribe_file(src_audio, model="large-v3", language="en", config_path=cfg, sanitize_output=True)
        assert params_file.read_text(encoding="utf-8") == before

    def test_unsupported_extension_raises_media_decode_error(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        bad = tmp_path / "file.pdf"
        bad.write_bytes(b"x")
        import pytest

        from src.utils import MediaDecodeError

        with pytest.raises(MediaDecodeError):
            service.transcribe_file(bad, model="base", language="en", config_path=cfg)

    def test_cleanup_failure_keeps_raw_and_warns(self, monkeypatch, tmp_path):
        self._patch(monkeypatch)
        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)

        from src.utils import ProcessingError

        def boom(*a, **k):
            raise ProcessingError("Ollama down")

        monkeypatch.setattr("src.pipeline.steps_cleanup.format_document_with_speakers", lambda d: "text")
        monkeypatch.setattr(service, "cleanup_with_ollama", boom)
        result = service.transcribe_file(
            src_audio, model="base", language="en", cleanup=True, config_path=cfg, sanitize_output=True
        )
        assert result.status == "warning"
        assert Path(result.txt_path).exists()        # raw transcript saved
        assert result.clean_path is None

    def test_auto_detect_passes_no_language_kwarg(self, monkeypatch, tmp_path):
        # Integration: real transcribe_audio + real pipeline; only the Whisper
        # model and the decode probe are mocked. Asserts language=None reaches
        # model.transcribe as "no language kwarg" and the detected language is
        # captured end-to-end. (Does NOT use self._patch — uses the real
        # transcription path so the Task 3 auto-detect fix is exercised here.)
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "hola", "segments": [], "language": "es"}
        monkeypatch.setattr(service.whisper, "load_model", lambda name, device=None: mock_model)
        monkeypatch.setattr(service, "_resolve_device", lambda logger: "cpu")
        # AudioIngestionStep probes decodability (Task 4); force it to pass.
        monkeypatch.setattr("src.pipeline.steps_ingestion._probe_duration", lambda p: 1.0)

        cfg = _tmp_config(tmp_path)
        src_audio = tmp_path / "clip.mp3"
        src_audio.write_bytes(b"x")
        result = service.transcribe_file(
            src_audio, model="base", language=None, config_path=cfg, sanitize_output=True
        )
        _, kwargs = mock_model.transcribe.call_args
        assert "language" not in kwargs  # auto-detect: no language kwarg
        assert result.language == "es"  # detected language captured end-to-end


def _tmp_config(tmp_path) -> str:
    """Write a minimal general_config + params + prompts under tmp_path; return config path."""
    import textwrap

    (tmp_path / "params.yaml").write_text("transcription_model: base\ncleanup_model: x\n", encoding="utf-8")
    (tmp_path / "prompts.yaml").write_text("cleanup_prompt: clean this\n", encoding="utf-8")
    (tmp_path / "diarization.yaml").write_text("enabled: false\n", encoding="utf-8")
    cfg = tmp_path / "general_config.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            paths:
              videos: "{tmp_path}/videos"
              audios: "{tmp_path}/audios"
              transcripts: "{tmp_path}/transcripts"
              logs: "{tmp_path}/logs"
            files:
              params: "{tmp_path}/params.yaml"
              prompts: "{tmp_path}/prompts.yaml"
              diarization: "{tmp_path}/diarization.yaml"
            extensions:
              video: [".mp4", ".mov", ".avi", ".mkv", ".webm"]
              audio: [".mp3", ".m4a"]
            output:
              transcript_extension: ".txt"
              cleaned_suffix: "_clean"
              extracted_audio_extension: ".mp3"
            ollama:
              url: "http://localhost:11434/api/generate"
              timeout_seconds: 600
              request_content_type: "application/json"
            processing:
              progress_update_interval_seconds: 0.25
            dependencies:
              ffmpeg_executable: "ffmpeg"
            logging:
              level: "INFO"
              file_name: "transcriber.log"
              format: "%(message)s"
            """
        ),
        encoding="utf-8",
    )
    return str(cfg)
