from __future__ import annotations

import gc
import logging
import threading

import whisper

_MODEL_LOCK = threading.Lock()
_resident_model: tuple[str, object] | None = None  # (name, model)
_diarization_backend: tuple[str, str, object] | None = None  # (backend, model_id, obj)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_whisper_model(name: str, device: str, logger: logging.Logger | None = None):
    """Return a resident Whisper model, holding at most one in memory.

    Reuses the resident model when ``name`` matches; otherwise evicts the
    previous model (drop ref + gc + empty CUDA cache) before loading the new
    one. Jobs run one at a time, so only one model is ever needed at once.
    """
    global _resident_model
    log = logger or logging.getLogger("video_to_text")
    with _MODEL_LOCK:
        if _resident_model is not None and _resident_model[0] == name:
            return _resident_model[1]
        if _resident_model is not None:
            log.info("Evicting Whisper model '%s' to load '%s'", _resident_model[0], name)
            _resident_model = None
            gc.collect()
            _empty_cuda_cache()
        log.info("Loading Whisper model: %s", name)
        model = whisper.load_model(name, device=device)
        _resident_model = (name, model)
        log.info("Whisper model loaded: %s", name)
        return model


def reset_caches() -> None:
    """Drop the resident model and diarization backend (tests/teardown)."""
    global _resident_model, _diarization_backend
    with _MODEL_LOCK:
        _resident_model = None
        _diarization_backend = None
        gc.collect()
        _empty_cuda_cache()
