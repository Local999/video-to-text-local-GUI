import logging
from pathlib import Path

from src.models import PipelineContext
from src.pipeline import PipelineOrchestrator, PipelineStep
from src.utils import MediaDecodeError, ProcessingError


class RaisingStep(PipelineStep):
    def __init__(self, exc: Exception):
        self._exc = exc

    def execute(self, context, logger):
        raise self._exc


def _ctx():
    return PipelineContext(source_path=Path("/tmp/x.mp3"), input_type="audio")


def test_media_decode_error_is_processing_error():
    assert issubclass(MediaDecodeError, ProcessingError)


def test_orchestrator_records_exception_on_context():
    logger = logging.getLogger("test")
    exc = MediaDecodeError("bad file")
    pipeline = PipelineOrchestrator(steps=[RaisingStep(exc)], logger=logger)
    ctx = pipeline.run(_ctx())
    assert ctx.exception is exc
    assert ctx.errors  # string error still recorded


def test_orchestrator_exception_none_on_success():
    logger = logging.getLogger("test")
    pipeline = PipelineOrchestrator(steps=[], logger=logger)
    ctx = pipeline.run(_ctx())
    assert ctx.exception is None
