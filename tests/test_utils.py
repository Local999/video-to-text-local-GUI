import os
import shutil
from pathlib import Path

import imageio_ffmpeg
import pytest

from src.utils.config import load_yaml_file
from src.utils.errors import ProcessingError
from src.utils.system import ensure_ffmpeg_on_path


class TestLoadYamlFile:
    def test_loads_valid_yaml(self, tmp_path: Path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n", encoding="utf-8")
        result = load_yaml_file(str(yaml_file))
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_raises_on_non_mapping(self, tmp_path: Path):
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ProcessingError, match="Expected YAML mapping"):
            load_yaml_file(str(yaml_file))

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_yaml_file("/nonexistent/path.yaml")


class TestProcessingError:
    def test_is_runtime_error(self):
        exc = ProcessingError("test")
        assert isinstance(exc, RuntimeError)
        assert str(exc) == "test"


class TestEnsureFfmpegOnPath:
    """Covers the ffmpeg auto-provisioning fix.

    Whisper's load_audio shells out to a bare ``ffmpeg`` and the project
    defaults ``dependencies.ffmpeg_executable`` to ``"ffmpeg"`` too. On a
    machine without a system ffmpeg (the GUI-hang bug), nothing was ever put
    on PATH, so transcription died with FileNotFoundError. ensure_ffmpeg_on_path
    falls back to the binary already bundled with imageio-ffmpeg (a moviepy
    dependency), exposing it under the name ``ffmpeg`` on PATH.
    """

    def _fake_binary(self, path: Path) -> Path:
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        return path

    def test_prefers_system_ffmpeg_and_leaves_path_untouched(self, monkeypatch, tmp_path):
        # A real system ffmpeg must win: no shim, no PATH mutation.
        sysbin = tmp_path / "sysbin"
        sysbin.mkdir()
        sys_ffmpeg = self._fake_binary(sysbin / "ffmpeg")
        monkeypatch.setenv("PATH", str(sysbin))
        before = os.environ["PATH"]
        shim_dir = tmp_path / "shim"

        result = ensure_ffmpeg_on_path(shim_dir=shim_dir)

        assert result == str(sys_ffmpeg)
        assert os.environ["PATH"] == before  # untouched
        assert not shim_dir.exists()  # no shim created when system ffmpeg exists

    def test_falls_back_to_bundled_binary_when_no_system_ffmpeg(self, monkeypatch, tmp_path):
        # Stand-in for the imageio-ffmpeg bundled binary (real + executable).
        bundled = self._fake_binary(tmp_path / "ffmpeg-bundled")
        empty = tmp_path / "empty"  # a PATH with no ffmpeg anywhere on it
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(bundled))
        shim_dir = tmp_path / "shim"

        result = ensure_ffmpeg_on_path(shim_dir=shim_dir)

        shim = shim_dir / "ffmpeg"
        assert result == str(shim)
        assert shim.is_symlink()
        assert os.path.realpath(shim) == os.path.realpath(bundled)
        # shim dir prepended so a *bare* `ffmpeg` resolves to the bundled binary
        assert os.environ["PATH"].split(os.pathsep)[0] == str(shim_dir)
        # The decisive assertion: a bare "ffmpeg" now resolves end-to-end,
        # which is exactly the lookup whisper's load_audio performs.
        assert shutil.which("ffmpeg") == str(shim)

    def test_idempotent_reuses_existing_shim(self, monkeypatch, tmp_path):
        bundled = self._fake_binary(tmp_path / "ffmpeg-bundled")
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(bundled))
        shim_dir = tmp_path / "shim"

        first = ensure_ffmpeg_on_path(shim_dir=shim_dir)
        second = ensure_ffmpeg_on_path(shim_dir=shim_dir)

        assert first == second == str(shim_dir / "ffmpeg")
        # PATH must not accumulate duplicate shim-dir entries across calls.
        assert os.environ["PATH"].split(os.pathsep).count(str(shim_dir)) == 1

    def test_returns_none_when_no_system_and_no_bundled(self, monkeypatch, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))
        before = os.environ["PATH"]

        def boom():
            raise OSError("imageio-ffmpeg ships no binary for this platform")

        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", boom)

        result = ensure_ffmpeg_on_path(shim_dir=tmp_path / "shim")

        assert result is None  # caller surfaces a clear ProcessingError
        assert os.environ["PATH"] == before  # no mutation on failure
