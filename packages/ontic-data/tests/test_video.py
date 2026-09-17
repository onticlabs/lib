"""Video backends are explicit; a missing one raises an ImportError naming the extra."""

import sys

import pytest

from ontic_data.video import VideoReader, has_video_backend


@pytest.fixture
def no_backends(monkeypatch):
    for name in ("torchcodec", "torchcodec.decoders", "decord"):
        monkeypatch.setitem(sys.modules, name, None)


def test_has_video_backend_false_when_blocked(no_backends):
    assert has_video_backend("torchcodec") is False
    assert has_video_backend("decord") is False
    assert has_video_backend() is False


def test_video_reader_errors_name_the_extra(no_backends, tmp_path):
    with pytest.raises(ImportError, match=r"ontic-data\[video\]"):
        VideoReader(tmp_path / "x.mp4")
    with pytest.raises(ImportError, match=r"ontic-data\[decord\]"):
        VideoReader(tmp_path / "x.mp4", backend="decord")


def test_unknown_backend_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="unknown video backend"):
        VideoReader(tmp_path / "x.mp4", backend="ffmpeg")
    with pytest.raises(ValueError):
        has_video_backend("ffmpeg")
