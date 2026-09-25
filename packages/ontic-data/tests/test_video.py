"""Video backends are explicit; a missing one raises an ImportError naming the extra."""

import sys

import pytest
import torch

from ontic_data.video import VideoReader, has_video_backend


@pytest.fixture
def no_backends(monkeypatch):
    for name in ("torchcodec", "torchcodec.decoders", "decord", "cv2"):
        monkeypatch.setitem(sys.modules, name, None)


def test_has_video_backend_false_when_blocked(no_backends):
    assert has_video_backend("torchcodec") is False
    assert has_video_backend("decord") is False
    assert has_video_backend("opencv") is False
    assert has_video_backend() is False


def test_video_reader_errors_name_the_extra(no_backends, tmp_path):
    with pytest.raises(ImportError, match=r"ontic-data\[video\]"):
        VideoReader(tmp_path / "x.mp4")
    with pytest.raises(ImportError, match=r"ontic-data\[decord\]"):
        VideoReader(tmp_path / "x.mp4", backend="decord")
    with pytest.raises(ImportError, match=r"ontic-data\[opencv\]"):
        VideoReader(tmp_path / "x.mp4", backend="opencv")


def test_unknown_backend_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="unknown video backend"):
        VideoReader(tmp_path / "x.mp4", backend="ffmpeg")
    with pytest.raises(ValueError):
        has_video_backend("ffmpeg")


def test_opencv_header_count_and_random_access_match_sequential_decode(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (32, 24))
    if not writer.isOpened():
        pytest.skip("OpenCV was built without an MJPEG encoder")
    for i in range(12):
        writer.write(np.full((24, 32, 3), (i * 17, 20, 240 - i * 13), dtype=np.uint8))
    writer.release()
    reader = VideoReader(path, backend="opencv")
    assert len(reader) == 12
    reference = reader.get_frames(list(range(12)))
    assert reference.dtype == torch.uint8 and reference.shape == (12, 24, 32, 3)
    assert reference[0, :, :, 0].float().mean() > reference[0, :, :, 2].float().mean()  # RGB
    assert torch.equal(reader.get_frames([11, 0, 7, 7, 3]), reference[[11, 0, 7, 7, 3]])
    with pytest.raises(IndexError):
        reader.get_frames([12])
    with pytest.raises(RuntimeError, match="Cannot open video"):
        VideoReader(tmp_path / "missing.mp4", backend="opencv")
