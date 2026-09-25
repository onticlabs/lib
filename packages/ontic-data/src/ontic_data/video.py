"""Random-access video frame reading.

One reader API over ``"torchcodec"`` (default), ``"decord"`` and ``"opencv"``.
OpenCV uses FFmpeg's container index without TorchCodec's full-file startup scan.
The backend is never picked from what happens to be installed: pass it, and check availability with
:func:`has_video_backend`.

Training readers are owned per sample. The viewer keeps readers for only its
current trajectory per source; avoid unbounded caches of open recordings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

VideoBackend = Literal["torchcodec", "decord", "opencv"]
DEFAULT_BACKEND: VideoBackend = "torchcodec"


def _import_backend(backend: str):
    if backend == "torchcodec":
        try:
            from torchcodec import decoders
        except (ImportError, RuntimeError) as e:  # RuntimeError: no FFmpeg shared libs
            raise ImportError(
                "video backend 'torchcodec' requires torchcodec (with FFmpeg shared libs); "
                "install ontic-data[video]"
            ) from e
        return decoders
    if backend == "decord":
        try:
            import decord
        except ImportError as e:
            raise ImportError(
                "video backend 'decord' requires decord; install ontic-data[decord]"
            ) from e
        return decord
    if backend == "opencv":
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                "video backend 'opencv' requires OpenCV; install ontic-data[opencv]"
            ) from e
        return cv2
    raise ValueError(f"unknown video backend {backend!r} (expected torchcodec, decord or opencv)")


def has_video_backend(backend: str = DEFAULT_BACKEND) -> bool:
    """Whether ``backend`` imports on this host (no fallback to the other one)."""
    try:
        _import_backend(backend)
    except ImportError:
        return False
    return True


class VideoReader:
    """Random-access RGB frame reader.

    ``get_frames(indices)`` returns ``(T, H, W, 3)`` uint8 CPU tensors in the requested
    order; indices must be in ``[0, len(self))`` (callers clamp).
    """

    def __init__(self, path: str | Path, backend: VideoBackend = DEFAULT_BACKEND):
        mod = _import_backend(backend)
        self._backend = backend
        self._path = str(path)
        if backend == "opencv":
            self._dec = mod.VideoCapture(self._path, mod.CAP_FFMPEG)
            if not self._dec.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            self._n = int(self._dec.get(mod.CAP_PROP_FRAME_COUNT))
            if self._n <= 0:
                raise RuntimeError(f"Video has no frame-count metadata: {path}")
            return
        if backend == "torchcodec":
            decoder_cls = getattr(mod, "VideoDecoder", None) or mod.SimpleVideoDecoder
            self._dec = decoder_cls(str(path))
            # torchcodec >= 0.1 has index-list APIs; 0.0.x only a strided range call.
            self._index_api = hasattr(self._dec, "get_frames_in_range")
        else:
            self._dec = mod.VideoReader(str(path), ctx=mod.cpu(0))
        self._n = len(self._dec)

    def __len__(self) -> int:
        return self._n

    def get_frames(self, indices: list[int]) -> torch.Tensor:
        ix = [int(i) for i in indices]
        if self._backend == "opencv":
            import cv2

            frames = []
            for i in ix:
                if not 0 <= i < self._n:
                    raise IndexError(f"Frame {i} is outside [0, {self._n}) in {self._path}")
                if int(self._dec.get(cv2.CAP_PROP_POS_FRAMES)) != i:
                    if not self._dec.set(cv2.CAP_PROP_POS_FRAMES, i):
                        raise RuntimeError(f"Cannot seek to frame {i} in {self._path}")
                ok, frame = self._dec.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {i} in {self._path}")
                frames.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return torch.stack(frames)
        if self._backend == "decord":
            batch = self._dec.get_batch(ix)
            # decord's torch bridge is thread-local, so the batch may be a torch tensor or
            # a decord NDArray depending on the calling thread; handle both.
            if isinstance(batch, torch.Tensor):
                return batch
            return torch.from_numpy(batch.asnumpy())
        if self._index_api:
            return self._dec.get_frames_at(ix).data.permute(0, 2, 3, 1)
        if len(ix) > 1:
            step = ix[1] - ix[0]
            if step > 0 and all(b - a == step for a, b in zip(ix, ix[1:])):
                return self._dec.get_frames_at(ix[0], ix[-1] + 1, step).data.permute(0, 2, 3, 1)
        return torch.stack([self._dec.get_frame_at(i).data for i in ix]).permute(0, 2, 3, 1)
