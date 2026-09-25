"""HO-Cap sequence access: YAML calibration plus per-frame files or packed h5 stores.

Frames come from per-frame files or the packed per-camera ``frames.h5`` / ``depth.h5`` /
``labels.h5`` / ``seg.h5`` stores. cv2, h5py and scipy are imported lazily (extra ``hocap``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import yaml


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise ImportError("HO-Cap frames need opencv; install ontic-data[hocap]") from e
    return cv2


def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError("HO-Cap packed stores need h5py; install ontic-data[hocap]") from e
    return h5py


def read_data_from_yaml(file_path: Union[str, Path]) -> Any:
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(str(file_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_rgb_image(file_path: Union[str, Path]) -> np.ndarray:
    """``(H, W, 3)`` uint8 RGB."""
    cv2 = _cv2()
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path))
    if image is None:
        raise ValueError(f"Failed to load image from '{file_path}'.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_depth_image(file_path: Union[str, Path], scale: float = 1.0) -> np.ndarray:
    """``(H, W)`` float32 depth, stored integer units divided by ``scale``."""
    cv2 = _cv2()
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Depth image file '{file_path}' does not exist.")
    image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise ValueError(f"Failed to load depth image from '{file_path}'.")
    return image.astype(np.float32) / scale


class PackedLabelView(Mapping):
    """Lazy npz-like view over one frame row of a packed ``labels.h5``.

    A key's dataset is read only when accessed. String-typed values are decoded back to
    numpy unicode arrays to match what ``np.load`` returns for the source npz.
    """

    def __init__(self, datasets: dict[str, Any], row: int):
        self._datasets = datasets
        self._row = row

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._datasets:
            raise KeyError(key)
        value = np.asarray(self._datasets[key][self._row])
        if value.dtype == object or value.dtype.kind == "S":
            decoded = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in value.ravel()]
            return np.array(decoded).reshape(value.shape)
        return value

    def __contains__(self, key: object) -> bool:
        return key in self._datasets

    def __iter__(self):
        return iter(self._datasets)

    def __len__(self) -> int:
        return len(self._datasets)

    @property
    def files(self) -> list[str]:
        return list(self._datasets)


class SequenceLoader:
    """One HO-Cap sequence: RealSense rig calibration + per-frame pixel/label access.

    ``rs_RTs`` are c2w ``(V, 4, 4)`` in the tag-1 world frame, ``rs_Ks`` pixel-space
    ``(V, 3, 3)``. File handles open lazily so instances are fork-safe.
    """

    def __init__(self, sequence_folder: str, device: str = "cpu"):
        self._data_folder = Path(sequence_folder)
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._models_folder = self._data_folder.parent.parent / "models"
        self._device = device
        self._load_metadata()
        # Per-camera seg.h5 handle cache: serial -> h5py.File | None
        self._seg_cache: dict[str, Any] = {}
        # Packed-store cache: (serial, file name) -> None | {"file", "data", "rows"}
        self._packed_cache: dict[tuple[str, str], Any] = {}

    def _load_metadata(self):
        data = read_data_from_yaml(self._data_folder / "meta.yaml")
        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]
        self._subject_id = data["subject_id"]
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cams = len(self._rs_serials)
        self._hl_serial = data["hololens"]["serial"]
        self._hl_pv_width = data["hololens"]["pv_width"]
        self._hl_pv_height = data["hololens"]["pv_height"]
        self._load_intrinsics()
        self._load_extrinsics(data["extrinsics"])
        self._mano_beta = self._load_mano_beta()

    def _load_intrinsics(self):
        def read_K_from_yaml(serial, cam_type="color"):
            data = read_data_from_yaml(self._calib_folder / "intrinsics" / f"{serial}.yaml")[
                cam_type
            ]
            return np.array(
                [[data["fx"], 0.0, data["ppx"]], [0.0, data["fy"], data["ppy"]], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

        rs_Ks = np.stack([read_K_from_yaml(serial) for serial in self._rs_serials], axis=0)
        rs_Ks_inv = np.stack([np.linalg.inv(K) for K in rs_Ks], axis=0)
        hl_K = read_K_from_yaml(self._hl_serial)
        self._rs_Ks = torch.from_numpy(rs_Ks).to(self._device)
        self._rs_Ks_inv = torch.from_numpy(rs_Ks_inv).to(self._device)
        self._hl_K = torch.from_numpy(hl_K).to(self._device)
        self._hl_K_inv = torch.from_numpy(np.linalg.inv(hl_K)).to(self._device)

    def _load_extrinsics(self, file_name):
        def create_mat(values):
            return np.array(
                [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]], dtype=np.float32
            )

        data = read_data_from_yaml(self._calib_folder / "extrinsics" / f"{file_name}")
        self._rs_master = data["rs_master"]
        extrinsics = data["extrinsics"]
        tag_0 = create_mat(extrinsics["tag_0"])
        tag_1 = create_mat(extrinsics["tag_1"])
        tag_1_inv = np.linalg.inv(tag_1)
        extr2master = np.stack([create_mat(extrinsics[s]) for s in self._rs_serials], axis=0)
        extr2world = np.stack([tag_1_inv @ t for t in extr2master], axis=0)

        dev = self._device
        self._tag_0 = torch.from_numpy(tag_0).to(dev)
        self._tag_0_inv = torch.from_numpy(np.linalg.inv(tag_0)).to(dev)
        self._tag_1 = torch.from_numpy(tag_1).to(dev)
        self._tag_1_inv = torch.from_numpy(tag_1_inv).to(dev)
        self._extr2master = torch.from_numpy(extr2master).to(dev)
        self._extr2master_inv = torch.from_numpy(
            np.stack([np.linalg.inv(t) for t in extr2master], axis=0)
        ).to(dev)
        self._rs_RTs = torch.from_numpy(extr2world).to(dev)
        self._rs_RTs_inv = torch.from_numpy(
            np.stack([np.linalg.inv(t) for t in extr2world], axis=0)
        ).to(dev)

    def _load_mano_beta(self) -> torch.Tensor:
        data = read_data_from_yaml(self._calib_folder / "mano" / f"{self._subject_id}.yaml")
        return torch.tensor(data["betas"], dtype=torch.float32, device=self._device)

    # ------------------------------------------------------------------ #
    # Packed stores
    # ------------------------------------------------------------------ #

    def _packed_store(self, serial: str, name: str, **h5_kwargs) -> Any:
        """Cached store for a packed ``<serial>/<name>``, or None when absent.

        Detected per artifact, so a tree with only some of frames.h5 / depth.h5 /
        labels.h5 falls back to the per-frame files for the rest.
        """
        key = (serial, name)
        if key not in self._packed_cache:
            path = self._data_folder / serial / name
            if not path.is_file():
                self._packed_cache[key] = None
            else:
                handle = _h5py().File(path, "r", **h5_kwargs)
                ids = np.asarray(handle["frame_ids"][:])
                rows: Any
                if ids.size and np.array_equal(ids, np.arange(ids.size, dtype=ids.dtype)):
                    rows = int(ids.size)  # contiguous ids from 0: row == frame_id
                else:
                    rows = {int(f): i for i, f in enumerate(ids)}
                data = {k: handle[k] for k in handle.keys() if k != "frame_ids"}
                self._packed_cache[key] = {"file": handle, "data": data, "rows": rows}
        return self._packed_cache[key]

    def _packed_row(
        self, store: dict[str, Any], serial: str, name: str, frame_id: int, required: bool = True
    ) -> int | None:
        """Map a frame id to its row; a store need not cover every frame of its camera."""
        rows = store["rows"]
        if isinstance(rows, int):
            if 0 <= frame_id < rows:
                return frame_id
        elif frame_id in rows:
            return rows[frame_id]
        if required:
            raise KeyError(f"frame id {frame_id} not in {self._data_folder / serial / name}")
        return None

    def get_rgb_image(self, frame_id: int, serial: str) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB."""
        store = self._packed_store(serial, "frames.h5")
        if store is not None:
            cv2 = _cv2()
            row = self._packed_row(store, serial, "frames.h5", frame_id)
            image = cv2.imdecode(store["data"]["color"][row], cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(
                    f"Failed to decode frame {frame_id} of {self._data_folder / serial / 'frames.h5'}"
                )
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return read_rgb_image(self._data_folder / f"{serial}/color_{frame_id:06d}.jpg")

    def get_depth_image(self, frame_id: int, serial: str) -> np.ndarray:
        """``(H, W)`` float32 metres."""
        store = self._packed_store(serial, "depth.h5")
        if store is not None:
            row = self._packed_row(store, serial, "depth.h5", frame_id)
            return store["data"]["depth"][row].astype(np.float32) / 1000.0
        return read_depth_image(
            self._data_folder / f"{serial}/depth_{frame_id:06d}.png", scale=1000.0
        )

    def get_image_label(self, frame_id: int, serial: str):
        """Label mapping for one frame (``{}`` when the frame is unlabelled)."""
        store = self._packed_store(serial, "labels.h5", rdcc_nbytes=4 * 1024 * 1024)
        if store is not None:
            row = self._packed_row(store, serial, "labels.h5", frame_id, required=False)
            return PackedLabelView(store["data"], row) if row is not None else {}
        label_file = self._data_folder / f"{serial}/label_{frame_id:06d}.npz"
        if not label_file.exists():
            return {}
        return np.load(label_file)

    def get_seg_mask(self, frame_id: int, serial: str) -> dict[str, np.ndarray]:
        """Bool ``(H, W)`` masks keyed ``human`` / ``table`` / ``object`` / ``combined``."""
        masks: dict[str, np.ndarray] = {}
        if serial not in self._seg_cache:
            seg_path = self._data_folder / f"{serial}/seg.h5"
            self._seg_cache[serial] = _h5py().File(seg_path, "r") if seg_path.exists() else None

        handle = self._seg_cache[serial]
        if handle is not None:
            for key in ("human", "table", "object"):
                if key in handle and 0 <= frame_id < handle[key].shape[0]:
                    masks[key] = handle[key][frame_id]

        if masks:
            try:
                from scipy.ndimage import binary_closing
            except ImportError as e:
                raise ImportError("HO-Cap seg masks need scipy; install ontic-data[hocap]") from e
            combined = np.zeros((self._rs_height, self._rs_width), dtype=bool)
            for v in masks.values():
                if v.shape == (self._rs_height, self._rs_width):
                    combined |= v
            masks["combined"] = binary_closing(combined, iterations=4)
        return masks

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    @property
    def object_ids(self) -> list:
        return self._object_ids

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def sequence_name(self) -> str:
        """Recording timestamp, e.g. ``20231027_112303`` — what separates two recordings
        of the same subject and task."""
        return self._data_folder.name

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def rs_width(self) -> int:
        return self._rs_width

    @property
    def rs_height(self) -> int:
        return self._rs_height

    @property
    def rs_serials(self) -> list:
        return self._rs_serials

    @property
    def num_cams(self) -> int:
        return self._num_cams

    @property
    def rs_master(self) -> str:
        return self._rs_master

    @property
    def mano_beta(self) -> torch.Tensor:
        return self._mano_beta

    @property
    def mano_sides(self) -> list:
        return self._mano_sides

    @property
    def rs_Ks(self) -> torch.Tensor:
        return self._rs_Ks

    @property
    def rs_Ks_inv(self) -> torch.Tensor:
        return self._rs_Ks_inv

    @property
    def rs_RTs(self) -> torch.Tensor:
        return self._rs_RTs

    @property
    def rs_RTs_inv(self) -> torch.Tensor:
        return self._rs_RTs_inv

    @property
    def tag_0(self) -> torch.Tensor:
        return self._tag_0

    @property
    def tag_1(self) -> torch.Tensor:
        return self._tag_1

    @property
    def object_textured_mesh_files(self) -> list:
        return [str(self._models_folder / f"{o}/textured_mesh.obj") for o in self._object_ids]

    @property
    def object_cleaned_mesh_files(self) -> list:
        return [str(self._models_folder / f"{o}/cleaned_mesh_10000.obj") for o in self._object_ids]
