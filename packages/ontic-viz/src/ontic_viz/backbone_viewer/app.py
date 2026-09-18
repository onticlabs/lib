"""Viser GUI wiring (:class:`BackboneViewer`) over the headless core.

Layout: dataset -> depth -> tracking; per-camera
"use as input" checkboxes (camera images live on the 3D frustums at the GT poses);
GT-camera conditioning + Run; explicitly applied point sampling (stride, confidence
percentile, voxel / SFC / FPS thinning); live colour, alignment and point size;
occupancy reprojection painted on the frustums; ruler; hands / workspace box;
robot overlay and action points (synthrobot).
"""

from __future__ import annotations

import threading
import zlib
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from ontic_lib.camera import overlay_masks_on_images, points_with_radius, project_occupancy
from ontic_lib.pointops import align_camera_poses_sim3, clamp_scale
from ontic_lib.structures import aabb_mask, nearest_point_to_ray, padded_aabb

from .config import PRESETS_PATH, ViewConfig, dataset_defaults, ensure_default_presets, save_preset
from .data_source import DEFAULT_ROOTS, build_source, dataset_has_gt_depth, dataset_names
from .errors import log_error, status_error
from .render import (
    COLOR_MODES,
    PointFilters,
    build_point_cloud,
    confidence_threshold,
    frustum_params,
    mat_to_wxyz,
    pred_cameras_in_display_frame,
)
from .runner import (
    ALIGN_MODES,
    AlignMode,
    BackboneResult,
    BackboneRunner,
    aligned_depth_and_cameras,
    backbone_accepts_gt_cameras,
    backbone_default_long_side,
    backbone_is_monocular,
    backbone_names,
    backbone_needs_gt_depth,
    metric_model_names,
)

_THUMB_W = 220
_FRUSTUM_SCALE = 0.08

_INPUT_COLOR = (45, 210, 90)
_OFF_COLOR = (110, 110, 110)
_INPUT_LINE = 3.0
_OFF_LINE = 1.5
_PRED_COLOR = (255, 140, 0)
_PRED_LINE = 2.0

_WORKSPACE_MARGIN = 0.15  # metres padded around hand keypoints / camera rig
_HAND_PALETTE = [(255, 220, 50), (50, 220, 100)]
_BOX_COLOR = (40, 220, 220)
_RED = '<span style="color:#e5534b">'

# Oblique unit direction the "frame scene" camera backs off along; the distance
# scales with the rig, never with how far the scene sits from the world origin.
_ORBIT_DIR = np.array([0.62, -0.62, 0.48], dtype=np.float32)
_ORBIT_DIR /= np.linalg.norm(_ORBIT_DIR)
_MIN_ORBIT_RADIUS = 0.25


def _link_color(name: str) -> np.ndarray:
    """A stable, well-spread colour per link name (hashed, no palette to maintain)."""
    h = zlib.crc32(name.encode()) & 0xFFFFFF
    return np.array([(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF], dtype=np.uint8) | 0x40


def _aabb_segments(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The 12 edges of the box ``[lo, hi]`` as ``(12, 2, 3)``."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    edges += [(0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.stack([c[a], c[b]]) for a, b in edges]).astype(np.float32)


def _camera_centres(extrinsics: torch.Tensor) -> np.ndarray:
    """World-space camera centres ``(V, 3)`` from c2w ``(V, 4, 4)``."""
    return extrinsics[:, :3, 3].detach().cpu().numpy().astype(np.float32)


def _scene_bbox(centres: np.ndarray, margin: float = _WORKSPACE_MARGIN):
    """AABB of the camera rig padded by ``margin``: the workspace box for datasets
    without hand keypoints (their scenes can sit metres from the origin)."""
    lo, hi = padded_aabb(torch.as_tensor(centres), margin)
    return lo.numpy(), hi.numpy()


def _orbit_pose(centres: np.ndarray, scale: float = 2.2) -> tuple[np.ndarray, np.ndarray]:
    """``(position, look_at)`` framing a camera rig from ``scale`` x its radius."""
    centres = np.asarray(centres, dtype=np.float32).reshape(-1, 3)
    look = centres.mean(0)
    radius = max(float(np.linalg.norm(centres - look, axis=1).max()), _MIN_ORBIT_RADIUS)
    return look + _ORBIT_DIR * (scale * radius), look


def _to_uint8_hwc(img_chw: torch.Tensor, max_w: int | None = None) -> np.ndarray:
    """``(3, H, W)`` float[0, 1] -> ``(h, w, 3)`` uint8, optionally width-limited."""
    _, h, w = img_chw.shape
    if max_w is not None and w > max_w:
        nh = max(1, int(round(h * max_w / w)))
        img_chw = F.interpolate(
            img_chw[None], size=(nh, max_w), mode="bilinear", align_corners=False
        )[0]
    return (img_chw.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


class BackboneViewer:
    """The viser app. ``backbones`` / ``datasets`` default to the ``ontic_nn`` /
    ``ontic_data`` registries; ``roots`` maps dataset name -> root path."""

    def __init__(
        self,
        server,
        device: str = "cuda",
        roots: dict[str, str] | None = None,
        stage: str = "val",
        presets_path=None,
        *,
        backbones: list[str] | None = None,
        datasets: list[str] | None = None,
        metric_models: list[str] | None = None,
    ):
        self.server = server
        self.device = device
        self.roots = dict(DEFAULT_ROOTS) if roots is None else roots
        self.stage = stage
        self.presets_path = presets_path or PRESETS_PATH
        self.backbones = backbones if backbones is not None else backbone_names()
        self.datasets = datasets if datasets is not None else ["demo", *dataset_names()]
        self.metric_models = (
            metric_models if metric_models is not None else metric_model_names()
        ) or ["da3"]

        self.runner = BackboneRunner(device=device)
        self._backbone_paths = {}
        self._backbone_sizes = {}
        self._backbone_size_owner = self.backbones[0]
        self._sources: dict = {}
        self._source = None
        self._traj_labels: list[str] = []
        self._frame = None
        self._result = None
        self._conditioned = False
        self._run_images: torch.Tensor | None = None
        self._run_ix: list[int] = []
        self._occupancy_on = False
        self._cloud_bounds = None
        self._cloud_pts = None
        self._applied_point_filters = PointFilters(stride=2, max_points=0)
        self._tc_a = self._tc_b = self._ruler_line = self._ruler_label = None
        self._ruler_next = 0
        self._ruler_picking = False
        self._hand_scene: list = []
        self._robot_scene: dict = {}
        self._robot_mdl = None
        self._action_pts_scene: list = []
        self._action_kin = None
        self._workspace_box = None
        self._framed_traj: int | None = None

        self._cam_checks: list = []
        self._cam_scene: list = []
        self._pred_cam_scene: list = []
        self._cam_context = None

        self._busy = False
        self._operation_lock = threading.Lock()
        self._tracking_frame = False
        self._camera_selection = {}
        self._selection_context = None
        self._suspend = False

        self._build_gui()
        self.server.gui.configure_theme(
            dark_mode=True, control_width="large", brand_color=(66, 170, 190)
        )
        self.server.scene.set_background_image(np.full((1, 1, 3), (24, 27, 33), dtype=np.uint8))
        self.server.scene.add_frame("/world", show_axes=True, axes_length=0.2, axes_radius=0.005)
        self.server.on_client_connect(self._frame_client)

    # ------------------------------------------------------------------ GUI
    def _build_gui(self) -> None:
        g = self.server.gui
        from .tracking_panel import TrackingPanel

        g.add_markdown("## Geometry & motion", order=-3)
        self.status = g.add_markdown(
            "Load the demo to explore, or select a dataset to begin.", order=-2
        )
        self.playback_panel = g.add_panel(order=-1)
        self.playback_tab = self.playback_panel.add_tab("Playback")
        self.playback_panel.dock_left()
        self.playback_panel.set_width(320)
        tabs = g.add_tab_group(order=0)
        self.dataset_tab = tabs.add_tab("1. Dataset")
        self.depth_tab = tabs.add_tab("2. Depth")
        self.track_tab = tabs.add_tab("3. Tracking")
        self.display_tab = tabs.add_tab("4. Display")
        with self.dataset_tab:
            g.add_markdown(
                "Select a dataset, trajectory, cameras and clip. Then open **2. Depth**.",
                order=-1,
            )
            with g.add_folder("Dataset", order=0):
                self.dd_dataset = g.add_dropdown(
                    "Dataset", self.datasets, initial_value=self.datasets[0]
                )
                self.btn_loadds = g.add_button("Load dataset", color="blue")
                self.dd_traj = g.add_dropdown(
                    "Trajectory", ["(load a dataset)"], initial_value="(load a dataset)"
                )
                self.sl_time = g.add_slider("Dataset frame", 0, 1, 1, 0)
                g.add_markdown(
                    "Browse any frame here. **Playback** plays a computed clip once depth or tracking is ready."
                )
            self.cam_folder = g.add_folder("Input cameras", order=2)
            with self.cam_folder:
                g.add_markdown(
                    "**Click a camera image or frustum in 3D to select / deselect it.** "
                    "Green = selected; grey = deselected. Clicking pauses playback. "
                    "Deselected cameras' cached points are hidden; depth runs use only selected cameras."
                )
                self.btn_all = g.add_button("Select all")
                self.btn_none = g.add_button("Select none")
            self.btn_all.on_click(lambda _: self._set_all(True))
            self.btn_none.on_click(lambda _: self._set_all(False))

        with self.depth_tab:
            g.add_markdown(
                "Depth places image points in 3D. Choose a depth source for the selected frames; "
                "**Run tracking** prepares it automatically. Running video depth separately is optional.",
                order=-1,
            )
            with g.add_folder("Model", order=1, expand_by_default=False):
                self.dd_backbone = g.add_dropdown(
                    "Backbone", self.backbones, initial_value=self.backbones[0]
                )
                self.depth_model_hint = g.add_markdown("")
                self.cb_condition = g.add_checkbox("Condition model on GT cameras", False)
                self.bb_checkpoint = g.add_text("Backbone checkpoint", "")
                self.bb_download = g.add_checkbox("Download missing backbone weights", False)
                self.sl_lside = g.add_slider(
                    "Model long side (px)",
                    14,
                    2048,
                    2,
                    backbone_default_long_side(self.dd_backbone.value),
                )
                self.btn_default_size = g.add_button("Use model default size")
                g.add_markdown("This input size applies to both single-frame and video depth runs.")
            with g.add_folder("Single-frame preview", order=2, expand_by_default=False):
                self.btn_run = g.add_button("Run depth on current frame")
                g.add_markdown(
                    "Inspect the model on **Dataset frame**. This computes one frame; "
                    "use **Run video depth** for playback."
                )
                with g.add_folder("Alignment", expand_by_default=False):
                    self.dd_align = g.add_dropdown(
                        "Metric alignment", ALIGN_MODES, initial_value=AlignMode.SIM3_POINTS.value
                    )
                    self.dd_metric = g.add_dropdown(
                        "Metric model (method A)",
                        self.metric_models,
                        initial_value=self.metric_models[0],
                    )
                    self.cb_pred_cams = g.add_checkbox("Show predicted cameras (orange)", False)

            self.depth_appearance_folder = g.add_folder(
                "Depth appearance", order=3, expand_by_default=False
            )
            with self.depth_appearance_folder:
                self.dd_color = g.add_dropdown(
                    "Color mode", COLOR_MODES, initial_value=COLOR_MODES[0]
                )
                self.sl_psize = g.add_slider("Point size", 0.001, 0.05, 0.001, 0.006)

            with g.add_folder("Point cloud", order=4, expand_by_default=False):
                self.sl_stride = g.add_slider("Pixel stride", 1, 16, 1, 2)
                self.sl_conf = g.add_slider("Drop lowest-conf %", 0, 95, 1, 0)
                self.sl_voxel = g.add_slider("Voxel size (m, 0=off)", 0.0, 0.1, 0.002, 0.0)
                self.sl_sfc = g.add_slider("SFC stride (1=off)", 1, 64, 1, 1)
                self.sl_fps = g.add_slider("Max sampled points", 0, 50000, 1000, 0)
                self.btn_fps = g.add_button("Apply point sampling")
                g.add_markdown(
                    "Changes to these controls take effect only when you click **Apply point sampling**. "
                    "Farthest-point sampling spreads points across the surface; 0 keeps all points. "
                    "These filters also determine where tracks can start."
                )

        self.tracking = TrackingPanel(self, self.track_tab, self.playback_tab)
        self.tracking.build_appearance()
        with self.display_tab:
            self._configs = self._all_configs()
            with g.add_folder("Presets", order=6, expand_by_default=False):
                initial = f"{self.datasets[0]}_default"
                self.dd_preset = g.add_dropdown(
                    "Config",
                    list(self._configs),
                    initial_value=initial
                    if initial in self._configs
                    else next(iter(self._configs)),
                )
                self.btn_apply_preset = g.add_button("Apply preset")
                self.tb_preset = g.add_text("Save as", "")
                self.btn_save_preset = g.add_button("Save preset")

            with g.add_folder("Occupancy (reproject cloud)", order=4, expand_by_default=False):
                self.btn_occ = g.add_button("Project -> occupancy")
                self.sl_splat = g.add_slider("Splat scale", 0.1, 10.0, 0.1, 1.0)
                self.sl_maxr = g.add_slider("Max radius (px)", 1, 30, 1, 6)

            with g.add_folder("Hands & workspace", order=2.5, expand_by_default=False):
                self.cb_crop = g.add_checkbox("Crop to workspace", False)
                self.cb_hands = g.add_checkbox("Show hand skeleton", False)
                self.cb_robot = g.add_checkbox("Show robot", False)
                self.md_robot = g.add_markdown("", visible=False)
                self.cb_action_pts = g.add_checkbox("Show action points (synthrobot)", False)
                self.dd_action_kind = g.add_dropdown(
                    "Action point kind", ("skeleton", "surface"), initial_value="surface"
                )
                self.sl_action_n = g.add_slider("Action points per link", 1, 512, 1, 128)
                self.cb_action_normals = g.add_checkbox("Show surface normals", False)
                self.vec_wmin = g.add_vector3(
                    "Workspace min", (-0.5, -0.5, -0.5), min=(-5, -5, -5), max=(5, 5, 5), step=0.01
                )
                self.vec_wmax = g.add_vector3(
                    "Workspace max", (0.5, 0.5, 0.5), min=(-5, -5, -5), max=(5, 5, 5), step=0.01
                )
                self.btn_winit = g.add_button("Init box from hands/cameras")
                self.btn_frame = g.add_button("Frame scene")

            with g.add_folder("Measure", order=4.5, expand_by_default=False):
                self.cb_ruler = g.add_checkbox(
                    "Ruler (click 2 cloud points; drag to adjust)", False
                )

        self.btn_loadds.on_click(lambda _: self._guarded(self._load_dataset))
        self.dd_traj.on_update(self._on_traj_or_time)
        self.sl_time.on_update(self._on_traj_or_time)
        self.btn_run.on_click(lambda _: self._guarded(self._run))
        self.dd_backbone.on_update(lambda _: self._on_backbone_change())
        self.bb_checkpoint.on_update(lambda _: self._remember_backbone_checkpoint())
        self.dd_metric.on_update(lambda _: self._on_metric_model_change())
        self.cb_pred_cams.on_update(lambda _: self._guarded(self._render_pred_cameras))
        # Sampling sliders are drafts; only the Apply button commits them.
        for h in (
            self.dd_color,
            self.dd_align,
            self.sl_psize,
        ):
            h.on_update(lambda _: self._cheap_update())
        self.sl_lside.on_update(lambda _: self._on_long_side())
        self.btn_default_size.on_click(
            lambda _: setattr(
                self.sl_lside, "value", backbone_default_long_side(self.dd_backbone.value)
            )
        )
        self.btn_fps.on_click(lambda _: self._guarded(self._apply_point_sampling))
        self.btn_occ.on_click(lambda _: self._toggle_occupancy())
        for h in (self.sl_splat, self.sl_maxr):
            h.on_update(lambda _: self._on_occ_param())
        self.cb_ruler.on_update(lambda _: self._toggle_ruler())
        self.cb_hands.on_update(lambda _: self._draw_hands())
        self.cb_robot.on_update(lambda _: self._guarded(self._draw_robot))
        for h in (
            self.cb_action_pts,
            self.dd_action_kind,
            self.sl_action_n,
            self.cb_action_normals,
        ):
            h.on_update(lambda _: self._guarded(self._draw_action_points))
        self.btn_winit.on_click(lambda _: self._init_workspace_from_hands())
        self.btn_frame.on_click(lambda _: self._frame_scene())
        for h in (self.vec_wmin, self.vec_wmax):
            h.on_update(lambda _: self._on_workspace_change())
        self.cb_crop.on_update(lambda _: self._on_workspace_change())
        self.dd_preset.on_update(lambda _: self._on_preset_selected())
        self.btn_apply_preset.on_click(lambda _: self._on_preset_selected())
        self.btn_save_preset.on_click(lambda _: self._save_preset())

        self.cb_condition.disabled = not backbone_accepts_gt_cameras(self.dd_backbone.value)
        self._update_depth_model_hint()

    def _set_status(self, msg: str) -> None:
        self.status.content = msg

    def _guarded(self, fn) -> None:
        if not self._operation_lock.acquire(blocking=False):
            return
        self._busy = True
        try:
            fn()
        except Exception as e:  # surface errors in the UI rather than dying
            summary = log_error(getattr(fn, "__name__", repr(fn)), e)
            self._set_status(status_error(summary))
        finally:
            self._busy = False
            self._operation_lock.release()

    # -------------------------------------------------------------- dataset
    def _load_dataset(self) -> None:
        name = self.dd_dataset.value
        self._set_status(f"Building **{name}**... (first build can take a while)")
        src = self._sources.get(name)
        if src is None:
            src = build_source(name, root=self.roots.get(name), stage=self.stage)
            self._sources[name] = src
        if not src.list_trajectories():
            raise ValueError(f"{name} contains no trajectories for this split")
        self.tracking.clear(clear_cache=True)
        self._source = src
        self._camera_selection.clear()
        self._cam_context = None
        self._selection_context = None
        self._framed_traj = None

        raw = src.list_trajectories()
        self._traj_labels = [f"{i:04d} - {t}" for i, t in enumerate(raw)]
        self._suspend = True
        self.dd_traj.options = self._traj_labels
        self.dd_traj.value = self._traj_labels[0]
        self.sl_time.value = 0
        self._suspend = False

        default_key = f"{name}_default"
        if default_key in self._configs:
            self._suspend = True
            self.dd_preset.value = default_key
            self._suspend = False
            self._apply_config(self._configs[default_key])

        self._load_frame()
        self._set_status(f"**{name}**: {len(raw)} trajectories loaded.")

    def _traj_index(self) -> int:
        return self._traj_labels.index(self.dd_traj.value)

    # --------------------------------------------------------------- presets
    def _all_configs(self) -> dict[str, ViewConfig]:
        """All presets (seeded ``<dataset>_default`` entries + user-saved ones)."""
        try:
            defaults = dataset_defaults()
        except ImportError:
            defaults = {name: ViewConfig() for name in self.datasets}
        return ensure_default_presets(self.presets_path, defaults=defaults)

    def _current_config(self) -> ViewConfig:
        return ViewConfig(
            stride=int(self.sl_stride.value),
            drop_conf_pct=float(self.sl_conf.value),
            voxel_size=float(self.sl_voxel.value),
            sfc_stride=int(self.sl_sfc.value),
            fps_max_points=int(self.sl_fps.value),
            point_size=float(self.sl_psize.value),
            color_mode=self.dd_color.value,
            align_mode=str(self.dd_align.value),
            crop_enabled=bool(self.cb_crop.value),
            ws_min=tuple(float(x) for x in self.vec_wmin.value),
            ws_max=tuple(float(x) for x in self.vec_wmax.value),
        )

    def _apply_config(self, cfg: ViewConfig) -> None:
        self._suspend = True
        self.sl_stride.value = int(cfg.stride)
        self.sl_conf.value = int(cfg.drop_conf_pct)
        self.sl_voxel.value = float(cfg.voxel_size)
        self.sl_sfc.value = int(cfg.sfc_stride)
        self.sl_fps.value = int(cfg.fps_max_points)
        self.sl_psize.value = float(cfg.point_size)
        self.dd_color.value = cfg.color_mode
        self.dd_align.value = (
            cfg.align_mode if cfg.align_mode in ALIGN_MODES else AlignMode.SIM3_POINTS.value
        )
        self.cb_crop.value = bool(cfg.crop_enabled)
        self.vec_wmin.value = tuple(cfg.ws_min)
        self.vec_wmax.value = tuple(cfg.ws_max)
        self._suspend = False
        self._draw_workspace_box()
        self._cheap_update()

    def _on_preset_selected(self) -> None:
        if self._suspend:
            return
        cfg = self._configs.get(self.dd_preset.value)
        if cfg is not None:
            self._apply_config(cfg)

    def _save_preset(self) -> None:
        name = self.tb_preset.value.strip()
        if not name:
            self._set_status("Type a name in **Save as** first, then Save preset.")
            return
        save_preset(name, self._current_config(), self.presets_path)
        self._configs = self._all_configs()
        self.dd_preset.options = list(self._configs)
        self._suspend = True
        self.dd_preset.value = name
        self._suspend = False
        self._set_status(f"Saved preset **{name}**.")

    def _on_traj_or_time(self, event=None) -> None:
        if self._suspend or self._source is None:
            return
        # Viser dispatches server-side updates asynchronously, often after
        # _suspend is reset. Only user browsing should pause the play loop.
        if event is None or event.client is not None:
            self.tracking.pause()
        self._guarded(self._load_frame)

    def _load_frame(self) -> None:
        ti = self._traj_index()
        context = (id(self._source), ti)
        if context != self._selection_context:
            self._camera_selection.clear()
            self._selection_context = context
        self.tracking.set_context(self._source, ti)
        nt = self._source.num_timesteps(ti)
        self._suspend = True
        self.sl_time.max = max(1, nt - 1)
        self._suspend = False
        t = min(int(self.sl_time.value), nt - 1)

        if self.tracking.show_frame(t):
            return
        self._tracking_frame = False
        self.tracking.hide()
        self.tracking.render()
        want_depth = backbone_needs_gt_depth(self.dd_backbone.value)
        frame = self._source.get_frame(ti, t, with_depth=want_depth)
        self._frame = frame
        self._result = None
        self._rebuild_cameras(frame)
        self._clear_cloud()
        # Follow the hands only while not cropping: a committed box survives frame changes.
        self._maybe_follow_workspace()
        self._maybe_frame_scene(ti)
        self._draw_hands()
        self._draw_robot()
        self._draw_action_points()
        self._draw_workspace_box()
        if self._gt_depth_gate():
            return
        self._set_status(
            f"{self.dd_traj.value} - t={t} - {frame.images.shape[0]} cameras. "
            "Click cameras in 3D (green = selected), then **Run depth on current frame**."
        )

    # -------------------------------------------------------------- cameras
    def _rebuild_cameras(self, frame) -> None:
        context = (self._selection_context, tuple(frame.cam_names))
        rebuild = context != self._cam_context
        for h in self._pred_cam_scene:
            h.remove()
        self._pred_cam_scene = []
        self._occupancy_on = False

        if rebuild:
            for h in self._cam_checks + self._cam_scene:
                h.remove()
            self._cam_checks, self._cam_scene = [], []
            self._cam_context = context
            with self.cam_folder:
                for i, name in enumerate(frame.cam_names):
                    cb = self.server.gui.add_checkbox(
                        f"Camera {i} · {name}", self._camera_selection.get(name, True)
                    )
                    cb.on_update(lambda _: self._recolor_cameras())
                    self._cam_checks.append(cb)

        for i, name in enumerate(frame.cam_names):
            ext = frame.extrinsics[i].numpy()
            fov, aspect = frustum_params(frame.intrinsics[i])
            image = _to_uint8_hwc(frame.images[i], max_w=_THUMB_W)
            if rebuild:
                fr = self.server.scene.add_camera_frustum(
                    f"/cams/{i}_{_safe(name)}",
                    fov=fov,
                    aspect=aspect,
                    scale=_FRUSTUM_SCALE,
                    image=image,
                    wxyz=mat_to_wxyz(ext[:3, :3]),
                    position=ext[:3, 3],
                    variant="filled",
                )
                fr.on_click(lambda event, camera=name: self._on_camera_click(event, camera))
                self._cam_scene.append(fr)
            else:
                # Keep the click target mounted while playback updates its image and pose.
                fr = self._cam_scene[i]
                fr.fov, fr.aspect = fov, aspect
                fr.image = image
                fr.wxyz = mat_to_wxyz(ext[:3, :3])
                fr.position = ext[:3, 3]
        self._recolor_cameras(redraw=False)

    def _on_camera_click(self, event, name: str) -> None:
        if self._busy and not self.tracking._playing.is_set():
            return
        self.tracking.pause()
        # Let an in-flight playback frame finish before changing its camera selection.
        with self._operation_lock:
            if event.target not in self._cam_scene + self._pred_cam_scene:
                return
            if self._frame is None or name not in self._frame.cam_names:
                return
            cb = self._cam_checks[self._frame.cam_names.index(name)]
            if cb.disabled:
                return
            cb.value = not cb.value
            self._recolor_cameras(redraw=False)
        self._cheap_update()

    def _set_all(self, value: bool) -> None:
        if self._busy:
            return
        for cb in self._cam_checks:
            cb.value = value
        self._recolor_cameras()

    def _toggle_cam(self, idx: int) -> None:
        if self._busy:
            return
        if 0 <= idx < len(self._cam_checks):
            self._cam_checks[idx].value = not self._cam_checks[idx].value
            self._recolor_cameras()

    def _recolor_cameras(self, *, redraw: bool = True) -> None:
        if self._frame is not None:
            self._camera_selection.update(
                {name: bool(cb.value) for name, cb in zip(self._frame.cam_names, self._cam_checks)}
            )
        self.tracking.sync_cameras()
        for cb, fr in zip(self._cam_checks, self._cam_scene):
            on = bool(cb.value)
            fr.color = _INPUT_COLOR if on else _OFF_COLOR
            fr.line_width = _INPUT_LINE if on else _OFF_LINE
        if redraw:
            self._cheap_update()

    def _render_pred_cameras(self, overlays=None) -> None:
        """Draw the run cameras' predicted poses as orange frustums in the current
        display frame. ``overlays (V, 3, H, W)`` paints images on them (and forces
        drawing regardless of the toggle)."""
        for h in self._pred_cam_scene:
            h.remove()
        self._pred_cam_scene = []
        if (
            (not self.cb_pred_cams.value and overlays is None)
            or self._result is None
            or self._frame is None
            or not self._run_ix
        ):
            return
        out = pred_cameras_in_display_frame(self._result, str(self.dd_align.value))
        if out is None:
            return
        ext, intr = out
        ext_np = ext.detach().cpu().numpy()
        selected = set(self._selected_ix())
        for k, cam_i in enumerate(self._run_ix):
            if cam_i not in selected:
                continue
            fov, aspect = frustum_params(intr[k])
            fr = self.server.scene.add_camera_frustum(
                f"/pred_cams/{k}_{_safe(self._frame.cam_names[cam_i])}",
                fov=fov,
                aspect=aspect,
                scale=_FRUSTUM_SCALE,
                color=_PRED_COLOR,
                line_width=_PRED_LINE,
                image=_to_uint8_hwc(overlays[k], max_w=_THUMB_W) if overlays is not None else None,
                wxyz=mat_to_wxyz(ext_np[k][:3, :3]),
                position=ext_np[k][:3, 3],
                variant="filled",
            )
            name = self._frame.cam_names[cam_i]
            fr.on_click(lambda event, camera=name: self._on_camera_click(event, camera))
            self._pred_cam_scene.append(fr)

    def _selected_ix(self) -> list[int]:
        return [i for i, cb in enumerate(self._cam_checks) if cb.value]

    def _visible_result(self) -> BackboneResult | None:
        """Mask disabled views without changing cached depth or camera alignment."""
        result = self._result
        if result is None:
            return None
        selected = set(self._selected_ix())
        keep = torch.tensor(
            [i in selected for i in self._run_ix], device=result.depth.device, dtype=torch.bool
        )
        if bool(keep.all()):
            return result
        return replace(
            result,
            depth=result.depth.masked_fill(~keep[:, None, None], 0),
        )

    def _remember_backbone_checkpoint(self) -> None:
        self._backbone_paths[self.dd_backbone.value] = self.bb_checkpoint.value.strip()

    def _configure_backbone(self) -> None:
        self.runner.configure(
            self.dd_backbone.value,
            checkpoint_path=self.bb_checkpoint.value.strip(),
            allow_download=self.bb_download.value,
        )

    def _on_backbone_change(self) -> None:
        name = self.dd_backbone.value
        if name != self._backbone_size_owner:
            self._backbone_sizes[self._backbone_size_owner] = int(self.sl_lside.value)
            self._backbone_size_owner = name
            self.sl_lside.value = self._backbone_sizes.get(name, backbone_default_long_side(name))
        self._update_depth_model_hint()
        self.tracking.update_depth_summary()
        self.bb_checkpoint.value = self._backbone_paths.get(self.dd_backbone.value, "")
        accepts = backbone_accepts_gt_cameras(self.dd_backbone.value)
        self.cb_condition.disabled = not accepts
        if not accepts and self.cb_condition.value:
            self.cb_condition.value = False
        if self._gt_depth_gate():
            return
        if self._suspend:
            return
        note = (
            "conditions on GT cameras when enabled."
            if accepts
            else "does not accept GT-camera conditioning."
        )
        if backbone_is_monocular(self.dd_backbone.value):
            note += (
                " Monocular: it predicts no camera poses, so `sim3_points` / `prescale_gt`"
                " fall back to `none`, which lifts with the GT cameras."
            )
        self._set_status(f"**{self.dd_backbone.value}** " + note)

    def _update_depth_model_hint(self) -> None:
        name = self.dd_backbone.value
        if name == "gtdepth":
            text = "Recorded-depth backbone."
        elif backbone_is_monocular(name):
            text = "**Single-view depth** · processes each enabled camera independently."
        else:
            text = "**Multi-view depth** · uses the enabled input cameras."
        self.depth_model_hint.content = (
            text + f" Default input: **{backbone_default_long_side(name)} px on the long side**."
        )

    # ------------------------------------------------------- GT-depth gating
    def _gt_depth_available(self) -> bool:
        """Whether GT depth can reach the backbone; re-fetches the frame with depth
        once when it was loaded without asking for it."""
        if self._frame is not None and self._frame.depth is None and self._source is not None:
            if dataset_has_gt_depth(self.dd_dataset.value):
                try:
                    self._frame = self._source.get_frame(
                        self._traj_index(), int(self.sl_time.value), with_depth=True
                    )
                except Exception:
                    return False
        if self._frame is not None:
            return self._frame.depth is not None
        return dataset_has_gt_depth(self.dd_dataset.value)

    def _apply_gt_depth_gating(self, available: bool) -> bool:
        """Disable Run + show a red notice when GT depth is required but absent."""
        name = self.dd_backbone.value
        if not backbone_needs_gt_depth(name) or available:
            self.btn_run.disabled = False
            return False
        self.btn_run.disabled = True
        ds = self.dd_dataset.value
        if dataset_has_gt_depth(ds):
            why = (
                f"**{ds}** has depth, but not for this recording/scene - its depth stream "
                "declares a codec the loader cannot honour, so it is refused rather than "
                "returned wrong."
            )
            how = "Try another trajectory, or a depth-predicting backbone."
        else:
            why = f"**{ds}** provides no ground-truth depth at all."
            with_depth = [d for d in self.datasets if dataset_has_gt_depth(d)]
            how = f"Pick {' / '.join(with_depth) or 'a dataset with depth'}, or a depth-predicting backbone."
        self._set_status(f"{_RED}**{name}** needs ground-truth depth. {why}</span> {how}")
        return True

    def _gt_depth_gate(self) -> bool:
        if not backbone_needs_gt_depth(self.dd_backbone.value):
            self.btn_run.disabled = False
            return False
        return self._apply_gt_depth_gating(self._gt_depth_available())

    def _on_metric_model_change(self) -> None:
        if self._suspend:
            return
        if self._result is not None:
            self._result.metric_scale = None
        self._cheap_update()

    def _on_long_side(self) -> None:
        if self._suspend:
            return
        v = int(self.sl_lside.value)
        self.runner.set_long_side(v if v > 0 else None)
        self._set_status(
            f"Input long side -> **{v if v > 0 else 'backbone default'}**. "
            "Applies to the next depth run."
        )

    # ----------------------------------------------------------- inference
    def _run(self) -> None:
        if self._frame is None:
            self._set_status("Load a dataset + frame first.")
            return
        ix = self._selected_ix()
        if not ix:
            self._set_status("Select at least one input camera.")
            return

        self.tracking.clear()
        # Playback frames may be downsampled; inference and its reusable result
        # must use the same original image pixels as clip preparation.
        if self._source is not None:
            self._frame = self._source.get_frame(
                self._traj_index(), int(self.sl_time.value), with_depth=True
            )
        name = self.dd_backbone.value
        self._set_status(f"Loading **{name}** + running on {len(ix)} view(s)...")
        self._configure_backbone()
        self.runner.set_long_side(int(self.sl_lside.value) or None)
        self.runner.load(name)
        res = self.runner.run(
            self._frame,
            ix,
            condition_on_gt_cameras=bool(self.cb_condition.value),
            depth=self._frame.depth,
        )

        if self._occupancy_on:
            self._restore_frustum_images()
            self._occupancy_on = False
        self._result = res
        self._conditioned = res.conditioned
        self._run_ix = list(ix)
        self._run_images = self._frame.images[torch.as_tensor(ix)]
        self._render_cloud()
        from .tracking import PREVIEW_ALIGNMENT, SENSOR

        if self._source is not None:
            key = self.tracking.runner.backbone_frame_key(
                self._source,
                self._traj_index(),
                int(self.sl_time.value),
                tuple(self._frame.cam_names[i] for i in ix),
                self.runner,
                name,
                int(self.sl_lside.value) or None,
                bool(self.cb_condition.value),
            )
            self.tracking.runner.backbone_frame = (key, res)
            # An explicit rerun supersedes any older clip with the same settings.
            self.tracking.runner.cached_clip = None
        self.tracking.geometry.value = SENSOR if name == "gtdepth" else PREVIEW_ALIGNMENT
        self.tracking.update_depth_summary()

    def _conf_thresh(self) -> float:
        conf = self._result.conf if self._result is not None else None
        return confidence_threshold(conf, self._applied_point_filters.drop_conf_pct)

    def _point_filters(self) -> PointFilters:
        return replace(
            self._applied_point_filters,
            crop=(tuple(self.vec_wmin.value), tuple(self.vec_wmax.value))
            if self.cb_crop.value
            else None,
        )

    def _apply_point_sampling(self) -> None:
        previous = self._applied_point_filters
        self._applied_point_filters = PointFilters(
            stride=int(self.sl_stride.value),
            drop_conf_pct=float(self.sl_conf.value),
            voxel_size=float(self.sl_voxel.value),
            sfc_stride=int(self.sl_sfc.value),
            max_points=int(self.sl_fps.value),
        )
        # Metric scale uses confident pixels; invalidate only on an applied change.
        if self._result is not None and (
            previous.drop_conf_pct != self._applied_point_filters.drop_conf_pct
        ):
            self._result.metric_scale = None
        self._render_cloud()
        if self._tracking_frame:
            self.tracking.render()

    def _umeyama_scale(self) -> float | None:
        """The clamped pred->GT Sim(3) scale (for the status line)."""
        res = self._result
        if res is None or res.pred_extrinsics is None or res.gt_extrinsics is None:
            return None
        _, _, s = align_camera_poses_sim3(
            res.pred_extrinsics.unsqueeze(0), res.gt_extrinsics.unsqueeze(0)
        )
        return float(clamp_scale(s)[0])

    def _render_cloud(self) -> None:
        """Re-render cached depth with the last applied sampling settings."""
        if self._result is None or self._run_images is None:
            return
        filters = self._point_filters()
        max_points = filters.max_points
        align_mode = AlignMode.NONE.value if self._tracking_frame else str(self.dd_align.value)
        if align_mode == AlignMode.METRIC_MONO.value and self._result.metric_scale is None:
            try:
                self._set_status(
                    f"Computing metric scale via **{self.dd_metric.value}**... "
                    "(loads the metric model)"
                )
                self._result.metric_scale = self.runner.compute_metric_scale(
                    self._result,
                    self._run_images,
                    self.dd_metric.value,
                    conf_thresh=self._conf_thresh(),
                )
            except Exception as e:
                self._clear_cloud()
                summary = log_error(f"metric_mono ({self.dd_metric.value})", e)
                self._set_status(
                    "Metric alignment (`metric_mono`) failed.\n\n" + status_error(summary)
                )
                return
        voxel = filters.voxel_size
        cloud = build_point_cloud(
            self._visible_result(),
            self._run_images,
            color_mode=self.dd_color.value,
            align_mode=align_mode,
            **filters.cloud_options(self._result.conf),
            return_source_indices=self._tracking_frame,
        )
        pts, cols = cloud[:2]
        self.server.scene.add_point_cloud(
            "/cloud",
            points=pts,
            colors=cols,
            point_size=float(self.sl_psize.value),
            visible=self.tracking.background.value,
        )
        self._cloud_pts = pts
        if pts.shape[0] > 0:
            self._cloud_bounds = (pts.min(0), pts.max(0))
        else:
            self._cloud_bounds = None
        sfc = filters.sfc_stride
        hd, wd = self._result.depth.shape[-2:]
        align_desc = align_mode
        if align_mode != AlignMode.NONE.value:
            s_um = self._umeyama_scale()
            if align_mode == AlignMode.METRIC_MONO.value and self._result.metric_scale is not None:
                align_desc = (
                    f"{align_mode}:{self.dd_metric.value} (s={self._result.metric_scale:.3f}"
                )
                align_desc += f" vs sim3 {s_um:.3f})" if s_um is not None else ")"
            elif s_um is not None:
                align_desc = f"{align_mode} (sim3 s={s_um:.3f})"
        cached = set(self._run_ix)
        selected = set(self._selected_ix())
        self._set_status(
            f"**{self.dd_backbone.value}** - depth {wd}x{hd} - {pts.shape[0]:,} points "
            f"- {len(cached & selected)}/{len(cached)} cameras "
            f"(stride {filters.stride}, drop {int(filters.drop_conf_pct)}% conf"
            f"{f', voxel {voxel:.3f}m' if voxel > 0 else ''}"
            f"{f', sfc/{sfc}' if sfc > 1 else ''}"
            f"{f', fps {max_points}' if max_points > 0 else ''}, "
            f"{self.dd_color.value}, align={align_desc}{', GT-cond' if self._conditioned else ''})."
            + (" Run depth again to include newly enabled cameras." if selected - cached else "")
        )
        if self._occupancy_on:
            self._show_occupancy()
        else:
            self._render_pred_cameras()

    def _cheap_update(self) -> None:
        if self._suspend or self._result is None:
            return

        def redraw():
            self._render_cloud()
            if self._tracking_frame:
                self.tracking.render()

        self._guarded(redraw)

    def _clear_cloud(self) -> None:
        self._cloud_pts = None
        self._cloud_bounds = None
        self.server.scene.add_point_cloud(
            "/cloud", points=np.zeros((0, 3), np.float32), colors=np.zeros((0, 3), np.uint8)
        )
        for h in self._pred_cam_scene:
            h.remove()
        self._pred_cam_scene = []

    # ----------------------------------------------------------- occupancy
    def _toggle_occupancy(self) -> None:
        if self._result is None or self._frame is None:
            self._set_status("Run inference first, then project occupancy.")
            return
        if self._occupancy_on:
            self._occupancy_on = False
            self._restore_frustum_images()
            self._render_pred_cameras()
            self._set_status("Occupancy overlay off.")
            return
        self._guarded(self._show_occupancy)

    def _on_occ_param(self) -> None:
        if self._occupancy_on:
            self._guarded(self._show_occupancy)

    def _show_occupancy(self) -> None:
        """Reproject the reconstruction into the cameras and paint coverage on the
        frustums: the predicted reconstruction into its own predicted views (orange
        frustums; skipped for monocular backbones), and — when aligned to GT — the
        aligned cloud into all GT cameras (GT frustums)."""
        res, mode = (
            self._visible_result(),
            (AlignMode.NONE.value if self._tracking_frame else str(self.dd_align.value)),
        )
        monocular = res.pred_extrinsics is None
        if monocular and mode in (AlignMode.SIM3_POINTS.value, AlignMode.PRESCALE_GT.value):
            mode = AlignMode.NONE.value
        stride, ct = self._point_filters().stride, self._conf_thresh()
        splat, maxr = float(self.sl_splat.value), int(self.sl_maxr.value)
        crop_lo, crop_hi = self._crop_box()

        def _masks(pts, r_world, ext, intr, hw):
            return project_occupancy(
                pts, r_world, ext, intr, hw, splat_scale=splat, max_radius=maxr
            )

        def _inside_box(pts):
            if crop_lo is None:
                return None
            lo = torch.as_tensor(crop_lo, dtype=pts.dtype)
            hi = torch.as_tensor(crop_hi, dtype=pts.dtype)
            return aabb_mask(pts, lo, hi).squeeze(-1)

        pts_p = r_p = None
        if not monocular:
            pred_depth, pred_cams, pred_intr = (res.depth, res.pred_extrinsics, res.pred_intrinsics)
            pts_p, r_p = points_with_radius(
                pred_depth,
                pred_cams,
                pred_intr,
                res.conf,
                stride=stride,
                confidence_threshold=ct,
            )

        cov_g = None
        gt_ready = (monocular or mode != AlignMode.NONE.value) and not (
            mode == AlignMode.METRIC_MONO.value and res.metric_scale is None
        )
        if gt_ready:
            depth, cams, intr = aligned_depth_and_cameras(res, mode)
            pts_g, r_g = points_with_radius(
                depth, cams, intr, res.conf, stride=stride, confidence_threshold=ct
            )
            # Crop in the GT frame; the same per-pixel mask applies 1:1 to the pred points.
            keep = _inside_box(pts_g)
            if keep is not None:
                pts_g, r_g = pts_g[keep], r_g[keep]
                if not monocular:
                    pts_p, r_p = pts_p[keep], r_p[keep]
            all_imgs = self._frame.images
            masks_g = _masks(
                pts_g, r_g, self._frame.extrinsics, self._frame.intrinsics, all_imgs.shape[-2:]
            )
            overlays_g = overlay_masks_on_images(all_imgs, masks_g)
            for i, fr in enumerate(self._cam_scene):
                fr.image = _to_uint8_hwc(overlays_g[i], max_w=_THUMB_W)
            cov_g = float(masks_g.float().mean()) * 100
        else:
            keep = None if monocular else _inside_box(pts_p)
            if keep is not None:
                pts_p, r_p = pts_p[keep], r_p[keep]
            self._restore_frustum_images()

        cov_p = None
        if monocular:
            self._render_pred_cameras()  # clears stale orange frustums from an earlier result
        else:
            run_imgs = self._frame.images[torch.as_tensor(self._run_ix)]
            masks_p = _masks(pts_p, r_p, pred_cams, pred_intr, run_imgs.shape[-2:])
            self._render_pred_cameras(overlay_masks_on_images(run_imgs, masks_p))
            cov_p = float(masks_p.float().mean()) * 100

        self._occupancy_on = True
        gt_txt = f"GT {cov_g:.0f}% (all cams)" if cov_g is not None else "GT skipped (align to GT)"
        pred_txt = (
            "pred N/A (monocular - no predicted poses)"
            if cov_p is None
            else f"pred {cov_p:.0f}% ({len(self._run_ix)} input cams)"
        )
        self._set_status(
            f"Occupancy -> {pred_txt} - {gt_txt} (splat x{splat:.1f}). Press again to clear."
        )

    def _restore_frustum_images(self) -> None:
        if self._frame is None:
            return
        for i, fr in enumerate(self._cam_scene):
            fr.image = _to_uint8_hwc(self._frame.images[i], max_w=_THUMB_W)

    # ------------------------------------------------------------- ruler
    def _toggle_ruler(self) -> None:
        if self.cb_ruler.value:
            self._show_ruler()
        else:
            self._remove_ruler()

    def _default_ruler_endpoints(self):
        if self._cloud_bounds is not None:
            lo, hi = self._cloud_bounds
            center = (lo + hi) / 2.0
            ext = hi - lo
            return (center - 0.25 * ext).astype(np.float32), (center + 0.25 * ext).astype(
                np.float32
            )
        return np.array([-0.1, 0, 0], np.float32), np.array([0.1, 0, 0], np.float32)

    def _show_ruler(self) -> None:
        if self._tc_a is not None:
            return
        a, b = self._default_ruler_endpoints()
        gscale = 0.1
        if self._cloud_bounds is not None:
            lo, hi = self._cloud_bounds
            gscale = max(0.02, 0.08 * float((hi - lo).max()))
        opts = dict(scale=gscale, disable_rotations=True, disable_sliders=True, depth_test=False)
        self._tc_a = self.server.scene.add_transform_controls("/ruler/a", position=a, **opts)
        self._tc_b = self.server.scene.add_transform_controls("/ruler/b", position=b, **opts)
        self._tc_a.on_update(lambda _: self._update_ruler())
        self._tc_b.on_update(lambda _: self._update_ruler())
        self.server.scene.on_click()(lambda ev: self._on_ruler_click(ev))
        self._ruler_picking = True
        self._ruler_next = 0
        self._update_ruler()

    def _on_ruler_click(self, event) -> None:
        if self._cloud_pts is None or self._tc_a is None or event.ray_origin is None:
            return
        picked = nearest_point_to_ray(
            torch.from_numpy(self._cloud_pts), event.ray_origin, event.ray_direction
        )
        if picked is None:
            return
        tc = self._tc_a if self._ruler_next == 0 else self._tc_b
        tc.position = picked.numpy().astype(np.float32)
        self._ruler_next ^= 1
        self._update_ruler()

    def _update_ruler(self) -> None:
        if self._tc_a is None or self._tc_b is None:
            return
        a = np.asarray(self._tc_a.position, dtype=np.float32)
        b = np.asarray(self._tc_b.position, dtype=np.float32)
        d = float(np.linalg.norm(b - a))
        self._ruler_line = self.server.scene.add_line_segments(
            "/ruler/line",
            points=np.array([[a, b]], dtype=np.float32),
            colors=np.array([[[255, 220, 40], [255, 220, 40]]], dtype=np.uint8),
            line_width=3.0,
        )
        self._ruler_label = self.server.scene.add_label(
            "/ruler/dist", text=f"{d:.4f} m", position=(a + b) / 2.0
        )
        self._set_status(
            f"Ruler: **{d:.4f} m**  (A {a.round(3).tolist()} -> B {b.round(3).tolist()})"
        )

    def _remove_ruler(self) -> None:
        if self._ruler_picking:
            try:
                self.server.scene.remove_click_callback()
            except Exception:
                pass
            self._ruler_picking = False
        for h in (self._tc_a, self._tc_b, self._ruler_line, self._ruler_label):
            if h is not None:
                h.remove()
        self._tc_a = self._tc_b = self._ruler_line = self._ruler_label = None

    # -------------------------------------------------------- hands & workspace
    def _draw_hands(self) -> None:
        for h in self._hand_scene:
            h.remove()
        self._hand_scene = []
        if not self.cb_hands.value or self._frame is None or self._frame.hands is None:
            return
        from ontic_data.hand import HAND_BONES

        hands = self._frame.hands  # (H, 21, 3) world
        for hi in range(hands.shape[0]):
            kp = hands[hi].cpu().numpy().astype(np.float32)
            color = _HAND_PALETTE[hi % len(_HAND_PALETTE)]
            segs = np.stack([np.stack([kp[a], kp[b]]) for a, b in HAND_BONES]).astype(np.float32)
            self._hand_scene.append(
                self.server.scene.add_line_segments(
                    f"/hands/{hi}/bones",
                    points=segs,
                    colors=np.broadcast_to(np.array(color, np.uint8), segs.shape).copy(),
                    line_width=3.0,
                )
            )
            self._hand_scene.append(
                self.server.scene.add_point_cloud(
                    f"/hands/{hi}/joints",
                    points=kp,
                    colors=np.broadcast_to(np.array(color, np.uint8), kp.shape).copy(),
                    point_size=0.006,
                )
            )

    def _robot_model(self):
        """The FR3 Duo model, built once per session; ``None`` when unavailable."""
        from .robot_model import DuoRobotModel, robot_model_available

        if self._robot_mdl is None:
            if not robot_model_available():
                return None
            self._robot_mdl = DuoRobotModel()
        return self._robot_mdl

    def _draw_robot(self) -> None:
        """Overlay the posed robot; meshes are uploaded once and only moved afterwards."""
        from .robot_model import describe_unavailable

        self.md_robot.visible = bool(self.cb_robot.value)
        if not self.cb_robot.value:
            for h in self._robot_scene.values():
                h.remove()
            self._robot_scene = {}
            return

        frame = self._frame
        if frame is None or frame.robot is None:
            for h in self._robot_scene.values():
                h.remove()
            self._robot_scene = {}
            self.md_robot.content = (
                "No robot pose for this frame. "
                "Use a recorded SynthRobot pose or a calibrated robot-dextris frame."
            )
            return

        model = self._robot_model()
        if model is None:
            self.cb_robot.value = False
            self._set_status(
                f"{_RED}Robot geometry unavailable: {describe_unavailable()}.</span> "
                "See the ontic-viz README."
            )
            return

        poses = model.geom_world_poses(frame.robot["qpos"], frame.robot["base_pose"])
        if not self._robot_scene:
            for geom in model.link_geoms:
                self._robot_scene[geom.name] = self.server.scene.add_mesh_simple(
                    f"/robot/{geom.name}",
                    vertices=geom.vertices,
                    faces=geom.faces,
                    color=tuple(int(255 * c) for c in geom.color),
                    flat_shading=False,
                    side="double",
                )
        for name, handle in self._robot_scene.items():
            pos, wxyz = poses[name]
            handle.position = pos.astype(np.float32)
            handle.wxyz = wxyz.astype(np.float32)
        if frame.robot.get("source") == "image_fit":
            self.md_robot.content = (
                f"**Franka Duo · estimated pose · frame {frame.robot['frame_index']}**\n\n"
                "Joint angles were fitted from images."
            )
        else:
            self.md_robot.content = "**Franka Duo · recorded joint angles**"

    def _draw_action_points(self) -> None:
        """Draw sampled robot action points: ``skeleton`` (on a line through each link)
        or ``surface`` (area-uniform on the link meshes, with colour + normal)."""
        from ontic_data.robot import (
            describe_unavailable,
            duo_joint_angles,
            franka_duo_kinematics,
            robot_model_available,
            sample_action_points,
            sample_surface_points,
        )

        for handle in self._action_pts_scene:
            handle.remove()
        self._action_pts_scene = []
        if not self.cb_action_pts.value:
            return

        frame = self._frame
        if frame is None or frame.robot is None:
            self.cb_action_pts.value = False
            self._set_status(
                f"{_RED}**Show action points** needs recorded joint angles, which only "
                "**synthrobot** provides.</span> Load a synthrobot trajectory first."
            )
            return
        if not robot_model_available():
            self.cb_action_pts.value = False
            self._set_status(
                f"{_RED}Robot kinematics unavailable: {describe_unavailable()}.</span>"
            )
            return

        if self._action_kin is None:
            self._action_kin = franka_duo_kinematics()
        angles = [duo_joint_angles(frame.robot["qpos"])]
        n = int(self.sl_action_n.value)
        surface = self.dd_action_kind.value == "surface"

        normals = None
        if surface:
            pts, normals, rgb, links = sample_surface_points(
                self._action_kin, angles, n_per_link=n, base_pose=frame.robot["base_pose"]
            )
            colors = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            pts, links = sample_action_points(
                self._action_kin, angles, n_per_link=n, base_pose=frame.robot["base_pose"]
            )
            colors = np.stack([_link_color(name) for name in links]).astype(np.uint8)

        self._action_pts_scene.append(
            self.server.scene.add_point_cloud(
                "/action_points",
                points=pts[0].astype(np.float32),
                colors=colors,
                point_size=0.006 if surface else 0.012,
            )
        )
        if normals is not None and self.cb_action_normals.value:
            step = max(1, pts.shape[1] // 2000)
            tail = pts[0, ::step].astype(np.float32)
            segs = np.stack([tail, tail + normals[0, ::step].astype(np.float32) * 0.01], axis=1)
            self._action_pts_scene.append(
                self.server.scene.add_line_segments(
                    "/action_normals",
                    points=segs,
                    colors=np.broadcast_to(np.array([255, 90, 90], np.uint8), segs.shape).copy(),
                    line_width=1.5,
                )
            )
        self._set_status(
            f"Action points: **{self.dd_action_kind.value}**, "
            f"{pts.shape[1]} points over {len(set(links))} links."
        )

    def _set_workspace_from_frame(self) -> None:
        """Fit the box to the hands when the dataset has them, else to the camera rig."""
        if self._frame is None:
            return
        if self._frame.hands is not None:
            lo, hi = padded_aabb(self._frame.hands, margin=_WORKSPACE_MARGIN)
            lo, hi = lo.tolist(), hi.tolist()
        else:
            lo, hi = _scene_bbox(_camera_centres(self._frame.extrinsics))
            lo, hi = lo.tolist(), hi.tolist()
        self._suspend = True
        self.vec_wmin.value = tuple(round(float(x), 3) for x in lo)
        self.vec_wmax.value = tuple(round(float(x), 3) for x in hi)
        self._suspend = False

    def _maybe_follow_workspace(self) -> None:
        if self.cb_crop.value:
            return
        self._set_workspace_from_frame()

    def _init_workspace_from_hands(self) -> None:
        if self._frame is None:
            self._set_status("Load a dataset first.")
            return
        self._set_workspace_from_frame()
        self._draw_workspace_box()
        if self.cb_crop.value:
            self._cheap_update()

    # ------------------------------------------------------------ framing
    def _frame_scene(self) -> None:
        """Point every connected client's camera at the current rig (no-op headless)."""
        if self._frame is None:
            return
        for client in self.server.get_clients().values():
            self._frame_client(client)

    def _frame_client(self, client) -> None:
        if self._frame is None:
            return
        result = self.tracking.result
        if result is not None:
            points = result.output.tracks_world[0, 0].numpy()
            points = points[result.output.valid[0, 0].numpy() & np.isfinite(points).all(-1)]
            if len(points):
                pos, look = _orbit_pose(points, scale=3.2)
            else:
                pos, look = _orbit_pose(_camera_centres(self._frame.extrinsics))
        elif self.tracking.active_clip is not None and self._cloud_bounds is not None:
            pos, look = _orbit_pose(np.stack(self._cloud_bounds), scale=2.8)
        else:
            pos, look = _orbit_pose(_camera_centres(self._frame.extrinsics))
        client.camera.position = pos
        client.camera.look_at = look

    def _maybe_frame_scene(self, traj: int) -> None:
        """Frame on trajectory change only, never on a timestep scrub."""
        if traj == self._framed_traj:
            return
        self._framed_traj = traj
        self._frame_scene()

    def _draw_workspace_box(self) -> None:
        if self._workspace_box is not None:
            self._workspace_box.remove()
            self._workspace_box = None
        lo = np.array(self.vec_wmin.value, np.float32)
        hi = np.array(self.vec_wmax.value, np.float32)
        if (hi <= lo).any():
            return
        segs = _aabb_segments(lo, hi)
        self._workspace_box = self.server.scene.add_line_segments(
            "/workspace",
            points=segs,
            colors=np.broadcast_to(np.array(_BOX_COLOR, np.uint8), segs.shape).copy(),
            line_width=2.0,
        )

    def _on_workspace_change(self) -> None:
        if self._suspend:
            return
        self._draw_workspace_box()
        self._cheap_update()

    def _crop_box(self):
        if not self.cb_crop.value:
            return None, None
        return torch.tensor(self.vec_wmin.value), torch.tensor(self.vec_wmax.value)

    def load_demo(self) -> None:
        """Open a reproducible animated scene without loading a neural model."""
        self.dd_dataset.value = "demo"
        self._load_dataset()
        self.tracking.dd_tracker.value = "Analytic demo (no model)"
        self.tracking.length.value = 48
        self.tracking.count.value = 768
        self.vec_wmin.value = (-1.2, -0.6, 1.9)
        self.vec_wmax.value = (1.2, 0.8, 3.1)
        self.cb_crop.value = True
        self.tracking.start_job(background=False)
        self.server.on_client_connect(lambda event: self._demo_camera(event))
        for client in self.server.get_clients().values():
            self._demo_camera(client)

    def _demo_camera(self, client):
        if self._source is None or self._source.name != "demo":
            return
        client.camera.position = (2.0, -2.0, -0.5)
        client.camera.look_at = (0.0, 0.0, 2.8)
        client.camera.up_direction = (0.0, -1.0, 0.0)

    def close(self) -> None:
        self.tracking.close()

    def register_recordings(self, paths) -> None:
        """Offer previously computed dataset runs without rerunning the models."""
        from .recording import recording_label

        records = {f"{i + 1}. {recording_label(p)}": str(p) for i, p in enumerate(paths)}
        with self.dataset_tab:
            with self.server.gui.add_folder("Saved runs", order=3, expand_by_default=False):
                picker = self.server.gui.add_dropdown("Open recording", list(records))
                self.recording_picker = picker
                self.server.gui.add_markdown(
                    "Open a cached run to compare, or follow steps 1–3 to compute another clip."
                )
        picker.on_update(
            lambda _: self._guarded(lambda: self._open_recording(records[picker.value]))
        )
        self._open_recording(next(iter(records.values())))

    def _open_recording(self, path) -> None:
        self.tracking.pause()
        self.tracking.play.disabled = self.tracking.restart.disabled = True
        self.tracking.clip_frame.disabled = True
        picker = getattr(self, "recording_picker", None)
        if picker is not None:
            picker.disabled = True
        try:
            self._load_recording(path)
        finally:
            self.tracking._update_playback_controls()
            if picker is not None:
                picker.disabled = False

    def _load_recording(self, path) -> None:
        from .recording import load_recording, recording_label
        from .tracking import TrackingRun

        value = load_recording(path)
        clip = value.clip if isinstance(value, TrackingRun) else value
        self.tracking.pause()
        if self._source is None or self._source.name != clip.source_name:
            self.dd_dataset.value = clip.source_name
            self._load_dataset()
        self._suspend = True
        self.dd_traj.value = self._traj_labels[clip.trajectory]
        self.sl_time.max = max(1, self._source.num_timesteps(clip.trajectory) - 1)
        self.sl_time.value = clip.indices[0]
        self._suspend = False
        self.tracking.set_context(self._source, clip.trajectory)
        self._selection_context = (id(self._source), clip.trajectory)
        self._camera_selection = {
            name: name in clip.camera_names for name in clip.frames[0].cam_names
        }
        self._cam_context = None
        panel = self.tracking
        panel.start.value = clip.indices[0]
        panel.length.value = len(clip.indices)
        panel.stride.value = clip.indices[1] - clip.indices[0]
        panel.resolution.value = max(128, min(1024, 16 * round(max(clip.images.shape[-2:]) / 16)))
        panel.geometry.value = clip.geometry.provenance["source"]
        video_config = clip.geometry.provenance.get("video_configuration")
        if video_config:
            from .video_depth import VIDEO_MODEL_LABELS

            label = next(
                (
                    label
                    for label, name in VIDEO_MODEL_LABELS.items()
                    if name == video_config["name"]
                ),
                None,
            )
            if label is not None:
                panel.video_model.value = label
                panel._on_video_model_change()
                panel.video_repo.value = video_config.get("repo_path", "")
                panel.video_checkpoint.value = video_config.get("checkpoint_path", "")
                defaults = panel.video_resolution_defaults()
                panel.video_input_size.value = (
                    video_config.get("input_size") or defaults["input_size"]
                )
                panel.video_resolution_level.value = video_config.get(
                    "resolution_level", defaults["resolution_level"]
                )
                panel.video_fp32.value = video_config.get("fp32", False)
                # Opening an archive does not enable future network downloads.
                panel.video_download.value = False
        panel.update_depth_summary()
        panel.point_size.value = 0.006
        backbone = clip.geometry.provenance.get("backbone")
        if backbone in self.backbones:
            self.dd_backbone.value = backbone
            if "backbone_long_side" in clip.geometry.provenance:
                self.sl_lside.value = clip.geometry.provenance[
                    "backbone_long_side"
                ] or backbone_default_long_side(backbone)
        if value is not clip:
            from .tracking import TRACKER_LABELS

            name = value.output.metadata.get("tracker")
            label = next((label for label, key in TRACKER_LABELS.items() if key == name), None)
            if label is not None:
                panel.dd_tracker.value = label
        panel.add_result(value, recording_label(path))
        self._frame_scene()
