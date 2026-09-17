"""Clip controls and cached playback for the geometry viewer."""

from __future__ import annotations

import importlib.util
import tempfile
import threading
from pathlib import Path


from .errors import log_error, status_error
from .query_regions import QueryRegions
from .tracking import (
    GEOMETRY_SOURCES,
    SENSOR,
    TRACKER_LABELS,
    ClipSpec,
    PreparedClip,
    TrackingRun,
    TrackerSettings,
    TrackingCancelled,
    TrackingRunner,
    export_tracks,
    render_tracks,
    region_mask,
    visible_query_mask,
    visible_track_samples,
)


class TrackingPanel:
    def __init__(self, app, tab, playback_folder):
        self.app = app
        self.runner = TrackingRunner(app.device)
        self.result = None
        self.preview = None
        self.history = {}
        self._history_serial = 0
        self._history_updating = False
        self.context = None
        self.cancel = threading.Event()
        self._stop = threading.Event()
        self._playing = threading.Event()
        self._play_thread = None
        self._worker = None
        self._handles = []
        self._disabled = []
        self._query_mask_cache = None
        self._sample_mask_cache = None
        self._model_files = {}
        self._model_files_lock = threading.RLock()
        self._model_files_owner = "mvtracker"
        g = app.server.gui
        with app.dataset_tab:
            with g.add_folder("Clip", order=1):
                self.start = g.add_number("Start frame", 0, min=0, step=1)
                self.length = g.add_number("Frames", 12, min=2, max=120, step=1)
                self.stride = g.add_number("Frame step", 1, min=1, step=1)
                self.use_frame = g.add_button("Start at current frame")
        with app.depth_tab:
            with g.add_folder("Depth for tracking", order=2):
                self.geometry = g.add_dropdown(
                    "Depth source", GEOMETRY_SOURCES, initial_value=SENSOR
                )
                self.resolution = g.add_slider("Clip long side", 128, 1024, 16, 512)
                g.add_markdown(
                    "Uses the clip and cameras from **1. Dataset**. Choose recorded depth "
                    "or the selected backbone with scale calibration. Preview it here, then open **3. Tracking**."
                )
                self.prepare_button = g.add_button("Preview depth clip", color="blue")
        with tab:
            g.add_markdown(
                "Choose a tracker and the points to follow. **Queries start on the first clip frame.**",
                order=-1,
            )
            self.depth_summary = g.add_markdown("")
            self.dd_tracker = g.add_dropdown(
                "Tracker", list(TRACKER_LABELS), initial_value="MVTracker"
            )
            self.model_hint = g.add_markdown("")
            with g.add_folder("Query points"):
                self.view = g.add_dropdown(
                    "Tracker camera", ["(load a scene)"], initial_value="(load a scene)"
                )
                self.count = g.add_slider("Point budget", 32, 4096, 32, 512)
                self.sampling = g.add_dropdown("Sampling", ["Even coverage", "Farthest points"])
                g.add_markdown(
                    "Seeds use the **shown depth points** on the first clip frame, including "
                    "**Display** workspace, confidence and point-count filters. Colors stay attached to point IDs."
                )
            self.regions = QueryRegions(app)
            self.run_button = g.add_button("Run tracking", color="blue")
            with g.add_folder("Model files", expand_by_default=False):
                self.repo = g.add_text("Research checkout", "")
                self.checkpoint = g.add_text("Checkpoint", "")
                self.wan_cache = g.add_text("Wan base model cache", "", visible=False)
                self.download = g.add_checkbox("Download missing weights", False)
                g.add_markdown(
                    "Optional upstream packages must be installed. TrackCraft3R also requires the Wan base model; downloads can be large."
                )
        self.message = g.add_markdown("", order=0.5)
        self.cancel_button = g.add_button(
            "Cancel after current operation", disabled=True, visible=False, order=0.6
        )
        with playback_folder:
            self.history_picker = g.add_dropdown("Cached result", ["(no results)"], disabled=True)
            self.play = g.add_button("Play", disabled=True)
            self.fps = g.add_slider("Playback FPS", 1, 30, 1, 8)
            self.loop = g.add_checkbox("Loop clip", True)
            self.timeline = g.add_markdown("No cached tracks yet.")
            self.download_button = g.add_button("Download tracks (.npz)", disabled=True)
            self.rerun_button = g.add_button(
                "Download Rerun (.rrd)",
                disabled=True,
                visible=importlib.util.find_spec("rerun") is not None,
            )
        self.rerun_button.on_click(self.download_rerun)
        self.inputs = [
            self.dd_tracker,
            self.start,
            self.length,
            self.stride,
            self.use_frame,
            self.geometry,
            self.resolution,
            self.view,
            self.count,
            self.sampling,
            self.repo,
            self.checkpoint,
            self.wan_cache,
            self.download,
        ] + self.regions.inputs
        self.dd_tracker.on_update(lambda _: self._on_tracker_change())
        self.geometry.on_update(lambda _: self.update_depth_summary())
        self.use_frame.on_click(lambda _: setattr(self.start, "value", int(app.sl_time.value)))
        self.run_button.on_click(lambda _: self.start_job())
        self.prepare_button.on_click(lambda _: self.start_job(track=False))
        self.history_picker.on_update(lambda _: self._select_history())
        self.cancel_button.on_click(lambda _: self.request_cancel())
        self.play.on_click(lambda _: self.toggle_playback())
        self.download_button.on_click(self.download_result)
        self.update_model_hint()
        self.update_depth_summary()

    def build_display(self):
        g = self.app.server.gui
        with g.add_folder("Track appearance", order=-1):
            self.show = g.add_checkbox("Show tracks", True)
            self.background = g.add_checkbox("Show depth cloud", True)
            self.trails = g.add_slider("Trail frames", 0, 60, 1, 10)
            self.point_size = g.add_slider("Track point size", 0.005, 0.06, 0.001, 0.018)
            self.occluded = g.add_checkbox("Show occluded tracks (dim)", True)
            self.visibility = g.add_slider("Visibility threshold", 0.0, 1.0, 0.05, 0.5)
        for handle in [self.show, self.trails, self.point_size, self.occluded, self.visibility]:
            handle.on_update(lambda _: self.app._guarded(self.render))
        self.background.on_update(lambda _: self.app._cheap_update())

    def update_model_hint(self):
        name = TRACKER_LABELS[self.dd_tracker.value]
        mono = name in ("tapip3d", "trackcraft3r")
        self.view.visible = mono
        self.wan_cache.visible = name == "trackcraft3r"
        hints = {
            "demo": "**Analytic demo** · known synthetic motion, no neural inference. Select the demo scene.",
            "mvtracker": "**Multi-view RGB-D** · jointly tracks enabled cameras. At least 7 frames; world-space trajectories.",
            "tapip3d": "**Single-view RGB-D** · choose one enabled tracker camera. Geometry can still use multiple views.",
            "trackcraft3r": "**Single-view RGB** · exactly 12 frames; native 480 × 832 grid. Depth anchors tracks into the calibrated world frame.",
        }
        self.model_hint.content = hints[name] + f" Device: **{self.app.device}**."

    def _file_controls(self):
        return {
            "repo_path": self.repo,
            "checkpoint_path": self.checkpoint,
            "base_model_cache_dir": self.wan_cache,
            "allow_download": self.download,
        }

    def _on_tracker_change(self):
        """Retain each tracker's settings instead of reusing another model's files."""
        with self._model_files_lock:
            name = TRACKER_LABELS[self.dd_tracker.value]
            if name != self._model_files_owner:
                self._model_files[self._model_files_owner] = {
                    key: control.value for key, control in self._file_controls().items()
                }
                self._model_files_owner = name
                self._show_model_files(name)
            self.update_model_hint()

    def _show_model_files(self, name):
        settings = self._model_files.get(name, {})
        for key, control in self._file_controls().items():
            control.value = settings.get(key, False if key == "allow_download" else "")

    def configure_model_files(self, settings):
        """Pre-fill a mapping keyed by tracker registry name without changing selection."""
        unknown = set(settings) - set(TRACKER_LABELS.values())
        if unknown:
            raise ValueError(f"Unknown trackers in model config: {sorted(unknown)}")
        with self._model_files_lock:
            self._on_tracker_change()
            self._model_files.update({name: dict(value) for name, value in settings.items()})
            if self._model_files_owner in settings:
                self._show_model_files(self._model_files_owner)

    def update_depth_summary(self):
        source = (
            "recorded sensor / GT depth"
            if self.geometry.value == SENSOR
            else f"{self.app.dd_backbone.value} · {self.geometry.value}"
        )
        self.depth_summary.content = (
            f"**Depth input:** {source}. Configure it in **2. Depth model**."
        )

    def set_context(self, source, traj):
        key = (id(source), traj)
        if key == self.context:
            return
        self.context = key
        if source.name == "demo":
            self.dd_tracker.value = "Analytic demo (no model)"
        elif TRACKER_LABELS[self.dd_tracker.value] == "demo":
            self.dd_tracker.value = "MVTracker"
        self.clear(clear_cache=True)
        self.start.value = 0
        self.length.value = max(2, min(12, source.num_timesteps(traj)))
        self.message.content = "Ready to prepare a clip."

    def sync_cameras(self):
        frame = self.app._frame
        if frame is None:
            return
        names = [frame.cam_names[i] for i in self.app._selected_ix()]
        previous = self.view.value
        self.view.options = names or ["(select an input camera)"]
        self.view.value = previous if previous in names else self.view.options[0]

    def clear(self, *, clear_cache=False):
        self.pause()
        self.result = None
        self.preview = None
        self.app._tracking_frame = False
        self.hide()
        if clear_cache:
            self.regions.clear()
            self._query_mask_cache = None
            self._sample_mask_cache = None
            self.runner.cached_clip = None
            self.history.clear()
            self._history_updating = True
            self.history_picker.options = ["(no results)"]
            self.history_picker.value = "(no results)"
            self.history_picker.disabled = True
            self._history_updating = False
        self.play.disabled = self.download_button.disabled = self.rerun_button.disabled = True
        self.app.dd_align.disabled = False
        self.app.dd_metric.disabled = False
        self.timeline.content = "No cached tracks yet."

    def start_job(self, *, background=True, track=True):
        app = self.app
        if not app._operation_lock.acquire(blocking=False):
            return
        app._busy = True
        self.pause()
        self.cancel.clear()
        try:
            # GUI updates are asynchronous; sync the fields before taking the job snapshot.
            self._on_tracker_change()
            if app._source is None or app._frame is None:
                raise ValueError("Load a dataset in 1. Dataset first")
            source, traj = app._source, app._traj_index()
            if app.dd_dataset.value != source.name:
                raise ValueError("Click Load dataset to open the selected scene first")
            spec = ClipSpec(
                int(self.start.value),
                int(self.length.value),
                int(self.stride.value),
                int(self.resolution.value),
            )
            spec.indices(source.num_timesteps(traj))
            settings = TrackerSettings(
                TRACKER_LABELS[self.dd_tracker.value],
                self.checkpoint.value.strip(),
                self.repo.value.strip(),
                self.wan_cache.value.strip(),
                self.download.value,
                spec.image_long_side,
            )
            if track and settings.name == "trackcraft3r" and spec.length != 12:
                raise ValueError("TrackCraft3R requires exactly 12 frames")
            if track and settings.name == "mvtracker" and spec.length < 7:
                raise ValueError("MVTracker requires at least 7 frames")
            if track and settings.name == "demo" and source.name != "demo":
                raise ValueError("Select the demo dataset for analytic motion")
            cameras = tuple(app._frame.cam_names[i] for i in app._selected_ix())
            geometry, backbone, conditioned = (
                self.geometry.value,
                app.dd_backbone.value,
                app.cb_condition.value,
            )
            query_view, count, sampling = (
                self.view.value,
                int(self.count.value),
                self.sampling.value,
            )
            point_filters = app._point_filters()
            regions = self.regions.bounds()
            if track and regions == ():
                raise ValueError("Add or include a query box, or turn off Use query boxes")
            if not cameras:
                raise ValueError("Select at least one input camera")
            if geometry != SENSOR:
                app._configure_backbone()
            controls = (
                self.inputs
                + [
                    self.run_button,
                    self.prepare_button,
                    self.history_picker,
                    app.bb_checkpoint,
                    app.bb_download,
                    app.btn_run,
                    app.btn_loadds,
                    app.dd_dataset,
                    app.dd_traj,
                    app.sl_time,
                    app.dd_backbone,
                    app.sl_lside,
                    app.cb_condition,
                    app.btn_all,
                    app.btn_none,
                    app.btn_apply_preset,
                    app.dd_preset,
                    app.vec_wmin,
                    app.vec_wmax,
                    app.btn_winit,
                    app.cb_crop,
                    app.sl_conf,
                    app.sl_stride,
                    app.sl_voxel,
                    app.sl_sfc,
                    app.sl_fps,
                    app.btn_fps,
                ]
                + app._cam_checks
            )
            self._disabled = [(h, h.disabled) for h in controls]
            for h, _ in self._disabled:
                h.disabled = True
            self.regions.set_busy(True)
            self.cancel_button.disabled = False
            self.cancel_button.visible = True
            self.play.disabled = self.download_button.disabled = self.rerun_button.disabled = True
        except Exception as e:
            self.message.content = status_error(log_error("tracking settings", e))
            app._busy = False
            app._operation_lock.release()
            return

        def work():
            try:
                self.message.content = "Preparing clip…"
                clip = self.runner.prepare(
                    source,
                    traj,
                    spec,
                    cameras,
                    geometry_source=geometry,
                    backbone=backbone,
                    backbone_runner=app.runner,
                    conditioned=conditioned,
                    progress=self._progress,
                    cancel=self.cancel,
                )
                result = (
                    self.runner.run(
                        clip,
                        settings,
                        source=source,
                        query_view=query_view,
                        count=count,
                        sampling=sampling,
                        point_filters=point_filters,
                        regions=regions,
                        progress=self._progress,
                        cancel=self.cancel,
                    )
                    if track
                    else None
                )
                self._restore_controls()
                value = result if result is not None else clip
                self.add_result(value)
                self.message.content = (
                    f"**{self.dd_tracker.value}** · {len(clip.indices)} frames · "
                    f"{result.output.ids.shape[1]:,} points · {result.elapsed:.1f}s tracking. "
                    "Clip cached; press Play or scrub the frame slider."
                    if result is not None
                    else f"**Depth preview** · {len(clip.indices)} frames cached. "
                    "Play or scrub to inspect geometry, then open 3. Tracking and Run tracking."
                )
            except TrackingCancelled:
                self.message.content = "Cancelled. Any previous completed result remains available."
            except Exception as e:
                self.message.content = status_error(log_error("prepare & track", e))
            finally:
                self._restore_controls()
                self.cancel_button.disabled = True
                self.cancel_button.visible = False
                self._update_playback_controls()
                app._busy = False
                app._operation_lock.release()

        if background:
            self._worker = threading.Thread(target=work, name="ontic-tracking", daemon=True)
            self._worker.start()
        else:
            work()

    @property
    def active_clip(self):
        return self.result.clip if self.result is not None else self.preview

    def _update_playback_controls(self):
        self.play.disabled = self.active_clip is None
        self.download_button.disabled = self.rerun_button.disabled = self.result is None
        self.app.dd_align.disabled = self.app.dd_metric.disabled = self.active_clip is not None
        self.history_picker.disabled = not self.history

    def add_result(self, value: TrackingRun | PreparedClip, label: str | None = None):
        clip = value.clip if isinstance(value, TrackingRun) else value
        geo = clip.geometry.provenance
        title = geo.get("backbone") or "sensor"
        kind = (
            value.output.metadata.get("tracker", "tracks")
            if isinstance(value, TrackingRun)
            else "depth preview"
        )
        self._history_serial += 1
        label = (
            label
            or f"{self._history_serial}. {title} · {kind} · {clip.indices[0]}–{clip.indices[-1]}"
        )
        # Bound retained clips; the current result and the last five comparisons fit typical sessions.
        self.history[label] = value
        while len(self.history) > 6:
            del self.history[next(iter(self.history))]
        self._history_updating = True
        self.history_picker.options = list(self.history)
        self.history_picker.value = label
        self._history_updating = False
        self._activate(value)

    def _activate(self, value):
        self.pause()
        self.result = value if isinstance(value, TrackingRun) else None
        self.preview = None if self.result is not None else value
        clip = self.active_clip
        self.app._suspend = True
        t = int(self.app.sl_time.value)
        if t not in clip.indices:
            self.app.sl_time.value = clip.indices[0]
        self.app._suspend = False
        self.show_frame(int(self.app.sl_time.value))
        self._update_playback_controls()

    def _select_history(self):
        if self._history_updating or self.history_picker.value not in self.history:
            return
        self.app._guarded(lambda: self._activate(self.history[self.history_picker.value]))

    def _restore_controls(self):
        for h, disabled in self._disabled:
            h.disabled = disabled
        self._disabled = []
        self.regions.set_busy(False)

    def _progress(self, message):
        self.message.content = message

    def request_cancel(self):
        self.cancel.set()
        self.message.content = (
            "Cancellation requested; waiting for the current load or inference call to return."
        )

    def show_frame(self, timestep):
        clip = self.active_clip
        if clip is None or timestep not in clip.indices:
            return False
        index = clip.indices.index(timestep)
        app, frame = self.app, clip.frames[index]
        app._frame = frame
        app._tracking_frame = True
        app._result = clip.display_result(index)
        app._run_images = clip.images[0, index]
        app._run_ix = [frame.cam_names.index(n) for n in clip.camera_names]
        app._conditioned = False
        app._rebuild_cameras(frame)
        app._draw_hands()
        app._draw_robot()
        app._draw_action_points()
        app._draw_workspace_box()
        app._render_cloud()
        self.render()
        return True

    def hide(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def render(self):
        self.hide()
        run, app = self.result, self.app
        clip = self.active_clip
        if clip is None:
            return
        t = int(app.sl_time.value)
        if not app._tracking_frame or t not in clip.indices:
            self.timeline.content = (
                f"Frame {t} is outside the cached clip. Press Play to return to it."
            )
            return
        index = clip.indices.index(t)
        if run is None:
            self.timeline.content = (
                f"**{index + 1} / {len(clip.indices)}** · dataset frame **{t}** · depth preview"
            )
            app._set_status(
                f"**{clip.geometry.provenance.get('backbone') or 'Sensor depth'}** · {clip.source_name} · "
                f"{clip.geometry.provenance['source']} · calibrated world (meters)"
            )
            return
        pts, colors, lines, line_colors = render_tracks(
            run,
            index,
            trail_length=int(self.trails.value),
            show_occluded=self.occluded.value,
            threshold=float(self.visibility.value),
            query_mask=self._visible_queries(),
            sample_mask=self._visible_samples(),
        )
        if self.show.value:
            self._handles = [
                app.server.scene.add_point_cloud(
                    "/tracks/points",
                    points=pts,
                    colors=colors,
                    point_size=float(self.point_size.value),
                    point_shape="circle",
                ),
                app.server.scene.add_line_segments(
                    "/tracks/trails", points=lines, colors=line_colors, line_width=2.0
                ),
            ]
        self.timeline.content = (
            f"**{index + 1} / {len(run.clip.indices)}** · dataset frame **{t}** · "
            f"{len(pts):,} / {run.output.ids.shape[1]:,} tracks shown"
        )
        app._set_status(
            f"**{run.output.metadata.get('tracker', 'Tracks')}** · {run.clip.source_name} · "
            f"{run.clip.geometry.provenance['source']} · calibrated world (meters)"
        )

    def _visible_queries(self):
        filters, regions = self.app._point_filters(), self.regions.bounds()
        key = (id(self.result), filters)
        if self._query_mask_cache is None or self._query_mask_cache[0] != key:
            self._query_mask_cache = (key, visible_query_mask(self.result, filters))
        keep = self._query_mask_cache[1]
        if regions is not None:
            keep = keep & region_mask(self.result.queries.xyz_world[0], regions)
        return keep

    def _visible_samples(self):
        filters = self.app._point_filters()
        key = (id(self.result), filters.crop, filters.drop_conf_pct)
        if self._sample_mask_cache is None or self._sample_mask_cache[0] != key:
            self._sample_mask_cache = (key, visible_track_samples(self.result, filters))
        return self._sample_mask_cache[1]

    def toggle_playback(self):
        if self._playing.is_set():
            self.pause()
            return
        if self.active_clip is None or self.app._busy:
            return
        self._playing.set()
        self.play.label = "Pause"
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(
                target=self._play_loop, name="ontic-playback", daemon=True
            )
            self._play_thread.start()

    def pause(self):
        self._playing.clear()
        self.play.label = "Play"

    def _play_loop(self):
        while not self._stop.wait(1 / max(1, int(self.fps.value))):
            if not self._playing.is_set():
                continue

            def advance():
                clip = self.active_clip
                if clip is None:
                    self.pause()
                    return
                indices = clip.indices
                t = int(self.app.sl_time.value)
                i = indices.index(t) + 1 if t in indices else 0
                if i == len(indices) and not self.loop.value:
                    self.pause()
                    return
                self.app._suspend = True
                self.app.sl_time.value = indices[i % len(indices)]
                self.app._suspend = False
                self.show_frame(indices[i % len(indices)])

            self.app._guarded(advance)

    def download_result(self, event):
        if self.result is not None and event.client is not None:
            event.client.send_file_download(
                "ontic-tracks.npz", export_tracks(self.result), save_immediately=True
            )

    def download_rerun(self, event):
        if self.result is None or event.client is None:
            return
        self.pause()

        def export():
            from .rerun_export import save_rerun

            self.rerun_button.disabled = True
            try:
                with tempfile.TemporaryDirectory(prefix="ontic-tracks-") as directory:
                    path = save_rerun(
                        self.result,
                        Path(directory) / "ontic-tracks.rrd",
                        trail_length=int(self.trails.value),
                    )
                    event.client.send_file_download(
                        path.name, path.read_bytes(), save_immediately=True
                    )
            finally:
                self.rerun_button.disabled = False

        self.app._guarded(export)

    def close(self):
        self.cancel.set()
        self._stop.set()
        self.pause()
