"""Editable world-space boxes for selecting track seeds across cameras."""

from __future__ import annotations

import numpy as np


class QueryRegions:
    def __init__(self, app):
        self.app = app
        self.boxes = {}
        self._serial = 0
        self._syncing = False
        self._handles = []
        self._gizmo = None
        g = app.server.gui
        with g.add_folder("Query boxes"):
            self.enabled = g.add_checkbox("Use query boxes", False)
            self.reference = g.add_button("Show query frame")
            self.add = g.add_button("Add box")
            self.selected = g.add_dropdown("Selected box", ["(no boxes)"], disabled=True)
            self.include = g.add_checkbox("Include selected box", True, disabled=True)
            self.center = g.add_vector3("Box center", (0.0, 0.0, 0.0), step=0.01, disabled=True)
            self.size = g.add_vector3(
                "Box size", (0.2, 0.2, 0.2), min=(0.001,) * 3, step=0.01, disabled=True
            )
            self.remove = g.add_button("Remove selected box", disabled=True)
            g.add_markdown(
                "**Add box** first shows the selected depth source on the first clip frame. "
                "After changing depth settings or browsing playback, use **Show query frame** "
                "before positioning boxes. Drag the axes to move a box; edit **Box size** to resize. "
                "Points inside **any included box** can seed tracks, after display filtering. "
                "Boxes select points on the **first clip frame**; tracks can move outside them."
            )
        self.inputs = [
            self.enabled,
            self.reference,
            self.add,
            self.selected,
            self.include,
            self.center,
            self.size,
            self.remove,
        ]
        self.reference.on_click(lambda _: app.tracking.show_query_frame())
        self.add.on_click(lambda _: app.tracking.show_query_frame(on_ready=self.add_box))
        self.remove.on_click(lambda _: app._guarded(self.remove_box))
        self.enabled.on_update(lambda _: app._guarded(self._changed))
        self.selected.on_update(lambda _: app._guarded(self._select))
        for h in (self.include, self.center, self.size):
            h.on_update(lambda _: app._guarded(self._edit))

    def bounds(self):
        if not self.enabled.value:
            return None
        return tuple(
            (
                tuple(np.asarray(b["center"]) - np.asarray(b["size"]) / 2),
                tuple(np.asarray(b["center"]) + np.asarray(b["size"]) / 2),
            )
            for b in self.boxes.values()
            if b["enabled"]
        )

    def add_box(self):
        cloud = self.app._cloud_pts
        points = np.empty((0, 3)) if cloud is None else np.asarray(cloud)
        points = points[np.isfinite(points).all(-1)]
        if len(points):
            lo, hi = np.quantile(points, (0.05, 0.95), axis=0)
            middle = np.median(points, axis=0)
            # A bounds midpoint may lie in empty space, especially when metric
            # depth contains distant outliers. Anchor the box on a shown sample.
            center = points[np.square(points - middle).sum(-1).argmin()]
        else:
            lo, hi = (np.asarray(x) for x in (self.app.vec_wmin.value, self.app.vec_wmax.value))
            center = (lo + hi) / 2
        self._serial += 1
        name = f"Box {self._serial}"
        self.boxes[name] = dict(
            center=tuple(center), size=tuple(np.maximum((hi - lo) / 3, 0.02)), enabled=True
        )
        self._syncing = True
        self.selected.options = list(self.boxes)
        self.selected.value = name
        self.enabled.value = True
        self._syncing = False
        self._select()

    def remove_box(self):
        self.boxes.pop(self.selected.value, None)
        self._syncing = True
        self.selected.options = list(self.boxes) or ["(no boxes)"]
        self.selected.value = self.selected.options[0]
        self._syncing = False
        self._select()

    def clear(self):
        self.boxes.clear()
        self._syncing = True
        self.selected.options = ["(no boxes)"]
        self.selected.value = "(no boxes)"
        self.enabled.value = False
        self._syncing = False
        self._select()

    def _select(self):
        if self._syncing:
            return
        box = self.boxes.get(self.selected.value)
        for h in (self.selected, self.include, self.center, self.size, self.remove):
            h.disabled = box is None
        if box is not None:
            self._syncing = True
            self.center.value, self.size.value = box["center"], box["size"]
            self.include.value = box["enabled"]
            self._syncing = False
        self._changed()

    def _edit(self):
        box = self.boxes.get(self.selected.value)
        if self._syncing or box is None:
            return
        values = dict(
            center=tuple(self.center.value), size=tuple(self.size.value), enabled=self.include.value
        )
        if box == values:
            return
        if min(values["size"]) <= 0:
            return
        self.boxes[self.selected.value] = values
        self._changed()

    def _drag(self, _):
        def update():
            box = self.boxes.get(self.selected.value)
            if box is None or self._gizmo is None:
                return
            box["center"] = tuple(self._gizmo.position)
            self.center.value = box["center"]
            for name, handle in self._handles:
                if name == self.selected.value:
                    handle.position = box["center"]
            self.app.tracking.render()

        self.app._guarded(update)

    def _changed(self):
        if self._syncing:
            return
        for _, h in self._handles:
            h.remove()
        self._handles = []
        if self._gizmo is not None:
            self._gizmo.remove()
            self._gizmo = None
        for name, box in self.boxes.items():
            selected = name == self.selected.value
            h = self.app.server.scene.add_box(
                f"/query_boxes/{name}",
                position=box["center"],
                dimensions=box["size"],
                color=(255, 190, 40) if selected else (80, 210, 255),
                wireframe=True,
                visible=self.enabled.value and box["enabled"],
            )
            self._handles.append((name, h))
            if selected and box["enabled"] and self.enabled.value:
                self._gizmo = self.app.server.scene.add_transform_controls(
                    "/query_boxes/move",
                    position=box["center"],
                    scale=max(0.05, min(max(box["size"]), 0.3)),
                    disable_rotations=True,
                )
                self._gizmo.on_update(self._drag)
        if hasattr(self.app, "tracking"):
            self.app.tracking.render()

    def set_busy(self, busy):
        if self._gizmo is not None:
            self._gizmo.visible = not busy and self.enabled.value
