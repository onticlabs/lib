"""Example schema: the dict every dataset ``__getitem__`` returns and its batched form.

Unbatched views carry ``(view, ...)`` or ``(time, view, ...)`` tensors; batched views
add a leading batch dim, and ``to_batched_views`` folds ``batch *time`` into one
leading axis so consumers can treat ``(batch*time, view, ...)`` uniformly.
"""

from __future__ import annotations

from typing import Optional, TypedDict

import einops as eo
from torch import Tensor

#: einops pack patterns per key (``{batch}`` is replaced by ``*``).
pattern_dict = {
    "extrinsics": "{batch} view four1 four2",
    "intrinsics": "{batch} view three1 three2",
    "image": "{batch} view channel height width",
    "depth": "{batch} view one height width",
    "state_mask": "{batch} view channel height width",
    "static_float": "{batch} view channel height width",
    "near": "{batch} view",
    "far": "{batch} view",
    "depth_is_metric": "{batch} view",
}


class BatchedTempViews(TypedDict, total=False):
    extrinsics: Tensor  # (batch, *time, view, 4, 4) c2w
    intrinsics: Tensor  # (batch, *time, view, 3, 3) normalised
    image: Tensor  # (batch, *time, view, 3, H, W) in [0, 1]
    depth: Tensor  # (batch, *time, view, 1, H, W) metres
    near: Tensor  # (batch, *time, view)
    far: Tensor  # (batch, *time, view)
    depth_is_metric: Tensor  # (batch, *time, view)
    index: Tensor  # (batch, *time, view) int64 camera indices
    state_mask: Optional[Tensor]  # (batch, *time, view, 1, H, W) bool
    static_float: Optional[Tensor]  # (batch, *time, view, 1, H, W)
    is_novel_view: Optional[Tensor]  # (batch, *time, view) bool


class BatchedTempExample(TypedDict, total=False):
    target: BatchedTempViews
    context: BatchedTempViews
    scene: list[str]
    actions: Optional[dict[str, Tensor]]  # (batch, *time, N_copies, N_points, 4)
    workspace_min: Tensor  # (batch, 3)
    workspace_max: Tensor  # (batch, 3)


def to_batched_example(any_example: BatchedTempExample) -> BatchedTempExample:
    """Fold ``batch *time`` into one leading axis; a no-op if already 5-D images."""
    if any_example["context"]["image"].dim() == 5:
        return any_example

    out = BatchedTempExample(
        target=to_batched_views(any_example["target"]),
        context=to_batched_views(any_example["context"]),
        scene=any_example["scene"],
    )
    actions = any_example.get("actions", None)
    if actions is not None:
        out["actions"] = {
            k: eo.pack([v], "* N_copies N_points four")[0] for k, v in actions.items()
        }
    return out


def to_batched_views(any_views: BatchedTempViews) -> BatchedTempViews:
    """Fold ``batch *time`` of every known key; ``index`` may be a tensor or a list."""
    if any_views["image"].dim() == 5:
        return any_views

    batch_t_v = BatchedTempViews(
        **{
            k: eo.pack([any_views[k]], v.format(batch="*"))[0]
            for k, v in pattern_dict.items()
            if any_views.get(k, None) is not None
        }
    )
    cam_index = any_views.get("index", None)
    if cam_index is not None:
        if isinstance(cam_index, Tensor):
            batch_t_v["index"] = eo.pack([cam_index], "* one")[0]
        else:
            if not isinstance(cam_index, list):
                raise TypeError("index must be a Tensor, a list or None")
            batch_t_v["index"] = [ix for vx in cam_index for ix in vx]
    return batch_t_v
