"""Numerical parity with the fwomo (frontier) PointTransformerV3.

Runs only where ``fwomo_3d`` and spconv import (the frontier venv):

    cd <frontier_world_model> && PYTHONPATH=<lib>/src:<lib>/packages/ontic-nn/src \\
        .venv/bin/python -m pytest <lib>/packages/ontic-nn/tests/test_ptv3_parity.py
"""

import dataclasses
import importlib

import pytest
import torch

from ontic_lib.structures import PointBatch
from ontic_nn.ptv3 import (
    AttentionCfg,
    PointTransformerV3,
    PoolingCfg,
    StageCfg,
    TemporalCfg,
    load_fwomo_state_dict,
    presets,
)


def _has(name):
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _has("spconv"), reason="needs spconv"),
    pytest.mark.skipif(
        not _has("fwomo_3d.model.dynamic_model.backbone.ptv3"), reason="needs fwomo_3d"
    ),
]

DEPTHS = (1, 2, 1)
CHANNELS = (12, 24, 24)
HEADS = (2, 2, 2)
PATCH = 8
DEC_DEPTHS = (1, 1)
DEC_CHANNELS = (12, 24)
DEC_HEADS = (2, 2)
GRID_SIZE = 0.02


def our_cfg(temporal: bool):
    return presets.base(
        encoder=tuple(StageCfg(d, c, h, PATCH) for d, c, h in zip(DEPTHS, CHANNELS, HEADS)),
        decoder=tuple(
            StageCfg(d, c, h, PATCH) for d, c, h in zip(DEC_DEPTHS, DEC_CHANNELS, DEC_HEADS)
        ),
        pooling=PoolingCfg(strides=(2, 2)),
        temporal=TemporalCfg(merge_window=(1, 2)) if temporal else None,
        attention=AttentionCfg(backend="sdpa"),
        conv_impl="torch",
        shuffle_orders=False,
        drop_path=0.0,
        grid_size=GRID_SIZE,
    )


def frontier_model(temporal: bool):
    from fwomo_3d.model.dynamic_model.backbone.ptv3 import (
        PointTrainsformerV3Cfg,
        PointTransformerV3 as FrontierPTv3,
    )
    from fwomo_3d.model.dynamic_model.backbone.pool_utils import SerializedPooling

    cfg = PointTrainsformerV3Cfg(
        in_channels=6,
        out_channels=3,
        grid_size=GRID_SIZE,
        temporal_merger=temporal,
        merge_window_size=(1, 2),
        stride=(2, 2),
        enc_depths=DEPTHS,
        enc_channels=CHANNELS,
        enc_num_head=HEADS,
        enc_patch_size=(PATCH,) * 3,
        dec_depths=DEC_DEPTHS,
        dec_channels=DEC_CHANNELS,
        dec_num_head=DEC_HEADS,
        dec_patch_size=(PATCH,) * 2,
        drop_path=0.0,
        shuffle_orders=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
    )
    model = FrontierPTv3(cfg)
    for module in model.modules():
        if isinstance(module, SerializedPooling):
            module.shuffle_orders = False  # frontier pooling ignores cfg.shuffle_orders
    return model


def unique_voxel_points(seed, temporal, device):
    """Groups of >= PATCH points whose voxels stay distinct through pooling and time merges.

    Each time step lives in its own x-slab of width 8 (aligned to the stride-4
    cells reached after two poolings), so merged groups never hold duplicate
    voxels either -- spconv's neighbour lookup on duplicate rows is hash-order
    dependent. An anchor point at cell (0, 0, 0) pins ``voxel_coords`` to the
    cell grid.
    """
    g = torch.Generator().manual_seed(seed)
    b, t = (2, 4) if temporal else (3, 1)
    side, slab, per_group = 24, 8, 100
    rows = []
    for bi in range(b):
        for ti in range(t):
            cells = torch.randperm(slab * side * side, generator=g)[:per_group]
            grid = torch.stack(
                [cells // (side * side) + slab * ti, cells // side % side, cells % side], -1
            )
            offset = 0.05 + 0.9 * torch.rand(per_group, 3, generator=g)
            if bi == 0 and ti == 0:
                grid[0] = 0
                offset[0] = 0.05
            rows.append((bi, ti, (grid.float() + offset) * GRID_SIZE))
    coord = torch.cat([r[2] for r in rows])
    batch = torch.cat([torch.full((per_group,), r[0]) for r in rows])
    time = torch.cat([torch.full((per_group,), r[1]) for r in rows])
    feat = torch.randn(coord.shape[0], 6, generator=g)
    points = PointBatch(coord, feat, batch, time=time if temporal else None).to(device)
    point_dict = {
        "coord": points.coord,
        "feat": points.feat,
        "batch": points.batch,
        "grid_size": GRID_SIZE,
    }
    if temporal:
        point_dict["time_coord"] = points.time
    return points, point_dict


@pytest.mark.parametrize("temporal", [False, True])
def test_matches_frontier(temporal):
    device = torch.device("cuda")
    torch.manual_seed(0)
    frontier = frontier_model(temporal).to(device).eval()
    ours = PointTransformerV3(our_cfg(temporal)).to(device).eval()
    result = load_fwomo_state_dict(ours, frontier.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    points, point_dict = unique_voxel_points(1, temporal, device)
    with torch.no_grad():
        expected = frontier(dict(point_dict)).feat
        actual = ours(points).feat
    diff = (expected - actual).abs().max().item()
    print(f"frontier parity (temporal={temporal}): max abs diff {diff:.3e}")
    assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), f"max abs diff {diff}"


@pytest.mark.parametrize("temporal", [False, True])
def test_spconv_matches_torch_conv(temporal):
    device = torch.device("cuda")
    torch.manual_seed(0)
    torch_model = PointTransformerV3(our_cfg(temporal)).to(device).eval()
    spconv_cfg = dataclasses.replace(our_cfg(temporal), conv_impl="spconv")
    spconv_model = PointTransformerV3(spconv_cfg).to(device).eval()
    spconv_model.load_state_dict(torch_model.state_dict())  # same keys and layouts

    points, _ = unique_voxel_points(2, temporal, device)
    with torch.no_grad():
        a = torch_model(points).feat
        b = spconv_model(points).feat
    diff = (a - b).abs().max().item()
    print(f"spconv vs torch conv (temporal={temporal}): max abs diff {diff:.3e}")
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), f"max abs diff {diff}"
