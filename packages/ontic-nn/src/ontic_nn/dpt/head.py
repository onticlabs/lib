"""DPT and DualDPT heads (Depth Anything V3 layout).

Attribute names follow ``depth_anything_3.model.dpt`` / ``dualdpt`` so released
weights load after a prefix strip on the state-dict keys.

Both heads take ``feats``: a sequence of four ``(B, S, N, C)`` ViT token maps
(coarse to fine block outputs), the image size ``H, W`` (multiples of
``patch_size``) and ``patch_start_idx`` (number of leading non-patch tokens to
drop). Outputs are dicts of ``(B, S, ...)`` tensors at ``(H, W) / down_ratio``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_INT_MAX = 1610612736


def _interpolate(
    x: Tensor,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Tensor:
    """``F.interpolate`` on ``(B, C, H, W)`` chunked over the batch to stay below INT_MAX elements."""
    if size is None:
        if scale_factor is None:
            raise ValueError("either size or scale_factor must be given")
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    total = size[0] * size[1] * x.shape[0] * x.shape[1]
    if total > _INT_MAX:
        chunks = torch.chunk(x, chunks=(total // _INT_MAX) + 1, dim=0)
        outs = [F.interpolate(c, size=size, mode=mode, align_corners=align_corners) for c in chunks]
        return torch.cat(outs, dim=0).contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


# ---------------------------------------------------------------------------
# Positional embedding
# ---------------------------------------------------------------------------
class Permute(nn.Module):
    """``Tensor.permute`` as a module."""

    def __init__(self, dims: Tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Normalised UV grid ``(H, W, 2)`` scaled so the image diagonal has unit half-length."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)
    diag = (aspect_ratio**2 + 1.0) ** 0.5
    sx, sy = aspect_ratio / diag, 1.0 / diag
    lx = -sx * (width - 1) / width
    rx = sx * (width - 1) / width
    ty = -sy * (height - 1) / height
    by = sy * (height - 1) / height
    xs = torch.linspace(lx, rx, steps=width, dtype=dtype, device=device)
    ys = torch.linspace(ty, by, steps=height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(xs, ys, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def _sincos_pos_embed(embed_dim: int, pos: Tensor, omega_0: float = 100) -> Tensor:
    """1-D sinusoidal positional embedding ``(M, embed_dim)``."""
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1.0 / omega_0 ** (omega / (embed_dim / 2.0))
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(pos_grid: Tensor, embed_dim: int, omega_0: float = 100) -> Tensor:
    """``(H, W, 2) -> (H, W, embed_dim)`` sinusoidal positional embedding."""
    h, w, _ = pos_grid.shape
    ex = _sincos_pos_embed(embed_dim // 2, pos_grid[..., 0].reshape(-1), omega_0)
    ey = _sincos_pos_embed(embed_dim // 2, pos_grid[..., 1].reshape(-1), omega_0)
    return torch.cat([ex, ey], dim=-1).view(h, w, embed_dim)


def _add_pos_embed(x: Tensor, w: int, h: int, ratio: float = 0.1) -> Tensor:
    """Add a sinusoidal UV embedding (image aspect ``w / h``) to ``x (B, C, h', w')``."""
    pw, ph = x.shape[-1], x.shape[-2]
    pe = create_uv_grid(pw, ph, aspect_ratio=w / h, dtype=x.dtype, device=x.device)
    pe = position_grid_to_embed(pe, x.shape[1]) * ratio
    pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pe


# ---------------------------------------------------------------------------
# Activations and norms
# ---------------------------------------------------------------------------
def _apply_activation(x: Tensor, activation: str) -> Tensor:
    act = activation.lower()
    if act == "exp":
        return torch.exp(x)
    if act == "expp1":
        return torch.exp(x) + 1
    if act == "expm1":
        return torch.expm1(x)
    if act == "relu":
        return torch.relu(x)
    if act == "sigmoid":
        return torch.sigmoid(x)
    if act == "softplus":
        return F.softplus(x)
    if act == "tanh":
        return torch.tanh(x)
    return x  # linear


NormType = Literal["noop", "idt", "batch_norm", "group_norm", "instance_norm"]


def _make_norm(num_features: int, norm_type: NormType, num_groups: int = 8) -> nn.Module:
    if norm_type in ("noop", "idt"):
        return nn.Identity()
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features=num_features)
    if norm_type == "group_norm":
        return nn.GroupNorm(num_channels=num_features, num_groups=num_groups)
    if norm_type == "instance_norm":
        return nn.InstanceNorm2d(num_features=num_features)
    raise ValueError(f"invalid normalization layer type: {norm_type!r}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ResidualConvUnit(nn.Module):
    """``x + conv2(act(norm2(conv1(act(norm1(x))))))`` with 3x3 convs on ``(B, C, H, W)``.

    Norms are identity for ``norm_type="idt"`` (the DA3 layout).
    """

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        groups: int = 1,
        norm_type: NormType = "idt",
    ) -> None:
        super().__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.norm1 = _make_norm(features, norm_type)
        self.norm2 = _make_norm(features, norm_type)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)
        out = self.activation(x)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Lateral merge + residual conv + bilinear upsample + 1x1 contraction.

    ``forward(x, lateral=None, size=None)``: ``x`` and ``lateral`` are ``(B, C, h, w)``;
    the output is resized to ``size`` (or ``self.size``, else 2x).
    """

    def __init__(
        self,
        features: int,
        activation: nn.Module,
        expand: bool = False,
        align_corners: bool = True,
        size: Optional[Tuple[int, int]] = None,
        has_residual: bool = True,
        groups: int = 1,
        norm_type: NormType = "idt",
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = (
            ResidualConvUnit(features, activation, groups=groups, norm_type=norm_type)
            if has_residual
            else None
        )
        self.resConfUnit2 = ResidualConvUnit(
            features, activation, groups=groups, norm_type=norm_type
        )
        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)

    def forward(
        self, x: Tensor, lateral: Optional[Tensor] = None, size: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        y = x
        if self.has_residual and lateral is not None and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(lateral)
        y = self.resConfUnit2(y)
        if size is None and self.size is None:
            y = _interpolate(y, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        else:
            target = size if size is not None else self.size
            y = _interpolate(y, size=target, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(y)


def _make_fusion_block(
    features: int,
    has_residual: bool = True,
    inplace: bool = False,
    norm_type: NormType = "idt",
) -> FeatureFusionBlock:
    return FeatureFusionBlock(
        features=features,
        activation=nn.ReLU(inplace=inplace),
        expand=False,
        align_corners=True,
        has_residual=has_residual,
        norm_type=norm_type,
    )


def _make_scratch(in_shape: Sequence[int], out_shape: int, groups: int = 1, expand: bool = False):
    scratch = nn.Module()
    c1, c2, c3, c4 = (out_shape * (2**i if expand else 1) for i in range(4))
    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch


def _make_resize_layers(out_channels: Sequence[int]) -> nn.ModuleList:
    """Spatial resizers for the four pyramid levels: x4, x2, x1, /2."""
    return nn.ModuleList(
        [
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ]
    )


def _flatten_views(feats: Sequence[Tensor]) -> Tuple[List[Tensor], int, int]:
    b, s, n, c = feats[0].shape
    return [f.reshape(b * s, n, c) for f in feats], b, s


def _run_chunked(
    fn, flat_feats: List[Tensor], b: int, s: int, chunk_size: Optional[int]
) -> Dict[str, Tensor]:
    if chunk_size is None or chunk_size >= s:
        out = fn(flat_feats)
    else:
        chunks: List[Dict[str, Tensor]] = []
        for s0 in range(0, b * s, chunk_size):
            s1 = min(s0 + chunk_size, b * s)
            chunks.append(fn([f[s0:s1] for f in flat_feats]))
        out = {k: torch.cat([c[k] for c in chunks], 0) for k in chunks[0]}
    return {k: v.reshape(b, s, *v.shape[1:]) for k, v in out.items()}


# ---------------------------------------------------------------------------
# DPTHead
# ---------------------------------------------------------------------------
class DPTHead(nn.Module):
    """Single-branch DPT head: main map (+ confidence) and an optional sky map.

    Output keys: ``head_name (B, S, H', W')`` (or ``(B, S, H', W', output_dim - 1)``
    when ``output_dim > 2``), ``f"{head_name}_conf"`` if ``output_dim > 1``, and
    ``sky_name (B, S, H', W')`` if ``use_sky_head``; ``H', W' = (H, W) / down_ratio``.
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 1,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = False,
        down_ratio: int = 1,
        head_name: str = "depth",
        use_sky_head: bool = True,
        sky_name: str = "sky",
        sky_activation: str = "relu",
        use_ln_for_heads: bool = False,
        norm_type: str = "idt",
        fusion_block_inplace: bool = False,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_main = head_name
        self.sky_name = sky_name
        self.out_dim = output_dim
        self.has_conf = output_dim > 1
        self.use_sky_head = use_sky_head
        self.sky_activation = sky_activation
        self.intermediate_layer_idx = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(dim_in) if norm_type == "layer" else nn.Identity()
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, 1) for oc in out_channels])
        self.resize_layers = _make_resize_layers(out_channels)

        self.scratch = _make_scratch(list(out_channels), features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet2 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet3 = _make_fusion_block(features, inplace=fusion_block_inplace)
        self.scratch.refinenet4 = _make_fusion_block(
            features, has_residual=False, inplace=fusion_block_inplace
        )

        hf1 = features
        hf2 = 32
        self.scratch.output_conv1 = nn.Conv2d(hf1, hf1 // 2, 3, 1, 1)

        ln_seq = (
            [Permute((0, 2, 3, 1)), nn.LayerNorm(hf2), Permute((0, 3, 1, 2))]
            if use_ln_for_heads
            else []
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(hf1 // 2, hf2, 3, 1, 1),
            *ln_seq,
            nn.ReLU(inplace=False),
            nn.Conv2d(hf2, output_dim, 1),
        )
        if use_sky_head:
            self.scratch.sky_output_conv2 = nn.Sequential(
                nn.Conv2d(hf1 // 2, hf2, 3, 1, 1),
                *ln_seq,
                nn.ReLU(inplace=False),
                nn.Conv2d(hf2, 1, 1),
            )

    def forward(
        self,
        feats: Sequence[Tensor],
        H: int,
        W: int,
        patch_start_idx: int = 0,
        chunk_size: Optional[int] = 8,
    ) -> Dict[str, Tensor]:
        flat_feats, b, s = _flatten_views(feats)
        return _run_chunked(
            lambda fs: self._forward_impl(fs, H, W, patch_start_idx), flat_feats, b, s, chunk_size
        )

    def _forward_impl(
        self, feats: List[Tensor], H: int, W: int, patch_start_idx: int
    ) -> Dict[str, Tensor]:
        b, _, c = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized = []
        for si, ti in enumerate(self.intermediate_layer_idx):
            x = feats[ti][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(b, c, ph, pw)
            x = self.projects[si](x)
            if self.pos_embed:
                x = _add_pos_embed(x, W, H)
            x = self.resize_layers[si](x)
            resized.append(x)

        fused = self._fuse(resized)
        fused = self.scratch.output_conv1(fused)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused = _interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)

        if self.pos_embed:
            fused = _add_pos_embed(fused, W, H)

        outs: Dict[str, Tensor] = {}
        logits = self.scratch.output_conv2(fused)
        if self.has_conf:
            fmap = logits.permute(0, 2, 3, 1)
            outs[self.head_main] = _apply_activation(fmap[..., :-1], self.activation).squeeze(-1)
            outs[f"{self.head_main}_conf"] = _apply_activation(fmap[..., -1], self.conf_activation)
        else:
            outs[self.head_main] = _apply_activation(logits, self.activation).squeeze(1)

        if self.use_sky_head:
            sky_logits = self.scratch.sky_output_conv2(fused)
            outs[self.sky_name] = _apply_activation(sky_logits, self.sky_activation).squeeze(1)
        return outs

    def _fuse(self, feats: List[Tensor]) -> Tensor:
        """Coarse-to-fine refinenet fusion; ``refinenet1`` uses its default 2x upsample."""
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out


# ---------------------------------------------------------------------------
# DualDPTHead
# ---------------------------------------------------------------------------
class DualDPTHead(nn.Module):
    """Dual-branch DPT head: main map (+ confidence) and a 6-channel auxiliary map.

    Output keys: ``head_names[0] (B, S, H', W')`` (or ``(..., output_dim - 1)`` when
    ``output_dim > 2``) and ``f"{head_names[0]}_conf" (B, S, H', W')`` at
    ``H', W' = (H, W) / down_ratio``; ``head_names[1] (B, S, H8, W8, 6)`` and
    ``f"{head_names[1]}_conf" (B, S, H8, W8)`` at the finest fusion grid
    ``H8, W8 = 8 * (H, W) // patch_size``.
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        head_names: Tuple[str, str] = ("depth", "ray"),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.aux_levels = aux_pyramid_levels
        self.aux_out1_conv_num = aux_out1_conv_num
        self.head_main, self.head_aux = head_names
        self.intermediate_layer_idx = (0, 1, 2, 3)

        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, 1) for oc in out_channels])
        self.resize_layers = _make_resize_layers(out_channels)

        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        hf1, hf2 = features, 32
        self.scratch.output_conv1 = nn.Conv2d(hf1, hf1 // 2, 3, 1, 1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(hf1 // 2, hf2, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hf2, output_dim, 1),
        )

        # Aux fusion chain (independent weights)
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1_aux = nn.ModuleList(
            [self._make_aux_out1_block(hf1) for _ in range(aux_pyramid_levels)]
        )
        ln_seq = [Permute((0, 2, 3, 1)), nn.LayerNorm(hf2), Permute((0, 3, 1, 2))]
        self.scratch.output_conv2_aux = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hf1 // 2, hf2, 3, 1, 1),
                    *ln_seq,
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hf2, 7, 1),
                )
                for _ in range(aux_pyramid_levels)
            ]
        )

    def forward(
        self,
        feats: Sequence[Tensor],
        H: int,
        W: int,
        patch_start_idx: int = 0,
        chunk_size: Optional[int] = 8,
    ) -> Dict[str, Tensor]:
        flat_feats, b, s = _flatten_views(feats)
        return _run_chunked(
            lambda fs: self._forward_impl(fs, H, W, patch_start_idx), flat_feats, b, s, chunk_size
        )

    def _forward_impl(
        self, feats: List[Tensor], H: int, W: int, patch_start_idx: int
    ) -> Dict[str, Tensor]:
        b, _, c = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size

        resized = []
        for si, ti in enumerate(self.intermediate_layer_idx):
            x = feats[ti][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(b, c, ph, pw)
            x = self.projects[si](x)
            if self.pos_embed:
                x = _add_pos_embed(x, W, H)
            x = self.resize_layers[si](x)
            resized.append(x)

        fused_main, fused_aux_pyr = self._fuse(resized)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)
        fused_main = _interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = _add_pos_embed(fused_main, W, H)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = _apply_activation(fmap[..., :-1], self.activation)
        main_conf = _apply_activation(fmap[..., -1], self.conf_activation)

        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = _add_pos_embed(last_aux, W, H)
        aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_aux = aux_logits.permute(0, 2, 3, 1)
        aux_pred = _apply_activation(fmap_aux[..., :-1], "linear")
        aux_conf = _apply_activation(fmap_aux[..., -1], self.conf_activation)

        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,
            f"{self.head_aux}_conf": aux_conf,
        }

    def _fuse(self, feats: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[Tensor] = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](a) for i, a in enumerate(aux_list)]
        return out, aux_list

    def _make_aux_out1_block(self, in_ch: int) -> nn.Sequential:
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 3:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.aux_out1_conv_num == 1:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1))
        raise ValueError(f"aux_out1_conv_num={self.aux_out1_conv_num} not in {{1, 3, 5}}")
