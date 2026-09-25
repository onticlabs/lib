"""Small analytic RGB-D scene for exercising playback without neural weights."""

from __future__ import annotations

import math

import torch

from ontic_nn.trackers import GeometrySequence, PointQueries, TrackerOutput
from ontic_nn.trackers.common import make_tracker_output

from .data_source import Frame


class DemoSource:
    name = "demo"
    height, width = 96, 128

    def list_trajectories(self):
        return ["Moving sphere · analytic RGB-D"]

    def num_timesteps(self, traj):
        return 72

    @staticmethod
    def center(t):
        phase = t * 2 * math.pi / 48
        return torch.tensor([0.55 * math.sin(phase), 0.12 * math.cos(phase), 2.5])

    def get_frame(self, traj, t, with_depth=False):
        h, w = self.height, self.width
        k = torch.tensor([[0.9, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]])
        y, x = torch.meshgrid(
            (torch.arange(h) + 0.5) / h, (torch.arange(w) + 0.5) / w, indexing="ij"
        )
        rays = torch.stack([(x - 0.5) / 0.9, (y - 0.5) / 1.2, torch.ones_like(x)], -1)
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[:, 0, 3] = torch.tensor([-0.3, 0.3])
        images, depths = [], []
        for pose in poses:
            offset = pose[:3, 3] - self.center(t)
            a = rays.square().sum(-1)
            b = 2 * (rays * offset).sum(-1)
            c = offset.square().sum() - 0.48**2
            discr = b.square() - 4 * a * c
            sphere_z = (-b - discr.clamp_min(0).sqrt()) / (2 * a)
            hit = (discr > 0) & (sphere_z > 0)
            depth = torch.where(hit, sphere_z, 4.0)
            world = rays * depth[..., None] + pose[:3, 3]
            normal = (world - self.center(t)) / 0.48
            sphere_rgb = (normal * 0.3 + torch.tensor([0.55, 0.55, 0.7])).clamp(0, 1)
            checker = ((world[..., 0] * 4).floor() + (world[..., 1] * 4).floor()).remainder(2)
            backdrop = (0.12 + 0.09 * checker)[..., None].expand(-1, -1, 3)
            images.append(torch.where(hit[..., None], sphere_rgb, backdrop).permute(2, 0, 1))
            depths.append(depth)
        return Frame(
            torch.stack(images),
            k.repeat(2, 1, 1),
            poses,
            ["left", "right"],
            depth=torch.stack(depths)[:, None] if with_depth else None,
        )

    def trajectories(
        self, indices, queries: PointQueries, geometry: GeometrySequence
    ) -> TrackerOutput:
        xyz = queries.xyz_world[0]
        on_sphere = (xyz - self.center(indices[0])).norm(dim=-1) < 0.49
        displacement = torch.stack([self.center(t) - self.center(indices[0]) for t in indices])
        tracks = xyz[None] + displacement[:, None] * on_sphere[None, :, None]
        t, v, h, w = geometry.depth.shape[1:]
        per_view = torch.zeros(t, v, len(xyz))
        for ti in range(t):
            for vi in range(v):
                pose = geometry.extrinsics[0, ti, vi]
                cam = (tracks[ti] - pose[:3, 3]) @ pose[:3, :3]
                projected = cam @ geometry.intrinsics[0, ti, vi].T
                uv = projected[:, :2] / projected[:, 2:]
                inside = (cam[:, 2] > 0) & ((uv >= 0) & (uv < 1)).all(-1)
                px = (uv[:, 0] * w).long().clamp(0, w - 1)
                py = (uv[:, 1] * h).long().clamp(0, h - 1)
                z = geometry.depth[0, ti, vi, py, px]
                per_view[ti, vi] = (inside & ((cam[:, 2] - z).abs() < 0.035)).float()
        output = make_tracker_output(
            queries,
            geometry,
            tracks[None],
            per_view.amax(1)[None],
            visibility_scope="any_view",
            metadata={
                "tracker": "analytic_demo",
                "neural_inference": False,
                "description": "Known sphere translation; visibility from synthetic depth",
            },
        )
        output.visibility_per_view = per_view[None]
        return output
