"""Calibrated pinhole rays and Top2Pano density alpha compositing."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pinhole_rays(K, T_wc, height, width):
    """CV camera (+X right, +Y down, +Z forward), pixel centers u/v + .5."""
    rows, columns = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype) + 0.5,
        torch.arange(width, device=K.device, dtype=K.dtype) + 0.5, indexing="ij")
    pixels = torch.stack((columns, rows, torch.ones_like(rows)), dim=-1).reshape(-1, 3)
    camera = torch.linalg.solve(K, pixels.T.unsqueeze(0).expand(K.shape[0], -1, -1)).transpose(1, 2)
    camera = F.normalize(camera, dim=-1)
    directions = torch.einsum("bij,brj->bri", T_wc[:, :3, :3], camera)
    return T_wc[:, :3, 3], directions, camera[..., 2]


def relative_map(values, valid=None):
    """Per-image min/max normalization; invalid pixels never set the range."""
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    axes = tuple(range(1, values.ndim))
    low = values.masked_fill(~valid, float("inf")).amin(dim=axes, keepdim=True)
    high = values.masked_fill(~valid, -float("inf")).amax(dim=axes, keepdim=True)
    present = valid.flatten(1).any(dim=1).reshape((-1,) + (1,) * (values.ndim - 1))
    low = torch.where(present, low, torch.zeros_like(low))
    high = torch.where(present, high, torch.ones_like(high))
    return torch.where(valid, (values - low) / (high - low).clamp_min(1e-6), torch.zeros_like(values))


class PerspectiveRenderer:
    def __init__(self, samples=150, chunk_rays=2048, max_height_m=3.0,
                 density_scale=10.0, rgb_distance_normalized=0.8):
        if samples <= 0 or chunk_rays <= 0 or max_height_m <= 0 or density_scale <= 0 or rgb_distance_normalized <= 0:
            raise ValueError("Renderer dimensions and scales must be positive")
        self.samples = samples
        self.chunk_rays = chunk_rays
        self.max_height_m = max_height_m
        self.density_scale = density_scale
        self.rgb_distance_normalized = rgb_distance_normalized

    def prepare_density(self, voxel, valid, wall_mask):
        # Retain the original hidden-channel vertical volume, density scaling,
        # solid walls and two solid floor slices. Padding contributes no density.
        valid = valid[:, None].to(voxel.dtype)
        sigma = relative_map(voxel) * self.density_scale
        sigma = torch.where(wall_mask[:, None], torch.full_like(sigma, 1000.0), sigma) * valid
        floors = (1000.0 * valid).expand(-1, 2, -1, -1)
        return torch.cat((floors, sigma), dim=1)

    def _integrate(self, sigma, color, origin, directions, camera_z, w2p,
                   floor_z, near, far, half_extent, height, width):
        outputs_rgb, outputs_depth, outputs_opacity = [], [], []
        batch, _, map_h, map_w = sigma.shape
        fractions = (torch.arange(self.samples, device=sigma.device, dtype=sigma.dtype) + 1) / self.samples
        distances = near[:, None] + (far - near)[:, None] * fractions
        # Density in the original renderer is per normalized map length, not /m.
        intervals = ((far - near) / self.samples / half_extent)[:, None, None]
        color = color.permute(0, 3, 1, 2)
        color_volume = color.unsqueeze(2).expand(-1, -1, sigma.shape[1], -1, -1)
        for start in range(0, height * width, self.chunk_rays):
            end = min(start + self.chunk_rays, height * width)
            rays = directions[:, start:end]
            points = origin[:, None, None, :] + rays[:, :, None, :] * distances[:, None, :, None]
            homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
            pixels = torch.einsum("bij,brsj->brsi", w2p, homogeneous)
            xy = torch.stack((2 * (pixels[..., 0] + 0.5) / map_w - 1,
                              2 * (pixels[..., 1] + 0.5) / map_h - 1), dim=-1)
            vertical = 2 * (points[..., 2] - floor_z[:, None, None]) / self.max_height_m - 1
            grid = torch.cat((xy, vertical[..., None]), dim=-1).permute(0, 2, 1, 3).unsqueeze(2)
            density = F.grid_sample(sigma[:, None], grid, align_corners=False,
                                    padding_mode="zeros")[:, 0, :, 0, :].transpose(1, 2)
            colors = F.grid_sample(color_volume, grid, align_corners=False,
                                   padding_mode="zeros")[:, :, :, 0, :].permute(0, 3, 2, 1)
            optical = density * intervals
            transmittance = torch.exp(-F.pad(optical[..., :-1].cumsum(dim=-1), (1, 0)))
            weights = transmittance * (-torch.expm1(-optical))
            outputs_rgb.append((weights[..., None] * colors).sum(dim=2))
            distance = (weights * distances[:, None, :]).sum(dim=-1)
            outputs_depth.append(distance * camera_z[:, start:end])
            outputs_opacity.append(weights.sum(dim=-1))
        return {
            "rgb": torch.cat(outputs_rgb, dim=1).reshape(batch, height, width, 3),
            "depth_z": torch.cat(outputs_depth, dim=1).reshape(batch, height, width),
            "opacity": torch.cat(outputs_opacity, dim=1).reshape(batch, height, width),
        }

    def __call__(self, voxel, batch, *, modality="rgb", prepared=False):
        if modality not in ("rgb", "depth"):
            raise ValueError("Choose rgb or depth camera")
        dtype, device = voxel.dtype, voxel.device
        def tensor(name):
            return batch[name].to(device=device, dtype=dtype)
        valid = batch["bev_valid"].to(device=device, dtype=torch.bool)
        walls = batch["wall_mask"].to(device=device, dtype=torch.bool)
        sigma = voxel if prepared else self.prepare_density(voxel, valid, walls)
        if modality == "rgb":
            height, width = map(int, batch["target_hw"][0].tolist())
        else:
            height, width = batch["target_depth_z"].shape[-2:]
        K, T = tensor("K_" + modality), tensor("T_wc_" + modality)
        origin, directions, camera_z = pinhole_rays(K, T, height, width)
        p2w = tensor("bev_to_world")
        half_extent = torch.linalg.vector_norm(p2w[:, :3, 0], dim=1) * sigma.shape[-1] / 2
        near, far = tensor("near_m").reshape(-1), tensor("far_m").reshape(-1)
        # Upstream samples colors from batch['jpg'], whose range is [-1, 1].
        source_color = tensor("bev_rgb") * 2.0 - 1.0
        full = self._integrate(sigma, source_color, origin, directions, camera_z,
                               tensor("world_to_bev"), tensor("floor_z").reshape(-1),
                               near, far, half_extent, height, width)
        if modality == "rgb":
            rgb_far = torch.minimum(far, half_extent * self.rgb_distance_normalized).clamp_min(near + 1e-3)
            cropped = self._integrate(sigma, source_color, origin, directions, camera_z,
                                      tensor("world_to_bev"), tensor("floor_z").reshape(-1),
                                      near, rgb_far, half_extent, height, width)
            full["rgb"] = relative_map(cropped["rgb"])
        full["depth_condition"] = relative_map(full["depth_z"])
        return full
