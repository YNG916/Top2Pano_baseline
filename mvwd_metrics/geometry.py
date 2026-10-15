"""Prediction-independent GT visible-surface overlap in calibrated CV cameras."""

from __future__ import annotations

import numpy as np


def camera(K, T_wc, depth_z, near=0.1, far=15.0):
    K, T, depth = (
        np.asarray(K, dtype=np.float64),
        np.asarray(T_wc, dtype=np.float64),
        np.asarray(depth_z, dtype=np.float64),
    )
    if K.shape != (3, 3) or T.shape != (4, 4) or depth.ndim != 2:
        raise ValueError(
            "Expected K3x3, CV camera_to_world4x4 and camera-forward depth HxW"
        )
    if (
        not np.isfinite(K).all()
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1])
    ):
        raise ValueError("Invalid camera calibration")
    if near < 0 or far <= near:
        raise ValueError("Invalid camera clipping range")
    return {"K": K, "T_wc": T, "depth_z": depth, "near": near, "far": far}


def _project(points, source, target):
    transform = np.linalg.inv(target["T_wc"]) @ source["T_wc"]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    projected = points @ target["K"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:, :2] / projected[:, 2:3]
    return uv, points[:, 2]


def register_depth(depth, K_depth, T_depth, K_rgb, T_rgb, target_hw, near, far):
    """Z-buffer native GT geometry into the target RGB pixel grid; no interpolated depth."""
    src = camera(K_depth, T_depth, depth, near, far)
    height, width = target_hw
    dst = camera(K_rgb, T_rgb, np.zeros((height, width)), near, far)
    valid = np.isfinite(depth) & (depth > near) & (depth <= far)
    y, x = np.nonzero(valid)
    pixels = np.stack((x + 0.5, y + 0.5, np.ones(len(x))), axis=1)
    points = (pixels @ np.linalg.inv(src["K"]).T) * np.asarray(depth)[y, x, None]
    uv, z = _project(points, src, dst)
    good = np.isfinite(uv).all(1) & (z > near) & (z <= far)
    uv, z = uv[good], z[good]
    xy = np.floor(uv).astype(np.int64)
    good = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    out = np.full(height * width, np.inf)
    np.minimum.at(out, xy[good, 1] * width + xy[good, 0], z[good])
    out[~np.isfinite(out)] = 0
    return out.reshape(height, width)


def _valid_surface(cam, boundary, atol, rtol):
    z = cam["depth_z"]
    valid = np.isfinite(z) & (z > cam["near"]) & (z <= cam["far"])
    # Reject both sides of a depth discontinuity and adjacent invalid samples.
    stable = valid.copy()
    for axis in (0, 1):
        sl0 = [slice(None), slice(None)]
        sl1 = list(sl0)
        sl0[axis], sl1[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(sl0), tuple(sl1)
        agree = (
            valid[a]
            & valid[b]
            & (np.abs(z[a] - z[b]) <= np.maximum(atol, rtol * np.maximum(z[a], z[b])))
        )
        stable[a] &= agree
        stable[b] &= agree
    if boundary:
        stable[:boundary] = False
        stable[-boundary:] = False
        stable[:, :boundary] = False
        stable[:, -boundary:] = False
    return stable


def directed_overlap(source, target, boundary=2, depth_atol_m=0.05, depth_rtol=0.01):
    if boundary < 0 or depth_atol_m <= 0 or depth_rtol < 0:
        raise ValueError("Invalid overlap tolerances")
    valid = _valid_surface(source, boundary, depth_atol_m, depth_rtol)
    target_valid = _valid_surface(target, boundary, depth_atol_m, depth_rtol)
    y, x = np.nonzero(valid)
    if not len(x):
        return {"ratio": 0.0, "visible_pixels": 0, "valid_pixels": 0}
    pixels = np.stack((x + 0.5, y + 0.5, np.ones(len(x))), axis=1)
    points = (pixels @ np.linalg.inv(source["K"]).T) * source["depth_z"][y, x, None]
    uv, z = _project(points, source, target)
    good = np.isfinite(uv).all(1) & (z > target["near"]) & (z <= target["far"])
    xy = np.zeros((len(x), 2), dtype=np.int64)
    xy[good] = np.floor(uv[good]).astype(np.int64)
    height, width = target["depth_z"].shape
    good &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    idx = np.flatnonzero(good)
    u, v = xy[idx, 0], xy[idx, 1]
    target_z = target["depth_z"][v, u]
    matches = target_valid[v, u] & (
        np.abs(target_z - z[idx]) <= np.maximum(depth_atol_m, depth_rtol * target_z)
    )
    visible = int(matches.sum())
    return {
        "ratio": visible / len(x),
        "visible_pixels": visible,
        "valid_pixels": len(x),
    }


def visible_surface_overlap(first, second, **kwargs):
    forward, backward = directed_overlap(first, second, **kwargs), directed_overlap(
        second, first, **kwargs
    )
    return {
        "ratio": min(forward["ratio"], backward["ratio"]),
        "forward": forward,
        "backward": backward,
        "source": "ground_truth_depth_and_camera",
        "definition": "minimum bidirectional GT visible-surface fraction",
    }


def directed_correspondences(source, target, boundary=2, depth_atol_m=0.05, depth_rtol=0.01):
    """GT-visible pixel-center matches, with a safe bilinear target footprint.

    uv is in continuous corner-origin coordinates: pixel (y,x) has center
    (x+.5,y+.5). All nonzero bilinear contributors must be stable surfaces
    at the projected depth. This is deliberately stricter than the nearest
    pixel overlap receipt used by the historical MEt3R eligibility filter.
    """
    if boundary < 0 or depth_atol_m <= 0 or depth_rtol < 0:
        raise ValueError("Invalid correspondence tolerances")
    valid = _valid_surface(source, boundary, depth_atol_m, depth_rtol)
    target_valid = _valid_surface(target, boundary, depth_atol_m, depth_rtol)
    y, x = np.nonzero(valid)
    pixels = np.stack((x + 0.5, y + 0.5, np.ones(len(x))), axis=1)
    points = (pixels @ np.linalg.inv(source["K"]).T) * source["depth_z"][y, x, None]
    uv, z = _project(points, source, target)
    height, width = target["depth_z"].shape
    good = np.isfinite(uv).all(1) & (z > target["near"]) & (z <= target["far"])
    # Snap only numerical roundoff at integer centers, never subpixel geometry.
    index = uv - 0.5
    rounded = np.rint(index)
    index = np.where(np.abs(index - rounded) < 1e-8, rounded, index)
    uv = index + 0.5
    good &= (index[:, 0] >= 0) & (index[:, 0] <= width - 1)
    good &= (index[:, 1] >= 0) & (index[:, 1] <= height - 1)
    idx = np.flatnonzero(good)
    base = np.floor(index[idx]).astype(np.int64)
    fraction = index[idx] - base
    visible = np.ones(len(idx), dtype=bool)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = (fraction[:, 0] if dx else 1 - fraction[:, 0]) * (
            fraction[:, 1] if dy else 1 - fraction[:, 1])
        active = weight > 1e-8
        u, v = base[:, 0] + dx, base[:, 1] + dy
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        safe_u, safe_v = u.clip(0, width - 1), v.clip(0, height - 1)
        tz = target["depth_z"][safe_v, safe_u]
        matches = inside & target_valid[safe_v, safe_u] & (
            np.abs(tz - z[idx]) <= np.maximum(depth_atol_m, depth_rtol * tz))
        visible &= ~active | matches
    selected = idx[visible]
    return {
        "source_yx": np.stack((y[selected], x[selected]), axis=1),
        "target_uv": uv[selected],
        "ratio": len(selected) / len(x) if len(x) else 0.0,
        "visible_pixels": len(selected),
        "valid_pixels": len(x),
    }


def gt_surface_correspondences(first, second, **kwargs):
    forward = directed_correspondences(first, second, **kwargs)
    backward = directed_correspondences(second, first, **kwargs)
    return {
        "forward": forward,
        "backward": backward,
        "ratio": min(forward["ratio"], backward["ratio"]),
        "source": "ground_truth_depth_and_camera",
        "definition": "minimum bidirectional GT visible-surface fraction, valid bilinear footprint",
    }


def correspondence_receipt(matches):
    """Small JSON receipt; never write dense coordinate arrays into summary logs."""
    return {
        **{k: matches[k] for k in ("ratio", "source", "definition")},
        **{direction: {k: matches[direction][k] for k in
                       ("ratio", "visible_pixels", "valid_pixels")}
           for direction in ("forward", "backward")},
    }
