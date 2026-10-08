"""Baseline-owned reader and textured mesh rays for the frozen robot bundle.

No simulator, producer, dataset API or uploaded loader is imported. The robot
surface is opaque; colors are a PBR base-color guide, not native RTX shading.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .mvwd_raw import read_json, sha256_file, within


def rigid_transform(matrix):
    value = np.asarray(matrix, dtype=np.float64)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-4)):
        raise ValueError("Robot base_to_world must be a finite rigid transform")
    return value


class RobotAssets:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.manifest = read_json(self.root / "manifest.json")
        if (self.manifest.get("format") != "mvwd-robot-assets-v1"
                or self.manifest.get("up_axis") != "Z"
                or self.manifest.get("meters_per_unit") != 1):
            raise ValueError("Expected a metric Z-up MVWD robot asset bundle")
        self.manifest_hash = sha256_file(self.root / "manifest.json")
        self.appearances = read_json(self.file("appearances.json"))
        self.parts = read_json(self.file("geometry/parts.json"))["parts"]
        self.mounts = read_json(self.file("camera_mount.json"))["mounts_by_height_m"]

    def file(self, relative):
        path = within(self.root, relative)
        receipt = self.manifest["files"].get(relative)
        if receipt is None or not path.is_file() or path.stat().st_size != receipt["size"]:
            raise ValueError("Missing or incorrect robot asset file: " + relative)
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError("Robot asset SHA256 mismatch: " + relative)
        return path

    def nominal_height(self, height):
        height = float(height)
        if not np.isfinite(height):
            raise ValueError("Non-finite robot camera height")
        nominal = min(map(float, self.mounts), key=lambda x: abs(x - height))
        # Measured PhysX poses have sub-mm deviations from the authored height.
        if abs(nominal - height) > 1e-3:
            raise ValueError("Unsupported robot camera height: " + str(height))
        return nominal

    def check_fingerprint(self, record):
        actual = record.get("provenance", {}).get("robot_fingerprint")
        if actual != self.manifest["template_fingerprint"]:
            raise ValueError("Robot bundle does not match episode robot fingerprint")

    def provenance(self):
        return {"asset_id": self.manifest["asset_id"], "manifest_sha256": self.manifest_hash,
                "template_fingerprint": self.manifest["template_fingerprint"],
                "isaac_asset_release": self.manifest["isaac_asset_release"],
                "color_policy": "GLB PBR base-color factor times embedded base-color texture; no scene lighting",
                "articulation_policy": "nominal mast height; other links in authored rest state",
                "pose_policy": "all planned base_to_world at the current physical frame"}


class MeshAtHeight:
    """One BVH shared by all identities/poses at this mast height."""
    def __init__(self, assets, height):
        try:
            import trimesh
            from trimesh.ray.ray_pyembree import RayMeshIntersector
        except ImportError as exc:
            raise ImportError("Robot rendering requires trimesh==4.5.3 and embreex==2.17.7") from exc
        self.assets = assets
        variant = next(v for v in assets.manifest["variants"]
                       if v["robot_id"] == "robot_00" and abs(v["camera_height_m"] - height) < 1e-6)
        scene = trimesh.load(assets.file(variant["glb"]), force="scene", process=False)
        if len(scene.geometry) != variant["parts"]:
            raise ValueError("Incomplete robot GLB geometry")
        vertices, faces, self.visuals, ends = [], [], [], []
        offset = count = 0
        for part in assets.parts:
            mesh = scene.geometry[part["node"]]
            transform = np.asarray(part["transform"], dtype=np.float64).copy()
            if part["mast"]:
                transform[2, 3] += height - 0.8
            # parts.json is authoritative for the base frame, not viewer axes.
            vertices.append(mesh.vertices @ transform[:3, :3].T + transform[:3, 3])
            faces.append(mesh.faces + offset)
            offset += len(mesh.vertices)
            count += len(mesh.faces)
            ends.append(count)
            accent = any(part["prim_path"] == p or part["prim_path"].startswith(p + "/")
                         for p in assets.appearances["accent_prim_paths"])
            self.visuals.append((mesh.faces, mesh.visual, accent))
        if count != variant["triangles"]:
            raise ValueError("Robot GLB triangle count differs from manifest")
        self.ends = np.asarray(ends)
        self.mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(faces), process=False)
        bounds = variant.get("base_bounds_m")
        if bounds is not None and not np.allclose(self.mesh.bounds, bounds, atol=1e-5):
            raise ValueError("Robot mesh transforms differ from the asset base frame")
        self.intersector = RayMeshIntersector(self.mesh)
        del scene

    def trace(self, origins, directions, robot_id):
        """Return nearest distance and interpolated material RGB; +inf means miss."""
        indices = self.intersector.intersects_first(origins, directions)
        ray = np.flatnonzero(indices >= 0)
        distance = np.full(len(origins), np.inf, dtype=np.float64)
        color = np.zeros((len(origins), 3), dtype=np.float32)
        if not len(ray):
            return distance, color
        face = indices[ray]
        triangle = self.mesh.vertices[self.mesh.faces[face]]
        e1, e2 = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        cross = np.cross(directions[ray], e2)
        det = np.einsum("ij,ij->i", e1, cross)
        valid = abs(det) > 1e-12
        inverse = np.divide(1, det, out=np.zeros_like(det), where=valid)
        delta = origins[ray] - triangle[:, 0]
        u = np.einsum("ij,ij->i", delta, cross) * inverse
        q = np.cross(delta, e1)
        v = np.einsum("ij,ij->i", directions[ray], q) * inverse
        t = np.einsum("ij,ij->i", e2, q) * inverse
        valid &= np.isfinite(t) & (t >= -1e-6)
        ray, face, u, v, t = ray[valid], face[valid], u[valid], v[valid], t[valid]
        distance[ray] = np.maximum(t, 0)
        part_ids = np.searchsorted(self.ends, face, side="right")
        for part_id in np.unique(part_ids):
            mask = part_ids == part_id
            local_face = face[mask] - (self.ends[part_id - 1] if part_id else 0)
            local_faces, visual, accent = self.visuals[part_id]
            if accent:
                rgb = np.asarray(self.assets.appearances["robots"][robot_id]["rgb"], dtype=np.float32)
            elif getattr(visual, "kind", None) == "texture":
                mat = visual.material
                factor = getattr(mat, "baseColorFactor", None)
                rgb = np.asarray(factor[:3] if factor is not None else [255, 255, 255], dtype=np.float32) / 255
                texture = getattr(mat, "baseColorTexture", None)
                if texture is not None:
                    if visual.uv is None:
                        raise ValueError("Textured robot part lacks UV coordinates")
                    uv = visual.uv[local_faces[local_face]]
                    weights = np.stack((1 - u[mask] - v[mask], u[mask], v[mask]), axis=1)
                    uv = (uv * weights[:, :, None]).sum(axis=1)
                    from trimesh.visual.color import uv_to_color
                    rgb = rgb * (uv_to_color(uv, texture)[:, :3].astype(np.float32) / 255)
            else:
                rgba = np.asarray(visual.face_colors)[local_face]
                rgb = rgba[:, :3].astype(np.float32) / 255
            color[ray[mask]] = np.clip(rgb, 0, 1)
        return distance, color


class RobotRayRenderer:
    def __init__(self, root):
        self.assets = RobotAssets(root)
        self._meshes = {}

    def trace(self, origin, directions, transforms, heights, ids, near, far):
        best = np.full(len(directions), np.inf, dtype=np.float64)
        rgb = np.zeros((len(directions), 3), dtype=np.float32)
        identity = np.full(len(directions), -1, dtype=np.int64)
        shifted = np.broadcast_to(origin, directions.shape) + directions * float(near)
        for transform, height, robot in zip(transforms, heights, ids):
            T = rigid_transform(transform)
            name = "robot_%02d" % int(robot)
            if name not in self.assets.appearances["robots"]:
                raise ValueError("Unknown robot identity: " + name)
            nominal = self.assets.nominal_height(height)
            if nominal not in self._meshes:
                self._meshes[nominal] = MeshAtHeight(self.assets, nominal)
            local_origins = (shifted - T[:3, 3]) @ T[:3, :3]
            local_directions = directions @ T[:3, :3]
            distance, colors = self._meshes[nominal].trace(local_origins, local_directions, name)
            distance += float(near)
            take = np.isfinite(distance) & (distance <= far) & (distance < best)
            best[take], rgb[take], identity[take] = distance[take], colors[take], robot
        return best.astype(np.float32), rgb, identity


@lru_cache(maxsize=2)
def robot_renderer(root):
    return RobotRayRenderer(root)


def robot_surfaces(batch, origin, directions):
    """CPU BVH rays; results follow the environment tensor's device and dtype."""
    import torch
    required = ("robot_assets_root", "robot_T_wb", "robot_camera_heights", "robot_ids")
    if not all(k in batch for k in required):
        raise ValueError("Robot rendering requires all robot poses, heights, identities and assets")
    surfaces = []
    for i in range(len(origin)):
        root = batch["robot_assets_root"][i]
        arrays = [batch[k][i].detach().cpu().numpy() for k in ("robot_T_wb", "robot_camera_heights", "robot_ids")]
        if len({len(a) for a in arrays}) != 1:
            raise ValueError("Robot state lengths differ")
        surfaces.append(robot_renderer(root).trace(origin[i].detach().cpu().numpy(),
                        directions[i].detach().cpu().numpy(), *arrays,
                        float(batch["near_m"][i]), float(batch["far_m"][i])))
    return {name: torch.as_tensor(np.stack([s[j] for s in surfaces]), device=origin.device,
                                 dtype=torch.long if name == "id" else origin.dtype)
            for j, name in enumerate(("distance", "rgb", "id"))}
