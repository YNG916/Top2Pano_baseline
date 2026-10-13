"""Known assets must respond to poses and obey predicted scene occlusion."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from adapters.camera_geometry import PerspectiveRenderer
from adapters.mvwd_raw import sha256_file
from adapters.robot_assets import RobotAssets, robot_renderer
from tests.test_perspective_renderer import room_batch


@pytest.fixture
def robot_bundle(tmp_path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    root = tmp_path / "robots"
    (root / "geometry").mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[.4, .2, 1.])
    mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(baseColorFactor=[230, 26, 26, 255]))
    transform = np.eye(4); transform[2, 3] = 1.5
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="part_000", node_name="part_000", transform=transform)
    (root / "geometry/body.glb").write_bytes(scene.export(file_type="glb"))
    (root / "geometry/parts.json").write_text(json.dumps({"parts": [{"node": "part_000", "prim_path": "accent",
        "material": "accent", "transform": transform.tolist(), "mast": False}]}))
    colors = [[.9, .1, .1], [.1, .1, .9], [.1, .8, .1]]
    (root / "appearances.json").write_text(json.dumps({"accent_prim_paths": ["accent"],
        "robots": {f"robot_{i:02d}": {"rgb": color, "material_prim": f"accent_{i}"} for i, color in enumerate(colors)}}))
    (root / "camera_mount.json").write_text(json.dumps({"mounts_by_height_m": {"1.0": {}}}))
    files = {str(p.relative_to(root)): {"size": p.stat().st_size, "sha256": sha256_file(p)}
             for p in root.rglob("*") if p.is_file()}
    manifest = {"format": "mvwd-robot-assets-v1", "asset_id": "fixture", "isaac_asset_release": "fixture",
        "template_fingerprint": "fixture-fingerprint", "meters_per_unit": 1, "up_axis": "Z", "files": files,
        "variants": [{"robot_id": "robot_00", "camera_height_m": 1., "glb": "geometry/body.glb",
                      "parts": 1, "triangles": len(mesh.faces)}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def bodies(batch, root, positions):
    poses = torch.eye(4).repeat(1, len(positions), 1, 1)
    for i, xyz in enumerate(positions):
        poses[0, i, :3, 3] = torch.tensor(xyz)
    return {**batch, "robot_assets_root": [str(root)], "robot_T_wb": poses,
            "robot_camera_heights": torch.ones(1, len(positions)),
            "robot_ids": torch.arange(len(positions))[None]}


def test_closest_body_identity_and_far_clip(robot_bundle):
    renderer = robot_renderer(str(robot_bundle))
    poses = np.tile(np.eye(4), (2, 1, 1)); poses[:, 1, 3] = [2., 3.]
    args = (np.array([0., 0., 1.5]), np.array([[0., 1., 0.]]), poses, [1., 1.], [0, 1], .1, 6.)
    distance, color, identity = renderer.trace(*args)
    np.testing.assert_allclose(distance, [1.9], atol=1e-5)
    np.testing.assert_allclose(color, [[.9, .1, .1]], atol=1e-5)
    assert identity.tolist() == [0]
    reversed_args = (*args[:2], poses[::-1], [1., 1.], [1, 0], .1, 6.)
    for a, b in zip(renderer.trace(*reversed_args), (distance, color, identity)):
        np.testing.assert_allclose(a, b)
    assert renderer.trace(*args[:-1], 1.)[2].tolist() == [-1]


def test_other_robot_pose_changes_fixed_camera_projection(robot_bundle):
    first = bodies(room_batch(), robot_bundle, [(0., 2., 0.)])
    renderer = PerspectiveRenderer(samples=80, chunk_rays=7, robot_rendering=True)
    empty = torch.zeros(1, 16, 32, 32)
    a = renderer(empty, first, prepared=True)
    second = bodies(first, robot_bundle, [(1., 2., 0.)])
    b = renderer(empty, second, prepared=True)
    assert a["robot_id"][0, 2, 3] == 0
    assert b["robot_id"][0, 2, 3] == -1
    assert not torch.allclose(a["rgb"], b["rgb"])
    assert float(a["depth_z"][0, 2, 3]) == pytest.approx(1.9, abs=1e-5)
    assert float(a["robot_weight"][0, 2, 3]) == pytest.approx(1.)


def test_predicted_wall_occludes_body_without_gt_depth(robot_bundle):
    batch = bodies(room_batch(), robot_bundle, [(0., 2., 0.)])
    sigma = torch.zeros(1, 16, 32, 32)
    sigma[:, :, 10:13, :] = 1000.  # y approximately .6..1.4, in front of the robot.
    out = PerspectiveRenderer(samples=160, robot_rendering=True)(sigma, batch, prepared=True)
    assert out["robot_id"][0, 2, 3] == 0  # Mesh alone intersects the ray.
    assert float(out["robot_weight"][0, 2, 3]) < 1e-6
    assert float(out["depth_z"][0, 2, 3]) < 1.5
    assert "target_depth_z" not in batch


def test_body_stops_environment_behind_it_and_survives_short_rgb_range(robot_bundle):
    batch = bodies(room_batch(), robot_bundle, [(0., 2., 0.)])
    sigma = torch.zeros(1, 16, 32, 32)
    sigma[:, :, 2:4, :] = 1000.  # Far wall behind the robot.
    renderer = PerspectiveRenderer(samples=120, robot_rendering=True, rgb_distance_normalized=.1)
    out = renderer(sigma, batch, prepared=True)
    assert float(out["depth_z"][0, 2, 3]) == pytest.approx(1.9, abs=1e-5)
    assert float(out["robot_weight"][0, 2, 3]) == pytest.approx(1.)
    assert out["rgb"][0, 2, 3, 0] > out["rgb"][0, 2, 3, 1]
    assert torch.isfinite(out["depth_condition"]).all()


def test_robot_scene_translation_and_chunking_preserve_result(robot_bundle):
    batch = bodies(room_batch(), robot_bundle, [(0., 2., 0.)])
    sigma = torch.zeros(1, 16, 32, 32)
    r = PerspectiveRenderer(samples=50, chunk_rays=3, robot_rendering=True)
    a = r(sigma, batch, prepared=True)
    b = PerspectiveRenderer(samples=50, chunk_rays=1000, robot_rendering=True)(sigma, batch, prepared=True)
    translation = torch.eye(4); translation[:3, 3] = torch.tensor([10., -20., 5.])
    moved = {**batch, "T_wc_rgb": translation[None] @ batch["T_wc_rgb"],
        "robot_T_wb": translation[None, None] @ batch["robot_T_wb"],
        "world_to_bev": batch["world_to_bev"] @ torch.linalg.inv(translation)[None],
        "bev_to_world": translation[None] @ batch["bev_to_world"], "floor_z": batch["floor_z"] + 5.}
    c = r(sigma, moved, prepared=True)
    for key in a:
        torch.testing.assert_close(a[key], b[key])
        torch.testing.assert_close(a[key], c[key], atol=1e-5, rtol=1e-5)


def test_assets_fail_on_corruption_and_unsupported_height(robot_bundle):
    assets = RobotAssets(robot_bundle)
    assert assets.nominal_height(1.0000008) == 1.
    with pytest.raises(ValueError, match="Unsupported"):
        assets.nominal_height(1.1)
    with pytest.raises(ValueError, match="fingerprint"):
        assets.check_fingerprint({"provenance": {"robot_fingerprint": "different"}})
    p = robot_bundle / "geometry/body.glb"
    raw = bytearray(p.read_bytes()); raw[-1] ^= 1; p.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA256"):
        assets.file("geometry/body.glb")


@pytest.mark.parametrize("include_targets", [False, True])
def test_adapter_reads_all_robots_when_only_one_view_is_requested(tmp_path, robot_bundle, include_targets):
    from tests.test_mvwd_level1 import payload as payload_fixture
    from adapters.mvwd_level1 import MVWDLevel1
    # Reuse the small pinned dataset, adding three known planned trajectories.
    root, segmentation, record = payload_fixture.__wrapped__(tmp_path)
    record["robots"] = 3
    record["provenance"] = {"robot_fingerprint": "fixture-fingerprint"}
    split = root / "splits/train.jsonl"
    split.write_text(json.dumps(record) + "\n")
    meta = json.loads((root / "dataset.json").read_text())
    meta["checksums"]["splits/train.jsonl"] = sha256_file(split)
    (root / "dataset.json").write_text(json.dumps(meta))
    ep = root / record["episode_path"]
    observations = json.loads((ep / "observations_before.json").read_text())
    base = np.tile(np.eye(4), (4, 1, 1))
    trajectories = {}
    all_observations = []
    for robot in range(3):
        pose = base.copy(); pose[:, 0, 3] = robot; pose[:, 1, 3] = np.arange(4) * .1
        trajectories[f"robot_{robot:02d}_base_to_world"] = pose
        for old in observations:
            camera = {**old["camera"], "camera_height_m": 1.0000005}
            camera_poses = dict(camera["modality_camera_to_world"])
            depth_pose = np.asarray(camera_poses["depth_linear"]).copy()
            depth_pose[0, 3] += .125  # Depth uses its own camera transform.
            camera_poses["depth_linear"] = depth_pose.tolist()
            camera["modality_camera_to_world"] = camera_poses
            all_observations.append({**old, "robot_id": f"robot_{robot:02d}", "camera": camera})
    np.savez(ep / "trajectories.npz", **trajectories)
    (ep / "observations_before.json").write_text(json.dumps(all_observations))
    if not include_targets:
        (ep / "robot_views/before/robot_00.npz").unlink()
    ds = MVWDLevel1(root, bev_size=64, target_size=(64, 128), robots=(0,), frames=[3],
                   segmentation_root=segmentation, include_targets=include_targets,
                   robot_rendering=True, robot_assets_root=robot_bundle)
    sample = ds[0]
    assert sample["robot_id"] == 0 and sample["robot_ids"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(sample["robot_T_wb"][:, 0, 3], [0, 1, 2])
    np.testing.assert_allclose(sample["robot_T_wb"][:, 1, 3], [.3, .3, .3])
    if include_targets:
        assert sample["target_rgb"].shape == (64, 128, 3)
        assert sample["target_depth_z"].shape == (64, 128)
        np.testing.assert_allclose(sample["T_wc_rgb"][:3, 3], [1, 2, 1])
        np.testing.assert_allclose(sample["T_wc_depth"][:3, 3], [1.125, 2, 1])
    else:
        assert "target_rgb" not in sample and "target_depth_z" not in sample


def test_embedded_texture_uv_and_material_factor(robot_bundle):
    import trimesh
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    texture = Image.fromarray(np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8))
    mesh = trimesh.Trimesh(vertices=[[-1, 2, 1], [1, 2, 1], [1, 2, 2], [-1, 2, 2]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
        material=PBRMaterial(baseColorFactor=[128, 255, 255, 255], baseColorTexture=texture))
    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="part_000", node_name="part_000")
    (robot_bundle / "geometry/body.glb").write_bytes(scene.export(file_type="glb"))
    parts = json.loads((robot_bundle / "geometry/parts.json").read_text())
    parts["parts"][0]["transform"] = np.eye(4).tolist()
    (robot_bundle / "geometry/parts.json").write_text(json.dumps(parts))
    appearance = json.loads((robot_bundle / "appearances.json").read_text())
    appearance["accent_prim_paths"] = []
    (robot_bundle / "appearances.json").write_text(json.dumps(appearance))
    manifest = json.loads((robot_bundle / "manifest.json").read_text())
    manifest["variants"][0]["triangles"] = 2
    for name, receipt in manifest["files"].items():
        p = robot_bundle / name
        receipt.update(size=p.stat().st_size, sha256=sha256_file(p))
    (robot_bundle / "manifest.json").write_text(json.dumps(manifest))
    origin = np.array([0., 0., 1.5])
    directions = np.array([[-.8, 2., 0.], [.8, 2., 0.]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distance, rgb, ids = robot_renderer(str(robot_bundle)).trace(
        origin, directions, np.eye(4)[None], [1.], [0], .1, 6.)
    assert np.isfinite(distance).all() and ids.tolist() == [0, 0]
    np.testing.assert_allclose(rgb, [[128 / 255, 0, 0], [0, 0, 1]], atol=1e-5)


def test_perspective_far_plane_uses_forward_depth(robot_bundle):
    renderer = robot_renderer(str(robot_bundle))
    pose = np.eye(4)[None]
    pose[0, :2, 3] = [1.45, 2.]
    origin = np.array([0., 0., 1.5])
    directions = np.array([[.6, .8, 0.]])
    args = (origin, directions, pose, [1.], [0], .1, 1.95)
    assert renderer.trace(*args)[2].tolist() == [-1]  # Historical radial range.
    distance, _, ids = renderer.trace(*args, camera_z=np.array([.8]))
    assert ids.tolist() == [0]
    assert float(distance[0] * .8) == pytest.approx(1.9, abs=1e-5)


def test_perspective_near_origin_lies_on_clip_plane(robot_bundle, monkeypatch):
    renderer = robot_renderer(str(robot_bundle))
    pose = np.eye(4)[None]
    origin = np.array([0., 0., 1.5])
    directions = np.array([[.6, .8, 0.], [0., 1., 0.]])
    renderer.trace(origin, directions, pose, [1.], [0], .1, 6.)
    def capture(origins, rays, identity):
        np.testing.assert_allclose(origins[:, 1], [.1, .1], atol=1e-8)
        np.testing.assert_allclose(origins[:, 0], [.075, 0.], atol=1e-8)
        return np.full(2, np.inf), np.zeros((2, 3))
    monkeypatch.setattr(renderer._meshes[1.], 'trace', capture)
    renderer.trace(origin, directions, pose, [1.], [0], .1, 6., camera_z=np.array([.8, 1.]))


def test_asset_visibility_is_specific_to_native_shell():
    from adapters.robot_assets import part_is_hidden
    white = {"prim_path": "chassis_link/visual/top_body", "material": "/Looks/Alabaster"}
    gray = {**white, "material": "/Looks/Gray"}
    head = {**white, "prim_path": "mast_carriage/sensor_head"}
    assert not part_is_hidden(white, "all")
    assert part_is_hidden(white, "hide_white_top_body")
    assert not part_is_hidden(gray, "hide_white_top_body")
    assert not part_is_hidden(head, "hide_white_top_body")
    assert part_is_hidden(gray, "hide_top_body")
    with pytest.raises(ValueError):
        part_is_hidden(white, "unknown")


def test_visibility_reveals_identity_deck_without_editing_assets(robot_bundle):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    scene = trimesh.Scene()
    parts = []
    for index, (y, color, prim, material) in enumerate([
        (1., [255, 255, 255, 255], "chassis_link/visual/top_body", "/Looks/Alabaster"),
        (2., [230, 26, 26, 255], "accent", "accent"),
    ]):
        mesh = trimesh.creation.box(extents=[.4, .2, 1.])
        mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(baseColorFactor=color))
        transform = np.eye(4); transform[:3, 3] = [0, y, 1.5]
        node = "part_%03d" % index
        scene.add_geometry(mesh, geom_name=node, node_name=node, transform=transform)
        parts.append({"node": node, "prim_path": prim, "material": material,
                      "transform": transform.tolist(), "mast": False})
    (robot_bundle / "geometry/body.glb").write_bytes(scene.export(file_type="glb"))
    (robot_bundle / "geometry/parts.json").write_text(json.dumps({"parts": parts}))
    manifest = json.loads((robot_bundle / "manifest.json").read_text())
    manifest["variants"][0].update(parts=2, triangles=24)
    for name in ["geometry/body.glb", "geometry/parts.json"]:
        path = robot_bundle / name
        manifest["files"][name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    (robot_bundle / "manifest.json").write_text(json.dumps(manifest))
    snapshot = {p: p.read_bytes() for p in robot_bundle.rglob("*") if p.is_file()}
    args = (np.array([0., 0., 1.5]), np.array([[0., 1., 0.]]), np.eye(4)[None], [1.], [0], .1, 6.)
    distance, color, identity = robot_renderer(str(robot_bundle)).trace(*args, camera_z=np.ones(1))
    np.testing.assert_allclose(distance, [.9], atol=1e-5)
    np.testing.assert_allclose(color, [[1., 1., 1.]])
    assert identity.tolist() == [0]
    for policy in ["hide_white_top_body", "hide_top_body"]:
        renderer = robot_renderer(str(robot_bundle), policy)
        distance, color, identity = renderer.trace(*args, camera_z=np.ones(1))
        np.testing.assert_allclose(distance, [1.9], atol=1e-5)
        np.testing.assert_allclose(color, [[.9, .1, .1]], atol=1e-5)
        assert identity.tolist() == [0]
        assert renderer._meshes[1.].source_triangles == 24
        assert renderer._meshes[1.].visible_triangles == 12
        batch = bodies(room_batch(), robot_bundle, [(0., 0., 0.)])
        coarse = PerspectiveRenderer(samples=20, robot_rendering=True,
                                     robot_visibility_policy=policy)(torch.zeros(1, 16, 32, 32), batch, prepared=True)
        assert float(coarse["depth_z"][0, 2, 3]) == pytest.approx(1.9, abs=1e-5)
        assert float(coarse["robot_weight"][0, 2, 3]) == pytest.approx(1.)
    assert all(p.read_bytes() == data for p, data in snapshot.items())
