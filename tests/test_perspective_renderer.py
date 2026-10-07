import torch

from adapters.camera_geometry import PerspectiveRenderer, pinhole_rays, relative_map


def room_batch(batch_size=1):
    w2p = torch.tensor([[4., 0, 0, 15.5], [0, -4., 0, 15.5], [0, 0, 1., 0], [0, 0, 0, 1.]])
    T = torch.tensor([[1., 0, 0, 0], [0, 0, 1., 0], [0, -1., 0, 1.5], [0, 0, 0, 1.]])
    K = torch.tensor([[8., 0, 3.5], [0, 8., 2.5], [0, 0, 1.]])
    return {"bev_rgb": torch.rand(batch_size, 32, 32, 3),
            "bev_valid": torch.ones(batch_size, 32, 32, dtype=torch.bool),
            "wall_mask": torch.zeros(batch_size, 32, 32, dtype=torch.bool),
            "world_to_bev": w2p.repeat(batch_size, 1, 1), "bev_to_world": torch.linalg.inv(w2p).repeat(batch_size, 1, 1),
            "floor_z": torch.zeros(batch_size), "K_rgb": K.repeat(batch_size, 1, 1),
            "T_wc_rgb": T.repeat(batch_size, 1, 1), "target_hw": torch.tensor([[5, 7]]).repeat(batch_size, 1),
            "near_m": torch.full((batch_size,), .1), "far_m": torch.full((batch_size,), 6.)}


def test_pinhole_center_and_cv_axes():
    batch = room_batch()
    origin, rays, dc_z = pinhole_rays(batch["K_rgb"], batch["T_wc_rgb"], 5, 7)
    center = 2 * 7 + 3
    torch.testing.assert_close(origin[0], torch.tensor([0., 0, 1.5]))
    torch.testing.assert_close(rays[0, center], torch.tensor([0., 1, 0]))
    assert rays[0, center + 1, 0] > 0
    assert rays[0, center + 7, 2] < 0
    assert dc_z[0, center] == 1 and dc_z[0, center + 1] < 1


def test_depth_is_forward_depth_not_ray_distance():
    batch = room_batch()
    sigma = torch.zeros(1, 32, 32, 32)
    sigma[:, :, 7:9, :] = 200  # A plane perpendicular to camera forward near y=2.
    out = PerspectiveRenderer(samples=300, chunk_rays=9)(sigma, batch, prepared=True)
    middle = out["depth_z"][0, 2]
    # Off-axis distances differ, while forward depths of this wall agree.
    assert float(middle.max() - middle.min()) < .04
    assert 1.6 < float(middle[3]) < 2.1
    assert out["rgb"].shape == (1, 5, 7, 3)
    assert torch.isfinite(out["depth_condition"]).all()


def test_scene_translation_and_floor_offset_do_not_change_rendering():
    batch = room_batch()
    voxel = torch.rand(1, 16, 32, 32)
    renderer = PerspectiveRenderer(samples=40, chunk_rays=11)
    first = renderer(voxel, batch)
    shift = torch.tensor([100., -50., 9.])
    translate = torch.eye(4); translate[:3, 3] = shift
    translated = {**batch, "T_wc_rgb": translate[None] @ batch["T_wc_rgb"],
                  "world_to_bev": batch["world_to_bev"] @ torch.linalg.inv(translate)[None],
                  "bev_to_world": translate[None] @ batch["bev_to_world"], "floor_z": batch["floor_z"] + shift[2]}
    second = renderer(voxel, translated)
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=3e-5, rtol=3e-5)


def test_chunking_batch_origins_and_renderer_gradients():
    batch = room_batch(2)
    batch["T_wc_rgb"][1, 0, 3] = .5
    batch["T_wc_rgb"][1, 2, 3] = 1.
    voxel = torch.rand(2, 12, 32, 32, requires_grad=True)
    one = PerspectiveRenderer(samples=30, chunk_rays=4)(voxel, batch)
    whole = PerspectiveRenderer(samples=30, chunk_rays=1000)(voxel, batch)
    for key in one:
        torch.testing.assert_close(one[key], whole[key])
    one["depth_z"].mean().backward()
    assert torch.isfinite(voxel.grad).all() and voxel.grad.abs().sum() > 0
    assert not torch.allclose(one["depth_z"][0], one["depth_z"][1])


def test_constant_normalization_and_invalid_depth_are_finite():
    values = torch.ones(2, 5, 7)
    valid = torch.zeros_like(values, dtype=torch.bool)
    normalized = relative_map(values, valid)
    assert torch.isfinite(normalized).all() and normalized.sum() == 0
    assert relative_map(values).sum() == 0
