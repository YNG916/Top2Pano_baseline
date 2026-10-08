"""Exercise actual ControlLDM/CombinedModel code, without pretrained weights."""
from pathlib import Path

import pytest
import torch
import yaml


from tests.test_robot_rendering import robot_bundle


@pytest.mark.parametrize("with_robots", [False, True])
def test_original_losses_backward_and_gt_free_refinement(tmp_path, robot_bundle, with_robots):
    pytest.importorskip("pytorch_lightning")
    pytest.importorskip("torchvision")
    pytest.importorskip("omegaconf")
    pytest.importorskip("einops")
    from mvwd_model import CombinedMVWDModel
    from adapters.camera_geometry import PerspectiveRenderer

    torch.set_num_threads(2)
    torch.manual_seed(42)
    repo = Path(__file__).resolve().parents[1]
    architecture = yaml.safe_load((repo / "models/cldm_v21.yaml").read_text())
    params = architecture["model"]["params"]
    params["timesteps"] = 20
    params["cond_stage_config"] = {"target": "tests.model_helpers.FrozenTextFixture"}
    for component in ("control_stage_config", "unet_config"):
        values = params[component]["params"]
        values.update(model_channels=32, num_res_blocks=1, context_dim=32,
                      num_head_channels=8, use_checkpoint=False)
    # Keep the original 128-channel decoder/occupancy/depth-head relationship.
    params["first_stage_config"]["params"]["ddconfig"]["num_res_blocks"] = 1
    model_config = tmp_path / "tiny.yaml"
    model_config.write_text(yaml.safe_dump(architecture))
    config = {"model": {"config": str(model_config), "sd_locked": True, "only_mid_control": False},
              "training": {"learning_rate": 1e-5}, "renderer": {"samples": 8, "chunk_rays": 1024},
              "prompts": {"occupancy": "room geometry", "refinement": "perspective room"}}
    if with_robots:
        from adapters.robot_assets import RobotAssets
        config["data"] = {"robot_rendering": True}
        config["renderer"]["robot_rendering"] = True
        config["robot_asset_provenance"] = RobotAssets(robot_bundle).provenance()
    model = CombinedMVWDModel(config, initialize=False)
    w2p = torch.tensor([[8., 0, 0, 31.5], [0, -8., 0, 31.5], [0, 0, 1., 0], [0, 0, 0, 1.]])[None]
    T = torch.tensor([[1., 0, 0, 0], [0, 0, 1., 0], [0, -1., 0, 1.2], [0, 0, 0, 1.]])[None]
    K = torch.tensor([[80., 0, 64.], [0, 80., 32.], [0, 0, 1.]])[None]
    batch = {"bev_rgb": torch.rand(1, 64, 64, 3), "segmentation": torch.rand(1, 64, 64, 3),
             "bev_valid": torch.ones(1, 64, 64, dtype=torch.bool), "wall_mask": torch.zeros(1, 64, 64, dtype=torch.bool),
             "world_to_bev": w2p, "bev_to_world": torch.linalg.inv(w2p), "floor_z": torch.zeros(1),
             "K_rgb": K, "T_wc_rgb": T, "K_depth": K, "T_wc_depth": T,
             "target_hw": torch.tensor([[64, 128]]), "near_m": torch.tensor([.1]), "far_m": torch.tensor([6.]),
             "target_rgb": torch.rand(1, 64, 128, 3) * 2 - 1,
             "target_depth_z": torch.rand(1, 64, 128) + 1,
             "depth_valid": torch.ones(1, 64, 128, dtype=torch.bool)}
    if with_robots:
        pose = torch.eye(4)[None, None]; pose[0, 0, 1, 3] = 2.
        batch.update(robot_assets_root=[str(robot_bundle)], robot_T_wb=pose,
                     robot_camera_heights=torch.ones(1, 1), robot_ids=torch.zeros(1, 1, dtype=torch.long))
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    # Explicitly verify the actual upstream behavior requested by the user.
    assert all(p.grad is None for p in model.density_model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.render_model.parameters())
    # Also exercise the real Trainer, validation monitor and resumable artifact.
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    loader = DataLoader([{key: (value[0].clone() if isinstance(value, torch.Tensor) else value[0]) for key, value in batch.items()}], batch_size=1)
    callback = ModelCheckpoint(dirpath=str(tmp_path / "checkpoints"), monitor="val/total_loss",
                               save_top_k=1, save_last=True)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=1, max_epochs=1,
                         logger=False, callbacks=[callback], num_sanity_val_steps=0,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, loader, loader)
    assert trainer.global_step == 1 and Path(callback.last_model_path).is_file()
    assert "val/total_loss" in trainer.callback_metrics
    from mvwd_model import load_trained
    model = load_trained(config, callback.last_model_path)
    model.eval()
    query = {key: value for key, value in batch.items()
             if key not in ("target_rgb", "target_depth_z", "depth_valid", "K_depth", "T_wc_depth")}
    with torch.no_grad():
        volume = model.predict_occupancy(query["bev_rgb"])
        coarse = PerspectiveRenderer(samples=8, chunk_rays=1024, robot_rendering=with_robots)(volume, query)
        generated = model.refine(coarse["rgb"], coarse["depth_condition"], steps=4, guidance=2.)
    assert generated.shape == (1, 3, 64, 128)
    assert torch.isfinite(generated).all()


def test_checkpoint_robot_protocol_mismatch_is_rejected():
    from mvwd_model import CombinedMVWDModel
    class Current:
        config = {"data": {"robot_rendering": True}, "robot_asset_provenance": {"manifest_sha256": "current"}}
    with pytest.raises(ValueError, match="conditioning differs"):
        CombinedMVWDModel.on_load_checkpoint(Current(), {"hyper_parameters": {"config": {}}})
    with pytest.raises(ValueError, match="manifest differs"):
        CombinedMVWDModel.on_load_checkpoint(Current(), {"hyper_parameters": {"config": {
            "data": {"robot_rendering": True}, "robot_asset_provenance": {"manifest_sha256": "old"}}}})
