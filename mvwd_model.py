"""Original CombinedModel training flow with an MVWD batch bridge."""
from __future__ import annotations

import pytorch_lightning as pl
import torch

from cldm.model import create_model, load_state_dict
from mvwd_runtime import original_batch


class CombinedMVWDModel(pl.LightningModule):
    def __init__(self, config, initialize=True):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.learning_rate = config["training"]["learning_rate"]
        self.density_model = create_model(config["model"]["config"])
        self.render_model = create_model(config["model"]["config"])
        for model in (self.density_model, self.render_model):
            model.learning_rate = self.learning_rate
            model.sd_locked = config["model"]["sd_locked"]
            model.only_mid_control = config["model"]["only_mid_control"]
        if initialize:
            state = load_state_dict(config["model"]["init_checkpoint"], location="cpu")
            for name in ("density_model", "render_model"):
                model = getattr(self, name)
                # Upstream initializes each stage from the same ControlNet file.
                result = model.load_state_dict(state, strict=False)
                print(f"{name}: initialization missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    def _step(self, batch, prefix):
        batch = original_batch(batch, self.config)
        # Preserve main.py exactly: density shared_step mutates conditions;
        # its return value/loss is intentionally not added to the render loss.
        self.density_model.shared_step(batch, model_name="density")
        loss, loss_dict = self.render_model.shared_step(batch, model_name="render")
        metrics = {f"{prefix}/{key.split('/', 1)[-1]}": value for key, value in loss_dict.items()}
        # Upstream records /loss before adding depth/color; monitor the returned
        # total without changing the optimization objective.
        metrics[f"{prefix}/total_loss"] = loss.detach()
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=prefix == "train",
                      on_epoch=True, batch_size=len(batch["jpg"]))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint.get("hyper_parameters", {}).get("config", {})
        enabled = bool(self.config.get("data", {}).get("robot_rendering", False))
        previous = bool(saved.get("data", {}).get("robot_rendering", False))
        if enabled != previous:
            raise ValueError("Checkpoint robot conditioning differs from current config; train a new robot run from initialization")
        current = self.config.get("robot_asset_provenance", {}).get("manifest_sha256")
        old = saved.get("robot_asset_provenance", {}).get("manifest_sha256")
        if enabled and (not current or current != old):
            raise ValueError("Checkpoint robot asset manifest differs from the current bundle")

    def configure_optimizers(self):
        # Retain the optimizer and parameter selection in upstream main.py.
        return torch.optim.Adam(list(self.density_model.parameters()) + list(self.render_model.parameters()),
                                lr=self.learning_rate)

    @torch.no_grad()
    def predict_occupancy(self, bev_rgb):
        # Matches p_losses_density: decode encoded source x_start, not denoised
        # model_output. No new occupancy head or training objective is added.
        x = (bev_rgb * 2 - 1).permute(0, 3, 1, 2).contiguous()
        posterior = self.density_model.encode_first_stage(x)
        z = self.density_model.get_first_stage_encoding(posterior)
        return self.density_model.decode_first_stage(z, density_map=True)

    @torch.no_grad()
    def refine(self, coarse_rgb, coarse_depth, steps=50, guidance=9.0):
        size = len(coarse_rgb)
        hint = torch.cat((coarse_rgb, coarse_depth[..., None].repeat(1, 1, 1, 3)), dim=-1).permute(0, 3, 1, 2)
        text = self.render_model.get_learned_conditioning([self.config["prompts"]["refinement"]] * size)
        condition = {"c_concat": [hint], "c_crossattn": [text]}
        kwargs = {}
        if guidance > 1:
            kwargs = {"unconditional_guidance_scale": guidance,
                      "unconditional_conditioning": {"c_concat": [hint], "c_crossattn": [self.render_model.get_unconditional_conditioning(size)]}}
        latent, _ = self.render_model.sample_log(condition, size, ddim=True,
                                                 ddim_steps=steps, eta=0.0, **kwargs)
        return self.render_model.decode_first_stage(latent)[:, :3].clamp(-1, 1)


def load_trained(config, path):
    model = CombinedMVWDModel(config, initialize=False)
    checkpoint = torch.load(path, map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    return model
