"""CLI plumbing, original batch fields and reproducible artifact metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent


def load_config(path):
    import yaml
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    # All relative paths in the shipped config are relative to this repository.
    for group, names in (("data", ("root", "segmentation_root", "cache_root")),
                         ("model", ("config", "init_checkpoint")),
                         ("training", ("output_dir",))):
        for name in names:
            value = config[group].get(name)
            if value:
                value = Path(value).expanduser()
                config[group][name] = str((REPO / value).resolve() if not value.is_absolute() else value)
    return config


def make_dataset(config, split, *, include_targets=True, scenes=None, max_episodes=None,
                 robots=None, frames=None, frame_stride=None):
    from adapters.mvwd_level1 import MVWDLevel1
    values = dict(config["data"])
    root = values.pop("root")
    if robots is not None:
        values["robots"] = robots
    if frame_stride is not None:
        values["frame_stride"] = frame_stride
    return MVWDLevel1(root, split, **values, include_targets=include_targets,
                     scenes=scenes, max_episodes=max_episodes, frames=frames)


def original_batch(batch, config):
    """Bridge to the unmodified loss routines; source/target shapes may differ."""
    import torch
    from adapters.camera_geometry import relative_map
    result = dict(batch)
    bev = batch["bev_rgb"]
    result["jpg"] = bev * 2.0 - 1.0
    result["hint"] = torch.cat((bev, batch["segmentation"]), dim=-1)
    size = len(bev)
    result["txt"] = [config["prompts"]["occupancy"]] * size
    result["render_prompt"] = [config["prompts"]["refinement"]] * size
    result["dataset_name"] = ["mvwd"] * size
    result["perspective_renderer_config"] = config["renderer"]
    if "target_rgb" in batch:
        result["ground_truth"] = batch["target_rgb"]
    if "target_depth_z" in batch:
        depth, valid = align_depth_to_rgb(batch)
        # Keep the original per-image normalized, three-channel GT depth.
        normalized = relative_map(depth, valid)
        result["ground_truth_depth"] = normalized[..., None].repeat(1, 1, 1, 3)
    return result


def align_depth_to_rgb(batch):
    """Reproject camera-forward depth using its own measured K/T, with a z-buffer.

    This is label alignment, not a new loss. Default MVWD RGB/geometry target
    sizes coincide; distinct measured poses are still respected.
    """
    import torch
    depth = batch["target_depth_z"]
    valid = batch["depth_valid"].bool()
    height, width = map(int, batch["target_hw"][0].tolist())
    aligned, masks = [], []
    for i in range(len(depth)):
        z = depth[i].detach().cpu().numpy().astype(np.float64)
        rows, columns = np.mgrid[:z.shape[0], :z.shape[1]]
        pixels = np.stack((columns + .5, rows + .5, np.ones_like(z)), axis=-1)
        kd = batch["K_depth"][i].detach().cpu().numpy().astype(np.float64)
        points = np.einsum("ij,hwj->hwi", np.linalg.inv(kd), pixels) * z[..., None]
        transform = np.linalg.inv(batch["T_wc_rgb"][i].detach().cpu().numpy().astype(np.float64)) @ batch["T_wc_depth"][i].detach().cpu().numpy()
        points = np.einsum("ij,hwj->hwi", transform[:3, :3], points) + transform[:3, 3]
        projected = np.einsum("ij,hwj->hwi", batch["K_rgb"][i].detach().cpu().numpy(), points)
        good = valid[i].detach().cpu().numpy() & np.isfinite(points).all(-1) & (points[..., 2] > 0)
        coords = projected[good, :2] / projected[good, 2:3]
        u, v = np.floor(coords[:, 0]).astype(int), np.floor(coords[:, 1]).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        out = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(out, v[inside] * width + u[inside], points[..., 2][good][inside])
        out = out.reshape(height, width)
        mask = np.isfinite(out)
        # The original loss has no valid-pixel mask. Invalid labels are encoded
        # as zero, explicitly preserving that behavior instead of adding losses.
        out[~mask] = 0
        aligned.append(out)
        masks.append(mask)
    return (torch.as_tensor(np.stack(aligned), device=depth.device),
            torch.as_tensor(np.stack(masks), device=depth.device))


def write_run_metadata(directory, config, datasets):
    from adapters.mvwd_raw import sha256_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"method": "Top2Pano-Persp", "implementation": "original-code-losses-v1",
                "config": config, "datasets": [ds.provenance() for ds in datasets],
                "changes": ["baseline-owned raw adapter", "calibrated perspective renderer",
                            "depth label registration", "non-square output layout"],
                "retained_code_behavior": ["density loss is not added by CombinedModel",
                                           "density decoded from encoded BEV x_start under no_grad",
                                           "histc color loss is non-differentiable",
                                           "refinement predicts clean latent from x_start, as in upstream"]}
    init = Path(config["model"]["init_checkpoint"])
    if init.exists():
        metadata["init_checkpoint_sha256"] = sha256_file(init)
    segmentation = datasets[0].segmentation_root
    if segmentation:
        recipe = segmentation / datasets[0].raw.metadata["release_id"] / "sam_metadata.json"
        if recipe.exists():
            metadata["sam_recipe"] = json.loads(recipe.read_text())
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for ds in datasets:
        source = ds.raw.root / "splits" / f"{ds.raw.split}.jsonl"
        (directory / f"{ds.raw.split}.jsonl").write_bytes(source.read_bytes())
    (directory / "dataset.json").write_bytes(datasets[0].raw.root.joinpath("dataset.json").read_bytes())
