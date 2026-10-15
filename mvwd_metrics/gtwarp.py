"""GT-constrained symmetric dense-feature consistency; not the official MEt3R."""
from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
from importlib.metadata import version
import numpy as np

from .core import FEATUP_COMMIT, _file_hash, _require_version, images01

METRIC_NAME = "mv_gtwarp_dino"
WEIGHTS = {
    "dino_deitsmall16_pretrain.pth": "1566d50496f27f52f07fea6094fa29b2fdd6fae89da65bdd3ebc3b24ef6b7eb7",
    "dino16_jbu_stack_cocostuff.ckpt": "407dff491058280abfbf38647e9365bf6397bc53d03b13251ed5a6715ef8d1b2",
}


class DenseDINOFeatUp:
    """Official frozen DINO ViT-S/16 + normalized FeatUp, no square resize."""
    def __init__(self, device="cuda", weights_dir="models/metrics"):
        import torch

        _require_version("featup", "0.1.2")
        self.device = device
        self.hub = Path(weights_dir).resolve() / "torch_hub"
        torch.hub.set_dir(str(self.hub))
        # Do not modify the caller's RNG as the upstream extension seeds on import.
        with torch.random.fork_rng():
            self.model = torch.hub.load(
                "mhamilton723/FeatUp:" + FEATUP_COMMIT,
                "dino16", use_norm=True, trust_repo=True).float().to(device).eval()
        self.model.requires_grad_(False)
        self.weight_hashes = {}
        for name, expected in WEIGHTS.items():
            actual = _file_hash(self.hub / "checkpoints" / name)
            if actual != expected:
                raise RuntimeError("Frozen GTWarp weight checksum mismatch: " + name)
            self.weight_hashes[name] = actual
        import featup

        root = Path(inspect.getfile(featup)).parent
        self.source_hashes = {
            str(p.relative_to(root)): _file_hash(p) for p in sorted(root.rglob("*.py"))}
        # Includes backbone, ChannelNorm and learned upsampler after all loads.
        digest = hashlib.sha256()
        for key, tensor in sorted(self.model.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.state_hash = digest.hexdigest()

    def __call__(self, images):
        import torch

        a = images01(images)
        if a.shape[-2] % 16 or a.shape[-1] % 16:
            raise ValueError("DINO16 requires image dimensions divisible by 16; no implicit resize")
        x = torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        with torch.inference_mode(), torch.autocast(device_type=x.device.type, enabled=False):
            # One image at a time bounds memory independently of LPIPS/FID batch size.
            features = torch.cat([self.model((im[None] - mean) / std) for im in x])
        if features.shape != (len(a), 384, *a.shape[-2:]):
            raise ValueError("Unexpected dense DINO feature dimensions")
        if not torch.isfinite(features).all():
            raise ValueError("Non-finite DINO features; evaluation aborted, no pixel reselection")
        return features

    def provenance(self):
        return {
            "feature_backbone": "DINO ViT-S/16, last-layer patch tokens, 384 channels",
            "upsampler": "official FeatUp dino16, use_norm=True",
            "featup_commit": FEATUP_COMMIT,
            "featup_source_sha256": self.source_hashes,
            "weights_sha256": self.weight_hashes,
            "loaded_state_sha256": self.state_hash,
            "preprocessing": "native full FOV, RGB[0,1], ImageNet mean/std; no crop or resize",
            "frozen": True,
            "precision": "float32, autocast disabled",
            "dependencies": {name: version(name)
                             for name in ("torch", "torchvision", "featup", "timm")},
        }


def _directed_feature_error(source, target, matches, chunk_points=8192):
    import torch
    import torch.nn.functional as F

    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0]:
        raise ValueError("Expected two dense CHW feature maps with equal channel count")
    count = len(matches["source_yx"])
    if not count:
        raise ValueError("No GT-visible correspondences; never score empty masks as zero")
    if count != len(matches["target_uv"]) or count != matches["visible_pixels"]:
        raise ValueError("Inconsistent GT correspondence count")
    total = 0.0
    height, width = target.shape[-2:]
    with torch.inference_mode():
        for start in range(0, count, chunk_points):
            yx = torch.as_tensor(matches["source_yx"][start:start + chunk_points],
                                 dtype=torch.long, device=source.device)
            uv = torch.as_tensor(matches["target_uv"][start:start + chunk_points],
                                 dtype=target.dtype, device=target.device)
            # align_corners=False: corner-origin UV maps to 2*u/W-1.
            grid = uv / uv.new_tensor([width, height]) * 2 - 1
            sampled = F.grid_sample(target[None], grid[None, None], mode="bilinear",
                                    padding_mode="zeros", align_corners=False)[0, :, 0].T
            first = source[:, yx[:, 0], yx[:, 1]].T
            if (not torch.isfinite(first).all() or not torch.isfinite(sampled).all()
                    or (first.norm(dim=1) <= 1e-8).any()
                    or (sampled.norm(dim=1) <= 1e-8).any()):
                raise ValueError("Invalid feature vector; no prediction-dependent pixel filtering")
            # Normalize AFTER bilinear sampling; clip cosine for numerical roundoff only.
            distances = 1 - (F.normalize(first, dim=1) * F.normalize(sampled, dim=1)).sum(1).clamp(-1, 1)
            total += distances.double().sum().item()
    return total / count


class MVGTWarpDINOMetric:
    def __init__(self, device="cuda", weights_dir="models/metrics", extractor=None):
        self.device, self.weights_dir = device, weights_dir
        self.extractor = extractor

    def features(self, images):
        if self.extractor is None:
            self.extractor = DenseDINOFeatUp(self.device, self.weights_dir)
        return self.extractor(images)

    def score_features(self, first, second, matches):
        if matches.get("source") != "ground_truth_depth_and_camera":
            raise ValueError("GTWarp requires GT-only correspondences")
        forward = _directed_feature_error(first, second, matches["forward"])
        backward = _directed_feature_error(second, first, matches["backward"])
        return {
            "score": (forward + backward) / 2,
            "forward": forward, "backward": backward,
            "forward_pixels": matches["forward"]["visible_pixels"],
            "backward_pixels": matches["backward"]["visible_pixels"],
        }

    def __call__(self, first, second, matches):
        a, b = images01(first), images01(second)
        if a.shape != b.shape or len(a) != 1:
            raise ValueError("GTWarp accepts one equally-sized view pair at a time")
        features = self.features(np.concatenate((a, b)))
        return self.score_features(features[0], features[1], matches)

    def provenance(self):
        return {
            "implementation": "MVWD GT-constrained feature correspondence error v1; not MEt3R",
            "name": "MV-GTWarp-DINO", "range": [0, 2], "direction": "lower is better",
            "distance": "one minus cosine of vectors normalized after target bilinear sampling",
            "pair_score": "arithmetic mean of two directed pixel means, not pooled pixel mean",
            "geometry": "GT depth and calibrated RGB camera, safe bilinear footprint visibility",
            "surface_scope": "all mutually visible GT surfaces including robots at the same time",
            "features": self.extractor.provenance() if self.extractor else None,
        }
