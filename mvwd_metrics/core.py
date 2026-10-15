"""Level-1 metrics: official learned backends and episode-cluster aggregation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import importlib.metadata
import inspect
import hashlib
import math

import numpy as np

MET3R_COMMIT = "ee0e1752898559e1a3e85e2e151d3edeb9b55f73"
FEATUP_COMMIT = "6b5a6c0e91f75e69194807128dcbc39c3084a30d"


def images01(images):
    """Explicit NCHW RGB float [0,1]; never silently normalize or clip inputs."""
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    images = np.asarray(images)
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or not np.issubdtype(images.dtype, np.floating)
    ):
        raise ValueError("Expected floating-point NCHW RGB in [0,1]")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Images must be finite and in [0,1]")
    return images.astype(np.float32, copy=False)


def _paired(a, b):
    a, b = images01(a), images01(b)
    if a.shape != b.shape:
        raise ValueError("Prediction and GT image shapes differ")
    return a, b


def _package(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_version(name, expected):
    actual = _package(name)
    if actual != expected:
        raise RuntimeError(
            "Frozen metric protocol requires %s==%s (found %s); use requirements_metrics.txt"
            % (name, expected, actual)
        )


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor(images, device):
    import torch

    return torch.from_numpy(np.ascontiguousarray(images01(images))).to(device)


class UndefinedMetricError(RuntimeError):
    """An official estimator cannot assign a scientifically valid value."""


class PSNRMetric:
    def __call__(self, predictions, targets):
        a, b = _paired(predictions, targets)
        mse = np.square(a.astype(np.float64) - b.astype(np.float64)).mean((1, 2, 3))
        with np.errstate(divide="ignore"):
            return -10 * np.log10(mse)

    def provenance(self):
        return {
            "definition": "-10 log10(mean RGB squared error)",
            "data_range": 1,
            "aggregation": "mean per-image PSNR, then seed/episode mean; not PSNR of pooled MSE",
        }


class LPIPSMetric:
    def __init__(self, device="cuda"):
        _require_version("lpips", "0.1.4")
        import lpips

        self.device = device
        self.model = (
            lpips.LPIPS(
                net="alex",
                version="0.1",
                spatial=False,
                pretrained=True,
                pnet_rand=False,
                verbose=False,
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)

    def __call__(self, predictions, targets):
        import torch

        a, b = _paired(predictions, targets)
        with torch.inference_mode():
            value = self.model(
                _tensor(a, self.device) * 2 - 1, _tensor(b, self.device) * 2 - 1
            )
        return value.flatten().cpu().numpy().astype(np.float64)

    def provenance(self):
        import lpips

        calibration = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
        return {
            "implementation": "official lpips",
            "package_version": _package("lpips"),
            "calibration_sha256": _file_hash(calibration),
            "net": "alex",
            "version": "0.1",
            "input_range": [-1, 1],
            "spatial": False,
            "pretrained_trunk_and_calibration": True,
        }


class FeatureMoments:
    """Mergeable float64 central moments; no images or all-test features retained."""

    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def update(self, features):
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected nonempty finite N x D features")
        n, mean = len(x), x.mean(0)
        centered = x - mean
        m2 = centered.T @ centered
        if not self.count:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        if mean.shape != self.mean.shape:
            raise ValueError("Feature dimension changed")
        delta = mean - self.mean
        total = self.count + n
        self.m2 += m2 + np.outer(delta, delta) * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def statistics(self):
        if self.count < 2:
            raise ValueError("FID requires at least two images in each population")
        return self.mean, self.m2 / (self.count - 1)


class FIDMetric:
    def __init__(self, device="cuda", weights_dir="models/metrics"):
        import torch

        _require_version("clean-fid", "0.1.35")
        from cleanfid.inception_torchscript import InceptionV3W
        from cleanfid.resize import build_resizer

        self.device = device
        self.weights_dir = Path(weights_dir).resolve()
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.model = (
            InceptionV3W(str(self.weights_dir), download=True, resize_inside=False)
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.resize = build_resizer("clean")

    def features(self, images):
        import torch

        # Match Clean-FID ResizeDataset: uint8 RGB -> per-channel PIL float bicubic.
        pixels = np.rint(images01(images).transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        batch = np.stack([self.resize(x).transpose(2, 0, 1) for x in pixels])
        with torch.inference_mode():
            feats = self.model(torch.from_numpy(batch).to(self.device))
        return feats.cpu().numpy().astype(np.float64)

    @staticmethod
    def distance(first, second):
        from cleanfid.fid import frechet_distance

        a, ac = first.statistics()
        b, bc = second.statistics()
        score = float(frechet_distance(a, ac, b, bc))
        if not np.isfinite(score) or score < -1e-3:
            raise ValueError("Invalid numerical FID: %r" % score)
        return max(
            0.0, score
        )  # Floating-point roundoff can make identical populations slightly negative.

    def provenance(self):
        weights = self.weights_dir / "inception-2015-12-05.pt"

        return {
            "implementation": "official clean-fid",
            "package_version": _package("clean-fid"),
            "mode": "clean",
            "model": "inception_v3",
            "features": 2048,
            "resize": "official clean PIL float bicubic 299x299, no requantization",
            "weights_sha256": _file_hash(weights),
            "aggregation": "one whole-population FID per seed, then arithmetic seed mean; no episode FID",
        }


class MEt3RMetric:
    def __init__(self, device="cuda", image_size=256):
        import torch

        if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) < (
            2,
            1,
        ):
            raise RuntimeError(
                "Official MEt3R requires PyTorch >=2.1; use the separate metrics environment"
            )
        for package, version in (
            ("met3r", "1.0.1"),
            ("pytorch3d", "0.7.7"),
            ("featup", "0.1.2"),
        ):
            _require_version(package, version)
        from met3r import MEt3R

        source = Path(inspect.getfile(MEt3R))
        # Pin implementation, including local/editable installs; do not silently use a changed API.
        import subprocess

        repo = source.parent.parent
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if revision.returncode or revision.stdout.strip() != MET3R_COMMIT:
            raise RuntimeError(
                "Install the pinned editable official MEt3R commit: " + MET3R_COMMIT
            )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                "--exit-code",
                "HEAD",
                "--",
                "met3r",
                "mast3r",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if status.returncode:
            raise RuntimeError(
                "MEt3R source/submodule differs from the pinned checkout"
            )
        self.device, self.image_size = device, image_size
        self.model = (
            MEt3R(
                img_size=image_size,
                use_norm=True,
                backbone="mast3r",
                feature_backbone="dino16",
                feature_backbone_weights="mhamilton723/FeatUp:" + FEATUP_COMMIT,
                upsampler="featup",
                distance="cosine",
                freeze=True,
            )
            .to(device)
            .eval()
        )
        self.source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        self.repo = repo

    def __call__(self, first, second):
        import torch
        import torch.nn.functional as F

        a, b = _paired(first, second)
        # Fixed full-FOV resize for non-square target views; no center crop or GT warping.
        x = torch.stack((_tensor(a, self.device), _tensor(b, self.device)), dim=1)
        shape = x.shape
        x = F.interpolate(
            x.flatten(0, 1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = x.reshape(shape[0], 2, 3, self.image_size, self.image_size) * 2 - 1
        with torch.inference_mode():
            scores, masks = self.model(images=x, return_overlap_mask=True)
        overlap_pixels = masks.flatten(1).sum(1)
        if not torch.isfinite(scores).all() or not (overlap_pixels > 0).all():
            raise UndefinedMetricError(
                "Official MEt3R invalid reconstruction: scores=%s, overlap_pixels=%s; not scored as zero"
                % (
                    scores.flatten().detach().cpu().tolist(),
                    overlap_pixels.detach().cpu().tolist(),
                )
            )
        return scores.flatten().cpu().numpy().astype(np.float64)

    def provenance(self):
        return {
            "implementation": "official met3r",
            "package_version": _package("met3r"),
            "commit": MET3R_COMMIT,
            "source_sha256": self.source_hash,
            "dependencies": {
                name: _package(name)
                for name in (
                    "torch",
                    "torchvision",
                    "pytorch3d",
                    "featup",
                    "timm",
                    "huggingface-hub",
                )
            },
            "backbone": "mast3r",
            "geometry_weights": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            "feature_backbone": "dino16",
            "featup_commit": FEATUP_COMMIT,
            "upsampler": "featup",
            "distance": "cosine",
            "image_size": self.image_size,
            "preprocessing": "full FOV bilinear antialiased square resize; [-1,1]",
            "pair_order": "ascending observer IDs, official directed score",
            "gt_geometry_use": "pair eligibility only, never substituted into official MEt3R",
        }


class EpisodeMetricAggregator:
    """Units -> seed means -> equal-weight episodes; bootstrap whole episodes."""

    def __init__(self, seeds, bootstrap_samples=2000, bootstrap_seed=20261015):
        self.seeds = tuple(seeds)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Choose distinct nonempty seeds")
        if bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        self.samples, self.seed = bootstrap_samples, bootstrap_seed
        self.values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    def add(self, metric, episode, seed, unit, value):
        value = float(value)
        if (
            seed not in self.seeds
            or math.isnan(value)
            or value == -math.inf
            or (not math.isfinite(value) and metric != "psnr")
        ):
            raise ValueError("Invalid metric value/seed")
        values = self.values[metric][str(episode)][seed]
        if unit in values:
            raise ValueError("Duplicate metric unit")
        values[unit] = value

    def summarize(self, metric):
        episodes = []
        for episode, by_seed in sorted(self.values[metric].items()):
            if set(by_seed) != set(self.seeds):
                raise ValueError("Seeds do not cover the same episodes")
            reference = set(by_seed[self.seeds[0]])
            if any(set(units) != reference for units in by_seed.values()):
                raise ValueError("Seeds do not cover the same metric units")
            means = {
                str(seed): float(np.mean(list(by_seed[seed].values())))
                for seed in self.seeds
            }
            episodes.append(
                {
                    "episode_id": episode,
                    "mean": float(np.mean(list(means.values()))),
                    "seed_means": means,
                    "units_per_seed": len(reference),
                }
            )
        if not episodes:
            return {
                "mean": None,
                "ci95": None,
                "episode_count": 0,
                "episodes": [],
                "status": "no_eligible_units",
            }
        values = np.asarray([r["mean"] for r in episodes])
        interval = None
        if len(values) > 1 and np.isfinite(values).all():
            rng = np.random.default_rng(self.seed)
            # Chunk to bound bootstrap memory for full datasets.
            boot = np.concatenate(
                [
                    values[
                        rng.integers(
                            len(values), size=(min(100, self.samples - i), len(values))
                        )
                    ].mean(1)
                    for i in range(0, self.samples, 100)
                ]
            )
            interval = np.quantile(boot, [0.025, 0.975]).tolist()
        return {
            "mean": float(values.mean()),
            "ci95": interval,
            "episode_count": len(episodes),
            "episodes": episodes,
            "status": "ok",
            "ci_unit": "episode",
            "ci_note": (
                None
                if interval is not None
                else "CI unavailable with one episode or infinite PSNR"
            ),
        }
