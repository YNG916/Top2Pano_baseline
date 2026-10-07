"""Build the original ControlNet initialization from SD 2.1 512-base weights.

Copy rule follows lllyasviel/ControlNet/tool_add_control_sd21.py:
control_* <- model.diffusion_*, other keys <- identical SD keys, new layers
keep the repository model's initialization. No trained Top2Pano checkpoint
or architecture/loss modification is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_initial_state(scratch, pretrained):
    target, copied, initialized = {}, [], []
    for name, value in scratch.items():
        source_name = "model.diffusion_" + name[len("control_"):] if name.startswith("control_") else name
        if source_name in pretrained:
            source = pretrained[source_name]
            if source.shape != value.shape:
                raise ValueError(f"Checkpoint shape mismatch for {name}: {tuple(source.shape)} != {tuple(value.shape)}; use SD 2.1 512-base")
            # These tensors are not modified after assembly; avoid a second
            # full CPU clone of multi-GB SD weights.
            target[name] = source
            copied.append(name)
        else:
            target[name] = value
            initialized.append(name)
    return target, copied, initialized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", required=True, help="v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--output", default=str(repo / "models/control_sd21_ini.ckpt"))
    parser.add_argument("--model-config", default=str(repo / "models/cldm_v21.yaml"))
    parser.add_argument("--seed", type=int, default=42, help="Seed for layers absent from SD")
    args = parser.parse_args()
    source, destination = Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Missing SD checkpoint: {source}")
    if destination.exists():
        parser.error(f"Output already exists: {destination}")
    import torch
    from cldm.model import create_model, load_state_dict
    from adapters.mvwd_raw import sha256_file
    torch.manual_seed(args.seed)
    pretrained = load_state_dict(str(source), location="cpu")
    required = "model.diffusion_model.input_blocks.0.0.weight"
    if required not in pretrained:
        parser.error("Expected an original SD checkpoint with model.diffusion_model.* keys, not Diffusers components or a CombinedModel checkpoint")
    # Model creation may fetch the original OpenCLIP pretrained resource on its
    # first use; run on a network-enabled node to populate that cache.
    model = create_model(args.model_config)
    target, copied, initialized = build_initial_state(model.state_dict(), pretrained)
    model.load_state_dict(target, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(destination)
    report = {"source": str(source), "source_sha256": sha256_file(source),
              "output_sha256": sha256_file(destination), "model_config": str(Path(args.model_config).resolve()),
              "seed": args.seed, "copied_key_count": len(copied), "new_key_count": len(initialized),
              "new_keys": initialized,
              "copy_rule_source": "https://github.com/lllyasviel/ControlNet/blob/main/tool_add_control_sd21.py"}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {destination}; copied {len(copied)} keys, initialized {len(initialized)} new keys")
    for name in initialized:
        print(f"Newly initialized: {name}")


if __name__ == "__main__":
    main()
