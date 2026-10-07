"""Train the minimally adapted perspective baseline on the pinned MVWD split."""
from __future__ import annotations

import argparse
from pathlib import Path

from mvwd_runtime import REPO, load_config, make_dataset, write_run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/mvwd_level1.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--scene", action="append", help="Train scene filter for debugging; split unchanged")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="Resume a CombinedMVWDModel Lightning checkpoint")
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.root:
        config["data"]["root"] = str(Path(args.root).expanduser().resolve())
    if not args.resume and not Path(config["model"]["init_checkpoint"]).is_file():
        parser.error(f"Missing initialization checkpoint: {config['model']['init_checkpoint']}")
    train = make_dataset(config, "train", scenes=args.scene, max_episodes=args.max_episodes,
                         frame_stride=args.frame_stride)
    datasets = [train]
    validation = None
    if not args.no_validation:
        validation = make_dataset(config, "val", max_episodes=args.max_episodes, frame_stride=args.frame_stride)
        datasets.append(validation)
    # Fail on missing SAM cache before allocating the two diffusion networks.
    train[0]
    if validation is not None:
        validation[0]
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from mvwd_model import CombinedMVWDModel
    settings = config["training"]
    pl.seed_everything(settings["seed"], workers=True)
    model = CombinedMVWDModel(config, initialize=not bool(args.resume))
    write_run_metadata(settings["output_dir"], config, datasets)
    loader_options = dict(batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                          pin_memory=settings["accelerator"] == "gpu")
    loader = torch.utils.data.DataLoader(train, shuffle=True, **loader_options)
    val_loader = torch.utils.data.DataLoader(validation, shuffle=False, **loader_options) if validation is not None else None
    callback = ModelCheckpoint(dirpath=str(Path(settings["output_dir"]) / "checkpoints"),
                               filename="epoch-{epoch:03d}", save_last=True,
                               save_top_k=1 if validation is not None else 0,
                               monitor="val/total_loss" if validation is not None else None,
                               mode="min")
    trainer = pl.Trainer(accelerator=settings["accelerator"], devices=settings["devices"],
                         precision=settings["precision"], max_epochs=settings["max_epochs"],
                         max_steps=args.max_steps if args.max_steps is not None else -1,
                         accumulate_grad_batches=settings["accumulate_grad_batches"],
                         callbacks=[callback], logger=CSVLogger(settings["output_dir"], name="logs"),
                         num_sanity_val_steps=0, log_every_n_steps=1)
    trainer.fit(model, loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
