"""Train the minimally adapted perspective baseline on the pinned MVWD split."""
from __future__ import annotations

import argparse
from pathlib import Path

from mvwd_runtime import REPO, load_config, make_dataset, write_run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/mvwd_level1.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--robot-assets")
    parser.add_argument("--scene", action="append", help="Train scene filter for debugging; split unchanged")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--frame-stride", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="Resume a CombinedMVWDModel Lightning checkpoint")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--devices", type=int, help="Number of GPUs visible in this allocation")
    parser.add_argument("--batch-size", type=int, help="Batch size per GPU")
    parser.add_argument("--accumulate-grad-batches", type=int, help="Gradient accumulation; effective batch is per-GPU batch x devices x this")
    parser.add_argument("--max-epochs", type=int, help="Epoch budget; -1 disables the epoch limit")
    parser.add_argument("--max-time", help="HH:MM:SS, D:HH:MM:SS, or auto (Slurm time left minus margin)")
    parser.add_argument("--time-margin-minutes", type=int, default=30)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=500)
    parser.add_argument("--val-every-n-batches", type=int, help="Validation interval in train batches")
    parser.add_argument("--val-episodes-per-scene", type=int, help="Fixed val subset from every held-out scene")
    parser.add_argument("--val-frame-stride", type=int)
    args = parser.parse_args()
    if args.checkpoint_every_n_steps < 1 or args.time_margin_minutes < 1:
        parser.error("Checkpoint interval and time margin must be positive")
    for name in ("val_every_n_batches", "val_episodes_per_scene", "val_frame_stride"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.max_epochs is not None and (args.max_epochs == 0 or args.max_epochs < -1):
        parser.error("max_epochs must be positive or -1")
    config = load_config(args.config)
    for key, value in (("devices", args.devices), ("batch_size", args.batch_size),
                       ("accumulate_grad_batches", args.accumulate_grad_batches)):
        if value is not None:
            if value < 1:
                parser.error(key + " must be positive")
            config["training"][key] = value
    if args.root:
        config["data"]["root"] = str(Path(args.root).expanduser().resolve())
    if args.robot_assets:
        config["data"]["robot_assets_root"] = str(Path(args.robot_assets).expanduser().resolve())
    if not args.resume and not Path(config["model"]["init_checkpoint"]).is_file():
        parser.error(f"Missing initialization checkpoint: {config['model']['init_checkpoint']}")
    train = make_dataset(config, "train", scenes=args.scene, max_episodes=args.max_episodes,
                         frame_stride=args.frame_stride)
    from mvwd_training_control import validation_datasets, checkpoint_callbacks, convergence_callbacks, time_budget
    val_parts = validation_datasets(config, args)
    datasets = [train] + val_parts
    validation = None
    # Fail on missing SAM cache before allocating the two diffusion networks.
    train[0]
    if train.robot_assets is not None:
        config["robot_asset_provenance"] = train.robot_assets.provenance()
    for part in val_parts:
        part[0]
    import torch
    if val_parts:
        validation = torch.utils.data.ConcatDataset(val_parts)
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
    from mvwd_model import CombinedMVWDModel
    settings = config["training"]
    pl.seed_everything(settings["seed"], workers=True)
    model = CombinedMVWDModel(config, initialize=not bool(args.resume))
    config["training_selection"] = vars(args)
    import os
    primary = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if primary:
        write_run_metadata(settings["output_dir"], config, datasets)
    import json
    import sys
    # Query time only after data/model initialization, so startup is included in Slurm accounting.
    try:
        duration = time_budget(args.max_time, args.time_margin_minutes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    control = {"train_queries": len(train), "val_queries": len(validation) if validation is not None else 0,
               "max_time_seconds": duration.total_seconds() if duration is not None else None,
               "checkpoint_every_n_steps": args.checkpoint_every_n_steps,
               "val_every_n_batches": args.val_every_n_batches,
               "max_steps": args.max_steps,
               "max_epochs": args.max_epochs if args.max_epochs is not None else settings["max_epochs"],
               "resume": str(Path(args.resume).resolve()) if args.resume else None,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.executable}
    control.update(devices=settings["devices"], batch_size_per_device=settings["batch_size"],
                   effective_batch_size=settings["devices"] * settings["batch_size"] * settings["accumulate_grad_batches"])
    if primary:
        (Path(settings["output_dir"]) / "training_control.json").write_text(json.dumps(control, indent=2) + "\n")
    print("Training schedule: " + json.dumps(control), flush=True)
    loader_options = dict(batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                          pin_memory=settings["accelerator"] == "gpu")
    loader = torch.utils.data.DataLoader(train, shuffle=True, **loader_options)
    val_loader = torch.utils.data.DataLoader(validation, shuffle=False, **loader_options) if validation is not None else None
    callbacks = checkpoint_callbacks(settings["output_dir"], validation is not None,
                                     args.checkpoint_every_n_steps, monitor=settings.get("monitor", "val/total_loss"))
    callbacks += convergence_callbacks(settings, validation is not None)
    options = {}
    if settings["devices"] > 1:
        from pytorch_lightning.plugins import DDPPlugin
        # Upstream density parameters deliberately receive no gradient.
        options["strategy"] = DDPPlugin(find_unused_parameters=True)
    if duration is not None:
        options.update(max_time=duration, min_steps=0)
    if args.val_every_n_batches is not None and validation is not None:
        options["val_check_interval"] = args.val_every_n_batches
    trainer = pl.Trainer(accelerator=settings["accelerator"], devices=settings["devices"],
                         precision=settings["precision"],
                         max_epochs=args.max_epochs if args.max_epochs is not None else settings["max_epochs"],
                         min_epochs=0 if duration is not None else int(settings.get("min_epochs", 0)),
                         max_steps=args.max_steps if args.max_steps is not None else -1,
                         accumulate_grad_batches=settings["accumulate_grad_batches"],
                         callbacks=callbacks, logger=CSVLogger(settings["output_dir"], name="logs", version=0, flush_logs_every_n_steps=10),
                         num_sanity_val_steps=0, log_every_n_steps=1, **options)
    trainer.fit(model, loader, val_loader, ckpt_path=args.resume)
    # ModelCheckpoint.on_train_end saves last.ckpt with Lightning 1.5's step
    # accounting. Calling save_checkpoint again after fit would add a spurious step.


if __name__ == "__main__":
    main()
