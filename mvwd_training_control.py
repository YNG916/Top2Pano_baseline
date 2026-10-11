"""Training scheduling/checkpoints only; no changes to model, losses or optimizer."""
from datetime import timedelta
import os
from pathlib import Path
import re
import subprocess


def parse_duration(value):
    """Accept HH:MM:SS, D:HH:MM:SS and Slurm D-HH:MM:SS."""
    parts = value.replace('-', ':').split(':')
    if len(parts) not in (3, 4) or any(not p.isdigit() for p in parts):
        raise ValueError('Time must be HH:MM:SS or D:HH:MM:SS')
    numbers = list(map(int, parts))
    if len(numbers) == 3:
        numbers.insert(0, 0)
    days, hours, minutes, seconds = numbers
    if minutes >= 60 or seconds >= 60:
        raise ValueError('Minutes and seconds must be below 60')
    duration = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    if duration.total_seconds() <= 0:
        raise ValueError('Time must be positive')
    return duration


def time_budget(value, margin_minutes=30):
    if value is None:
        return None
    if value != 'auto':
        return parse_duration(value)
    job = os.environ.get('SLURM_JOB_ID')
    if not job or not re.fullmatch(r'\d+', job):
        raise ValueError('--max-time auto requires a SLURM_JOB_ID inside an allocation')
    try:
        result = subprocess.run(['squeue', '-h', '-j', job, '-o', '%L'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Slurm time-left query timed out; specify an explicit --max-time') from exc
    if result.returncode:
        raise ValueError('Cannot query Slurm time left: ' + result.stderr.strip())
    remaining = result.stdout.strip().splitlines()
    if len(remaining) != 1:
        raise ValueError('Cannot determine a unique Slurm time-left value')
    duration = parse_duration(remaining[0]) - timedelta(minutes=margin_minutes)
    if duration.total_seconds() <= 60:
        raise ValueError('Allocation has too little time left after the safety margin')
    return duration


def checkpoint_callbacks(output, validation, every_steps):
    from pytorch_lightning.callbacks import ModelCheckpoint
    if every_steps < 1:
        raise ValueError('Checkpoint interval must be positive')
    class LatestStepsCheckpoint(ModelCheckpoint):
        def _validate_monitor_key(self, trainer):
            # Lightning 1.5 injects step into monitor_candidates but validates
            # against callback_metrics first. It is a built-in counter, not a loss.
            if self.monitor != 'step':
                super()._validate_monitor_key(trainer)

    directory = Path(output) / 'checkpoints'
    callbacks = [LatestStepsCheckpoint(dirpath=str(directory), filename='step-{step:09d}',
                                 auto_insert_metric_name=False, monitor='step', mode='max',
                                 save_top_k=2, save_last=True, every_n_train_steps=every_steps,
                                 every_n_epochs=0, save_on_train_epoch_end=False)]
    if validation:
        callbacks.append(ModelCheckpoint(dirpath=str(directory / 'best'), filename='best-{step:09d}',
                                         auto_insert_metric_name=False, monitor='val/total_loss', mode='min',
                                         save_top_k=1, save_last=False, every_n_epochs=1,
                                         save_on_train_epoch_end=False))
    return callbacks


def validation_datasets(config, args):
    from mvwd_runtime import make_dataset
    if args.no_validation:
        return []
    stride = args.val_frame_stride if args.val_frame_stride is not None else args.frame_stride
    if args.val_episodes_per_scene is None:
        return [make_dataset(config, 'val', max_episodes=args.max_episodes, frame_stride=stride)]
    from adapters.mvwd_raw import MVWDRaw
    scenes = sorted({r['scene'] for r in MVWDRaw(config['data']['root'], 'val').records})
    return [make_dataset(config, 'val', scenes=[scene], max_episodes=args.val_episodes_per_scene,
                         frame_stride=stride) for scene in scenes]
