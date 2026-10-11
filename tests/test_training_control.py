from argparse import Namespace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from mvwd_training_control import parse_duration, time_budget, validation_datasets


def test_slurm_remaining_time_and_margin(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '57269')
    monkeypatch.setattr('mvwd_training_control.subprocess.run',
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout='1-13:30:00\n', stderr=''))
    assert time_budget('auto', 30) == timedelta(hours=37)
    assert time_budget('12:00:00') == timedelta(hours=12)
    assert parse_duration('0:00:01') == timedelta(seconds=1)
    monkeypatch.setattr('mvwd_training_control.subprocess.run',
                        lambda *a, **k: SimpleNamespace(returncode=0, stdout='00:20:00\n', stderr=''))
    with pytest.raises(ValueError, match='too little'):
        time_budget('auto', 30)
    for invalid in ('UNLIMITED', '01:99:00', '00:00:00'):
        with pytest.raises(ValueError):
            parse_duration(invalid)


def test_balanced_validation_covers_all_held_out_scenes(monkeypatch):
    monkeypatch.setattr('adapters.mvwd_raw.MVWDRaw',
                        lambda *a: SimpleNamespace(records=[{'scene':'b'}, {'scene':'a'}, {'scene':'c'}, {'scene':'a'}]))
    calls = []
    monkeypatch.setattr('mvwd_runtime.make_dataset', lambda config, split, **kw: calls.append((split, kw)) or kw)
    args = Namespace(no_validation=False, val_frame_stride=20, frame_stride=10,
                     val_episodes_per_scene=4, max_episodes=999)
    parts = validation_datasets({'data':{'root':'unused'}}, args)
    assert [p['scenes'] for p in parts] == [['a'], ['b'], ['c']]
    assert all(p['max_episodes']==4 and p['frame_stride']==20 for p in parts)
    args.no_validation=True
    assert validation_datasets({'data':{}}, args) == []
