"""Protect experiment artifacts and retain failures from child processes."""
import json
import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_mvwd_experiment as runner


def fixture_experiment(tmp_path):
    directory = tmp_path / 'experiment'
    (directory / 'steps').mkdir(parents=True)
    (directory / 'logs').mkdir()
    (directory / 'config.yaml').write_text(yaml.safe_dump({'training': {'output_dir': '/legacy'}}))
    (directory / 'manifest.json').write_text('{}')
    return directory


def test_failure_keeps_combined_log_and_exit_status(tmp_path):
    log = tmp_path / 'run.log'
    status = tmp_path / 'run.json'
    code = runner.capture([sys.executable, '-u', '-c',
                           "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"],
                          log, status, {'name': 'failure'})
    assert code == 7
    assert 'out' in log.read_text() and 'err' in log.read_text()
    record = json.loads(status.read_text())
    assert record['status'] == 'failed' and record['exit_code'] == 7
    assert record['started_utc'] and record['finished_utc']


def test_command_routes_output_without_mutating_base_config(tmp_path):
    directory = fixture_experiment(tmp_path)
    before = (directory / 'config.yaml').read_bytes()
    config, snapshot, output, command = runner.plan_step(directory, 'fit_20', 'train', ['--max-steps', '20'])
    assert config['training']['output_dir'] == str(directory / 'training/fit_20')
    assert command[command.index('--config') + 1] == str(snapshot)
    assert command[-2:] == ['--max-steps', '20']
    _, _, output, command = runner.plan_step(directory, 'sensitivity', 'diagnose', [])
    assert command[command.index('--output') + 1] == str(directory / 'diagnostics/sensitivity')
    assert not output.exists()
    assert (directory / 'config.yaml').read_bytes() == before
    for extra in (['--output=/tmp/wrong'], ['--config', '/tmp/wrong'], ['--mode=all'], ['--out=/tmp/wrong'], ['--conf=/tmp/wrong']):
        with pytest.raises(ValueError):
            runner.plan_step(directory, 'bad', 'infer', extra)


def test_duplicate_step_and_path_escape_are_rejected(tmp_path, monkeypatch):
    directory = fixture_experiment(tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    existing = directory / 'steps/fit.json'
    existing.write_text('original status')
    with pytest.raises(FileExistsError):
        runner.run_step('experiment', 'fit', 'train', [])
    assert existing.read_text() == 'original status'
    for value in ('../escape', 'x/y', '', '.hidden'):
        with pytest.raises(ValueError):
            runner.identifier(value)
