"""Create named MVWD experiments and capture every command, log and exit status."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'artifacts/mvwd/experiments'
KINDS = {
    'train': ('train_mvwd.py', 'training'),
    'infer': ('infer_mvwd.py', 'predictions'),
    'diagnose': ('scripts/diagnose_robot_conditions.py', 'diagnostics'),
    'inspect': ('scripts/inspect_mvwd.py', 'diagnostics'),
    'robots': ('scripts/inspect_robot_rendering.py', 'diagnostics'),
    'check': ('scripts/check_mvwd_checkpoint.py', 'evaluations'),
    'sam': ('scripts/prepare_mvwd.py', None),
    'views': ('scripts/prepare_mvwd.py', None),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', value):
        raise ValueError('Names must use 1-120 letters, digits, underscores or hyphens')
    return value


def write_json(path, value):
    # Atomic replacement prevents a reader seeing a half-written status file.
    temp = path.with_name(path.name + '.%s.tmp' % os.getpid())
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def environment():
    return {'python': sys.executable, 'python_version': sys.version,
            'slurm': {k: os.environ[k] for k in
                      ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_NODELIST',
                       'CUDA_VISIBLE_DEVICES') if k in os.environ}}


def git_output(*args):
    result = subprocess.run(['git', *args], cwd=str(REPO), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() if result.returncode == 0 else None


def create_experiment(experiment, config_path, purpose):
    import yaml
    sys.path.insert(0, str(REPO))
    from mvwd_runtime import load_config
    config = load_config(config_path)
    directory = ROOT / identifier(experiment)
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('training', 'predictions', 'diagnostics', 'evaluations', 'logs', 'steps', 'reports'):
        (directory / name).mkdir()
    config['training']['output_dir'] = str(directory / 'training')
    (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    source = Path(config_path).expanduser().resolve()
    (directory / 'source_config.yaml').write_bytes(source.read_bytes())
    write_json(directory / 'manifest.json', {
        'schema_version': 1, 'experiment': experiment, 'purpose': purpose,
        'created_utc': now(), 'source_config': str(source),
        'source_config_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'git_commit': git_output('rev-parse', 'HEAD'),
        'git_status': git_output('status', '--short'), 'environment': environment(),
        'note': 'Step status is recorded separately in steps/*.json; caches are shared.'})
    (directory / 'reports/README.md').write_text(
        '# %s\n\n%s\n\n' % (experiment, purpose) +
        '汇总报告写在本目录；原始命令与状态见 ../steps/，终端日志见 ../logs/。\n')
    print(directory)
    return directory


def plan_step(directory, name, kind, extra):
    import yaml
    identifier(name)
    if any(arg.startswith('--') and any(owned.startswith(arg.split('=', 1)[0])
                                         for owned in ('--config', '--output', '--mode'))
           for arg in extra):
        raise ValueError('The runner owns --config, --output and --mode; edit the experiment config instead')
    config = yaml.safe_load((directory / 'config.yaml').read_text())
    config = copy.deepcopy(config)
    script, group = KINDS[kind]
    output = directory / group / name if group else None
    command = [sys.executable, '-u', str(REPO / script)]
    snapshot = directory / 'steps' / (name + '.config.yaml')
    if kind == 'train':
        config['training']['output_dir'] = str(output)
    if kind != 'check':
        command += ['--config', str(snapshot)]
    if kind in ('infer', 'diagnose', 'inspect', 'robots'):
        command += ['--output', str(output)]
    elif kind == 'check':
        command += ['--output', str(output / 'checkpoint_finiteness.json')]
    elif kind in ('sam', 'views'):
        command += ['--mode', kind]
    command += extra
    return config, snapshot, output, command


def capture(command, log_path, status_path, record):
    process = None
    record.update(status='running', started_utc=now(), environment=environment())
    write_json(status_path, record)
    try:
        with log_path.open('x') as log:
            process = subprocess.Popen(command, cwd=str(REPO), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
        record.update(status='completed' if code == 0 else 'failed', exit_code=code)
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        record.update(status='interrupted', exit_code=130)
        code = 130
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        record.update(status='failed', exit_code=1, error=str(exc))
        code = 1
    record['finished_utc'] = now()
    write_json(status_path, record)
    return code


def run_step(experiment, name, kind, extra):
    import yaml
    directory = ROOT / identifier(experiment)
    if not (directory / 'manifest.json').is_file():
        raise ValueError('Create this experiment first')
    config, snapshot, output, command = plan_step(directory, name, kind, extra)
    status = directory / 'steps' / (name + '.json')
    log = directory / 'logs' / (name + '.log')
    if snapshot.exists() or log.exists() or (output is not None and output.exists()):
        raise FileExistsError('Step/output already exists; use a new step name')
    # Exclusive reservation also prevents two concurrent launches using one name.
    with status.open('x') as stream:
        json.dump({'status': 'reserved', 'name': name}, stream)
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
    # The diagnostic entry point deliberately requires a non-existing output.
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    record = {'name': name, 'kind': kind, 'command': command, 'cwd': str(REPO),
              'config': str(snapshot), 'config_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              'output': str(output) if output is not None else None, 'log': str(log)}
    print('Experiment: %s | Step: %s | Log: %s' % (experiment, name, log), flush=True)
    return capture(command, log, status, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    create = sub.add_parser('create', help='Create a new named experiment without starting any training')
    create.add_argument('experiment', help='e.g. 20261009_control_condition_audit')
    create.add_argument('--config', default=str(REPO / 'configs/mvwd_level1.yaml'))
    create.add_argument('--purpose', required=True)
    run = sub.add_parser('run', help='Run one step in the current interactive allocation')
    run.add_argument('experiment')
    run.add_argument('--name', required=True, help='Unique within this experiment')
    run.add_argument('--kind', required=True, choices=sorted(KINDS))
    args, extra = parser.parse_known_args()
    if extra[:1] == ['--']:
        extra = extra[1:]
    try:
        if args.action == 'create':
            if extra:
                parser.error('Unexpected arguments: ' + ' '.join(extra))
            create_experiment(args.experiment, args.config, args.purpose)
            return 0
        return run_step(args.experiment, args.name, args.kind, extra)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
