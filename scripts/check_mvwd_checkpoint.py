"""Check all saved model and optimizer tensors without constructing networks."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(path, map_location='cpu')
    counts = Counter()
    invalid = []

    def visit(value, location):
        if isinstance(value, torch.Tensor):
            counts['tensors'] += 1
            counts['elements'] += value.numel()
            if not (value.is_floating_point() or value.is_complex()):
                return
            counts['floating_tensors'] += 1
            flat = value.detach().reshape(-1)
            bad = 0
            # Bound temporary boolean allocations even for large matrices.
            for start in range(0, flat.numel(), 1_048_576):
                bad += int((~torch.isfinite(flat[start:start + 1_048_576])).sum())
            if bad:
                invalid.append({'path': location, 'nonfinite_elements': bad})
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, location + '/' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, location + '/' + str(index))

    sections = {}
    for name in ('state_dict', 'optimizer_states'):
        before = counts.copy()
        visit(checkpoint.get(name, {}), name)
        sections[name] = dict(counts - before)
        print(name + ': ' + json.dumps(sections[name]), flush=True)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint': str(path), 'bytes': path.stat().st_size,
              'global_step': checkpoint.get('global_step'), 'epoch': checkpoint.get('epoch'),
              'checked': sections, 'all_tensors_finite': not invalid, 'invalid_tensors': invalid,
              'note': 'Finiteness checks do not establish image quality or convergence.'}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('all_tensors_finite: ' + str(not invalid), flush=True)
    if invalid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
