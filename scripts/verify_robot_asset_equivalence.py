"""GT-free equivalence of source+static hiding and the shared pruned asset bundle."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from adapters.camera_geometry import pinhole_rays
from adapters.robot_assets import robot_renderer
from mvwd_runtime import load_config, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config(args.config)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    candidate = robot_renderer(str(Path(args.candidate).resolve()), 'all')
    source = robot_renderer(str(Path(args.source).resolve()), 'hide_top_body')
    rows = []
    for split, scene in [('train', 'Rs_int'), ('val', 'office_vendor_machine')]:
        dataset = make_dataset(config, split, include_targets=False, scenes=[scene], max_episodes=1,
                               frames=[0, 20, 40], robots=[0, 1, 2])
        for index in range(len(dataset)):
            sample = dataset[index]; batch = default_collate([sample])
            height, width = map(int, sample['target_hw'])
            origin, rays, z = pinhole_rays(batch['K_rgb'], batch['T_wc_rgb'], height, width)
            inputs = (origin[0].numpy(), rays[0].numpy(), sample['robot_T_wb'], sample['robot_camera_heights'],
                      sample['robot_ids'], float(sample['near_m']), float(sample['far_m']))
            reference = source.trace(*inputs, camera_z=z[0].numpy())
            actual = candidate.trace(*inputs, camera_z=z[0].numpy())
            np.testing.assert_array_equal(actual[2], reference[2])
            np.testing.assert_array_equal(np.isfinite(actual[0]), np.isfinite(reference[0]))
            np.testing.assert_allclose(actual[0], reference[0], rtol=0, atol=1e-5)
            np.testing.assert_allclose(actual[1], reference[1], rtol=0, atol=1e-6)
            hit = np.isfinite(actual[0])
            row = {'scene': scene, 'episode_id': sample['episode_id'], 'robot_id': sample['robot_id'],
                   'frame_index': sample['frame_index'], 'hit_pixels': int(hit.sum()),
                   'max_distance_delta_m': float(np.max(np.abs(actual[0][hit] - reference[0][hit]))) if hit.any() else 0.,
                   'max_RGB_delta': float(np.max(np.abs(actual[1] - reference[1])))}
            rows.append(row); print(json.dumps(row), flush=True)
    report = {'passed': True, 'queries': rows, 'GT_loaded': False, 'source_policy': 'hide_top_body', 'source_asset': source.assets.provenance(),
              'candidate_policy': 'all', 'candidate_asset': candidate.assets.provenance()}
    (output / 'equivalence.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
