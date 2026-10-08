"""GT-free, fixed-noise robot-conditioning interventions on trained checkpoints.

This diagnostic does not change training, model weights or the baseline's
normal inference entry. Counterfactual outputs have no corresponding GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

MODES = ('original', 'remove_others', 'move_others_right', 'swap_other_identities')


def perturb_sample(sample, mode, shift_m=.75):
    """Change other robots only; preserve observer, camera and environment."""
    if mode not in MODES:
        raise ValueError('Unknown intervention: ' + mode)
    if not np.isfinite(shift_m) or shift_m <= 0:
        raise ValueError('shift_m must be positive and finite')
    result = dict(sample)
    for name in ('robot_T_wb', 'robot_camera_heights', 'robot_ids'):
        result[name] = np.asarray(sample[name]).copy()
    other = result['robot_ids'] != sample['robot_id']
    if np.count_nonzero(~other) != 1:
        raise ValueError('Expected exactly one observer state')
    details = {'mode': mode, 'observer_id': int(sample['robot_id'])}
    if mode == 'remove_others':
        for name in ('robot_T_wb', 'robot_camera_heights', 'robot_ids'):
            result[name] = result[name][~other]
        details['removed_ids'] = sample['robot_ids'][other].tolist()
    elif mode == 'move_others_right':
        # Planar camera-right direction; do not change authored mast height.
        right = np.asarray(sample['T_wc_rgb'][:3, 0], dtype=np.float64).copy()
        right[2] = 0
        norm = np.linalg.norm(right)
        if norm < 1e-6:
            raise ValueError('Camera right axis has no horizontal component')
        delta = right / norm * shift_m
        result['robot_T_wb'][other, :3, 3] += delta
        details['world_translation_m'] = delta.tolist()
        details['moved_ids'] = sample['robot_ids'][other].tolist()
    elif mode == 'swap_other_identities':
        slots = np.flatnonzero(other)
        if len(slots) < 2:
            raise ValueError('Identity swap requires at least two other robots')
        # IDs determine known asset accent colors; poses/heights stay fixed.
        result['robot_ids'][slots] = result['robot_ids'][slots[::-1]]
        details['old_ids'] = sample['robot_ids'].tolist()
        details['new_ids'] = result['robot_ids'].tolist()
    return result, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('train','val','test'), default='train')
    parser.add_argument('--scene', action='append')
    parser.add_argument('--max-episodes', type=int, default=2)
    parser.add_argument('--robots', nargs='+', type=int, default=[1,2])
    parser.add_argument('--frames', nargs='+', type=int, default=[10])
    parser.add_argument('--guidances', nargs='+', type=float, default=[1,9])
    parser.add_argument('--shift-m', type=float, default=.75)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.guidances or any(not np.isfinite(g) or g < 1 for g in args.guidances):
        parser.error('Use finite guidance values >= 1')
    if len(set(args.guidances)) != len(args.guidances):
        parser.error('Guidance values must be unique')
    from mvwd_runtime import load_config, make_dataset
    config = load_config(args.config)
    dataset = make_dataset(config, args.split, include_targets=False, scenes=args.scene,
                           max_episodes=args.max_episodes, robots=args.robots, frames=args.frames)
    dataset[0]
    if dataset.robot_assets is None:
        parser.error('Robot conditioning must be enabled')
    config['robot_asset_provenance'] = dataset.robot_assets.provenance()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    output.mkdir(parents=True)
    import torch
    from PIL import Image
    from torch.utils.data._utils.collate import default_collate
    from adapters.camera_geometry import PerspectiveRenderer
    from adapters.mvwd_raw import sha256_file
    from mvwd_model import load_trained
    model = load_trained(config, args.checkpoint).to(args.device).eval()
    renderer = PerspectiveRenderer(**config['renderer'])
    metadata = {'checkpoint_sha256':sha256_file(args.checkpoint), 'config':config,
                'dataset':dataset.provenance(), 'ground_truth_loaded':False,
                'weights_modified':False, 'guidances':args.guidances,'shift_m':args.shift_m,
                'modes':MODES,'scope':'Counterfactual conditioning diagnostic, not benchmark predictions or GT accuracy.',
                'seed_policy':'Same geometry/query seeds as infer_mvwd.py; reset query seed before every variant/guidance.',
                'limitations':['Changes can include global per-image coarse min/max normalization effects.',
                               'Pixel sensitivity does not establish correct robot count, identity or motion.',
                               'Removed/moved/recolored conditions have no matching GT; no PSNR/SSIM is computed.']}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[];geometry_key=None;voxel=None
    def collate(sample):
        batch=default_collate([sample])
        return {k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    def seed_for(identity):return int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample=dataset[index]
            assert not any(k in sample for k in ('target_rgb','target_depth_z','depth_valid'))
            if sample['bev_key'] != geometry_key:
                identity='%s:%s:%s'%(config['inference']['seed'],dataset.raw.metadata['release_id'],sample['bev_key'])
                torch.manual_seed(seed_for(identity));voxel=model.predict_occupancy(collate(sample)['bev_rgb']);geometry_key=sample['bev_key']
            identity='%s:%s:%s:%s:%s'%(config['inference']['seed'],dataset.raw.metadata['release_id'],sample['episode_id'],sample['robot_id'],sample['frame_index'])
            seed=seed_for(identity)
            coarse={};details={};generated={}
            directory=output/sample['episode_id']/('robot_%02d'%sample['robot_id'])/('frame_%03d'%sample['frame_index']);directory.mkdir(parents=True)
            for mode in MODES:
                modified,details[mode]=perturb_sample(sample,mode,args.shift_m)
                coarse[mode]=renderer(voxel,collate(modified))
                saved={k:coarse[mode][k][0].cpu().numpy() for k in ('rgb','depth_condition','depth_z','robot_id','robot_weight','robot_depth_z')}
                if not all(np.isfinite(v).all() for v in saved.values()):raise ValueError('Non-finite coarse output')
                np.savez_compressed(directory/(mode+'.coarse.npz'),**saved)
                Image.fromarray(np.rint(saved['rgb'].clip(0,1)*255).astype(np.uint8)).save(directory/(mode+'.coarse.png'))
            original=coarse['original']
            original_mask=(original['robot_id'][0]>=0)&(original['robot_id'][0]!=sample['robot_id'])&(original['robot_weight'][0]>.1)
            for guidance in args.guidances:
                gdir=directory/('guidance_%g'%guidance);gdir.mkdir()
                for mode in MODES:
                    torch.manual_seed(seed)
                    c=coarse[mode]
                    result=model.refine(c['rgb'],c['depth_condition'],steps=config['inference']['ddim_steps'],guidance=guidance)
                    if not torch.isfinite(result).all():raise ValueError('Non-finite final output')
                    signed=result[0].permute(1,2,0).cpu().numpy()
                    rgb=((signed+1)/2).clip(0,1)
                    generated[mode]=rgb
                    Image.fromarray(((signed+1)*127.5).clip(0,255).astype(np.uint8)).save(gdir/(mode+'.png'))
                    if mode=='original':continue
                    variant_mask=(c['robot_id'][0]>=0)&(c['robot_id'][0]!=sample['robot_id'])&(c['robot_weight'][0]>.1)
                    union=(original_mask|variant_mask).cpu().numpy()
                    difference=np.abs(rgb-generated['original']).mean(axis=-1)
                    rows.append({'episode_id':sample['episode_id'],'observer':sample['robot_id'],'frame':sample['frame_index'],
                        'seed':seed,'guidance':guidance,'mode':mode,'intervention':details[mode],
                        'original_other_robot_pixels':int(original_mask.sum()),'variant_other_robot_pixels':int(variant_mask.sum()),
                        'coarse_rgb_mean_abs_delta':float((c['rgb']-original['rgb']).abs().mean()),
                        'coarse_depth_condition_mean_abs_delta':float((c['depth_condition']-original['depth_condition']).abs().mean()),
                        'final_rgb_mean_abs_delta':float(difference.mean()),
                        'final_rgb_delta_in_union_predicted_robot_region':float(difference[union].mean()) if union.any() else None,
                        'final_rgb_delta_outside_region':float(difference[~union].mean()) if (~union).any() else None,
                        'original_path':str((gdir/'original.png').relative_to(output)),
                        'variant_path':str((gdir/(mode+'.png')).relative_to(output))})
            print('Diagnosed %d/%d queries'%(index+1,len(dataset)),flush=True)
            (output/'sensitivity.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'query_count':len(dataset),'generated_final_images':len(dataset)*len(MODES)*len(args.guidances),'comparisons':len(rows),'ground_truth_loaded':False,'weights_modified':False},indent=2)+'\n')


if __name__=='__main__':main()
