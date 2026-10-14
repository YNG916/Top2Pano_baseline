"""Compare completed predictions against seeded noise and a fixed gray image."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from mvwd_runtime import load_config
from mvwd_prediction_comparison import prediction_context
from adapters.mvwd_raw import sha256_file, within


def psnr(image, target):
    difference = np.asarray(image, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    mse = float(np.mean(difference ** 2))
    return float(-10 * np.log10(mse)) if mse > 0 else float('inf')


def random_scores(target, identity, seed=42, repetitions=16):
    scores = []
    for trial in range(repetitions):
        material = '%s:%s:%s' % (seed, identity, trial)
        query_seed = int(hashlib.sha256(material.encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(query_seed)
        noise = rng.integers(0, 256, target.shape, dtype=np.uint8).astype(np.float64) / 255.
        scores.append(psnr(noise, target))
    # U{0,...,255}/255 has mean .5 and variance 257/(12*255).
    expected_mse = 257. / (12 * 255) + float(np.mean((target - .5) ** 2))
    return scores, float(-10 * np.log10(expected_mse))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--random-repetitions', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pair', action='append', default=[], help='Compare group names BEST:INITIALIZATION on identical queries')
    args = parser.parse_args()
    if args.random_repetitions <= 0:
        parser.error('random-repetitions must be positive')
    load_config(args.config)  # Validate the captured experiment config; no model is loaded.
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    gt_cache = {}; rows = []; groups = []; contexts = {}
    for predictions in args.predictions:
        predictions = predictions.resolve()
        metadata, manifest, raw, records = prediction_context(predictions)
        selected = []
        if predictions.name in contexts:raise ValueError('Prediction group names must be unique')
        contexts[predictions.name] = (predictions, metadata, {(r['episode_id'],r['robot_id'],r['frame_index']):r for r in manifest})
        for query in manifest:
            path = within(predictions, query['path'])
            image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.
            height, width = image.shape[:2]
            key = (raw.metadata['release_id'], query['episode_id'], query['robot_id'], query['frame_index'], height, width)
            if key not in gt_cache:
                video = raw.views(records[query['episode_id']], query['robot_id'], ('rgb',),
                                  metadata['config']['data'].get('cache_root'))['rgb']
                target = np.asarray(Image.fromarray(video[query['frame_index']]).resize((width, height), Image.Resampling.BOX), dtype=np.float64) / 255.
                identity = ':'.join(map(str, key))
                trials, expected = random_scores(target, identity, args.seed, args.random_repetitions)
                gt_cache[key] = target, trials, expected
            target, trials, expected = gt_cache[key]
            with np.load(path.with_suffix('.coarse.npz'), allow_pickle=False) as packet:
                coarse = packet['rgb'].astype(np.float64)
            if coarse.shape != target.shape:
                raise ValueError('Coarse/GT dimensions differ')
            final_score = psnr(image, target)
            row = {'group': predictions.name, 'episode_id': query['episode_id'],
                   'scene': records[query['episode_id']]['scene'], 'robot_id': query['robot_id'],
                   'frame_index': query['frame_index'], 'final_psnr_db': final_score,
                   'coarse_psnr_db': psnr(coarse, target),
                   'uniform_noise_trial_psnr_db': trials, 'uniform_noise_mean_psnr_db': float(np.mean(trials)),
                   'uniform_noise_psnr_of_expected_mse_db': expected,
                   'fixed_gray_128_psnr_db': psnr(np.full_like(target, 128. / 255), target),
                   'final_minus_noise_mean_db': final_score - float(np.mean(trials))}
            rows.append(row); selected.append(row)
        summary = {'group': predictions.name, 'path': str(predictions), 'query_count': len(selected),
                   'scene_count': len({r['scene'] for r in selected}),
                   'checkpoint_sha256': metadata['checkpoint_sha256'],
                   'checkpoint_kind': metadata.get('checkpoint_kind', 'trained'),
                   'renderer': metadata['config']['renderer'],
                   'robot_assets': metadata['dataset'].get('robot_assets'),
                   'prediction_manifest_sha256': sha256_file(predictions / 'predictions.jsonl'),
                   'metadata_sha256': sha256_file(predictions / 'metadata.json')}
        for name in ['final_psnr_db', 'coarse_psnr_db', 'uniform_noise_mean_psnr_db', 'fixed_gray_128_psnr_db', 'final_minus_noise_mean_db']:
            summary['mean_per_query_' + name] = float(np.mean([r[name] for r in selected]))
        by_trial = np.asarray([r['uniform_noise_trial_psnr_db'] for r in selected]).mean(axis=0)
        summary['uniform_noise_trial_group_mean_std_db'] = float(by_trial.std())
        summary['queries_final_above_noise'] = sum(r['final_minus_noise_mean_db'] > 0 for r in selected)
        summary['queries_final_above_gray'] = sum(r['final_psnr_db'] > r['fixed_gray_128_psnr_db'] for r in selected)
        groups.append(summary); print(json.dumps(summary, indent=2), flush=True)
    pairs = []
    for pair in args.pair:
        current, reference = pair.split(':',1)
        a, am, aq = contexts[current]; b, bm, bq = contexts[reference]
        if set(aq) != set(bq):raise ValueError('Paired predictions have different queries')
        for key in ('data','renderer','inference','prompts'):
            if am['config'][key] != bm['config'][key]:raise ValueError('Paired generation protocols differ: '+key)
        if am['dataset']['release_id'] != bm['dataset']['release_id']:raise ValueError('Paired releases differ')
        paired = []
        for identity in aq:
            ar, br = aq[identity], bq[identity]
            if ar['seed'] != br['seed']:raise ValueError('Paired generation seeds differ')
            ap = within(a,ar['path']); bp = within(b,br['path'])
            with np.load(ap.with_suffix('.coarse.npz'),allow_pickle=False) as av, np.load(bp.with_suffix('.coarse.npz'),allow_pickle=False) as bv:
                delta=float(np.max(np.abs(av['rgb']-bv['rgb'])))
                depth_delta=float(np.max(np.abs(av['depth_z']-bv['depth_z'])))
                np.testing.assert_array_equal(av['robot_id'],bv['robot_id'])
            ra=next(r for r in rows if r['group']==current and (r['episode_id'],r['robot_id'],r['frame_index'])==identity)
            rb=next(r for r in rows if r['group']==reference and (r['episode_id'],r['robot_id'],r['frame_index'])==identity)
            paired.append({'episode_id':identity[0],'robot_id':identity[1],'frame_index':identity[2],
                'coarse_RGB_max_abs_delta':delta,'coarse_depth_max_abs_delta_m':depth_delta,
                'best_minus_initialization_psnr_db':ra['final_psnr_db']-rb['final_psnr_db']})
            # Cross-model comparison stays next to BEST predictions, per directory convention.
            from PIL import ImageDraw
            target_key=(am['dataset']['release_id'],*identity,*np.asarray(Image.open(ap)).shape[:2])
            target=Image.fromarray(np.round(gt_cache[target_key][0]*255).astype(np.uint8))
            with np.load(ap.with_suffix('.coarse.npz'),allow_pickle=False) as av:
                coarse_image=Image.fromarray(np.round(np.clip(av['rgb'],0,1)*255).astype(np.uint8))
            images=[target,coarse_image,Image.open(bp).convert('RGB'),Image.open(ap).convert('RGB')]
            width,height=target.size
            sheet=Image.new('RGB',(width*4,height+35),'white');draw=ImageDraw.Draw(sheet)
            for i,(label,image) in enumerate(zip(['GT','Shared coarse','SD initialization','Trained best'],images)):
                draw.text((i*width+8,8),label,fill='black');sheet.paste(image,(i*width,35))
            destination=ap.with_suffix('.initialization_comparison.png')
            if destination.exists():raise FileExistsError('Paired comparison already exists: '+str(destination))
            sheet.save(destination)
        result={'best_group':current,'initialization_group':reference,'query_count':len(paired),
            'mean_best_minus_initialization_psnr_db':float(np.mean([r['best_minus_initialization_psnr_db'] for r in paired])),
            'queries_best_above_initialization':sum(r['best_minus_initialization_psnr_db']>0 for r in paired),
            'max_coarse_RGB_pair_delta':max(r['coarse_RGB_max_abs_delta'] for r in paired),
            'max_coarse_depth_pair_delta_m':max(r['coarse_depth_max_abs_delta_m'] for r in paired),'queries':paired}
        pairs.append(result);print(json.dumps(result,indent=2),flush=True)
    report = {'groups': groups,'pairs':pairs, 'queries': rows, 'peak': 1., 'aggregation': 'arithmetic mean of per-query PSNR, full RGB, no masks',
              'random_reference': 'Independent uint8 uniform RGB pixels; deterministic physical-query seeds, shared across matched groups',
              'seed': args.seed, 'random_repetitions': args.random_repetitions,
              'GT_used_for_evaluation_only': True, 'prediction_files_modified': False,
              'limitations': ['Selected preview queries, not full val/test.',
                  'Uniform noise and fixed gray are diagnostic references, not strong image generation baselines.',
                  'PSNR does not establish spatial, temporal, robot identity or perceptual correctness.',
                  'A pretrained SD/ControlNet initialization is not random weights; compare it separately.']}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
