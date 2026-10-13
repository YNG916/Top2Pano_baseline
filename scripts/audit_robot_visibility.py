"""Compare fixed asset visibility policies without modifying assets or GT."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from mvwd_runtime import load_config,make_dataset,align_depth_to_rgb
from adapters.camera_geometry import pinhole_rays
from adapters.robot_assets import robot_renderer,VISIBILITY_POLICIES,part_is_hidden
from adapters.mvwd_raw import within,read_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--cases',nargs='+',default=['train:Rs_int','val:office_vendor_machine'])
    parser.add_argument('--frames',type=int,nargs='+',default=[0,20,40])
    parser.add_argument('--robots',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--policies',nargs='+',choices=VISIBILITY_POLICIES,default=list(VISIBILITY_POLICIES))
    args=parser.parse_args();torch.set_num_threads(4)
    config=load_config(args.config);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    datasets=[]
    for case in args.cases:
        split,scene=case.split(':',1)
        datasets.append(make_dataset(config,split,include_targets=False,scenes=[scene],max_episodes=1,frames=args.frames,robots=args.robots))
    rows=[];hidden={}
    # Policy is fixed per asset prim/material, independent of query and GT.
    for policy in args.policies:
        for ds in datasets:
            for index in range(len(ds)):
                sample=ds[index];batch=default_collate([sample]);record,robot,frame=ds.query(index)
                h,w=map(int,sample['target_hw']);origin,directions,z=pinhole_rays(batch['K_rgb'],batch['T_wc_rgb'],h,w)
                renderer=robot_renderer(sample['robot_assets_root'],policy)
                distance,rgb,ids=renderer.trace(origin[0].numpy(),directions[0].numpy(),sample['robot_T_wb'],
                  sample['robot_camera_heights'],sample['robot_ids'],float(sample['near_m']),float(sample['far_m']),camera_z=z[0].numpy())
                hidden[policy]=[{'prim_path':p['prim_path'],'material':p['material']} for p in renderer.assets.parts if part_is_hidden(p,policy)]
                predicted=(ids>=0).reshape(h,w);depth=(distance*z[0].numpy()).reshape(h,w)
                gt_rgb=ds.raw.views(record,robot,('rgb',),ds.cache_root)['rgb'][frame]
                gt_sem=ds.raw.views(record,robot,('semantic',),None)['semantic'][frame]
                labels=read_json(within(ds.raw.root,record['taxonomy_path']))['semantic_labels']
                robot_labels=[int(k) for k,v in labels.items() if v=='robot' or (isinstance(v,dict) and 'robot' in str(v).lower())]
                if not robot_labels:raise ValueError('Taxonomy contains no robot semantic labels')
                gt_mask=np.isin(gt_sem,robot_labels)
                if gt_mask.shape!=(h,w):gt_mask=np.asarray(Image.fromarray((gt_mask*255).astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST))>0
                camera=ds.raw.camera(record,robot,frame);poses=camera.get('modality_camera_to_world',{})
                gt_z=ds.raw.views(record,robot,('depth_linear',),ds.cache_root)['depth_linear'][frame]
                label_batch={**batch,'target_depth_z':torch.from_numpy(gt_z.copy())[None],
                  'depth_valid':torch.from_numpy(np.isfinite(gt_z)&(gt_z>0)&(gt_z<=sample['far_m']))[None],
                  'K_depth':torch.tensor(camera['geometry_pixel_intrinsics'],dtype=torch.float32)[None],
                  'T_wc_depth':torch.tensor(poses.get('depth_linear',camera['camera_to_world']),dtype=torch.float32)[None]}
                aligned,valid=align_depth_to_rgb(label_batch);target_z=aligned[0].numpy();valid=valid[0].numpy()
                union=predicted|gt_mask;intersection=predicted&gt_mask
                # Remove predicted mesh points behind GT environment from the audit metric only.
                epsilon=.03
                gt_occluded=predicted&valid&(~gt_mask)&(target_z+epsilon<depth)
                known_visible=predicted&(~gt_occluded)
                visible_union=known_visible|gt_mask
                common=intersection&valid
                row={'policy':policy,'episode_id':record['id'],'scene':record['scene'],'robot_id':robot,'frame_index':frame,
                     'GT_robot_pixels':int(gt_mask.sum()),'mesh_pixels':int(predicted.sum()),
                     'self_pixels':int((ids==robot).sum()),'IoU':float(intersection.sum()/max(1,union.sum())),
                     'GT_occlusion_filtered_IoU':float((known_visible&gt_mask).sum()/max(1,visible_union.sum())),
                     'depth_MAE_m_on_GT_robot_overlap':float(np.abs(depth[common]-target_z[common]).mean()) if common.any() else None,
                     'false_positive_pixels':int((predicted&(~gt_mask)).sum()),'false_negative_pixels':int((gt_mask&(~predicted)).sum())}
                directory=output/record['id']/('robot_%02d'%robot);directory.mkdir(parents=True,exist_ok=True);prefix='frame_%03d'%frame
                display=np.where(predicted[...,None],rgb.reshape(h,w,3),.65)
                Image.fromarray((display*255).astype(np.uint8)).save(directory/(prefix+'.'+policy+'.png'))
                np.savez_compressed(directory/(prefix+'.'+policy+'.npz'),rgb=rgb.reshape(h,w,3),ids=ids.reshape(h,w),depth_z=depth)
                if not (directory/(prefix+'.gt.png')).exists():Image.fromarray(gt_rgb).resize((w,h),Image.Resampling.BOX).save(directory/(prefix+'.gt.png'))
                rows.append(row);print(json.dumps(row),flush=True)
    def mean_or_none(values):
        return float(np.mean(values)) if values else None
    summary={}
    for policy in args.policies:
        selected=[r for r in rows if r['policy']==policy and r['GT_robot_pixels']>0]
        summary[policy]={'queries_with_GT_robots':len(selected),'mean_IoU':mean_or_none([r['IoU'] for r in selected]),
                         'mean_GT_occlusion_filtered_IoU':mean_or_none([r['GT_occlusion_filtered_IoU'] for r in selected]),
                         'mean_depth_MAE_m':mean_or_none([r['depth_MAE_m_on_GT_robot_overlap'] for r in selected if r['depth_MAE_m_on_GT_robot_overlap'] is not None])}
    report={'config':config,'asset_provenance':renderer.assets.provenance(),'summary':summary,'queries':rows,'hidden_parts':hidden,'GT_used_for_audit_only':True,
            'asset_and_GT_files_modified':False,'policy_selected_from_known_asset_metadata_not_GT_per_query':True,
            'limitations':['Selected preview episodes only; not formal evaluation.','GT occlusion filtering is a diagnostic metric only, never a render input.','Improved agreement does not prove original Isaac visibility settings.']}
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
