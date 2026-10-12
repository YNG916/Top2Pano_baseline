"""CPU robot projection audit: clipping, self-hit parts and GT silhouette."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from mvwd_runtime import load_config, make_dataset
from adapters.camera_geometry import pinhole_rays
from adapters.robot_assets import robot_renderer
from adapters.mvwd_raw import read_json, within


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--robots", type=int, nargs="+", default=[0,1,2])
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config(args.config)
    ds = make_dataset(config,args.split,include_targets=False,scenes=[args.scene],max_episodes=1,
                      robots=args.robots,frames=args.frames)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    for index in range(len(ds)):
        sample=ds[index];batch=default_collate([sample]);h,w=map(int,sample["target_hw"])
        origin,directions,cz=pinhole_rays(batch["K_rgb"],batch["T_wc_rgb"],h,w)
        o,d,z=origin[0].numpy(),directions[0].numpy(),cz[0].numpy()
        renderer=robot_renderer(sample["robot_assets_root"])
        trace_args=(o,d,sample["robot_T_wb"],sample["robot_camera_heights"],sample["robot_ids"],float(sample["near_m"]),float(sample["far_m"]))
        legacy=renderer.trace(*trace_args)
        planar=renderer.trace(*trace_args,camera_z=z)
        record,robot,frame=ds.query(index)
        directory=output/sample["episode_id"]/('robot_%02d'%robot);directory.mkdir(parents=True,exist_ok=True)
        prefix='frame_%03d'%frame
        for label,packet in [('radial',legacy),('planar',planar)]:
            distance,rgb,ids=packet
            Image.fromarray((rgb.reshape(h,w,3)*255).astype(np.uint8)).save(directory/(prefix+'.'+label+'.png'))
            np.savez_compressed(directory/(prefix+'.'+label+'.npz'),rgb=rgb.reshape(h,w,3),ids=ids.reshape(h,w),depth_z=(distance*z).reshape(h,w))
        gt=ds.raw.views(record,robot,("rgb",),ds.cache_root)
        gt={**gt,**ds.raw.views(record,robot,("semantic",),None)}
        image=Image.fromarray(gt['rgb'][frame]).resize((w,h),Image.Resampling.BOX)
        image.save(directory/(prefix+'.gt.png'))
        tax=read_json(within(ds.raw.root,record['taxonomy_path']))
        labels=tax['semantic_labels']
        robot_label_ids=[int(k) for k,v in labels.items() if (v=='robot' or (isinstance(v,dict) and 'robot' in str(v).lower()))]
        semantic=gt['semantic'][frame]
        mask=np.isin(semantic,robot_label_ids)
        if mask.shape!=(h,w):
            mask=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((w,h),Image.Resampling.NEAREST))>0
        Image.fromarray((mask*255).astype(np.uint8)).save(directory/(prefix+'.gt_robot_mask.png'))
        # Find which observer parts the legacy rays intersect near the camera.
        pose=sample['robot_T_wb'][robot];height=renderer.assets.nominal_height(sample['robot_camera_heights'][robot]);mesh=renderer._meshes[height]
        shifted=o+d*float(sample['near_m']);local_o=(shifted-pose[:3,3])@pose[:3,:3];local_d=d@pose[:3,:3]
        faces=mesh.intersector.intersects_first(local_o,local_d)
        self_pixels=(legacy[2]==robot)&(faces>=0)
        part_ids=np.searchsorted(mesh.ends,faces[self_pixels],side='right')
        counts=np.bincount(part_ids,minlength=len(renderer.assets.parts))
        details=[]
        for part_id in np.flatnonzero(counts):
            part=renderer.assets.parts[part_id];visual=mesh.visuals[part_id][1]
            material=getattr(visual,'material',None)
            factor=getattr(material,'baseColorFactor',None)
            details.append({'part':part['prim_path'],'material':part['material'],'pixels':int(counts[part_id]),
                            'baseColorFactor':np.asarray(factor).tolist() if factor is not None else None})
        depth_packet=ds.raw.views(record,robot,('depth_linear',),ds.cache_root)['depth_linear'][frame]
        gt_depth=depth_packet.reshape(-1)
        material_sources=read_json(renderer.assets.file('materials.json'))['materials']
        visible_parts=[]
        for body in sample['robot_ids']:
            pose=sample['robot_T_wb'][body];height=renderer.assets.nominal_height(sample['robot_camera_heights'][body]);mesh=renderer._meshes[height]
            shifted=o+d*(float(sample['near_m'])/z)[:,None]
            faces=mesh.intersector.intersects_first((shifted-pose[:3,3])@pose[:3,:3],d@pose[:3,:3])
            seen=(planar[2]==body)&(faces>=0)
            part_ids=np.searchsorted(mesh.ends,faces[seen],side='right')
            counts=np.bincount(part_ids,minlength=len(renderer.assets.parts))
            for part_id in np.flatnonzero(counts):
                part=renderer.assets.parts[part_id];visual=mesh.visuals[part_id][1]
                material=getattr(visual,'material',None);factor=getattr(material,'baseColorFactor',None)
                selected=np.flatnonzero(seen)[part_ids==part_id]
                gt_points=o+d[selected]*(gt_depth[selected]/z[selected])[:,None]
                mesh_points=o+d[selected]*planar[0][selected,None]
                local_gt=(gt_points-pose[:3,3])@pose[:3,:3]
                local_mesh=(mesh_points-pose[:3,3])@pose[:3,:3]
                visible_parts.append({'robot_id':int(body),'part':part['prim_path'],'material':part['material'],
                                      'pixels':int(counts[part_id]),'baseColorFactor':np.asarray(factor).tolist() if factor is not None else None,
                                      'USD_inputs':material_sources.get(part['material'],{}).get('shader_inputs',{}),
                                       'mean_base_color':planar[1][selected].mean(0).tolist(),
                                      'GT_robot_fraction':float(mask.reshape(-1)[selected].mean()),
                                      'mesh_local_z_median':float(np.median(local_mesh[:,2])),
                                      'GT_surface_local_z_median':float(np.median(local_gt[:,2])),
                                      'mean_GT_RGB_at_projection':(np.asarray(image).reshape(-1,3)[selected]/255.).mean(0).tolist()})
        pose_errors=[]
        for body in sample['robot_ids']:
            measured=ds.raw.camera(record,int(body),frame).get('robot_base_to_world')
            if measured is not None:
                delta=np.linalg.inv(sample['robot_T_wb'][body])@np.asarray(measured)
                pose_errors.append({'robot_id':int(body),'translation_m':float(np.linalg.norm(delta[:3,3])),
                                    'rotation_rad':float(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))})
        comparisons={}
        for label,packet in [('radial',legacy),('planar',planar)]:
            distance,rgb,ids=packet;pred=ids.reshape(h,w)>=0;union=(pred|mask).sum()
            comparisons[label]={'self_pixels':int((ids==robot).sum()),'other_pixels':int(((ids>=0)&(ids!=robot)).sum()),
                                'mesh_vs_GT_robot_IoU':float((pred&mask).sum()/max(1,union)),
                                'near_plane_violations':int((np.isfinite(distance)&(distance*z<float(sample['near_m'])-1e-5)).sum())}
        results.append({'episode_id':record['id'],'robot':robot,'frame':frame,'metrics':comparisons,
                        'legacy_self_parts':details,'visible_parts':visible_parts,'planned_vs_actual_base':pose_errors,'GT_robot_pixels':int(mask.sum()),'GT_semantic_robot_ids':robot_label_ids})
        print(json.dumps({key:value for key,value in results[-1].items() if key!='visible_parts'}),flush=True)
    (output/'audit.json').write_text(json.dumps({'queries':results,'GT_used_for_audit_only':True,'no_model_loaded':True},indent=2)+'\n')

if __name__=='__main__':main()
