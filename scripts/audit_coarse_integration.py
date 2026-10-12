"""Audit density and color integration on CPU, loading only the frozen VAE."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
from torch.utils.data._utils.collate import default_collate

from mvwd_runtime import load_config,make_dataset,align_depth_to_rgb
from adapters.camera_geometry import PerspectiveRenderer, relative_map
from scripts.audit_upstream_weights import Archive


def load_vae(config):
    import yaml
    from ldm.models.autoencoder import AutoencoderKL
    params=yaml.safe_load(Path(config['model']['config']).read_text())['model']['params']['first_stage_config']['params']
    model=AutoencoderKL(**params).eval().requires_grad_(False)
    archive=Archive(config['model']['init_checkpoint'])
    prefix='first_stage_model.';state={}
    for name,metadata in archive.state.items():
        if name.startswith(prefix):
            if metadata['dtype']!='float32':raise ValueError('Unexpected VAE dtype')
            data=np.frombuffer(archive.zip.read(archive.entry(metadata).filename),dtype=np.float32).copy()
            state[name[len(prefix):]]=torch.from_numpy(data.reshape(metadata['shape']))
    model.load_state_dict(state,strict=True)
    archive.zip.close()
    return model


def stat(x):
    return {k:float(np.quantile(x,q)) for k,q in [('min',0),('p10',.1),('median',.5),('p90',.9),('max',1)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--split',default='val');parser.add_argument('--scene',required=True)
    parser.add_argument('--robot',type=int,default=0);parser.add_argument('--frame',type=int,default=0)
    args=parser.parse_args();torch.set_num_threads(4)
    config=load_config(args.config);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    ds=make_dataset(config,args.split,include_targets=False,scenes=[args.scene],max_episodes=1,robots=[args.robot],frames=[args.frame])
    sample=ds[0];batch=default_collate([sample]);vae=load_vae(config)
    identity="%s:%s:%s"%(config['inference']['seed'],ds.raw.metadata['release_id'],sample['bev_key'])
    torch.manual_seed(int(hashlib.sha256(identity.encode()).hexdigest()[:8],16))
    with torch.no_grad():
        latent=vae.encode((batch['bev_rgb']*2-1).permute(0,3,1,2)).sample()
        voxel=vae.decode(latent,density_map=True)
        del vae
        renderer=PerspectiveRenderer(**config['renderer'],diagnostics=True)
        result=renderer(voxel,batch)
        sigma=renderer.prepare_density(voxel,batch['bev_valid'],batch['wall_mask'])
        valid_volume=batch['bev_valid'][:,None].expand_as(voxel)
        valid_sigma=relative_map(voxel,valid_volume)*renderer.density_scale
        valid_sigma=torch.where(batch['wall_mask'][:,None],torch.full_like(valid_sigma,1000.),valid_sigma)
        valid_sigma=valid_sigma*batch['bev_valid'][:,None]
        valid_sigma=torch.cat((sigma[:,:2],valid_sigma),dim=1)
        valid_normalization=renderer(valid_sigma,batch,prepared=True)
        # Counterfactual density ablation only, never a new inference or training variant.
        floor_wall=sigma.clone();floor_wall[:,2:]=torch.where(batch['wall_mask'][:,None],sigma[:,2:],torch.zeros_like(sigma[:,2:]))
        prior_only=renderer(floor_wall,batch,prepared=True)
        no_short=PerspectiveRenderer(**{**config['renderer'],'rgb_distance_normalized':100.},diagnostics=True)(voxel,batch)
    record,robot,frame=ds.query(0)
    gt=ds.raw.views(record,robot,('rgb','depth_linear'),ds.cache_root)
    image=Image.fromarray(gt['rgb'][frame]).resize(tuple(sample['target_hw'][::-1]),Image.Resampling.BOX)
    image.save(output/'gt.png')
    camera=ds.raw.camera(record,robot,frame);poses=camera.get('modality_camera_to_world',{})
    labels={**batch,'target_depth_z':torch.from_numpy(gt['depth_linear'][frame].copy())[None],
            'K_depth':torch.tensor(camera['geometry_pixel_intrinsics'],dtype=torch.float32)[None],
            'T_wc_depth':torch.tensor(poses.get('depth_linear',camera['camera_to_world']),dtype=torch.float32)[None]}
    labels['depth_valid']=torch.isfinite(labels['target_depth_z'])&(labels['target_depth_z']>0)&(labels['target_depth_z']<=sample['far_m'])
    depth,valid=align_depth_to_rgb(labels)
    arrays={key:v[0].numpy() for key,v in result.items()}
    gray=(arrays['rgb'].max(-1)-arrays['rgb'].min(-1)<.08)&(arrays['rgb'].mean(-1)>.3)&(arrays['rgb'].mean(-1)<.7)&(arrays['robot_id']==-1)
    gt_depth=depth[0].numpy();gt_valid=valid[0].numpy();mask=gray&gt_valid
    stats={'scene':record['scene'],'episode_id':record['id'],'robot_id':robot,'frame_index':frame,
           'gray_pixels':int(gray.sum()),'all_pixels':int(gray.size),
           'voxel_raw':stat(voxel.numpy()),'voxel_valid':stat(voxel[valid_volume].numpy()),
           'sigma_no_floor':stat(sigma[:,2:].numpy()),
           'wall_pixel_count':int(batch['wall_mask'].sum()),'valid_bev_pixels':int(batch['bev_valid'].sum()),
           'camera_height_above_floor_m':float(sample['T_wc_rgb'][2,3]-sample['floor_z']),
           'rgb_far_m':float(result['rgb_far_m'][0]),'depth_far_m':float(sample['far_m']),
           'gray_rgb_opacity':stat(arrays['rgb_environment_opacity'][gray]),
           'gray_full_opacity':stat(arrays['environment_opacity'][gray]),
           'gray_coarse_depth':stat(arrays['depth_z'][gray]),'gray_GT_depth':stat(gt_depth[mask]),
           'gray_depth_divided_by_opacity':stat(arrays['depth_z'][gray]/np.maximum(arrays['environment_opacity'][gray],1e-6)),
           'gray_valid_norm_opacity':stat(valid_normalization['rgb_environment_opacity'][0].numpy()[gray]),
           'rgb_raw_range':[float(arrays['rgb_raw'].min()),float(arrays['rgb_raw'].max())],
           'gray_opacity_lt_01_fraction':float((arrays['rgb_environment_opacity'][gray]<.1).mean()),
           'gray_opacity_gt_09_fraction':float((arrays['rgb_environment_opacity'][gray]>.9).mean()),
           'GT_beyond_rgb_far_fraction':float((gt_depth[mask]>float(result['rgb_far_m'][0])).mean()),
           'short_vs_full_RGB_mean_delta':float((result['rgb']-no_short['rgb']).abs().mean()),
           'GT_used_for_diagnostic_only':True,'only_frozen_VAE_loaded':True}
    np.savez_compressed(output/'integration.npz',**arrays,gray_mask=gray,gt_depth_z=gt_depth,
                        sigma_height_mean=sigma[0].mean(0).numpy(),floor_wall_rgb=prior_only['rgb'][0].numpy(),full_range_rgb=no_short['rgb'][0].numpy(),valid_normalization_rgb=valid_normalization['rgb'][0].numpy())
    Image.fromarray((arrays['rgb']*255).astype(np.uint8)).save(output/'coarse_fixed.png')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,10))
    panels=[(np.asarray(image),'GT RGB'),(arrays['rgb'],'Coarse after clipping fix'),
            (arrays['rgb_environment_opacity'],'RGB environment opacity'),(arrays['environment_opacity'],'Full environment opacity'),
            (prior_only['rgb'][0].numpy(),'Only wall and floor priors'),(no_short['rgb'][0].numpy(),'Full RGB sampling range'),
            (gt_depth,'GT forward depth'),(arrays['depth_z'],'Coarse forward depth'),
            (valid_normalization['rgb'][0].numpy(),'Ablation: normalize valid BEV only')]
    for ax,(value,title) in zip(axes.flat,panels):
        artist=ax.imshow(value,cmap='viridis' if value.ndim==2 else None);ax.set_title(title);ax.axis('off')
        if value.ndim==2:fig.colorbar(artist,ax=ax,fraction=.04)
    fig.tight_layout();fig.savefig(output/'integration.png',dpi=120);plt.close(fig)
    (output/'audit.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
