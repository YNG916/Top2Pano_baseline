"""Measure real FP32 forward/backward/Adam batch memory; no checkpoint is written."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvwd_runtime import load_config, make_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--iterations',type=int,default=4)
    args=parser.parse_args()
    if args.batch_size<1 or args.iterations<3:parser.error('Positive batch size and at least 3 iterations required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=load_config(args.config)
    import torch
    import pytorch_lightning as pl
    from torch.utils.data._utils.collate import default_collate
    from mvwd_model import CombinedMVWDModel
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    pl.seed_everything(config['training']['seed'])
    ds=make_dataset(config,'train',include_targets=True)
    config['robot_asset_provenance']=ds.robot_assets.provenance()
    report={'batch_size':args.batch_size,'precision':32,'iterations':args.iterations,
            'device':torch.cuda.get_device_name(0),'total_GPU_bytes':torch.cuda.get_device_properties(0).total_memory,
            'train_queries':len(ds),'episode_count':len(ds.raw.records),'frame_sampling':ds.frame_sampling,
            'profiling_subset':'First batch from the full dataset index, repeated; not a training-quality experiment',
            'optimizer_states_included':True,'checkpoint_written':False}
    try:
        batch=default_collate([ds[i] for i in range(args.batch_size)])
        batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        model=CombinedMVWDModel(config,initialize=True).to('cuda').train()
        optimizer=model.configure_optimizers()
        losses=[];times=[]
        for step in range(args.iterations):
            if step==1:torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();start=time.perf_counter()
            loss=model.training_step({**batch,"target_rgb":batch["target_rgb"].clone()},step)
            if not torch.isfinite(loss):raise ValueError('Nonfinite profile loss')
            loss.backward();optimizer.step();torch.cuda.synchronize()
            times.append(time.perf_counter()-start);losses.append(float(loss.detach().cpu()))
            print(json.dumps({'profile_step':step,'seconds':times[-1],'loss':losses[-1],
                              'allocated_GiB':torch.cuda.memory_allocated()/2**30,
                              'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30}),flush=True)
        steady=sum(times[1:])/len(times[1:])
        report.update(fits=True,losses=losses,step_seconds=times,steady_step_seconds=steady,
            queries_per_second=args.batch_size/steady,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),reserved_fraction=torch.cuda.max_memory_reserved()/report['total_GPU_bytes'])
        model.eval()
        cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
        with torch.no_grad():
            first=model.validation_step({**batch,"target_rgb":batch["target_rgb"].clone()},0)
            second=model.validation_step({**batch,"target_rgb":batch["target_rgb"].clone()},0)
        if not torch.equal(cpu_rng,torch.get_rng_state()):raise ValueError('Validation changed CPU RNG')
        if any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())):raise ValueError('Validation changed CUDA RNG')
        torch.testing.assert_close(first,second,rtol=0,atol=1e-6)
        report.update(validation_reproducible=True,validation_loss_pair=[float(first),float(second)],
                      validation_preserves_training_rng=True,GT_normalization_does_not_mutate_profile_source_batch=True)
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():raise
        report.update(fits=False,error=str(exc),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
