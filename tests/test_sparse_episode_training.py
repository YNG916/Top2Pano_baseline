"""Sparse sampling covers episodes; fixed validation RNG does not disturb training."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adapters.mvwd_level1 import MVWDLevel1, stratified_episode_frames
from adapters.mvwd_raw import sha256_file
from tests.test_mvwd_level1 import payload


def test_two_frames_cover_temporal_halves_without_using_labels():
    frames = stratified_episode_frames(60, 2, 'episode', 42)
    assert 0 <= frames[0] < 30 <= frames[1] < 60
    assert frames == stratified_episode_frames(60, 2, 'episode', 42)
    assert len({stratified_episode_frames(60, 2, str(i), 42) for i in range(100)}) > 20
    assert len(stratified_episode_frames(60, 1, 'episode', 42)) == 1
    for count in [0, 61, 1.5, True]:
        with pytest.raises(ValueError):
            stratified_episode_frames(60, count, 'episode')


def test_sparse_dataset_keeps_every_episode_and_all_observers(payload):
    root, segmentation, _ = payload
    split = root / 'splits/train.jsonl'
    record = json.loads(split.read_text())
    records = [{**record, 'id': 'ep_%d' % i, 'robots': 3} for i in range(5)]
    split.write_text(''.join(json.dumps(r)+'\n' for r in records))
    meta = json.loads((root/'dataset.json').read_text())
    meta['splits']['train']['count'] = 5
    meta['checksums']['splits/train.jsonl'] = sha256_file(split)
    (root/'dataset.json').write_text(json.dumps(meta))
    ds = MVWDLevel1(root, bev_size=64, target_size=(64,128), robots=(0,1,2),
                   frames_per_episode=2, segmentation_root=segmentation)
    assert len(ds) == 5*2*3
    tuples = {(r['id'], robot, frame) for r,robot,frame in map(ds.query,range(len(ds)))}
    for record in records:
        frames = stratified_episode_frames(record['frames'],2,record['id'])
        assert all((record['id'],r,f) in tuples for r in range(3) for f in frames)
    with pytest.raises(ValueError, match='explicit frames'):
        MVWDLevel1(root, bev_size=64, target_size=(64,128),frames=[0],frames_per_episode=2)


def test_fixed_validation_repeats_noise_and_preserves_training_rng():
    from mvwd_model import CombinedMVWDModel
    current = SimpleNamespace(config={'training':{'validation_seed':42}},
                              _step=lambda batch,prefix:torch.rand(10))
    batch={'episode_id':['a'],'robot_id':torch.tensor([0]),'frame_index':torch.tensor([15])}
    torch.manual_seed(123); before=torch.get_rng_state().clone()
    first=CombinedMVWDModel.validation_step(current,batch,0)
    assert torch.equal(before,torch.get_rng_state())
    torch.rand(30); before=torch.get_rng_state().clone()
    second=CombinedMVWDModel.validation_step(current,batch,0)
    assert torch.equal(first,second) and torch.equal(before,torch.get_rng_state())
    other=CombinedMVWDModel.validation_step(current,{**batch,'frame_index':torch.tensor([45])},0)
    assert not torch.equal(first,other)


def test_validation_monitor_checkpoint_and_plateau_stop(tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader,TensorDataset
    from mvwd_training_control import checkpoint_callbacks,convergence_callbacks
    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
        def training_step(self,batch,index):return self.weight.square().mean()
        def validation_step(self,batch,index):self.log('val/loss_simple',torch.tensor(.5),batch_size=1)
        def configure_optimizers(self):return torch.optim.Adam(self.parameters(),lr=.01)
    settings={'monitor':'val/loss_simple','early_stopping':{'patience':2,'min_delta':.001}}
    callbacks=checkpoint_callbacks(tmp_path,True,2,monitor=settings['monitor'])+convergence_callbacks(settings,True)
    data=DataLoader(TensorDataset(torch.ones(2,1)),batch_size=1)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=10,min_epochs=3,logger=False,
                       callbacks=callbacks,num_sanity_val_steps=0,enable_progress_bar=False,
                       enable_model_summary=False)
    trainer.fit(Toy(),data,data)
    assert trainer.should_stop and trainer.global_step==6
    assert len(list((tmp_path/'checkpoints/best').glob('*.ckpt')))==1
