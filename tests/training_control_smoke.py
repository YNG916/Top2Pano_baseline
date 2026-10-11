"""CPU smoke check of Lightning 1.5 scheduling, unused parameters and Adam resume."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader, TensorDataset
from mvwd_training_control import checkpoint_callbacks


class Toy(pl.LightningModule):
    def __init__(self, expected=None):
        super().__init__()
        self.live = torch.nn.Linear(1, 1)
        self.unused = torch.nn.Parameter(torch.ones(1))
        self.expected = expected

    def training_step(self, batch, index):
        loss = self.live(batch[0]).square().mean()
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, index):
        self.log('val/total_loss', self.live(batch[0]).square().mean(), sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.01)

    def on_train_start(self):
        if self.expected is not None:
            for name, value in self.expected['state_dict'].items():
                torch.testing.assert_close(self.state_dict()[name].cpu(), value)
            actual = self.trainer.optimizers[0].state_dict()
            expected = self.expected['optimizer_states'][0]
            assert actual['param_groups'] == expected['param_groups']
            for index, state in expected['state'].items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(actual['state'][index][name].cpu(), value)
                    else:
                        assert actual['state'][index][name] == value
            assert self.global_step == self.expected['global_step']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    root = Path(args.output)
    torch.set_num_threads(1)
    data = DataLoader(TensorDataset(torch.ones(40, 1)), batch_size=1)
    for part, target in (('first', 5), ('resume', 7)):
        source = root / 'first/checkpoints/last.ckpt'
        expected = torch.load(source, map_location='cpu') if part == 'resume' else None
        model = Toy(expected)
        options = {'strategy': DDPPlugin(find_unused_parameters=True)} if args.devices > 1 else {}
        if args.validate:
            options['val_check_interval'] = 2
        trainer = pl.Trainer(accelerator='cpu', devices=args.devices, max_epochs=-1, min_epochs=0,
                             max_steps=target, callbacks=checkpoint_callbacks(root/part, args.validate, 2),
                             logger=False, enable_progress_bar=False, enable_model_summary=False,
                             num_sanity_val_steps=0, **options)
        trainer.fit(model, data, data if args.validate else None, ckpt_path=str(source) if expected is not None else None)
        assert trainer.global_step == target
        if trainer.is_global_zero:
            directory = root/part/'checkpoints'
            saved = torch.load(directory/'last.ckpt', map_location='cpu')
            assert saved['global_step'] == target and saved['optimizer_states'][0]['state']
            assert 1 <= len(list(directory.glob('step-*.ckpt'))) <= 2
            if args.validate:
                assert len(list((directory/'best').glob('*.ckpt'))) == 1
            assert torch.equal(saved['state_dict']['unused'], torch.ones(1))
        # This script re-enters for local DDP children; do not launch a second DDP
        # process group from the child after its first fit. A fresh invocation below
        # is used for the multi-process check.
        if args.devices > 1:
            break
    if int(os.environ.get('LOCAL_RANK', '0')) == 0:
        print('TRAINING_CONTROL_SMOKE_PASSED devices=%s' % args.devices)


if __name__ == '__main__':
    main()
