"""Audit official Top2Pano ZIP checkpoints with bounded HTTP range reads.

No network tensor is deserialized or executed. Only allowlisted metadata
constructors are decoded. ZIP CRCs screen every tensor; selected matching
storages are independently checked with SHA256. This is an audit, not an
initializer, downloader, or change to the released training algorithm.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import pickle
from pathlib import Path
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
HF_API = 'https://huggingface.co/api/models/freeA1/top2pano'


class PassiveMetadata:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class Storage:
    def __init__(self, kind, key, location, numel):
        self.kind, self.key, self.location, self.numel = kind, key, location, numel


def tensor_metadata(storage, offset, shape, stride, *unused):
    return dict(storage=storage.key, dtype=storage.kind, offset=offset,
                shape=list(shape), stride=list(stride), storage_numel=storage.numel)


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('collections', 'OrderedDict'): collections.OrderedDict,
            ('torch', 'FloatStorage'): 'float32',
            ('torch', 'IntStorage'): 'int32',
            ('torch', 'LongStorage'): 'int64',
            ('torch', 'HalfStorage'): 'float16',
            ('torch._utils', '_rebuild_tensor_v2'): tensor_metadata,
            ('_codecs', 'encode'): lambda value, encoding: value.encode(encoding),
            ('numpy', 'dtype'): PassiveMetadata,
            ('numpy.core.multiarray', 'scalar'): PassiveMetadata,
            ('pytorch_lightning.callbacks.model_checkpoint', 'ModelCheckpoint'): PassiveMetadata,
        }
        if (module, name) not in allowed:
            raise ValueError('Unrecognized checkpoint metadata constructor: %s.%s' % (module, name))
        return allowed[module, name]

    def persistent_load(self, value):
        if value[0] != 'storage' or len(value) != 5:
            raise ValueError('Unexpected persistent metadata')
        return Storage(*value[1:])


class RangeFile(io.RawIOBase):
    """Read a pinned HF file without accidentally downloading the full file."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError('Negative range offset')
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if not 0 <= size <= 8 * 1024 * 1024:
            raise ValueError('Refusing an HTTP read exceeding 8 MiB')
        if not size:
            return b''
        start, end = self.pos, self.pos + size - 1
        # A different URL per range prevents proxies reusing a cached 206 body.
        url = self.url + '?audit_range=%d-%d' % (start, end)
        request = urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (start, end)})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected = 'bytes %d-%d/%d' % (start, end, self.size)
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server did not honor the requested range')
            data = response.read(size + 1)
        if len(data) != size:
            raise ValueError('Unexpected HTTP range length')
        self.pos += size
        self.bytes_read += size
        return data


class Archive:
    def __init__(self, source):
        self.zip = zipfile.ZipFile(source)
        name = next(name for name in self.zip.namelist() if name.endswith('/data.pkl'))
        self.prefix = name.rsplit('/', 1)[0]
        self.metadata = MetadataUnpickler(io.BytesIO(self.zip.read(name))).load()
        self.state = self.metadata.get('state_dict', self.metadata)

    def entry(self, tensor):
        if tensor['offset'] != 0 or math.prod(tensor['shape']) != tensor['storage_numel']:
            raise ValueError('Audit requires complete, unsliced tensor storage')
        return self.zip.getinfo(self.prefix + '/data/' + str(tensor['storage']))

    def same_storage(self, name, other, other_name):
        left, right = self.state[name], other.state[other_name]
        fields = ('dtype', 'shape', 'stride', 'offset', 'storage_numel')
        if any(left[field] != right[field] for field in fields):
            return False
        a, b = self.entry(left), other.entry(right)
        return (a.file_size, a.CRC) == (b.file_size, b.CRC)

    def sha256(self, name):
        entry = self.entry(self.state[name])
        # Anchors should be small even for local checkpoints.
        if entry.file_size > 8 * 1024 * 1024:
            raise ValueError('Anchor tensor exceeds 8 MiB')
        return hashlib.sha256(self.zip.read(entry.filename)).hexdigest()


def compare_prefix(left, right, prefix, right_prefix=''):
    names = [name for name in left.state if name.startswith(prefix)]
    missing, changed, equal = [], [], 0
    for name in names:
        target = right_prefix + name[len(prefix):]
        if target not in right.state:
            missing.append(name)
        elif left.same_storage(name, right, target):
            equal += 1
        else:
            changed.append(name)
    return dict(tensor_count=len(names), matching_storage_crc_count=equal,
                differing_tensors=changed, missing_tensors=missing)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init', default=str(REPO / 'models/control_sd21_ini.ckpt'))
    parser.add_argument('--sd', default=str(REPO / 'models/v2-1_512-ema-pruned.ckpt'))
    parser.add_argument('--local-trained', help='Optional MVWD checkpoint to compare against initialization')
    parser.add_argument('--output', default=str(REPO / 'artifacts/mvwd/upstream_audit/weights.json'))
    args = parser.parse_args()
    initial = Archive(args.init)
    source = Archive(args.sd)
    receipt = json.loads(Path(args.init).with_suffix('.json').read_text())
    copied, mismatched, new = [], [], []
    for name in initial.state:
        target = 'model.diffusion_' + name[len('control_'):] if name.startswith('control_') else name
        if target not in source.state:
            new.append(name)
        elif initial.same_storage(name, source, target):
            copied.append(name)
        else:
            mismatched.append(name)
    report = {
        'method': 'All full tensor storages: dtype/shape/stride/size and ZIP CRC32; selected anchors: SHA256. CRC32 is a screening check, not a cryptographic proof for every tensor.',
        'initialization': {'path': str(Path(args.init).resolve()), 'sd_path': str(Path(args.sd).resolve()),
                           'copied_matching_storage_count': len(copied), 'copy_mismatches': mismatched,
                           'new_keys': new, 'receipt_new_keys_match': sorted(new) == sorted(receipt['new_keys']),
                           'source_sha256_matches_receipt': file_sha256(args.sd) == receipt['source_sha256'],
                           'init_sha256_matches_receipt': file_sha256(args.init) == receipt['output_sha256']},
        'official': {},
    }
    with urllib.request.urlopen(HF_API, timeout=45) as response:
        info = json.load(response)
    revision = info['sha']
    with urllib.request.urlopen(HF_API + '/tree/' + revision + '?recursive=true&expand=false', timeout=45) as response:
        files = json.load(response)
    report['hf_revision'] = revision
    for name in ('gibson.ckpt', 'matterport.ckpt'):
        item = next(item for item in files if item['path'] == name)
        url = 'https://huggingface.co/freeA1/top2pano/resolve/' + revision + '/' + name
        remote = RangeFile(url, item['size'])
        author = Archive(remote)
        result = {'url': url, 'size': item['size'], 'lfs_sha256': item.get('lfs', {}).get('oid'),
                  'epoch': author.metadata.get('epoch'), 'global_step': author.metadata.get('global_step'),
                  'lightning_version': author.metadata.get('pytorch-lightning_version'),
                  'has_hyper_parameters': 'hyper_parameters' in author.metadata,
                  'state_tensor_count': len(author.state),
                  'optimizer_state_count': len(author.metadata['optimizer_states'][0]['state']),
                  'density_vs_initialization': compare_prefix(author, initial, 'density_model.'),
                  'density_vae_vs_sd': compare_prefix(author, source, 'density_model.first_stage_model.', 'first_stage_model.'),
                  'render_vs_initialization': compare_prefix(author, initial, 'render_model.'),
                  'render_vae_vs_initialization': compare_prefix(author, initial, 'render_model.first_stage_model.', 'first_stage_model.'),
                  'sha256_anchors': {}}
        for branch in ('density_model.', 'render_model.'):
            for suffix in ('first_stage_model.decoder.conv_in.weight', 'first_stage_model.decoder.norm_out.weight',
                           'first_stage_model.post_quant_conv.weight', 'model.diffusion_model.input_blocks.0.0.weight',
                           'control_model.zero_convs.0.0.weight'):
                remote_name = branch + suffix
                a, b = author.sha256(remote_name), initial.sha256(suffix)
                result['sha256_anchors'][remote_name] = {'author': a, 'initialization': b, 'equal': a == b}
        result['http_bytes_read'] = remote.bytes_read
        report['official'][name] = result
        print('%s: epoch=%s step=%s density CRC matches=%s/%s, downloaded %.2f MiB' %
              (name, result['epoch'], result['global_step'],
               result['density_vs_initialization']['matching_storage_crc_count'],
               result['density_vs_initialization']['tensor_count'], remote.bytes_read / 1024**2), flush=True)
    if args.local_trained:
        trained = Archive(args.local_trained)
        report['local_trained'] = {'path': str(Path(args.local_trained).resolve()),
                                   'global_step': trained.metadata.get('global_step'),
                                   'density_vs_initialization': compare_prefix(trained, initial, 'density_model.'),
                                   'render_vs_initialization': compare_prefix(trained, initial, 'render_model.'),
                                   'render_vae_vs_initialization': compare_prefix(trained, initial, 'render_model.first_stage_model.', 'first_stage_model.')}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved ' + str(output))
    if mismatched or not all(report['initialization'][key] for key in
                             ('receipt_new_keys_match', 'source_sha256_matches_receipt', 'init_sha256_matches_receipt')):
        raise SystemExit('Initialization audit failed; see report')


if __name__ == '__main__':
    main()
