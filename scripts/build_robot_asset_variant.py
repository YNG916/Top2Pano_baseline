"""Build a portable visibility-aligned derivative; never edit the source bundle."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mvwd_runtime import load_config

HIDDEN_PRIM = 'chassis_link/visual/top_body'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def remove_glb_parts(path, hidden_nodes):
    """Rewrite only glTF JSON; retain all surviving binary geometry/textures exactly."""
    with path.open('rb') as stream:
        magic, version, total = struct.unpack('<III', stream.read(12))
        length, kind = struct.unpack('<II', stream.read(8))
        if magic != 0x46546c67 or version != 2 or kind != 0x4e4f534a or total != path.stat().st_size:
            raise ValueError('Expected a complete glTF 2 GLB')
        data = json.loads(stream.read(length))
        if any(k in data for k in ('skins', 'animations')):
            raise ValueError('Unexpected animated/skinned GLB')
        removed = {i for i, n in enumerate(data['nodes']) if n.get('name') in hidden_nodes}
        if len(removed) != len(hidden_nodes):
            raise ValueError('Missing expected native shell nodes')
        removed_meshes = {data['nodes'][i]['mesh'] for i in removed}
        keep = {i: j for j, i in enumerate(i for i in range(len(data['nodes'])) if i not in removed)}
        mesh_keep = {i: j for j, i in enumerate(i for i in range(len(data['meshes'])) if i not in removed_meshes)}
        nodes = []
        for i, node in enumerate(data['nodes']):
            if i not in keep:
                continue
            node = copy.deepcopy(node)
            if 'children' in node:
                node['children'] = [keep[n] for n in node['children'] if n in keep]
            if 'mesh' in node:
                node['mesh'] = mesh_keep[node['mesh']]
            nodes.append(node)
        data['nodes'] = nodes
        data['meshes'] = [m for i, m in enumerate(data['meshes']) if i in mesh_keep]
        for scene in data['scenes']:
            scene['nodes'] = [keep[n] for n in scene['nodes'] if n in keep]
        header = json.dumps(data, separators=(',', ':')).encode('utf-8')
        header += b' ' * (-len(header) % 4)
        tail_bytes = total - stream.tell()
        temporary = path.with_suffix('.glb.partial')
        with temporary.open('wb') as dest:
            dest.write(struct.pack('<III', magic, version, 20 + len(header) + tail_bytes))
            dest.write(struct.pack('<II', len(header), kind)); dest.write(header)
            shutil.copyfileobj(stream, dest, 4 * 1024 * 1024)
    temporary.replace(path)


def build(source, destination, output):
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom

    source = source.resolve(); destination = destination.resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('Derivative must be a separate sibling, not overwrite the source')
    staging = destination.with_name(destination.name + '.building')
    if destination.exists() or staging.exists():
        raise FileExistsError('Keep existing bundles/builds; choose a new destination')
    source_manifest_bytes = (source / 'manifest.json').read_bytes()
    manifest = json.loads(source_manifest_bytes)
    if manifest.get('format') != 'mvwd-robot-assets-v1' or manifest.get('adaptation'):
        raise ValueError('Expected original v1 bundle')
    source_hash = sha256(source / 'manifest.json')
    # Verify the original once before constructing a derivative.
    for name, receipt in manifest['files'].items():
        path = source / name
        if not path.is_file() or path.stat().st_size != receipt['size'] or sha256(path) != receipt['sha256']:
            raise ValueError('Source checksum mismatch: ' + name)
    print('Source checksums verified:', len(manifest['files']), flush=True)
    staging.mkdir()
    subprocess.run(['cp', '--reflink=auto', '-a', str(source) + '/.', str(staging)], check=True)
    parts_path = staging / 'geometry/parts.json'
    parts = json.loads(parts_path.read_text())
    removed_parts = [p for p in parts['parts'] if p['prim_path'] == HIDDEN_PRIM]
    if not removed_parts:
        raise ValueError('Source has no expected native top_body')
    hidden_nodes = {p['node'] for p in removed_parts}
    parts['parts'] = [p for p in parts['parts'] if p['prim_path'] != HIDDEN_PRIM]
    write_json(parts_path, parts)
    removed_triangles = 0
    for part in removed_parts:
        with np.load(staging / part['mesh_path'], allow_pickle=False) as mesh:
            removed_triangles += len(mesh['faces'])
        (staging / part['mesh_path']).unlink()
    for variant in manifest['variants']:
        path = staging / variant['glb']
        original_triangles = variant['triangles']
        remove_glb_parts(path, hidden_nodes)
        scene = trimesh.load(path, force='scene', process=False)
        variant['parts'] = len(scene.geometry)
        variant['triangles'] = sum(len(m.faces) for m in scene.geometry.values())
        variant['base_bounds_m'] = scene.bounds.tolist()
        if variant['parts'] != len(parts['parts']):
            raise ValueError('GLB/parts count mismatch')
        if variant['triangles'] != original_triangles - removed_triangles:
            raise ValueError('Triangle count mismatch')
        if any(n in scene.geometry for n in hidden_nodes):
            raise ValueError('Native top_body survived GLB pruning')
        print('Filtered', variant['name'], variant['parts'], variant['triangles'], flush=True)
        del scene
    usd_checks = []
    for relative in [manifest['entry_usd'], *[v['usd'] for v in manifest['variants']]]:
        stage = Usd.Stage.Open(str(staging / relative), load=Usd.Stage.LoadNone)
        prim = stage.GetPrimAtPath('/mobile_sensor_robot_v1/' + HIDDEN_PRIM)
        if not prim or not prim.IsA(UsdGeom.Imageable):
            raise ValueError('Missing native shell USD prim: ' + relative)
        stage.SetEditTarget(stage.GetRootLayer())
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        stage.GetRootLayer().Save()
        if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
            raise ValueError('USD shell still visible')
        usd_checks.append({'entry': relative, 'prim': str(prim.GetPath()), 'effective_visibility': 'invisible'})
        print('USD visibility authored:', relative, flush=True)
        del stage
    # Preserve the source manifest as a checked lineage receipt, not current checksums.
    (staging / 'SOURCE_MANIFEST.json').write_bytes(source_manifest_bytes)
    source_id = manifest['asset_id']
    manifest['asset_id'] = source_id + '-gt-aligned-top-body-hidden-v1'
    manifest['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    manifest['adaptation'] = {'kind': 'native_top_body_hidden', 'revision': 1,
        'source_asset_id': source_id, 'source_manifest_sha256': source_hash,
        'source_manifest_file': 'SOURCE_MANIFEST.json', 'hidden_prim_paths': [HIDDEN_PRIM],
        'removed_parts': [p['node'] for p in removed_parts], 'removed_triangles_per_variant': removed_triangles,
        'reason': 'Fixed visibility alignment supported by preview geometry/depth audit; original Isaac runtime state unconfirmed.',
        'ground_truth_data_modified': False, 'GT_visibility_used_at_runtime': False,
        'glb_policy': 'Remove nodes/meshes; surviving BIN data unchanged; unused BIN bytes remain.',
        'usd_policy': 'Author invisible on the native top_body prim in each operational entry layer.',
        'source_template_fingerprint_preserved_for_lineage': True}
    manifest['limitations'].append('Derived visibility-aligned asset; template fingerprint refers to source, not identity of adapted geometry.')
    loader = staging / 'load_robot_assets.py'
    text = loader.read_text()
    text = text.replace('def safe_file(root, relative):', '''def is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def safe_file(root, relative):''')
    text = text.replace('path.is_relative_to(root)', 'is_within(path, root)')
    text = text.replace('Path(asset).resolve().is_relative_to(root)', 'is_within(Path(asset).resolve(), root)')
    text = text.replace('Path(layer.realPath).resolve().is_relative_to(root)', 'is_within(Path(layer.realPath).resolve(), root)')
    loader.write_text(text)
    (staging / 'README.md').write_text('''# MVWD shared visibility-aligned robot assets

This is a derivative of MVWD_RobotAssets_v1. The source remains unchanged beside it.
The native chassis_link/visual/top_body is excluded from all GLB mesh records and
geometry/parts.json, and is invisible in every operational USD entry. Other parts,
identity colors, mast heights, camera mounts and coordinates remain unchanged.

Use this same asset root for all baselines and the proposed method. Ordinary GLB
loading and load_parts do not need extra filtering. USD renderers honor visibility;
custom USD mesh extractors must honor Imageable.ComputeVisibility() as usual.
Do not load vendor layers or provenance templates as operational robot entries.

SOURCE_MANIFEST.json and manifest.adaptation record exact source lineage. The
preserved template_fingerprint links dataset episodes to the source template; it
does not claim that the derivative is byte-identical to the production asset.
This fixed alignment is based on preview geometry/depth audits and does not prove
the original Isaac runtime visibility state. Dataset observations and GT are unchanged.

Portable standalone loader: load_robot_assets.py (NumPy/trimesh/Pillow; Python 3.8+).
Recheck: python load_robot_assets.py /path/to/this/bundle --verify
Optional USD closure checks additionally require usd-core in a compatible Python.
The inherited fidelity/material/articulation limitations and NVIDIA terms still apply.
Unused GLB BIN bytes retain original hidden geometry data, but hidden nodes and
meshes are absent: standard loaders/renderers cannot instantiate that geometry.
''')
    for name in ['EXPORT_STATUS.json', 'EXPORT_OWNERSHIP.json']:
        write_json(staging / name, {'status': 'derived_visibility_aligned_bundle',
            'source_manifest_sha256': source_hash, 'source_root': str(source),
            'destination_root': str(destination), 'adaptation': manifest['adaptation']})
    # Old verification is not evidence for the derivative; replace only after fresh checks.
    (staging / 'VERIFICATION.json').unlink(missing_ok=True)
    manifest['files'] = {str(p.relative_to(staging)): {'size': p.stat().st_size, 'sha256': sha256(p)}
        for p in sorted(staging.rglob('*')) if p.is_file() and p.name not in ('manifest.json', 'VERIFICATION.json')}
    write_json(staging / 'manifest.json', manifest)
    spec = importlib.util.spec_from_file_location('aligned_asset_loader', loader)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    report = module.verify_bundle(staging, usd_checks=True)
    report['effective_visibility'] = usd_checks
    report['adaptation'] = manifest['adaptation']
    write_json(output / 'verification.json', report)
    if not report['passed']:
        raise ValueError('Derived asset verification failed; staging preserved: ' + str(staging))
    # Confirm nothing in the source changed during this build.
    if (source / 'manifest.json').read_bytes() != source_manifest_bytes:
        raise ValueError('Source manifest changed during build')
    for name, receipt in json.loads(source_manifest_bytes)['files'].items():
        if sha256(source / name) != receipt['sha256']:
            raise ValueError('Source changed during build: ' + name)
    report['source_unchanged'] = True
    write_json(staging / 'VERIFICATION.json', report)
    staging.rename(destination)
    write_json(output / 'build.json', {'source': str(source), 'destination': str(destination),
        'manifest_sha256': sha256(destination / 'manifest.json'), 'verification': report,
        'numpy': np.__version__, 'trimesh': trimesh.__version__, 'usd_version': list(Usd.GetVersion())})
    print('Verified shared asset variant:', destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.source or Path(config['data']['root']) / 'robot_assets/MVWD_RobotAssets_v1'
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    build(source, args.destination, output)


if __name__ == '__main__':
    main()
