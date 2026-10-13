"""A portable derivative must prune all loader paths and retain exact source lineage."""
import copy
import json
import struct

import numpy as np
import pytest
import trimesh

from adapters.mvwd_raw import sha256_file
from adapters.robot_assets import RobotAssets
from scripts.build_robot_asset_variant import remove_glb_parts
from tests.test_robot_rendering import robot_bundle


def test_glb_pruning_preserves_other_geometry_and_binary_data(tmp_path):
    scene = trimesh.Scene()
    for name, x in [('shell', 0.), ('deck', 2.)]:
        transform = np.eye(4); transform[0, 3] = x
        scene.add_geometry(trimesh.creation.box(), geom_name=name, node_name=name, transform=transform)
    path = tmp_path / 'robot.glb'; path.write_bytes(scene.export(file_type='glb'))
    original = trimesh.load(path, force='scene', process=False)
    def tail(data):
        return data[20 + struct.unpack('<I', data[12:16])[0]:]
    before = path.read_bytes()
    remove_glb_parts(path, {'shell'})
    result = trimesh.load(path, force='scene', process=False)
    assert set(result.geometry) == {'deck'}
    assert tail(before) == tail(path.read_bytes())
    np.testing.assert_array_equal(original.geometry['deck'].vertices, result.geometry['deck'].vertices)
    np.testing.assert_array_equal(original.geometry['deck'].faces, result.geometry['deck'].faces)
    np.testing.assert_array_equal(original.graph['deck'][0], result.graph['deck'][0])
    with pytest.raises(ValueError, match='Missing expected'):
        remove_glb_parts(path, {'missing'})


def test_asset_derivative_lineage_is_checked(robot_bundle):
    path = robot_bundle / 'manifest.json'
    source = path.read_bytes(); source_hash = sha256_file(path)
    (robot_bundle / 'SOURCE_MANIFEST.json').write_bytes(source)
    manifest = json.loads(source)
    manifest['asset_id'] += '-gt-aligned'
    manifest['adaptation'] = {'kind': 'native_top_body_hidden', 'revision': 1,
        'source_asset_id': 'fixture', 'source_manifest_file': 'SOURCE_MANIFEST.json',
        'source_manifest_sha256': source_hash, 'hidden_prim_paths': ['chassis_link/visual/top_body']}
    receipt = robot_bundle / 'SOURCE_MANIFEST.json'
    manifest['files']['SOURCE_MANIFEST.json'] = {'size': receipt.stat().st_size, 'sha256': sha256_file(receipt)}
    path.write_text(json.dumps(manifest))
    assert RobotAssets(robot_bundle).provenance()['source_manifest_sha256'] == source_hash
    manifest['adaptation']['source_manifest_sha256'] = 'incorrect'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='lineage mismatch'):
        RobotAssets(robot_bundle)


def test_source_checkpoint_requires_opt_in_and_exact_lineage():
    from mvwd_model import CombinedMVWDModel
    class Current:
        config = {'data': {'robot_rendering': True}, 'model': {},
                  'robot_asset_provenance': {'manifest_sha256': 'derived', 'source_manifest_sha256': 'source'}}
    checkpoint = {'hyper_parameters': {'config': {'data': {'robot_rendering': True},
                                                 'robot_asset_provenance': {'manifest_sha256': 'source'}}}}
    current = Current(); current.config = copy.deepcopy(current.config)
    with pytest.raises(ValueError, match='manifest differs'):
        CombinedMVWDModel.on_load_checkpoint(current, checkpoint)
    current.config['model']['allow_source_asset_checkpoint'] = True
    CombinedMVWDModel.on_load_checkpoint(current, checkpoint)
    current.config['robot_asset_provenance']['source_manifest_sha256'] = 'unrelated'
    with pytest.raises(ValueError, match='manifest differs'):
        CombinedMVWDModel.on_load_checkpoint(current, checkpoint)
