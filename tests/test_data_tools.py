"""Synthetic unit tests. These do not claim to validate real nuScenes files."""
import importlib.util
import json
from pathlib import Path
import pytest


def module(name):
    p = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mat = module('materialize_paper_data')
raw = module('check_nuscenes_assets')


@pytest.mark.parametrize('name', ['/b4dl_eval/a.csv', 'b4dl_eval/../a.csv',
                                  'other/a.csv', r'b4dl_eval\a.csv'])
def test_reject_unsafe_member(name):
    with pytest.raises(ValueError):
        mat.member_path(name, 'b4dl_eval')


def test_original_bytes_are_immutable(tmp_path):
    p = tmp_path / 'a.json'
    mat.write_immutable(p, b'{"answer":"A"}\n')
    mat.write_immutable(p, b'{"answer":"A"}\n')
    with pytest.raises(ValueError):
        mat.write_immutable(p, b'{"answer":"B"}\n')
    assert p.read_bytes() == b'{"answer":"A"}\n'


def make_metadata(tmp_path):
    root = tmp_path / 'nuscenes'
    meta = root / 'v1.0-trainval'
    meta.mkdir(parents=True)
    tables = {name: [] for name in raw.TABLES}
    tables.update({
        'scene': [{'token': 'scene1', 'name': 'scene-test', 'first_sample_token': 's0', 'nbr_samples': 2}],
        'sample': [{'token': 's0', 'scene_token': 'scene1', 'next': 's1', 'timestamp': 0},
                   {'token': 's1', 'scene_token': 'scene1', 'next': '', 'timestamp': 500000}],
        'sensor': [{'token': 'lidar', 'channel': 'LIDAR_TOP'}],
        'calibrated_sensor': [{'token': 'cal', 'sensor_token': 'lidar'}],
        'sample_data': [{'sample_token': s, 'calibrated_sensor_token': 'cal',
                         'is_key_frame': True, 'filename': f'samples/LIDAR_TOP/{s}.bin'}
                        for s in ('s0', 's1')],
    })
    for name, rows in tables.items():
        (meta / (name + '.json')).write_text(json.dumps(rows))
    scenes = tmp_path / 'scenes.json'
    scenes.write_text(json.dumps({'scenes': [{'scene_id': 'id1', 'scene_token': 'scene1',
                                             'reference_samples': {'0': 's0', '1': 's1'}}]}))
    return root, scenes


def test_manifest_then_missing_then_present(tmp_path):
    root, scenes = make_metadata(tmp_path)
    out = tmp_path / 'report.json'
    assert raw.inspect(root, scenes, out, manifest_only=True)
    assert json.loads(out.read_text())['status'] == 'MANIFEST_ONLY'
    assert not raw.inspect(root, scenes, out)
    assert json.loads(out.read_text())['missing_file_count'] == 2
    for s in ('s0', 's1'):
        p = root / f'samples/LIDAR_TOP/{s}.bin'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'not-real-points')
    assert raw.inspect(root, scenes, out)
    assert json.loads(out.read_text())['keyframe_count'] == 2


def test_frame_mismatch_fails(tmp_path):
    root, scenes = make_metadata(tmp_path)
    rows = json.loads(scenes.read_text())
    rows['scenes'][0]['reference_samples']['1'] = 's0'
    scenes.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='frame-order mismatch'):
        raw.inspect(root, scenes, tmp_path / 'out.json', manifest_only=True)
