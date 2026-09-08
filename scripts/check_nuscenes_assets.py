#!/usr/bin/env python3
"""Map benchmark scenes to official nuScenes keyframes and check local files.

No SDK or checkpoint is required. --manifest-only needs metadata, not point
clouds. Unknown/mismatching B4DL frame mappings cause a failure, never a guess.
The check verifies metadata alignment and file presence, not sensor calibration,
point-cloud contents, or fidelity to a historical learned feature cache.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path, PurePosixPath

TABLES = ('scene', 'sample', 'sample_data', 'sensor', 'calibrated_sensor',
          'ego_pose', 'sample_annotation', 'instance', 'category')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_table(meta, name):
    path = meta / (name + '.json')
    require(path.is_file(), f'Missing metadata: {path}')
    return json.loads(path.read_text())


def inspect(dataroot: Path, scenes_path: Path, output: Path,
            manifest_only=False, include_sweeps=False, camera_channels=()):
    dataroot = dataroot.expanduser().resolve()
    meta = dataroot / 'v1.0-trainval'
    require(meta.is_dir(), f'Expected {meta}. Use the dataset root, not the metadata subdirectory.')
    for name in TABLES:
        require((meta / (name + '.json')).is_file(), f'Missing metadata table: {name}.json')
    requested = json.loads(scenes_path.read_text())['scenes']
    scenes = {r['token']: r for r in read_table(meta, 'scene')}
    samples = {r['token']: r for r in read_table(meta, 'sample')}
    sensors = {r['token']: r['channel'] for r in read_table(meta, 'sensor')}
    channels = {r['token']: sensors[r['sensor_token']]
                for r in read_table(meta, 'calibrated_sensor')}
    sample_data = read_table(meta, 'sample_data')
    camera_channels = set(camera_channels)
    require(camera_channels <= set(sensors.values()), 'Unknown camera channel.')
    selected_channels = {'LIDAR_TOP'} | camera_channels
    frame_records, sample_ids = [], set()
    for entry in requested:
        token = entry['scene_token']
        require(token in scenes, f'Scene {entry["scene_id"]} is absent from v1.0-trainval.')
        scene = scenes[token]
        chain, visited, current = [], set(), scene['first_sample_token']
        while current:
            require(current not in visited and current in samples, f'Broken sample chain: {token}')
            visited.add(current)
            sample = samples[current]
            require(sample['scene_token'] == token, 'Sample chain crosses scene boundaries.')
            chain.append(sample)
            current = sample['next']
        require(len(chain) == scene['nbr_samples'], f'Frame count mismatch: {token}')
        for frame, sample_token in entry.get('reference_samples', {}).items():
            i = int(frame)
            require(0 <= i < len(chain) and chain[i]['token'] == sample_token,
                    f'B4DL frame-order mismatch for {entry["scene_id"]}, frame {frame}. '
                    'Restore the original B4DL scene metadata instead of guessing.')
        for i, sample in enumerate(chain):
            sample_ids.add(sample['token'])
            frame_records.append({'scene_id': entry['scene_id'], 'scene_token': token,
                                  'scene_name': scene['name'], 'frame': i,
                                  'sample_token': sample['token'], 'timestamp': sample['timestamp']})
    by_sample, selected = {}, []
    for sd in sample_data:
        if sd['sample_token'] not in sample_ids:
            continue
        channel = channels[sd['calibrated_sensor_token']]
        if channel not in selected_channels:
            continue
        if not sd['is_key_frame'] and not (include_sweeps and channel == 'LIDAR_TOP'):
            continue
        filename = sd['filename']
        p = PurePosixPath(filename)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in filename,
                f'Unsafe sensor filename: {filename}')
        selected.append(filename)
        if sd['is_key_frame']:
            k = (sd['sample_token'], channel)
            require(k not in by_sample, f'Duplicate keyframe sensor record: {k}')
            by_sample[k] = filename
    for frame in frame_records:
        frame['files'] = {}
        for channel in sorted(selected_channels):
            k = (frame['sample_token'], channel)
            require(k in by_sample, f'Missing keyframe metadata: {k}')
            frame['files'][channel] = by_sample[k]
    filenames = sorted(set(selected))
    missing = [] if manifest_only else [f for f in filenames
              if not (dataroot / f).is_file() or (dataroot / f).stat().st_size == 0]
    status = 'MANIFEST_ONLY' if manifest_only else ('FAIL' if missing else 'PASS')
    report = {'status': status, 'version': 'v1.0-trainval',
              'scene_count': len(requested), 'keyframe_count': len(frame_records),
              'selected_channels': sorted(selected_channels),
              'include_lidar_sweeps': include_sweeps, 'required_sensor_files': len(filenames),
              'missing_file_count': None if manifest_only else len(missing),
              'scope': 'Metadata alignment and nonempty file presence only. Not a model/feature test.',
              'frames': frame_records, 'missing_files': missing}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    output.with_suffix('.files.txt').write_text('\n'.join(filenames) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('frames', 'missing_files')}, indent=2))
    return not missing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot', type=Path, required=True)
    p.add_argument('--scenes', type=Path, default=Path('data/manifests/benchmark_scenes.json'))
    p.add_argument('--output', type=Path, default=Path('data/local/nuscenes_assets.json'))
    p.add_argument('--manifest-only', action='store_true')
    p.add_argument('--include-sweeps', action='store_true')
    p.add_argument('--camera-channels', nargs='*', default=[],
                   choices=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
    a = p.parse_args()
    try:
        ok = inspect(a.dataroot, a.scenes, a.output, a.manifest_only,
                     a.include_sweeps, a.camera_channels)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
