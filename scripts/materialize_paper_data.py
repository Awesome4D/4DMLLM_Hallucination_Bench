#!/usr/bin/env python3
"""Preserve and validate the exact thesis question/prediction archives.

Default execution is offline. --download retrieves only the pinned original
archives when absent, and validates their hashes before extracting any member.
No predictions, labels, whitespace, or historical scores are rewritten.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVES = {
    'b4dl_dataset.zip': ('1FCSGJ2uNjNngHYlKhtdqxxtUzHqlmE-j', 4771522,
        'face9fed1cac30689a3891b38f33b5299e7dbf12faaf4e0f25044360e241a207'),
    'b4dl_eval.zip': ('1i5yQCf_J8E3zs7GGCcJkc3gwVrvjD5g8', 15886960,
        '04d3ab515a802d1440ce56d71b4654f271ddecef800bdb3f42caf4976c9097ed'),
}
CATEGORIES = ('object_existence', 'temporal_grounding', 'ego_relative_spatial',
              'motion_action', 'distance_depth')
VARIANTS = ('hallucination_v2', 'hallucination_v2_metatoken',
            'hallucination_v2_no_position_bias',
            'hallucination_v2_no_position_bias_metatoken')
CONDITIONS = ('hallucination_v2', 'hallucination_v2_metatoken',
              'hallucination_v2_no_position_bias',
              'hallucination_v2_no_position_bias_metatoken',
              'hallucination_v2_temporal_shuffle_cd',
              'hallucination_v2_temporal_shuffle_cd_metatoken',
              'hallucination_v2_temporal_shuffle_cd_alpha05',
              'hallucination_v2_temporal_shuffle_cd_alpha05_metatoken',
              'hallucination_v2_temporal_shuffle_cd_alpha15',
              'hallucination_v2_temporal_shuffle_cd_alpha15_metatoken')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> dict:
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(data)).encode()
                                        + b'\0' + data).hexdigest()}


def member_path(name: str, root: str) -> PurePosixPath:
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
            f'Unsafe archive member: {name}')
    require(bool(p.parts) and p.parts[0] == root, f'Unexpected root: {name}')
    return p


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        require(path.read_bytes() == payload,
                f'Refusing to overwrite changed source data: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def obtain(archive_dir: Path, name: str, download: bool) -> Path:
    drive_id, size, sha = ARCHIVES[name]
    path = archive_dir / name
    if not path.exists():
        require(download, f'Missing {path}. Supply the original ZIP or use --download.')
        import gdown
        archive_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.zip.part')
        gdown.download(id=drive_id, output=str(temporary), quiet=False)
        require(temporary.exists(), f'Download failed: {name}')
        require(digest(temporary.read_bytes())['sha256'] == sha,
                f'Download hash mismatch: {name}')
        temporary.replace(path)
    data = path.read_bytes()
    require(len(data) == size and digest(data)['sha256'] == sha,
            f'Archive differs from the user-provided original: {name}')
    return path


def materialize(archive_dir: Path, data_dir: Path, download: bool) -> tuple[list, list]:
    # Validate both archives before any extraction.
    archives = [obtain(archive_dir, name, download) for name in ARCHIVES]
    manifest, excluded = [], []
    for path in archives:
        root = path.stem
        with zipfile.ZipFile(path) as z:
            names = set()
            for info in sorted(z.infolist(), key=lambda i: i.filename):
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                if '__MACOSX' in parts or PurePosixPath(info.filename).name == '.DS_Store':
                    excluded.append({'archive': path.name, 'member': info.filename})
                    continue
                p = member_path(info.filename, root)
                require(info.filename not in names, f'Duplicate archive member: {p}')
                names.add(info.filename)
                require(not stat.S_ISLNK(info.external_attr >> 16), f'Symlink: {p}')
                allowed = {'.json'} if root == 'b4dl_dataset' else {'.csv', '.jsonl'}
                require(p.suffix in allowed, f'Unexpected data type: {p}')
                payload = z.read(info)
                target = data_dir.joinpath(*p.parts)
                require(target.resolve().is_relative_to(data_dir.resolve()),
                        f'Target escapes data directory: {target}')
                write_immutable(target, payload)
                manifest.append({'path': 'data/' + str(p), 'archive': path.name,
                                 'member': info.filename, **digest(payload)})
    expected = {m['path'].removeprefix('data/') for m in manifest}
    actual = {p.relative_to(data_dir).as_posix() for root in ('b4dl_dataset', 'b4dl_eval')
              for p in (data_dir / root).rglob('*') if p.is_file()}
    require(actual == expected, 'Unexpected or missing files in the immutable data trees.')
    return manifest, excluded


def key(row: dict) -> tuple:
    return row['scene_id'], row['question']


def validate_records(data_dir: Path) -> tuple[dict, dict, dict]:
    sets, rowcounts, scene_map = {}, {}, {}
    for variant in VARIANTS:
        for category in CATEGORIES:
            rel = f'b4dl_dataset/{variant}/{category}.json'
            rows = json.loads((data_dir / rel).read_text())
            require(len(rows) == 2000, f'Expected 2000 questions: {rel}')
            lookup = {key(r): r for r in rows}
            require(len(lookup) == 2000, f'Duplicate question key: {rel}')
            sets[variant, category] = lookup
            rowcounts['data/' + rel] = len(rows)
            if variant == 'hallucination_v2':
                for r in rows:
                    sid, token = r['scene_id'], r.get('scene_token')
                    s = scene_map.setdefault(sid, {'scene_id': sid, 'scene_token': token,
                                                   'reference_samples': {}})
                    require(s['scene_token'] == token, f'Conflicting scene tokens: {sid}')
                    if r.get('sample_token') is not None:
                        f = str(r['frame'])
                        old = s['reference_samples'].setdefault(f, r['sample_token'])
                        require(old == r['sample_token'], f'Conflicting frame mapping: {sid}/{f}')
    errors = 0
    for condition in CONDITIONS:
        variant = condition if 'no_position_bias' in condition else (
            'hallucination_v2_metatoken' if condition.endswith('_metatoken') else 'hallucination_v2')
        for category in CATEGORIES:
            rel = f'b4dl_eval/{condition}/{category}_predictions'
            with (data_dir / (rel + '.jsonl')).open() as f:
                rows = [json.loads(line) for line in f if line.strip()]
            with (data_dir / (rel + '.csv')).open(newline='') as f:
                csv_rows = list(csv.DictReader(f))
            require(len(rows) == len(csv_rows) == 2000, f'Prediction count mismatch: {rel}')
            lookup = sets[variant, category]
            require({key(r) for r in rows} == set(lookup), f'Question coverage mismatch: {rel}')
            for r, c in zip(rows, csv_rows):
                q = lookup[key(r)]
                for field in ('scene_id', 'question', 'prediction', 'reference_answer', 'parsed_prediction'):
                    expected = '' if r.get(field) is None else str(r[field])
                    require(c.get(field, '') == expected, f'CSV/JSONL mismatch: {rel}/{field}')
                for pred_field, question_field in [('reference_answer', 'answer'),
                        ('correct_option', 'correct_option'), ('options', 'options'), ('prompt', 'prompt')]:
                    require(r.get(pred_field) == q.get(question_field),
                            f'Prediction/reference mismatch: {rel}/{pred_field}')
                errors += bool(r.get('error'))
            rowcounts['data/' + rel + '.csv'] = 2000
            rowcounts['data/' + rel + '.jsonl'] = 2000
    require(len(scene_map) == 150, 'Expected exactly 150 scene IDs.')
    require(all(s['scene_token'] for s in scene_map.values()), 'Missing scene tokens.')
    require(len({s['scene_token'] for s in scene_map.values()}) == 150,
            'Scene ID to token mapping is not one-to-one.')
    summary = {'status': 'PASS', 'unique_questions': 10000, 'unique_scenes': 150,
               'question_variants': 4, 'question_records_all_variants': 40000,
               'model_configurations': 2, 'conditions_per_configuration': 5,
               'prediction_runs': 10, 'unique_saved_predictions': 100000,
               'benchmark_json_files': 20, 'prediction_jsonl_files': 50,
               'prediction_csv_files': 50, 'csv_duplicates_jsonl_records': True,
               'logged_inference_errors': errors,
               'reference_matching': 'All predictions match exact question, prompt, options and answer.',
               'scope': 'All outputs in the supplied thesis archives, not unprovided experiments or logits.'}
    return summary, rowcounts, scene_map


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive-dir', type=Path, default=Path('data/archives'))
    ap.add_argument('--data-dir', type=Path, default=Path('data'))
    ap.add_argument('--download', action='store_true')
    args = ap.parse_args()
    files, excluded = materialize(args.archive_dir, args.data_dir, args.download)
    summary, rowcounts, scenes = validate_records(args.data_dir)
    for f in files:
        f['records'] = rowcounts[f['path']]
    out = args.data_dir / 'manifests'
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'source': 'User-provided original archives',
                'archives': {n: {'bytes': a[1], 'sha256': a[2], 'google_drive_id': a[0]}
                             for n, a in ARCHIVES.items()},
                'files': files, 'excluded_os_metadata': excluded, 'counts': summary}
    for name, value in [('assets.json', manifest), ('verification.json', summary),
                        ('benchmark_scenes.json', {'scenes': [scenes[k] for k in sorted(scenes)]})]:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
