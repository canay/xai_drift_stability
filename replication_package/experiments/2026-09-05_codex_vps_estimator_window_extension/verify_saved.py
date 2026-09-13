"""Verify the portable saved-run payload; optionally extract to a new directory.

Uses the Python standard library. It runs no estimators or statistical analyses.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import stat
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ('..', '.') for p in path.parts):
        raise ValueError('unsafe archive member: ' + name)
    if ':' in name or chr(92) in name or str(path) != name:
        raise ValueError('nonportable archive member: ' + name)
    return path


def verify(extract=None):
    sums = ROOT / 'SHA256SUMS'
    checked = set()
    for line in sums.read_text(encoding='utf8').splitlines():
        expected, name = line.split('  ', 1)
        rel = safe_relative(name)
        if name in checked:
            raise ValueError('duplicate checksum entry: ' + name)
        checked.add(name)
        if digest((ROOT / rel).read_bytes()) != expected:
            raise ValueError('payload checksum mismatch: ' + name)
    required = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
                if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
    if checked != required:
        raise ValueError('checksum inventory differs from payload inventory')
    manifest_raw = (ROOT / 'raw_metadata/SNAPSHOT_MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_raw)
    expected = manifest['files']
    if manifest['source_exit_code'] != 0:
        raise ValueError('saved source run was not successful')
    seen = set()
    archives = [ROOT / ('saved_units_' + name + '.zip') for name in ('adult', 'electricity')]
    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                name = item.filename
                safe_relative(name)
                if name in seen or name not in expected or stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError('unexpected/duplicate/link archive member: ' + name)
                if digest(zipped.read(item)) != expected[name]:
                    raise ValueError('saved-unit checksum mismatch: ' + name)
                seen.add(name)
    for name in ('environment.json', 'run_status.json'):
        if digest((ROOT / 'raw_metadata' / name).read_bytes()) != expected[name]:
            raise ValueError('metadata checksum mismatch: ' + name)
        seen.add(name)
    if seen != set(expected):
        raise ValueError('incomplete saved snapshot')
    units = sum(name.endswith('/completed.json') for name in seen)
    if units != 720:
        raise ValueError('expected 720 saved units')
    if extract is not None:
        destination = Path(extract).resolve()
        if destination.exists():
            raise FileExistsError('extraction requires a new directory')
        if destination.is_relative_to(ROOT) or ROOT.is_relative_to(destination):
            raise ValueError('extract outside the release directory')
        destination.mkdir(parents=True, exist_ok=False)
        for archive in archives:
            with zipfile.ZipFile(archive) as zipped:
                for item in zipped.infolist():
                    path = destination / safe_relative(item.filename)
                    if not path.resolve().is_relative_to(destination):
                        raise ValueError('extraction target escapes destination')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as stream:
                        stream.write(zipped.read(item))
        for name in ('environment.json', 'run_status.json', 'SNAPSHOT_MANIFEST.json'):
            with (destination / name).open('xb') as stream:
                stream.write((ROOT / 'raw_metadata' / name).read_bytes())
        for name, expected_hash in expected.items():
            if digest((destination / name).read_bytes()) != expected_hash:
                raise ValueError('extracted checksum mismatch: ' + name)
    return {'status': 'PASS', 'payload_files': len(checked), 'saved_units': units,
            'snapshot_files': len(seen), 'extracted': extract is not None,
            'scientific_fingerprint': manifest['scientific_fingerprint'],
            'experiments_run': False, 'analyses_run': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', type=Path, help='new directory outside this release')
    args = parser.parse_args()
    print(json.dumps(verify(args.extract), indent=2))
