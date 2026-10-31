"""Retrieve only course-provided files, requiring the supplied SHA-256 hashes.

The default action is a plan. --download is explicit; existing matching files
are retained and an existing mismatching file is never replaced. This program
does not parse examples, fit a model, make predictions or score answers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_COMMIT = 'bb801f4f2e7629d270df765ab12c333b2dc616bb'
MANIFEST_SHA256 = '2e9b714598e1addba61545f0d7ce4891d9d11aaf567fb341420d36694fcfd656'
UPSTREAM = f'https://raw.githubusercontent.com/xudongwu-0/DASE7506-MP2/{UPSTREAM_COMMIT}/'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def files_for(splits):
    path=ROOT/'code/PACKAGE_MANIFEST.json'
    if sha(path)!=MANIFEST_SHA256:raise RuntimeError('Supplied package manifest changed')
    manifest = json.loads(path.read_text(encoding='utf-8'))['files']
    wanted = {'code/data/LICENSE.txt', 'code/data/NOTICE.txt', 'code/data/manifest.json'}
    for split in splits:
        wanted.update({f'code/data/gsm8k_{split}.jsonl', f'code/data/candidates/gsm8k_{split}.jsonl',
                       f'code/data/candidates/gsm8k_{split}.npz'})
    result = {}
    for name in sorted(wanted):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Unsafe upstream path')
        digest = manifest[name]
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid supplied SHA-256')
        result[name] = digest
    return result


def retrieve(name, expected):
    target = ROOT/name
    if ROOT not in target.resolve().parents:raise ValueError('Data destination escapes the repository')
    if target.exists():
        if not target.is_file() or sha(target) != expected:
            raise RuntimeError('Existing file differs; preserve and inspect it: '+str(target))
        return 'existing verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.download-'+uuid.uuid4().hex)
    try:
        request = urllib.request.Request(UPSTREAM+name, headers={'User-Agent': 'DASE7506-MP2-reproduction'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('xb') as stream:
            for block in iter(lambda: response.read(1024*1024), b''):
                stream.write(block)
        if sha(temporary) != expected:
            raise RuntimeError('Upstream bytes do not match the supplied manifest: '+name)
        # Exclusive destination creation also protects against another fetcher.
        with temporary.open('rb') as source, target.open('xb') as destination:
            for block in iter(lambda: source.read(1024*1024), b''):
                destination.write(block)
        return 'downloaded and verified'
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=('train','validation','test'), default=['train'])
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    files = files_for(args.splits)
    if not args.download:
        print(json.dumps({'action':'plan only', 'upstream':UPSTREAM, 'files':files}, indent=2))
        return
    for name, expected in files.items():
        print(json.dumps({'file':name,'status':retrieve(name, expected)}), flush=True)


if __name__ == '__main__':
    main()
