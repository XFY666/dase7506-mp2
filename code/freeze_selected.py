"""Freeze or verify a complete selected checkpoint and its local source closure."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_inventory(checkpoint):
    checkpoint=Path(checkpoint).resolve();files={};total=0;disk=0
    for path in sorted(checkpoint.rglob('*')):
        if not path.is_file():continue
        if checkpoint not in path.resolve().parents:raise RuntimeError('Asset escapes checkpoint')
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as archive:size=sum(x.file_size for x in archive.infolist())
        else:size=path.stat().st_size
        files[path.relative_to(checkpoint).as_posix()]=dict(sha256=sha(path),uncompressed_bytes=size,file_bytes=path.stat().st_size)
        total+=size;disk+=path.stat().st_size
    if not files:raise ValueError('Empty checkpoint')
    return dict(files=files,uncompressed_bytes=total,file_bytes=disk)


def verify(manifest):
    record=json.loads(Path(manifest).read_text(encoding='utf-8'))
    for relative,digest in record['source_sha256'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Frozen source changed: '+relative)
    if sha(ROOT/record['selection_record'])!=record['selection_record_sha256']:raise RuntimeError('Selection record changed')
    if sha(ROOT/record['prevalidation_record'])!=record['prevalidation_record_sha256']:raise RuntimeError('Prevalidation freeze changed')
    if sha(ROOT.parent/'docs/optimization_selection_policy.md')!=record['selection_policy_sha256']:raise RuntimeError('Selection policy changed')
    if asset_inventory(ROOT/record['checkpoint'])!=record['asset_inventory']:raise RuntimeError('Frozen checkpoint changed')
    if record['asset_inventory']['uncompressed_bytes']>16*2**20:raise RuntimeError('Asset size limit exceeded')
    return record


def freeze(checkpoint,manifest,method):
    from optimization_finalists import source_closure
    checkpoint=Path(checkpoint).resolve();manifest=Path(manifest).resolve()
    if manifest.exists():raise FileExistsError('Preserve an existing freeze; use --verify')
    choice=ROOT/'runs/optimization_final_selection/selection.json'
    selection=json.loads(choice.read_text(encoding='utf-8'));selected=selection['selected']
    prepath=ROOT/'runs/optimization_final_selection/prevalidation_freeze.json'
    if sha(prepath)!=selection['prevalidation_freeze_sha256']:raise RuntimeError('Changed prevalidation evidence')
    pre=json.loads(prepath.read_text(encoding='utf-8'))
    if sha(ROOT/'runs/optimization_finalists/shortlist.json')!=pre['shortlist_sha256']:raise RuntimeError('Changed finalist shortlist')
    if sha(ROOT.parent/'docs/optimization_selection_policy.md')!=selection['selection_policy_sha256']:raise RuntimeError('Changed prospective selection policy')
    if selected['name']=='D':raise ValueError('D retains its historical freeze; do not create a new test selection')
    inventory=asset_inventory(checkpoint)
    expected={k:v['sha256'] for k,v in inventory['files'].items()}
    if expected!=selected['checkpoint_sha256']:raise RuntimeError('Deployment differs from selected checkpoint')
    if inventory['uncompressed_bytes']>16*2**20:raise RuntimeError('Asset budget exceeded')
    hashes=source_closure()
    if hashes!=pre['source_sha256']:raise RuntimeError('Model implementation changed after validation freeze')
    method=Path(method).resolve()
    if method.parent!=ROOT or method.name not in ('submission.py','optimized_selection.py'):
        raise ValueError('Use the audited submission or optimized_selection entry point')
    for name in ('submission.py','ensemble.py','freeze_selected.py','measure_selected.py','optimization_validate_finalists.py','train_optimized.py','verify_optimized_checkpoint.py'):
        hashes[name]=sha(ROOT/name)
    record=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        method=method.relative_to(ROOT).as_posix(),selected_name=selected['name'],source_sha256=hashes,
        selection_record=choice.relative_to(ROOT).as_posix(),selection_record_sha256=sha(choice),asset_inventory=inventory,
        prevalidation_record=prepath.relative_to(ROOT).as_posix(),prevalidation_record_sha256=sha(prepath),
        selection_policy_sha256=selection['selection_policy_sha256'],
        policy='Selected before final test; no test-based rollback',
        prior_exposure='Earlier frozen methods were scored historically. All new experiments/selection use supplied training and validation evidence only.')
    manifest.parent.mkdir(parents=True,exist_ok=True)
    with manifest.open('x',encoding='utf-8') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    verify(manifest)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=ROOT/'runs/optimized_bundle')
    p.add_argument('--manifest',type=Path,default=ROOT/'runs/optimized_freeze.json')
    p.add_argument('--method',type=Path,default=ROOT/'submission.py');p.add_argument('--verify',action='store_true');a=p.parse_args()
    result=verify(a.manifest) if a.verify else freeze(a.checkpoint,a.manifest,a.method)
    print(json.dumps(dict(status='verified',selected=result['selected_name'],uncompressed_bytes=result['asset_inventory']['uncompressed_bytes'])))
