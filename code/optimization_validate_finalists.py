"""One prospective validation pass for at most two frozen checkpoints."""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode=True
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parents[1]
OUT=ROOT/'runs/optimization_final_selection'
POLICY=ROOT.parent/'docs/optimization_selection_policy.md'
SHORTLIST=ROOT/'runs/optimization_finalists/shortlist.json'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def payload_size(directory):
    total=0
    for path in directory.rglob('*'):
        if not path.is_file():continue
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as z:total+=sum(i.file_size for i in z.infolist())
        else:total+=path.stat().st_size
    return total


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    value=dict(policy_sha256=sha(POLICY),source_sha256=sha(Path(__file__)),maximum_new_validation_checkpoints=2,
               minimum_hard_validation_correct=294,index='mean repeated hard OOF correct + hard validation correct',
               test_access=False,method_promotion=False,
               reference_d_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [
                   *sorted((ROOT/'runs/model_bundle').glob('*')),
                   ROOT/'runs/evaluation/final-validation-score.json',ROOT/'runs/analysis_validation/summary.json',
                   ROOT/'runs/evaluation/final-test-resources.json',
                   *[ROOT/n for n in ('submission.py','ensemble.py','selection.py','text_features.py','tree_model.py','common.py')]] if p.is_file()})
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==value,'Changed final validation protocol'
    else:dump(path,value)
    return value


def run():
    protocol=register()
    if (OUT/'selection.json').exists():raise RuntimeError('Final decision already recorded; do not reselect')
    final=read(SHORTLIST);candidates=final['candidates']
    assert 0<=len(candidates)<=2
    assert final['selection_policy_sha256']==protocol['policy_sha256']
    d_official=read(ROOT/'runs/evaluation/final-validation-score.json')
    d_analysis=read(ROOT/'runs/analysis_validation/summary.json')
    d_resources=read(ROOT/'runs/evaluation/final-test-resources.json')
    assert d_official['correct']==754 and d_analysis['metrics']['fusion']['hard_correct']==311
    assert d_analysis['configuration']['fusion']==read(ROOT/'runs/model_bundle/config.json')
    for file,digest in final['source_sha256'].items():assert sha(ROOT/file)==digest,file
    for candidate in candidates:
        checkpoint=(ROOT/candidate['checkpoint']).resolve()
        assert ROOT/'runs' in checkpoint.parents
        actual={p.relative_to(checkpoint).as_posix():sha(p) for p in checkpoint.rglob('*') if p.is_file()}
        assert actual==candidate['checkpoint_sha256'],candidate['name']
        assert payload_size(checkpoint)==candidate['uncompressed_bytes']<=16*2**20
        assert candidate['probe_seconds']<=25
    freeze=dict(shortlist_sha256=sha(SHORTLIST),policy_sha256=protocol['policy_sha256'],
                candidates=candidates,source_sha256=final['source_sha256'])
    if (OUT/'prevalidation_freeze.json').exists():assert read(OUT/'prevalidation_freeze.json')==freeze,'Checkpoint changed after validation began'
    else:dump(OUT/'prevalidation_freeze.json',freeze)
    # No training is reachable from this program. Test files are prohibited.
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).name.startswith('gsm8k_test'):
            raise RuntimeError('No test access during validation selection')
    sys.addaudithook(guard)
    from common import DATA,parse_answer,read_jsonl
    bank=read_jsonl(DATA/'candidates/gsm8k_validation.jsonl');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    assert len(bank)==len(refs)==798
    hard=np.array([len({parse_answer(c['text']) for c in b['candidates'] if parse_answer(c['text']) is not None})>=2 for b in bank])
    assert int(hard.sum())==355
    labels=np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(r['answer']) for c in b['candidates']] for b,r in zip(bank,refs)])
    python=TASK/'.local/checkenv/Scripts/python.exe'
    results=[]
    for ordinal,candidate in enumerate(candidates):
        folder=OUT/f'candidate{ordinal+1}';folder.mkdir(exist_ok=True)
        predictions=folder/'validation_predictions.json';score=folder/'validation_score.json'
        tick=time.perf_counter()
        commands=[([str(python),str(ROOT/'predict.py'),'--method',str(ROOT/'optimized_selection.py'),'--checkpoint',str(ROOT/candidate['checkpoint']),
                    '--split','validation','--output',str(predictions)],predictions),
                  ([str(python),str(ROOT/'score_predictions.py'),'--split','validation','--predictions',str(predictions),'--output',str(score)],score)]
        for command,target in commands:
            process_record=folder/(target.stem+'_process.json')
            if target.exists():
                assert process_record.exists(),'Unverified partial output; preserve and inspect before resuming'
                previous=read(process_record)
                assert previous['command']==command and previous['returncode']==0 and previous['output_sha256']==sha(target),'Changed/stale validation output'
                continue
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONUTF8':'1'},
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            dump(process_record,dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                                     output_sha256=sha(target) if target.exists() else None))
            if result.returncode:raise RuntimeError('Validation process failed: '+result.stderr[-1000:])
        predicted=read(predictions);official=read(score)
        assert predicted['ids']==[b['id'] for b in bank] and len(predicted['selected'])==798
        selected=np.array(predicted['selected']);correct=labels[np.arange(len(bank)),selected]
        assert int(correct.sum())==official['correct']
        count=int(correct[hard].sum())
        menu=read(ROOT/'runs/optimization_fusion/protocol.json')['menu']
        menu_index=next(i for i,r in enumerate(menu) if r['name']==candidate['name'])
        row=dict(**candidate,menu_index=menu_index,validation_correct=int(correct.sum()),validation_n=798,validation_hard_correct=count,
                 validation_hard_n=355,passes_guard=count>=294,deployment_index=float(candidate['oof_mean_hard_correct'])+count,
                 validation_process_seconds=time.perf_counter()-tick,predictions_sha256=sha(predictions),official_score=official)
        results.append(row);dump(OUT/'validation_results.json',results)
    baseline_oof=read(ROOT/'runs/optimization_baselines/summary.json')
    d_oof=float(np.mean([r['results']['D']['hard_correct'] for r in baseline_oof['repeats']]))
    assert d_oof==1727.5
    reference=dict(name='D',checkpoint='runs/model_bundle',oof_mean_hard_correct=d_oof,
                   validation_correct=754,validation_n=798,validation_hard_correct=311,validation_hard_n=355,
                   deployment_index=d_oof+311,passes_guard=True,uncompressed_bytes=payload_size(ROOT/'runs/model_bundle'),
                   probe_seconds=float(d_resources['total_fresh_process_seconds']),menu_index=-1,
                   measurement_scope='Previously frozen unchanged D checkpoint and its recorded validation/resource measurement')
    choices=[reference]+[r for r in results if r['passes_guard']]
    order=sorted(range(len(choices)),key=lambda i:(-choices[i]['deployment_index'],choices[i]['uncompressed_bytes'],
                 choices[i]['probe_seconds'],0 if choices[i]['name']=='D' else 1,choices[i]['menu_index']))
    winner=choices[order[0]]
    selection=dict(selected=winner,all_new_validation_results=results,reference=reference,
        decision_criterion='Prespecified deployment index after validation guard and resource checks; not an accuracy estimate',
        selection_policy_sha256=sha(POLICY),prevalidation_freeze_sha256=sha(OUT/'prevalidation_freeze.json'),
        selected_utc=datetime.now(timezone.utc).isoformat(),test_access=False,method_promoted=False,
        next_action='Verify interfaces, freeze exact deployment, then one final official test evaluation if method changes; never choose rollback from test')
    assert all(sha(ROOT/file)==digest for file,digest in final['source_sha256'].items())
    dump(OUT/'selection.json',selection)
    dump(OUT/'summary.json',dict(status='complete',new_validation_evaluations=len(results),selected_name=winner['name'],
         validation_hard_correct=winner['validation_hard_correct'],model_fit_calls=0,test_access=False))
    dump(OUT/'progress.json',dict(stage='complete',completed_fits=0,planned_fits=0,
         detail='预先规定的验证与选模完成；尚未修改正式入口或进行测试',updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    p=argparse.ArgumentParser();a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--register-only',action='store_true');a.add_argument('--run',action='store_true');args=p.parse_args()
    if args.register_only:register()
    else:run()
