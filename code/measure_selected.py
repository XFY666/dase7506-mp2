"""One fresh official test evaluation of an already frozen selected method."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from freeze_selected import verify,sha

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,default=ROOT/'runs/optimized_freeze.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/optimization_evaluation');args=p.parse_args()
    frozen=verify(args.freeze);output=args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Preserve evaluation outputs; use a new directory only for explicitly requested reproduction')
    output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):env[key]='4'
    pred=output/'test_predictions.json';score=output/'test_score.json'
    commands=[[sys.executable,str(ROOT/'predict.py'),'--method',str(ROOT/frozen['method']),'--checkpoint',str(ROOT/frozen['checkpoint']),
               '--split','test','--output',str(pred)],
              [sys.executable,str(ROOT/'score_predictions.py'),'--split','test','--predictions',str(pred),'--output',str(score)]]
    with (output/'evaluation_started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(),freeze_sha256=sha(args.freeze),
                       selected_name=frozen['selected_name'],commands=commands),stream,indent=2)
    processes=[];start=time.perf_counter()
    for command in commands:
        tick=time.perf_counter()
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
        processes.append(dict(command=command,returncode=result.returncode,seconds=time.perf_counter()-tick,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:
            (output/'process_failure.json').write_text(json.dumps(processes,indent=2),encoding='utf-8')
            raise RuntimeError('Official evaluation failed: '+result.stderr[-1500:])
    elapsed=time.perf_counter()-start
    verify(args.freeze)
    record=dict(selected_name=frozen['selected_name'],split='test',threads=4,total_fresh_process_seconds=elapsed,
        prediction_fresh_process_seconds=processes[0]['seconds'],scoring_fresh_process_seconds=processes[1]['seconds'],
        asset_uncompressed_bytes=frozen['asset_inventory']['uncompressed_bytes'],asset_file_bytes=frozen['asset_inventory']['file_bytes'],
        asset_inventory=frozen['asset_inventory'],source_sha256=frozen['source_sha256'],processes=processes,
        local_cpu='AMD Ryzen 5 7600',platform_processor=platform.processor(),platform=platform.platform(),python=sys.version,
        local_within_limits=elapsed<=30 and frozen['asset_inventory']['uncompressed_bytes']<=16*2**20,
        eligibility_note='Local measurement; the reference Xeon Platinum8457C determines eligibility. All startup/import/loading/preprocessing and both processes are included.',
        selection_changed_after_test=False)
    (output/'test_resources.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(score=json.loads(score.read_text(encoding='utf-8')),seconds=elapsed,asset_uncompressed_bytes=record['asset_uncompressed_bytes'])))


if __name__=='__main__':main()
