"""Descriptive validation analysis of the already-selected frozen method."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from common import DATA,load_inputs,read_jsonl,parse_answer
from answer_group_model import group_weights
from selection import majority,select
from optimized_selection import score_candidates

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/optimization_final_analysis'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def rescue_title(row, baseline_index):
    """Describe votes without calling a tied plurality a strict majority."""
    answers=[parse_answer(candidate['text']) for candidate in row['candidates']]
    counts=Counter(answer for answer in answers if answer is not None)
    baseline=answers[baseline_index]
    if baseline is not None and 2*counts[baseline]>sum(counts.values()):
        return 'Correct despite an incorrect strict majority'
    if counts and sum(count==max(counts.values()) for count in counts.values())>1:
        return 'Correct choice after an incorrect tied-plurality baseline'
    return 'Correct despite an incorrect plurality baseline'


REVIEWED_EXPLANATIONS={
    'gsm8k:fd23d86676d457a5edd2ad70cc5accd6b3e3c36e975e3b85316a5e7990357a36':
        'The return from school to home is another 50 m, so 50+50+x=140 gives x=40. '
        'The selected trace instead sets the return leg to x and solves 50+2x=140, obtaining 45. '
        'Its algebra is consistent with the wrong equation; the error is assigning the route quantities. '
        'Three traces state 40, but two lack a recognized final-answer marker; official parses are None, 45, 40, None. '
        'The valid comparison is therefore one candidate per answer group. A large score margin does not certify semantic correctness.',
    'gsm8k:eb1fbf0cbd85808aad3b8deda1374acf76a5cbb9327fc93e661b01da20249712':
        'The four candidate answers are 0, 0, 12 and 12: a tied plurality, not an incorrect strict majority. '
        'All use 27 red and 48 blue flowers. The first two incorrectly evaluate 96-9-27-48 as 0; '
        'the selected third candidate obtains 12, matching the fourth and the reference. '
        'The baseline resolves the tie to 0; the learned scores select the correct answer group.'
}


def main():
    if OUT.exists():raise FileExistsError('Preserve completed analysis')
    selection_path=ROOT/'runs/optimization_final_selection/selection.json'
    decision=read(selection_path);chosen=decision['selected']
    if chosen['name']=='D':raise ValueError('Unchanged D uses the prior validation analysis')
    OUT.mkdir()
    bank,arrays=load_inputs('validation');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    labels=np.array([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(ref['answer']) for c in b['candidates']] for b,ref in zip(bank,refs)])
    w=group_weights(bank).astype(np.float64);gv=w.sum(2)>0;hard=gv.sum(1)>=2
    score,_=score_candidates(bank,arrays,ROOT/chosen['checkpoint'])
    predictions,margin,support=select(bank,score,'mean',return_details=True)
    predictions=np.array(predictions);correct=labels[np.arange(len(bank)),predictions]
    groups=np.einsum('ngc,nc->ng',w,score)
    masked=np.where(gv,groups,-np.inf);prob=np.exp(masked-masked.max(1)[:,None]);prob/=prob.sum(1)[:,None]
    confidence=prob.max(1)
    def metric(choice,mask):
        yes=labels[np.arange(len(bank)),choice]
        return dict(correct=int(yes[mask].sum()),n=int(mask.sum()),accuracy=float(100*yes[mask].mean()))
    full=metric(predictions,np.ones(len(bank),bool));difficult=metric(predictions,hard)
    assert full['correct']==chosen['validation_correct'] and difficult['correct']==chosen['validation_hard_correct']
    sources={'SELECTION':dict(path=str(selection_path.relative_to(ROOT)),sha256=sha(selection_path)),
             'ANALYSIS_CODE':dict(path=Path(__file__).name,sha256=sha(Path(__file__))),
             'VALIDATION_INPUTS':dict(path='data/candidates/gsm8k_validation.jsonl',sha256=sha(DATA/'candidates/gsm8k_validation.jsonl')),
             'VALIDATION_REFERENCES':dict(path='data/gsm8k_validation.jsonl',sha256=sha(DATA/'gsm8k_validation.jsonl'))}
    baseline_choices=[('Majority vote',np.array(majority(bank)))]
    dpath=ROOT/'runs/evaluation/final-validation-predictions.json';d=read(dpath)
    assert d['ids']==[b['id'] for b in bank]
    baseline_choices.append(('Historical D',np.array(d['selected'])))
    sources['HISTORICAL_D']=dict(path=str(dpath.relative_to(ROOT)),sha256=sha(dpath))
    for i,result in enumerate(decision['all_new_validation_results']):
        path=ROOT/f'runs/optimization_final_selection/candidate{i+1}/validation_predictions.json'
        saved=read(path);assert saved['ids']==[b['id'] for b in bank]
        baseline_choices.append((result['name'],np.array(saved['selected'])))
        sources[f'FINALIST_{i+1}']=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
        if result['name']==chosen['name']:assert np.array_equal(predictions,saved['selected'])
    baselines=[dict(name=name,full=metric(choice,np.ones(len(bank),bool)),hard=metric(choice,hard)) for name,choice in baseline_choices]
    strata=[]
    ix=np.flatnonzero(hard);order=ix[np.argsort(margin[ix],kind='stable')]
    for i,part in enumerate(np.array_split(order,4)):
        mask=np.zeros(len(bank),bool);mask[part]=True
        strata.append(dict(name=f'Margin quartile {i+1}',kind='margin_quartile',definition='Ascending answer-group score gap; difficult validation only',
                          **metric(predictions,mask),minimum_margin=float(margin[part].min()),maximum_margin=float(margin[part].max())))
    for n in (2,3,4):
        mask=hard&(gv.sum(1)==n)
        if mask.any():strata.append(dict(name=f'{n} distinct answers',kind='answer_count',definition='Difficult validation only',**metric(predictions,mask)))
    for n in (1,2,3):
        mask=hard&(support==n)
        if mask.any():strata.append(dict(name=f'Selected support {n}',kind='selected_support',definition='Difficult validation only',**metric(predictions,mask)))
    errors=ix[~correct[ix]];rescues=ix[correct[ix]&~labels[np.arange(len(bank)),baseline_choices[0][1]][ix]]
    examples=[]
    selections=[]
    if len(errors):selections.append(('High-margin incorrect choice',int(errors[np.argmax(margin[errors])])))
    if len(rescues):
        rescued=int(rescues[np.argmax(margin[rescues])])
        selections.append((rescue_title(bank[rescued],int(baseline_choices[0][1][rescued])),rescued))
    for title,i in selections:
        good=int(np.flatnonzero(labels[i])[0]);picked=int(predictions[i])
        examples.append(dict(title=title,id=bank[i]['id'],index=i,question=bank[i]['question'],
            selected_index=picked,selected_answer=parse_answer(bank[i]['candidates'][picked]['text']),
            correct_index=good,reference_answer=parse_answer(refs[i]['answer']),
            selected_text=bank[i]['candidates'][picked]['text'],correct_text=bank[i]['candidates'][good]['text'],
            reference_text=refs[i]['answer'],margin=float(margin[i]),
            explanation=REVIEWED_EXPLANATIONS.get(bank[i]['id'],'Concrete mathematical interpretation pending source review; selection/counts are programmatically verified.'),
            evidence_ids=['VALIDATION_INPUTS','VALIDATION_REFERENCES','SELECTION']))
    paired=[]
    for name,choice in baseline_choices:
        base=labels[np.arange(len(bank)),choice]
        paired.append(dict(reference=name,repairs=int((correct&~base&hard).sum()),regressions=int((~correct&base&hard).sum())))
    summary=dict(status='complete',selected_name=chosen['name'],validation=dict(full=full,hard=difficult),
        baselines=baselines,strata=strata,examples=examples,paired_comparisons=paired,
        reliability=dict(scope='Descriptive selected-validation analysis, not a new held-out confirmation',
            score_confidence='Softmax of normalized answer-group scores; not a fitted probability calibration',
            hard_mean_confidence=float(confidence[hard].mean()),hard_correct=int(correct[hard].sum()),hard_n=int(hard.sum())),
        explanations=[dict(text='Score-margin strata describe validation error concentration after model selection; they do not authorize an abstention rule or further tuning.',evidence_ids=['SELECTION','VALIDATION_INPUTS','VALIDATION_REFERENCES'])],
        evidence=sources,human_review='pending',test_inputs_read=False,model_changes=False,new_model_fits=0)
    np.savez_compressed(OUT/'validation_scores.npz',ids=np.array([b['id'] for b in bank]),scores=score,group_scores=groups,
        predictions=predictions,labels=labels,hard=hard,margins=margin,confidence=confidence)
    dump(OUT/'summary.json',summary)
    print(json.dumps(dict(status='complete',validation=summary['validation'],examples=[v['id'] for v in examples])))


if __name__=='__main__':main()
