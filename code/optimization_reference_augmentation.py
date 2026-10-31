"""C4 reference-corruption auxiliary control, with lazy training dependencies.

Importing predict/score_candidates requires NumPy only. This is not candidate
label augmentation: real final-answer labels are never changed.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np

from common import DATA, parse_answer, read_jsonl
from experiment_utils import dump, metrics
from selection import select
from text_features import _safe_arithmetic
from sequence_model import numpy_forward

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"runs/optimization_reference_augmentation"
FOLD_SEEDS = (20261027, 20261028)
CONFIGS = (("digit_text_control", 0.), ("reference_corruption01", .1))
PLANNED_FITS = 24
TIME_CAP = 7200.
SPECIAL = ("<pad>", "<unk>", "<sep>", "<gap>", "<number>", "<number_end>",
           "<decimal>", "<comma>") + tuple(f"<digit_{i}>" for i in range(10))
PATTERN = re.compile(r"####|\\[a-z]+|[a-z]+(?:'[a-z]+)?|\d[\d,]*(?:\.\d+)?|[^\w\s]", re.I)
ANNOTATION = re.compile(r"<<([^<>]*)>>")
LITERAL = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:/[-+]?\d+)?")
SOURCES = ("optimization_reference_augmentation.py", "sequence_model.py", "sequence_experiment.py",
           "common.py", "experiment_utils.py", "selection.py", "text_features.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: sha(ROOT/name) for name in SOURCES}


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def digits(text):
    text = str(text).lower()
    if len(text) > 24000:
        return digits(text[:12000])+["<gap>"]+digits(text[-12000:])
    result = []
    for part in PATTERN.findall(text):
        if part and part[0].isdigit():
            result.append("<number>")
            result.extend(f"<digit_{c}>" if c.isdigit() else "<decimal>" if c == "." else "<comma>" for c in part)
            result.append("<number_end>")
        else:
            result.append(part[:64])
    return result


def input_sequence(question, text):
    q = digits(question)
    if len(q) > 64:
        q = q[:32]+q[-32:]
    body = digits(text)
    if len(body) > 256:
        body = body[:128]+["<gap>"]+body[-128:]
    return q+["<sep>"]+body


def numeric_literal(text):
    token = text.strip()
    if not LITERAL.fullmatch(token):
        return None
    try:
        value = Fraction(token.replace(",", ""))
        return value if abs(value) <= 10**12 else None
    except (ValueError, ZeroDivisionError):
        return None


def changed_literal(value, old):
    changed = value+1
    compact = old.strip().replace(",", "")
    if "/" in compact:
        return f"{changed.numerator}/{changed.denominator}"
    if "." in compact:
        places = len(compact.rsplit(".",1)[1])
        with localcontext() as ctx:
            ctx.prec = 64
            text = format(Decimal(changed.numerator)/Decimal(changed.denominator), f".{places}f")
        if compact.startswith(".") and text.startswith("0."):
            text = text[1:]
        return text
    if changed.denominator != 1:
        return str(changed)
    return f"{changed.numerator:,}" if "," in old else str(changed.numerator)


def corrupt_reference(question, reference):
    positive = input_sequence(question, reference)
    attempted = 0
    for annotation in ANNOTATION.finditer(reference):
        expression = annotation.group(1)
        if "=" not in expression:
            continue
        left, right = expression.rsplit("=",1)
        value = numeric_literal(right)
        if value is None:
            continue
        lhs = left.replace(",", "").replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
        evaluated = _safe_arithmetic(lhs)
        if evaluated is None or abs(evaluated-float(value)) > 1e-6*max(1.,abs(evaluated),abs(float(value))):
            continue
        attempted += 1
        new = changed_literal(value, right)
        assert numeric_literal(new) == value+1
        # Change only the literal RHS inside one originally consistent annotation.
        leading = len(right)-len(right.lstrip())
        trailing = len(right)-len(right.rstrip())
        replacement = left+"="+right[:leading]+new+(right[-trailing:] if trailing else "")
        negative_text = reference[:annotation.start(1)]+replacement+reference[annotation.end(1):]
        negative = input_sequence(question, negative_text)
        if positive == negative:
            continue
        return positive, negative, {"observed": True, "valid_annotations_tried": attempted,
            "annotation_start": annotation.start(), "old_rhs": str(value), "new_rhs": str(value+1),
            "only_annotation_rhs_changed": True, "encoded_visible_pre_vocab": True}
    return positive, positive.copy(), {"observed": False, "valid_annotations_tried": attempted}


def make_sequences(bank, references):
    actual, augmented, observed, records = [], [], [], []
    for row, ref in zip(bank, references):
        assert row["id"] == ref["id"] and row["question"] == ref["question"]
        actual.append([input_sequence(row["question"], c["text"]) for c in row["candidates"]])
        positive, negative, info = corrupt_reference(row["question"], ref["answer"])
        augmented.append([positive, negative])
        observed.append(info["observed"])
        records.append(info)
    return actual, augmented, np.asarray(observed, bool), records


def vocabulary(actual, augmented, ids):
    counts = Counter()
    for i in ids:
        for sequence in actual[int(i)]+augmented[int(i)]:
            counts.update(sequence)
    words = sorted((token for token in counts if token not in SPECIAL), key=lambda token:(-counts[token],token))
    return np.array(list(SPECIAL)+words[:4096-len(SPECIAL)])


def encode(sequence_bank, vocab, candidates):
    lookup = {token:i for i,token in enumerate(vocab.tolist())}
    codes = np.zeros((len(sequence_bank),candidates,322),np.int64)
    for i,row in enumerate(sequence_bank):
        assert len(row)==candidates
        for j,seq in enumerate(row):
            assert len(seq)<=322
            codes[i,j,:len(seq)] = [lookup.get(token,1) for token in seq]
    return codes


def score_candidates(bank, embeddings, asset):
    if not bank:
        return np.empty((0,4),np.float32)
    seqs = [[input_sequence(row["question"], c["text"]) for c in row["candidates"]] for row in bank]
    codes = encode(seqs,asset["vocabulary"],4)
    return numpy_forward(codes,np.zeros((len(bank),4,0),np.float32),asset)


def predict(bank, arrays, checkpoint_path):
    path=Path(checkpoint_path)
    if path.is_dir():path/= "selected_model.npz"
    with np.load(path,allow_pickle=False) as saved:asset={k:saved[k] for k in saved.files}
    return select(bank,score_candidates(bank,None,asset),"mean")


def torch_runtime():
    (OUT/"tmp").mkdir(parents=True,exist_ok=True)
    os.environ["TMP"]=os.environ["TEMP"]=str(OUT/"tmp")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert torch.version.cuda is None
    return torch


def build_model(vocab_size):
    import torch
    from torch import nn
    import sequence_experiment as base
    class DigitCNN(base.WordCNN):
        def __init__(self):
            super().__init__(vocab_size,False)
            self.auxiliary=nn.Linear(64,1)

        def hidden_values(self,codes):
            shape=codes.shape
            flat=codes.reshape(-1,shape[-1]);mask=flat.ne(0)
            x=self.embedding(flat).transpose(1,2);pooled=[]
            for conv in (self.conv3,self.conv5):
                z=torch.relu(conv(x)).transpose(1,2)
                maximum=z.masked_fill(~mask[:,:,None],-1e9).amax(dim=1)
                mean=(z*mask[:,:,None]).sum(dim=1)/mask.sum(dim=1,keepdim=True).clamp_min(1)
                pooled.extend((maximum,mean))
            return torch.tanh(self.hidden(torch.cat(pooled,dim=1))).reshape(shape[0],shape[1],64)

        def forward(self,codes,numeric=None):
            return self.output(self.hidden_values(codes)).squeeze(-1)

        def auxiliary_scores(self,codes):
            return self.auxiliary(self.hidden_values(codes)).squeeze(-1)
    return DigitCNN()


def combined_loss(main_scores,aux_scores,structure,eligible,observed,strength):
    import torch
    import torch.nn.functional as F
    import sequence_experiment as base
    weights,gv,good=structure
    main=base.objective(main_scores[eligible],weights[eligible],gv[eligible],good[eligible]) if eligible.any() else main_scores.sum()*0
    target=torch.tensor([1.,0.],dtype=aux_scores.dtype,device=aux_scores.device).expand_as(aux_scores)
    paired=F.binary_cross_entropy_with_logits(aux_scores,target,reduction="none").mean(dim=1)
    auxiliary=paired[observed].mean() if observed.any() else aux_scores.sum()*0
    return main+strength*auxiliary,main,auxiliary


def native_scores(model,codes):
    import torch
    out=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(codes),64):out.append(model(codes[start:start+64]).numpy())
    return np.concatenate(out)


class TimeBudget(Exception):pass


def train(codes,augmented,structure,eligible,observed,fit_ids,strength,epochs,deadline,held_ids=None):
    import torch
    import sequence_experiment as base
    torch.manual_seed(7506);rng=np.random.default_rng(7507)
    model=build_model(max(int(codes.max()),int(augmented.max()))+1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best,selected,stale=float("inf"),0,0;curves=[];tick=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(fit_ids);sums=np.zeros(3)
        for start in range(0,len(order),64):
            if time.perf_counter()>deadline:raise TimeBudget("Registered two-hour family cap reached")
            ids=order[start:start+64];optimizer.zero_grad(set_to_none=True)
            z=model(codes[ids]);aux=model.auxiliary_scores(augmented[ids])
            loss,main,weak=combined_loss(z,aux,tuple(v[ids] for v in structure),eligible[ids],observed[ids],strength)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            sums+=np.array([float(loss.detach()),float(main.detach()),float(weak.detach())])*len(ids)
        row={"epoch":epoch,"objective":sums[0]/len(fit_ids),"main":sums[1]/len(fit_ids),
             "weak_auxiliary":sums[2]/len(fit_ids),"elapsed_seconds":time.perf_counter()-tick}
        if held_ids is not None:
            model.eval();active=held_ids[eligible[held_ids].numpy()];total=0.
            with torch.no_grad():
                for start in range(0,len(active),64):
                    ids=active[start:start+64]
                    total+=float(base.objective(model(codes[ids]),*(v[ids] for v in structure)))*len(ids)
            current=total/len(active);row["heldout_main_group_ce"]=current
            if current<best-1e-4:best,selected,stale=current,epoch,0
            else:stale+=1
        curves.append(row)
        if held_ids is not None and stale>=4:break
    return model,curves,selected if held_ids is not None else epochs


def prepare_inputs(actual,augmented,observed,fit_ids):
    tick=time.perf_counter();vocab=vocabulary(actual,augmented,fit_ids)
    codes=encode(actual,vocab,4);extra=encode(augmented,vocab,2)
    visible=np.any(extra[:,0]!=extra[:,1],axis=1)
    active=observed&visible
    info={"fit_questions":len(fit_ids),"fit_indices_sha256":hashlib.sha256(np.ascontiguousarray(fit_ids).tobytes()).hexdigest(),
          "vocabulary_size":len(vocab),"fit_augmented_observed":int(active[fit_ids].sum()),
          "fit_pairs_identical_after_encoding":int((observed[fit_ids]&~visible[fit_ids]).sum()),
          "seconds":time.perf_counter()-tick}
    return codes,extra,active,vocab,info


def protocol():
    return {"task":"C4 pure-text matched control for training-reference corruption auxiliary",
        "configs":[{"name":n,"auxiliary_weight":w} for n,w in CONFIGS],
        "fold_seeds":list(FOLD_SEEDS),"folds":3,
        "fold_assignment":"fold[default_rng(seed).permutation(N)]=arange(N)%3",
        "inner_split":"85/15 outer-fitting questions, seed outer_seed+100+fold",
        "real_labels":"unchanged official final-answer correctness for actual candidate groups; no real candidate relabeled",
        "corruption":"scan reference <<expr=result>> in order; first literal RHS whose bounded arithmetic LHS agrees within1e-6 relative tolerance and whose +1 exact Fraction change is visible after truncation; change only RHS inside that annotation",
        "corruption_rejection":"unparseable, inconsistent, out-of-range or invisible annotations skipped; no qualifying visible pair means unobserved, no auxiliary gradient; final encoded identical pairs masked",
        "meaning":"original versus one process-corrupted reference, NOT final-answer correctness; reference final answer is unchanged",
        "tokenizer":"both arms use identical fixed digit expansion with number boundaries/digit0..9/decimal/comma tokens; numeric values remain distinguishable; question first32+last32 if>64, bodyfirst128+gap+last128 if>256, total322",
        "vocabulary":"max4096; fit on actual candidates plus both reference pair versions only from applicable inner/outer fitting questions; identical corpus policy both arms; no heldout reference in fitting vocabulary",
        "inputs":"pure text only; no1024provided embeddings, no75numeric features, no reference-derived inference features or artificial reference/corruption prefix",
        "model":"original pure-text WordCNN embedding32,Conv3/5each32,ReLU,mean/maxpool,tanh64,scalar; separate auxiliary scalar on shared64hidden, discarded at export",
        "loss":"question-balanced answer-group CE on hard questions + lambda * mean paired original1/corrupt0 BCE over observed questions",
        "batches":"both arms all-fitting-question batches64; same optimizer steps and auxiliary forward; zero weight is a fresh matched control",
        "training":"seed7506/shuffle7507; AdamW lr.001 weight_decay.001; gradientclip5; max24epochs/patience4/minchange1e-4; stop on inner MAIN groupCE only; fresh outer refit",
        "weakness":"native reference annotation style and duplicate result literals may allow local consistency shortcuts; improvement would not prove reasoning verification or transfer to other corruption mechanisms",
        "comparison":"interpret auxiliary versus matched new digit-text control only; differences from old joint/number-masked CNN cannot be assigned solely to augmentation",
        "planned_fits":PLANNED_FITS,"family_runtime_cap_seconds":TIME_CAP,
        "selection":"mean hard correct over repeated OOF then mean normalized groupCE thenzero auxiliary",
        "same_questions_repeated":True,"threads":4,"no_validation_test_full_refit":True,
        "inference":"NumPy-only existing convolution forward plus input-only digit tokenizer; no reference access"}


def progress(stage,completed,detail,started):
    atomic(OUT/"progress.json",{"stage":stage,"completed_fits":completed,"planned_fits":PLANNED_FITS,
        "detail":detail,"elapsed_seconds":time.perf_counter()-started,"updated_utc":datetime.now(timezone.utc).isoformat()})


def prepare_registration():
    if (OUT/"protocol.json").exists():raise FileExistsError("Preserve existing registration")
    torch=torch_runtime()
    import sequence_experiment as base
    question="There are 3 boxes with 4 apples. How many apples?"
    reference="There are 3*4=<<3*4=12>>12 apples. #### 12"
    positive,negative,info=corrupt_reference(question,reference)
    assert info["observed"] and positive!=negative and info["old_rhs"]=="12" and info["new_rhs"]=="13"
    assert not corrupt_reference(question,"No arithmetic annotations. #### 12")[2]["observed"]
    assert not corrupt_reference(question,"Bad annotation <<3*4=15>>. #### 12")[2]["observed"]
    actual=[[input_sequence(question,text) for text in ("3*4=12. #### 12","3+4=7. #### 7","12 apples. #### 12","4 apples. #### 4")]]
    vocab=vocabulary(actual,[[positive,negative]],[0]);codes=encode(actual,vocab,4);extra=encode([[positive,negative]],vocab,2)
    assert not np.array_equal(extra[:,0],extra[:,1])
    torch.manual_seed(7506);model=build_model(len(vocab)).eval()
    with torch.no_grad():expected=model(torch.from_numpy(codes)).numpy()
    asset=base.export(model,{"vocabulary":vocab})
    replay=numpy_forward(codes,np.zeros((1,4,0),np.float32),asset)
    error=float(np.max(abs(replay-expected)));assert error<1e-5
    np.savez(OUT/"readiness_asset.npz",**asset)
    np.savez(OUT/"readiness_inputs.npz",codes=codes,expected=expected)
    # Loss-level finite differences verify both the main mask and auxiliary pair averaging.
    score=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    aux=torch.randn(2,2,dtype=torch.float64,requires_grad=True)
    structure=(torch.eye(4,dtype=torch.float64).expand(2,4,4),torch.ones(2,4,dtype=torch.bool),
               torch.tensor([[1,0,0,0],[1,1,1,1]],dtype=torch.bool))
    eligible=torch.tensor([True,False]);observed=torch.tensor([True,False])
    assert torch.autograd.gradcheck(lambda x,y:combined_loss(x,y,structure,eligible,observed,.1)[0],(score,aux),eps=1e-6,atol=1e-5)
    assert torch.allclose(combined_loss(score,aux,structure,eligible,observed,0.)[0],base.objective(score[:1],*(x[:1] for x in structure)))
    dump(OUT/"readiness_checks.json",{"digit_change_visible":True,"invalid_annotation_rejected":True,
        "torch_numpy_max_abs_error":error,"combined_loss_gradcheck":True,"zero_weight_control":True,"model_fits":0})
    dump(OUT/"protocol.json",protocol());dump(OUT/"source_evidence.json",sources())
    progress("ready",0,"registered; independent NumPy readiness still required; not launched",time.perf_counter())
    print(json.dumps({"ready":True,"planned_fits":PLANNED_FITS,"preparation_model_fits":0}))


def check_numpy():
    with np.load(OUT/"readiness_asset.npz",allow_pickle=False) as saved:asset={k:saved[k] for k in saved.files}
    with np.load(OUT/"readiness_inputs.npz",allow_pickle=False) as saved:
        codes,expected=saved["codes"],saved["expected"]
    numeric=np.zeros((len(codes),4,0),np.float32)
    z=numpy_forward(codes,numeric,asset);error=float(np.max(abs(z-expected)));assert error<1e-5
    perm=np.array([2,0,3,1]);zp=numpy_forward(codes[:,perm],numeric[:,perm],asset)
    perr=float(np.max(abs(zp-z[:,perm])));assert perr<1e-5
    assert score_candidates([],None,asset).shape==(0,4)
    result={"numpy":np.__version__,"torch_imported":"torch" in sys.modules,"sklearn_imported":"sklearn" in sys.modules,
            "torch_numpy_error":error,"permutation_error":perr,"empty":True,"model_fits":0}
    assert not result["torch_imported"] and not result["sklearn_imported"]
    dump(OUT/"numpy_readiness.json",result);print(json.dumps(result,indent=2))


def audit_io():
    accesses=set()
    def hook(event,args):
        if event!="open" or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation","gsm8k_test")):raise RuntimeError(f"Forbidden split {path}")
        if DATA.resolve() in path.parents:accesses.add(path.relative_to(DATA.resolve()).as_posix())
        mode,flags=args[1],args[2] or 0
        writing=isinstance(mode,str) and any(c in mode for c in "wax+") or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if writing and OUT.resolve() not in path.parents:raise RuntimeError(f"Persistent write outside assigned directory {path}")
    sys.addaudithook(hook);return accesses


def run():
    started=time.perf_counter()
    expected=json.loads((OUT/"source_evidence.json").read_text(encoding="utf-8"));assert expected==sources()
    assert json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))==protocol()
    assert not json.loads((OUT/"numpy_readiness.json").read_text(encoding="utf-8"))["torch_imported"]
    if (OUT/"started.json").exists():raise FileExistsError("Preserve existing run")
    torch=torch_runtime()
    import sequence_experiment as base
    accesses=audit_io();dump(OUT/"started.json",{"utc":datetime.now(timezone.utc).isoformat(),"sources":expected})
    progress("prepare_train_only",0,"training text and deterministic reference corruption",started)
    bank=read_jsonl(DATA/"candidates/gsm8k_train.jsonl");refs=read_jsonl(DATA/"gsm8k_train.jsonl")
    answers=[[parse_answer(c["text"]) for c in row["candidates"]] for row in bank]
    labels=np.array([[a is not None and a==parse_answer(ref["answer"]) for a in aa] for aa,ref in zip(answers,refs)],np.float32)
    hard=np.array([len({a for a in aa if a is not None})>=2 for aa in answers])
    weights,gv,good,valid,eligible=base.group_structure(bank,labels);assert np.array_equal(hard,eligible)
    actual,augmented,observed,corruptions=make_sequences(bank,refs)
    dump(OUT/"corruption_records.json",corruptions)
    dump(OUT/"coverage.json",{"training_questions":len(bank),"observed_pairs":int(observed.sum()),
         "observed_easy_questions":int((observed&~hard).sum()),"observed_hard_questions":int((observed&hard).sum()),
         "real_candidate_labels_unchanged":True,"meaning":"Process corruption weak labels, not final correctness labels"})
    folds=np.empty((2,len(bank)),np.int8)
    for r,seed in enumerate(FOLD_SEEDS):folds[r,np.random.default_rng(seed).permutation(len(bank))]=np.arange(len(bank))%3
    ids=np.array([row["id"] for row in bank]);raw=np.full((2,len(bank),4,2),np.nan,np.float32);normalized=raw.copy()
    structure_t=tuple(torch.from_numpy(x) for x in (weights,gv,good));eligible_t=torch.from_numpy(eligible)
    records=[];completed=0
    try:
        for r,seed in enumerate(FOLD_SEEDS):
            for f in range(3):
                fit_ids,held_ids=np.flatnonzero(folds[r]!=f),np.flatnonzero(folds[r]==f)
                order=np.random.default_rng(seed+100+f).permutation(fit_ids);cut=int(.85*len(order));inner,early_held=order[:cut],order[cut:]
                np.savez_compressed(OUT/f"repeat{r}_fold{f}_indices.npz",fit=fit_ids,held=held_ids,inner=inner,early_held=early_held)
                for c,(name,strength) in enumerate(CONFIGS):
                    stem=f"repeat{r}_fold{f}_{name}";progress("early_fit",completed,stem,started)
                    codes,extra,active,vocab,info=prepare_inputs(actual,augmented,observed,inner)
                    ct,at,ot=map(torch.from_numpy,(codes,extra,active))
                    model,curve,epochs=train(ct,at,structure_t,eligible_t,ot,inner,strength,24,started+TIME_CAP,early_held)
                    completed+=1;dump(OUT/f"{stem}_early_curve.json",curve);early_seconds=curve[-1]["elapsed_seconds"]
                    progress("outer_refit",completed,f"{stem}; epochs={epochs}",started)
                    codes,extra,active,vocab,outer=prepare_inputs(actual,augmented,observed,fit_ids)
                    ct,at,ot=map(torch.from_numpy,(codes,extra,active))
                    model,curve,_=train(ct,at,structure_t,eligible_t,ot,fit_ids,strength,epochs,started+TIME_CAP)
                    completed+=1;dump(OUT/f"{stem}_refit_curve.json",curve)
                    z=native_scores(model,ct);asset=base.export(model,{"vocabulary":vocab})
                    scale=float(max(np.std((z[fit_ids]-z[fit_ids].mean(1,keepdims=True))[valid[fit_ids]]),1e-6))
                    asset["score_scale"]=np.array(scale);np.savez(OUT/f"{stem}.npz",**asset)
                    replay=numpy_forward(codes[held_ids[:32]],np.zeros((min(32,len(held_ids)),4,0),np.float32),asset)
                    error=float(np.max(abs(replay-z[held_ids[:32]])));assert error<1e-4
                    assert select([bank[i] for i in held_ids[:32]],replay,"mean")==select([bank[i] for i in held_ids[:32]],z[held_ids[:32]],"mean")
                    raw[r,held_ids,:,c]=z[held_ids];normalized[r,held_ids,:,c]=z[held_ids]/scale
                    rec={"repeat":r,"fold":f,"config":name,"auxiliary_weight":strength,"selected_epochs":epochs,
                        "early_fit_seconds":early_seconds,"refit_seconds":curve[-1]["elapsed_seconds"],
                        "inner_vocabulary":info,"outer_vocabulary":outer,"score_scale":scale,
                        "asset_bytes":(OUT/f"{stem}.npz").stat().st_size,"asset_sha256":sha(OUT/f"{stem}.npz"),
                        "export_error":error,"metrics":metrics(select([bank[i] for i in held_ids],z[held_ids],"mean"),labels[held_ids],hard[held_ids])}
                    records.append(rec);atomic(OUT/"fit_records.json",records)
                    np.savez_compressed(OUT/"partial_oof.npz",raw=raw,normalized=normalized,ids=ids,labels=labels,
                        valid=valid,hard=hard,folds=folds,configs=np.array([n for n,_ in CONFIGS]))
                    progress("outer_complete",completed,stem,started)
    except TimeBudget as error:
        dump(OUT/"resource_stop.json",{"reason":str(error),"model_fit_calls_completed":completed,"partial_fit_started":True,
            "seconds":time.perf_counter()-started,"does_not_establish_accuracy_ceiling":True})
        progress("resource_stopped",completed,str(error),started);return
    assert np.isfinite(raw).all();rows=[]
    for c,(name,strength) in enumerate(CONFIGS):
        repeats=[]
        for r in range(2):
            choices=np.asarray(select(bank,raw[r,:,:,c],"mean"));baseline=np.asarray(select(bank,raw[r,:,:,0],"mean"))
            right=labels[np.arange(len(bank)),choices].astype(bool);control=labels[np.arange(len(bank)),baseline].astype(bool)
            repeats.append({"seed":FOLD_SEEDS[r],"metrics":metrics(choices,labels,hard),
                "group_ce":base.numpy_group_loss(normalized[r,:,:,c],weights,gv,good,eligible),
                "repairs_vs_control":int((hard&right&~control).sum()),"regressions_vs_control":int((hard&control&~right).sum())})
        rows.append({"config":name,"auxiliary_weight":strength,"repeats":repeats,
            "mean_hard_correct":float(np.mean([v["metrics"]["hard_correct"] for v in repeats])),
            "mean_group_ce":float(np.mean([v["group_ce"] for v in repeats]))})
    selected=min(rows,key=lambda x:(-x["mean_hard_correct"],x["mean_group_ce"],x["auxiliary_weight"]))
    np.savez_compressed(OUT/"oof_logits.npz",raw=raw,normalized=normalized,ids=ids,labels=labels,valid=valid,
        hard=hard,folds=folds,configs=np.array([n for n,_ in CONFIGS]))
    assert expected==sources()
    result={"configs":rows,"selected":selected,"model_fit_calls":completed,"vocabulary_fits":completed,
        "max_asset_bytes":max(v["asset_bytes"] for v in records),"max_export_error":max(v["export_error"] for v in records),
        "seconds":time.perf_counter()-started,"data_files_read":sorted(accesses),"validation_evaluations":0,
        "test_access":False,"full_model_fitted":False,
        "interpretation":"Matched digit-preserving pure-text auxiliary control; reference-domain style and duplicate-result shortcuts remain possible; no final candidate relabeling"}
    dump(OUT/"summary.json",result);progress("complete",completed,"selected "+selected["config"],started)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--run",action="store_true");parser.add_argument("--check-assets",action="store_true")
    args=parser.parse_args()
    if sum((args.prepare_only,args.run,args.check_assets))!=1:parser.error("Choose exactly one stage")
    if args.prepare_only:prepare_registration()
    elif args.run:run()
    else:check_numpy()
