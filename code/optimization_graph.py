"""Queued matched node-MLP versus two-layer equation GCN study, training only."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="4"
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from common import DATA,parse_answer
from equation_graph_model import graph_inputs,numpy_forward,FEATURE_NAMES,MAX_NODES,synthetic_checks
from experiment_utils import development_data,dump,metrics
from selection import select
from sequence_experiment import group_structure,objective,numpy_group_loss
from sequence_model import numeric_inputs,transform_numeric

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"runs"/"optimization_graph"
FOLD_SEEDS=(20261027,20261028);ARMS=("node_mlp","equation_gcn")
MAX_EPOCHS=24;PATIENCE=4;BATCH=64;INIT_SEED=7506
SOURCES=("optimization_graph.py","equation_graph_model.py","quantitative_features.py","text_features.py",
         "sequence_model.py","sequence_experiment.py","experiment_utils.py","common.py","selection.py")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrhash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def progress(stage,done,started,detail):
    dump(OUT/"progress.json",{"stage":stage,"completed_fits":done,"planned_fits":24,"detail":detail,
        "elapsed_seconds":time.perf_counter()-started,"updated_utc":datetime.now(timezone.utc).isoformat()})


class EquationNet(nn.Module):
    def __init__(self,graph):
        super().__init__();self.graph=graph
        self.node1=nn.Linear(len(FEATURE_NAMES),32);self.node2=nn.Linear(32,32)
        self.hidden=nn.Linear(64+1099,64);self.output=nn.Linear(64,1)

    def forward(self,nodes,adjacency,mask,numeric):
        n=len(nodes);h=nodes.reshape(-1,MAX_NODES,len(FEATURE_NAMES))
        a=adjacency.reshape(-1,MAX_NODES,MAX_NODES);m=mask.reshape(-1,MAX_NODES)
        for layer in (self.node1,self.node2):
            if self.graph:h=torch.bmm(a,h)
            h=torch.relu(layer(h))*m[:,:,None]
        maximum=h.masked_fill(~m[:,:,None],-1e9).amax(dim=1)
        maximum=torch.where(m.any(dim=1)[:,None],maximum,torch.zeros_like(maximum))
        mean=(h*m[:,:,None]).sum(dim=1)/m.sum(dim=1,keepdim=True).clamp_min(1)
        pooled=torch.cat((mean,maximum,numeric.reshape(-1,1099)),dim=1)
        return self.output(torch.tanh(self.hidden(pooled))).reshape(n,4)


def export(model,transform):
    asset={"graph":np.array(model.graph,dtype=np.int8),**transform}
    for name in ("node1","node2","hidden","output"):
        layer=getattr(model,name)
        asset[name+"_weight"]=layer.weight.detach().numpy().copy()
        asset[name+"_bias"]=layer.bias.detach().numpy().copy()
    return asset


def export_synthetic_checks():
    torch.manual_seed(INIT_SEED)
    rows=[{"question":"There are 3 and 4 items.","candidates":[{"text":"3 + 4 = 7. #### 7"},{"text":"3 * 4 = 12. #### 12"},{"text":"3 + 4 = 8. #### 8"},{"text":"#### 7"}]}]
    nodes,adj,mask=graph_inputs(rows);numeric=np.zeros((1,4,1099),dtype=np.float32)
    arrays=tuple(torch.from_numpy(a) for a in (nodes,adj,mask,numeric));errors={}
    for graph in (False,True):
        torch.manual_seed(INIT_SEED);model=EquationNet(graph).eval()
        with torch.no_grad():expected=model(*arrays).numpy()
        asset=export(model,{"numeric_mean":np.zeros(1099,np.float32),"numeric_scale":np.ones(1099,np.float32)})
        actual=numpy_forward(nodes,adj,mask,numeric,asset)
        error=float(np.max(np.abs(actual-expected)));assert error<1e-5
        assert numpy_forward(nodes[:0],adj[:0],mask[:0],numeric[:0],asset).shape==(0,4)
        errors[str(graph)]=error
    return {"numpy_torch_synthetic_max_errors":errors,"empty_inference":True}


def protocol():
    return {"created_utc":datetime.now(timezone.utc).isoformat(),"scope":"E4 equation structure: same-node MLP control versus2-layer GCN",
        "data_scope":["train"],"validation_access":False,"test_access":False,"fold_seeds":FOLD_SEEDS,"folds":3,
        "fold_assignment":"fold[default_rng(seed).permutation(N)]=arange(N)%3",
        "inner_split":"85/15 whole outer-fitting questions, default_rng(20264000+100*repeat+fold), identical for both arms",
        "arms":ARMS,"planned_model_fits":24,"full_training_refits":0,"threads":4,
        "graph":{"question_nodes":8,"equation_nodes":32,"maximum_nodes":40,"numeric_features":FEATURE_NAMES,
            "equations":"Reuse existing bounded numeric _equations checker; preserve boxed results in equations, drop #### final text",
            "dependencies":"Equation operands matching question values or earlier equation results; incoming row normalized with selfloops; no future edges",
            "labels":"None used to construct nodes or edges; role flags use input final answers only",
            "limitation":"numeric dependency proxies and a bounded equation parser; not semantic equivalence or a complete proof graph"},
        "architecture":{"shared":"2node layers32D/ReLU, masked mean+max pool; concatenate1099numeric channels; tanh64 then scalar",
            "control":"independent node MLP, no adjacency aggregation","treatment":"incoming adjacency aggregation before each of the same two linear layers",
            "numeric":"1024providedembedding+75existing precision-stable text features; inner/outer fitting-side valid-candidate mean/std, clip[-8,8]",
            "initialization":"all weights random; same seed7506 and parameter shapes in both arms; no external weights"},
        "optimizer":{"name":"AdamW","lr":.001,"weight_decay":.001,"gradient_clip":5.,"batch_questions":64},
        "objective":"equal-question cross-entropy on official parsed-answer group mean scores",
        "eligible":"correct and incorrect valid groups both present; normalizers may use easy fitting inputs but losses use eligible questions only",
        "early_stop":{"maximum_epochs":MAX_EPOCHS,"patience":PATIENCE,"minimum_improvement":1e-4,"criterion":"inner group CE"},
        "refit":"Fresh same-seed initialization at independently inner-selected epoch count; outer-fit normalizer rebuilt without outer-heldout inputs",
        "normalizer_fits":12,"selection":"summed OOF hard correct across repeats, lower mean normalized group CE, no-edge control on tie",
        "scale":"fitting-side std of all-four centered scores at valid candidate entries",
        "numpy_export":"All outer-heldout predictions replayed in NumPy; OOF uses NumPy scores; max allowed error1e-4; answer differences recorded rather than hidden",
        "cost_probe":"First complete outer partition, planned fits retained; report fitting and NumPy inference timing estimates, no scope expansion",
        "adaptation":"Fixed two arms only. No additional parameters, validation/test or full-data checkpoint in this stage.",
        "interpretation":"Two repeats reuse the same questions; OOF selection remains exploratory after historical search"}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"protocol.json").exists():raise SystemExit("Registered protocol exists")
    torch.set_num_threads(4)
    checks={**synthetic_checks(),**export_synthetic_checks()}
    dump(OUT/"synthetic_checks.json",checks);dump(OUT/"protocol.json",protocol())
    dump(OUT/"source_hashes.json",{name:digest(ROOT/name) for name in SOURCES})
    dump(OUT/"input_hashes.json",{str(p.relative_to(ROOT)):digest(p) for p in (
        DATA/"candidates"/"gsm8k_train.jsonl",DATA/"candidates"/"gsm8k_train.npz",DATA/"gsm8k_train.jsonl")})
    progress("ready",0,time.perf_counter(),"Graph and NumPy export synthetic checks passed; awaiting queue")
    print(json.dumps({"stage":"ready","command":"sequenceenv/python optimization_graph.py --run","model_fits":24,"training_started":False}))


def normalizer(raw,valid,indices):
    values=raw[indices][valid[indices]].astype(np.float64)
    return {"numeric_mean":values.mean(axis=0).astype(np.float32),"numeric_scale":np.maximum(values.std(axis=0),1e-3).astype(np.float32)}


def native_scores(model,inputs):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(inputs[0]),BATCH):result.append(model(*[a[start:start+BATCH] for a in inputs]).numpy())
    return np.concatenate(result)


def train(inputs,structure,fit_ids,graph,epochs,held_ids=None):
    torch.manual_seed(INIT_SEED);rng=np.random.default_rng(INIT_SEED+1)
    model=EquationNet(graph);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=float("inf");selected=0;stale=0;curves=[];started=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(fit_ids);total=0.
        for start in range(0,len(order),BATCH):
            ids=order[start:start+BATCH];optimizer.zero_grad(set_to_none=True)
            z=model(*[a[ids] for a in inputs]);loss=objective(z,*[a[ids] for a in structure])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=float(loss.detach())*len(ids)
        row={"epoch":epoch,"fit_loss":total/len(fit_ids),"elapsed_seconds":time.perf_counter()-started}
        if held_ids is not None:
            model.eval();loss_sum=0.
            with torch.no_grad():
                for start in range(0,len(held_ids),BATCH):
                    ids=held_ids[start:start+BATCH];loss_sum+=float(objective(model(*[a[ids] for a in inputs]),*[a[ids] for a in structure]))*len(ids)
            value=loss_sum/len(held_ids);row["heldout_group_ce"]=value
            if value<best-1e-4:best=value;selected=epoch;stale=0
            else:stale+=1
        curves.append(row)
        if held_ids is not None and stale>=PATIENCE:break
    assert all(torch.isfinite(p).all() for p in model.parameters())
    return model,curves,(selected if held_ids is not None else epochs)


def run():
    started=time.perf_counter()
    sources=json.loads((OUT/"source_hashes.json").read_text(encoding="utf-8"));inputs_hash=json.loads((OUT/"input_hashes.json").read_text(encoding="utf-8"))
    if any(digest(ROOT/name)!=h for name,h in {**sources,**inputs_hash}.items()):raise SystemExit("Registered source/input changed")
    if (OUT/"summary.json").exists():print("Already complete; no new fits");return
    if (OUT/"run_started.json").exists():raise SystemExit("Incomplete run exists; preserve evidence before recovery")
    dump(OUT/"run_started.json",{"utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(OUT/"protocol.json")})
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.version.cuda is None
    progress("features",0,started,"Extracting bounded graph nodes and dependencies from training inputs")
    bank,embeddings,labels,hard=development_data("train");n=len(bank)
    weights,valid_groups,good,valid,eligible=group_structure(bank,labels)
    tick=time.perf_counter();nodes,adjacency,mask,coverage=graph_inputs(bank,True);raw_numeric=numeric_inputs(bank,embeddings);feature_seconds=time.perf_counter()-tick
    dump(OUT/"graph_coverage.json",{"question_candidates":len(coverage),"feature_seconds":feature_seconds,
        "candidates_with_equations":sum(r["equation_nodes"]>0 for r in coverage),
        "mean_equation_nodes":float(np.mean([r["equation_nodes"] for r in coverage])),
        "mean_dependency_edges":float(np.mean([r["dependency_edges"] for r in coverage])),
        "question_quantity_truncation_fraction":float(np.mean([r["question_numbers_truncated"] for r in coverage])),
        "equation_truncation_fraction":float(np.mean([r["equations_truncated"] for r in coverage]))})
    np.savez_compressed(OUT/"training_graph_inputs.npz",ids=np.array([r["id"] for r in bank]),nodes=nodes,adjacency=adjacency,mask=mask)
    base_t=tuple(torch.from_numpy(x) for x in (nodes,adjacency,mask))
    structure=tuple(torch.from_numpy(x) for x in (weights,valid_groups,good))
    folds=np.empty((2,n),dtype=np.int8)
    for r,seed in enumerate(FOLD_SEEDS):folds[r,np.random.default_rng(seed).permutation(n)]=np.arange(n)%3
    raw=np.full((2,n,4,2),np.nan,dtype=np.float32);normalized=raw.copy();scales=np.zeros((2,3,2));records=[];done=0
    for repeat in range(2):
        for fold in range(3):
            fitting=np.where(folds[repeat]!=fold)[0];held=np.where(folds[repeat]==fold)[0]
            order=np.random.default_rng(20264000+100*repeat+fold).permutation(fitting);cut=int(.85*len(order));inner=order[:cut];stop=order[cut:]
            early_transform=normalizer(raw_numeric,valid,inner);outer_transform=normalizer(raw_numeric,valid,fitting)
            early_numeric=transform_numeric(raw_numeric,early_transform);outer_numeric=transform_numeric(raw_numeric,outer_transform)
            early_inputs=base_t+(torch.from_numpy(early_numeric),);outer_inputs=base_t+(torch.from_numpy(outer_numeric),)
            for arm,name in enumerate(ARMS):
                stem=f"repeat{repeat}_fold{fold}_{name}"
                progress("early_stopping",done,started,f"{stem}: inner training-only epoch selection")
                _,early_curve,epochs=train(early_inputs,structure,inner[eligible[inner]],arm==1,MAX_EPOCHS,stop[eligible[stop]])
                done+=1;dump(OUT/f"{stem}_early_curve.json",early_curve)
                progress("refitting",done,started,f"{stem}: fresh outer-fit model at epoch{epochs}")
                model,curve,_=train(outer_inputs,structure,fitting[eligible[fitting]],arm==1,epochs)
                done+=1;dump(OUT/f"{stem}_refit_curve.json",curve)
                native=native_scores(model,outer_inputs);asset=export(model,outer_transform)
                scale=float(max(np.std((native[fitting]-native[fitting].mean(axis=1,keepdims=True))[valid[fitting]]),1e-6))
                asset["score_scale"]=np.array(scale);path=OUT/f"{stem}.npz";np.savez_compressed(path,**asset)
                tick=time.perf_counter();replay=numpy_forward(nodes[held],adjacency[held],mask[held],outer_numeric[held],asset);inference_seconds=time.perf_counter()-tick
                error=float(np.max(np.abs(replay-native[held])));assert error<1e-4
                hb=[bank[i] for i in held];native_choices=select(hb,native[held],"mean");numpy_choices=select(hb,replay,"mean")
                answer_differences=sum(parse_answer(row["candidates"][a]["text"])!=parse_answer(row["candidates"][b]["text"]) for row,a,b in zip(hb,native_choices,numpy_choices))
                raw[repeat,held,:,arm]=replay;normalized[repeat,held,:,arm]=replay/scale;scales[repeat,fold,arm]=scale
                record={"repeat":repeat,"fold":fold,"arm":name,"selected_epochs":epochs,"model_fit_calls":2,
                    "inner_fit_question_sha256":arrhash(inner.astype(np.int64)),"inner_stop_question_sha256":arrhash(stop.astype(np.int64)),
                    "outer_fit_question_sha256":arrhash(fitting.astype(np.int64)),"outer_held_question_sha256":arrhash(held.astype(np.int64)),
                    "inner_fit_questions":len(inner),"inner_eligible":int(eligible[inner].sum()),"outer_fit_questions":len(fitting),
                    "early_seconds":early_curve[-1]["elapsed_seconds"],"refit_seconds":curve[-1]["elapsed_seconds"],
                    "scale":scale,"numpy_held_inference_seconds":inference_seconds,"held_questions":len(held),
                    "numpy_export_max_error":error,"numpy_native_answer_differences":answer_differences,
                    "metrics":metrics(numpy_choices,labels[held],hard[held]),"asset":path.name,"asset_sha256":digest(path),
                    "uncompressed_asset_bytes":sum(a.nbytes for a in asset.values()),
                    "inner_normalizer_mean_sha256":arrhash(early_transform["numeric_mean"]),"outer_normalizer_mean_sha256":arrhash(outer_transform["numeric_mean"])}
                records.append(record);dump(OUT/"fit_records.json",records)
                print(json.dumps({"fits":done,"of":24,"name":stem,"epochs":epochs,"hard_correct":record["metrics"]["hard_correct"]}),flush=True)
            if repeat==0 and fold==0:
                dump(OUT/"cost_probe.json",{"planned_fits_retained":4,"estimated_remaining_fit_seconds":sum(r["early_seconds"]+r["refit_seconds"] for r in records)*5,
                    "estimated_numpy467_seconds":[r["numpy_held_inference_seconds"]*467/r["held_questions"] for r in records],
                    "estimates_exclude_preprocessing_startup_and_scoring":True})
            progress("fitting",done,started,f"Completed repeat{repeat+1} fold{fold+1}")
    assert np.isfinite(raw).all()
    correct=np.zeros((2,n,2),dtype=bool);choices=np.zeros((2,n,2),dtype=np.int32);results=[]
    for arm,name in enumerate(ARMS):
        repeats=[]
        for r in range(2):
            choices[r,:,arm]=select(bank,raw[r,:,:,arm],"mean");correct[r,:,arm]=labels[np.arange(n),choices[r,:,arm]]
            result=metrics(choices[r,:,arm],labels,hard)
            result["group_ce_hard"]=numpy_group_loss(normalized[r,:,:,arm],weights,valid_groups,good,eligible);repeats.append(result)
        results.append({"name":name,"repeats":repeats,"summed_hard_correct":sum(r["hard_correct"] for r in repeats),
            "mean_group_ce":float(np.mean([r["group_ce_hard"] for r in repeats]))})
    for arm,result in enumerate(results):
        result["paired_vs_node_mlp"]=[{"repaired":int((~correct[r,:,0]&correct[r,:,arm]&hard).sum()),
            "regressed":int((correct[r,:,0]&~correct[r,:,arm]&hard).sum()),"gain":int(correct[r,hard,arm].sum()-correct[r,hard,0].sum())} for r in range(2)]
    chosen=max(range(2),key=lambda a:(results[a]["summed_hard_correct"],-results[a]["mean_group_ce"],-a))
    np.savez_compressed(OUT/"oof_logits.npz",ids=np.array([r["id"] for r in bank]),raw=raw,normalized=normalized,labels=labels,valid=valid,hard=hard,
        folds=folds,scales=scales,choices=choices,correct=correct,configuration_names=np.array(ARMS),fold_seeds=np.array(FOLD_SEEDS))
    dump(OUT/"oof_schema.json",{"raw_normalized_shape":[2,n,4,2],"axes":["repeat","question","candidate","configuration"],"label_scope":"train only","raw_source":"NumPy exported inference"})
    summary={"configurations":results,"selected_config_index":chosen,"training_selected_arm":ARMS[chosen],"model_fit_calls":done,
        "normalizer_fits":12,"full_refit_calls":0,"validation_evaluations":0,"test_access":False,"feature_seconds":feature_seconds,
        "elapsed_seconds":time.perf_counter()-started,"training_questions":n,"hard_questions":int(hard.sum()),
        "numpy_max_error":max(r["numpy_export_max_error"] for r in records),"numpy_native_answer_differences":sum(r["numpy_native_answer_differences"] for r in records),
        "threads":4,"torch":torch.__version__,"numpy":np.__version__,"python":platform.python_version(),
        "source_hashes":sources,"input_hashes":inputs_hash,"sources_unchanged":all(digest(ROOT/name)==h for name,h in sources.items()),
        "interpretation":"Exploratory repeated OOF after historical selection; no validation/test score or deployment claim"}
    assert summary["sources_unchanged"]
    dump(OUT/"results.json",results);dump(OUT/"summary.json",summary)
    progress("complete",done,started,"24 fits complete; NumPy OOF saved; no validation/test/fullrefit")


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare",action="store_true");g.add_argument("--run",action="store_true")
    args=parser.parse_args();prepare() if args.prepare else run()


if __name__=="__main__":main()
