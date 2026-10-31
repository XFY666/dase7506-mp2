"""Selected-branch full-training refit and NumPy-only scoring adapters.

Import and inference use NumPy and the standard library. Training is explicit:
train_branch(...) or --fit --branch NAME --output DIRECTORY. No automatic fit,
validation/test loading, model selection, or prediction cache is provided.
"""
from __future__ import annotations
import os
import sys
sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT.parent/"docs/optimization_refit_protocol.md"
FORMAT = "mp2_sequence_branch_v1"
EPOCH_POLICY = "max(1, floor(median(six outer-fold selected epochs)+0.5))"
BRANCHES = {
    "headtail_roles": ("optimization_sequence", "role_cnn", "recipe"),
    "control_allq": ("optimization_sequence_aux", "aux_cnn", "config"),
    "token_aux01": ("optimization_sequence_aux", "aux_cnn", "config"),
    "token_attention": ("optimization_transformer", "transformer", "config"),
    "token_set_attention": ("optimization_transformer", "transformer", "config"),
    "digit_text_control": ("optimization_reference_augmentation", "digit_cnn", "config"),
    "reference_corruption01": ("optimization_reference_augmentation", "digit_cnn", "config"),
}
_DIGIT_PATTERN = re.compile(r"####|\\[a-z]+|[a-z]+(?:'[a-z]+)?|\d[\d,]*(?:\.\d+)?|[^\w\s]", re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def _inference_sources(branch):
    if branch not in BRANCHES:
        raise ValueError("Unsupported branch: "+str(branch))
    names = ["optimization_sequence_refit.py", "common.py", "selection.py", "sequence_model.py", "text_features.py"]
    family = BRANCHES[branch][1]
    if family != "digit_cnn":
        names.append("optimization_sequence_model.py")
    if family == "transformer":
        names.append("optimization_transformer_model.py")
    return {name: sha(ROOT/name) for name in names}


def epoch_provenance(name):
    """Inspect completed outer-fold evidence, without fitting or reading data."""
    if name not in BRANCHES:
        raise ValueError("Unsupported branch: "+str(name))
    run, family, key = BRANCHES[name]
    directory = ROOT/"runs"/run
    required = ["summary.json", "fit_records.json", "source_evidence.json", "protocol.json"]
    if not all((directory/x).is_file() for x in required):
        raise RuntimeError(name+": complete study evidence is required before full refit")
    summary, protocol = _json(directory/"summary.json"), _json(directory/"protocol.json")
    planned = protocol.get("planned_fits", protocol.get("max_model_fits"))
    if planned is None or summary.get("model_fit_calls") != planned:
        raise RuntimeError(name+": source study has not completed its registered fits")
    evidence = _json(directory/"source_evidence.json")
    if evidence != {file: sha(ROOT/file) for file in evidence}:
        raise RuntimeError(name+": a registered source changed")
    selected = [row for row in _json(directory/"fit_records.json") if row.get(key) == name]
    if len(selected) != 6 or {(row["repeat"],row["fold"]) for row in selected} != {(r,f) for r in range(2) for f in range(3)}:
        raise RuntimeError(name+": exactly six unique completed outer folds are required")
    selected.sort(key=lambda row:(row["repeat"],row["fold"]))
    epochs = [int(row["selected_epochs"]) for row in selected]
    if not all(1 <= e <= 24 for e in epochs):
        raise RuntimeError("Outer-selected epoch outside registered bounds")
    final_epochs = max(1, math.floor(float(np.median(epochs))+.5))
    return {"branch":name, "family":family, "policy":EPOCH_POLICY, "epochs":final_epochs,
        "outer_selected_epochs":epochs,
        "outer_records":[{"repeat":row["repeat"],"fold":row["fold"],"selected_epochs":row["selected_epochs"]} for row in selected],
        "source_run":"runs/"+run, "source_files":evidence,
        "evidence_files_sha256":{file:sha(directory/file) for file in required},
        "refit_protocol_sha256":sha(PROTOCOL)}


def _digits(text):
    # Identical input-only C4 tokenization, kept here to avoid importing its
    # training/diagnostic module in a deployed inference process.
    text = str(text).lower()
    if len(text) > 24000:
        return _digits(text[:12000])+["<gap>"]+_digits(text[-12000:])
    result = []
    for part in _DIGIT_PATTERN.findall(text):
        if part and part[0].isdigit():
            result.append("<number>")
            result.extend(f"<digit_{c}>" if c.isdigit() else "<decimal>" if c == "." else "<comma>" for c in part)
            result.append("<number_end>")
        else:
            result.append(part[:64])
    return result


def _digit_sequence(question, text):
    q, body = _digits(question), _digits(text)
    if len(q) > 64:
        q = q[:32]+q[-32:]
    if len(body) > 256:
        body = body[:128]+["<gap>"]+body[-128:]
    return q+["<sep>"]+body


def _score_digit(bank, asset):
    from sequence_model import numpy_forward
    if not bank:
        return np.empty((0,4),np.float32)
    lookup = {token:i for i,token in enumerate(asset["vocabulary"].tolist())}
    codes = np.zeros((len(bank),4,322),np.int64)
    for i,row in enumerate(bank):
        if len(row["candidates"]) != 4:
            raise ValueError("Expected four candidates")
        for j,candidate in enumerate(row["candidates"]):
            seq = _digit_sequence(row["question"], candidate["text"])
            codes[i,j,:len(seq)] = [lookup.get(token,1) for token in seq]
    return numpy_forward(codes,np.zeros((len(bank),4,0),np.float32),asset)


def _score_asset(name, bank, embeddings, asset):
    family = BRANCHES[name][1]
    if family == "digit_cnn":
        return _score_digit(bank,asset)
    if family == "transformer":
        from optimization_transformer_model import score_candidates
    else:
        from optimization_sequence_model import score_candidates
    return score_candidates(bank,embeddings,asset)


def score_branch(bank, emb, checkpoint_manifest, *, normalize=False):
    """Return fresh (N,4) scores; normalize=True divides by full-train scale.

    checkpoint_manifest is a manifest JSON path, its directory, or a dictionary.
    A dictionary needs an absolute asset_path or a directory for relative paths.
    """
    if isinstance(checkpoint_manifest, dict):
        manifest = checkpoint_manifest
        directory = Path(manifest.get("directory", ".")).resolve()
        if not Path(manifest["asset_path"]).is_absolute() and "directory" not in manifest:
            raise ValueError("Dictionary manifest requires directory or absolute asset_path")
    else:
        path = Path(checkpoint_manifest)
        if path.is_dir():
            path /= "manifest.json"
        directory, manifest = path.resolve().parent, _json(path)
    if manifest.get("format") != FORMAT or manifest.get("branch") not in BRANCHES:
        raise ValueError("Unsupported checkpoint manifest")
    name = manifest["branch"]
    if manifest["inference_sources_sha256"] != _inference_sources(name):
        raise RuntimeError("Inference implementation differs from checkpoint manifest")
    path = Path(manifest["asset_path"])
    if not path.is_absolute():
        path = directory/path
    if sha(path) != manifest["asset_sha256"]:
        raise RuntimeError("Checkpoint asset checksum mismatch")
    with np.load(path,allow_pickle=False) as saved:
        asset = {key:saved[key] for key in saved.files}
    scale = float(manifest["score_scale"])
    if not np.isfinite(scale) or scale <= 0 or float(asset.get("score_scale",scale)) != scale:
        raise ValueError("Invalid score scale")
    scores = _score_asset(name,bank,emb,asset)
    if scores.shape != (len(bank),4) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid fresh branch scores")
    return scores/scale if normalize else scores


def predict(bank, arrays, checkpoint_path):
    from selection import select
    return select(bank,score_branch(bank,arrays.get("embeddings"),checkpoint_path),"mean")


@contextmanager
def _training_io(output):
    """A scoped audit hook; disabled after return to permit another branch call."""
    from common import DATA
    active = [True]
    accesses = set()
    def hook(event,args):
        if not active[0] or event != "open" or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.name.startswith(("gsm8k_validation","gsm8k_test","gsm8k_audit")):
            raise RuntimeError("Only training data may be read during full refit")
        if DATA.resolve() in path.parents:
            accesses.add(path.relative_to(DATA.resolve()).as_posix())
        mode,flags = args[1],args[2] or 0
        writing = (isinstance(mode,str) and any(c in mode for c in "wax+")) or bool(flags&(os.O_WRONLY|os.O_RDWR))
        if writing and output not in path.parents:
            raise RuntimeError("Refit output outside its new output directory: "+str(path))
    sys.addaudithook(hook)
    try:
        yield accesses
    finally:
        active[0] = False


def _validate_full_training(name, bank, emb, labels, refs):
    from common import DATA, load_inputs, parse_answer, read_jsonl
    official_bank = read_jsonl(DATA/"candidates/gsm8k_train.jsonl")
    official_refs = read_jsonl(DATA/"gsm8k_train.jsonl")
    if bank != official_bank or len(bank) != 4755:
        raise ValueError("Full-training bank must exactly match the supplied course training bank")
    if refs is not None and refs != official_refs:
        raise ValueError("References must exactly match full training references")
    expected = np.array([[parse_answer(c["text"]) is not None and parse_answer(c["text"]) == parse_answer(ref["answer"])
                         for c in row["candidates"]] for row,ref in zip(bank,official_refs)],np.float32)
    if not np.array_equal(labels,expected):
        raise ValueError("Labels differ from official final-answer training labels")
    if BRANCHES[name][1] != "digit_cnn":
        _, arrays = load_inputs("train")
        if emb is None or not np.array_equal(np.asarray(emb),arrays["embeddings"]):
            raise ValueError("Dense branch requires unmodified full-training embeddings")
    return expected, official_refs


def train_branch(name, output_directory, training_bank, emb, labels, refs=None):
    """Fit one explicitly selected branch once and return its manifest path.

    This function has side effects only when explicitly called. The output must
    not already exist. It never evaluates validation/test or picks a branch.
    """
    plan = epoch_provenance(name)
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing output; choose a new refit directory")
    output.mkdir(parents=True)
    (output/"tmp").mkdir()
    previous_env = {key:os.environ.get(key) for key in ("TMP","TEMP")}
    os.environ["TMP"] = os.environ["TEMP"] = str(output/"tmp")
    start_wall,start_cpu = time.perf_counter(),time.process_time()
    state = {"branch":name,"stage":"checking_inputs","completed_fits":0,"fit_calls_started":0,"planned_fits":1,
             "updated_utc":datetime.now(timezone.utc).isoformat()}
    _dump(output/"progress.json",state)
    _dump(output/"plan.json",plan)
    try:
        with _training_io(output) as accesses:
            y, references = _validate_full_training(name,training_bank,emb,labels,refs)
            import torch
            import sequence_experiment as base
            torch.set_num_threads(4)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            if torch.version.cuda is not None:
                raise RuntimeError("Use the registered CPU-only PyTorch environment")
            weights,gv,good,valid,eligible = base.group_structure(training_bank,y)
            all_ids = np.arange(len(training_bank))
            structure = weights,gv,good
            structure_t = tuple(torch.from_numpy(x) for x in structure)
            family,epochs = plan["family"],plan["epochs"]
            target_info = None
            if family == "digit_cnn":
                import optimization_reference_augmentation as task
                actual,augmented,observed,_ = task.make_sequences(training_bank,references)
                codes,extra,active,vocab,transform_info = task.prepare_inputs(actual,augmented,observed,all_ids)
                transform = {"vocabulary":vocab}
                ct,at,ot = map(torch.from_numpy,(codes,extra,active))
                et = torch.from_numpy(eligible)
                strength = .1 if name == "reference_corruption01" else 0.
                target_info = {"observed_questions":int(active.sum()),"positive_original_negative_corrupted":True,
                               "final_candidate_labels_unchanged":True}
                training_ids = all_ids
            else:
                import optimization_sequence as screen
                import optimization_sequence_model as variants
                import sequence_model as original
                sequences = variants.sequences(training_bank,"headtail_roles")
                raw = original.numeric_inputs(training_bank,emb)
                codes,numeric,transform,transform_info = screen.prepare(sequences,"headtail_roles",raw,valid,all_ids)
                ct,nt,structure_t = base.tensor_inputs(codes,numeric,structure)
                training_ids = all_ids[eligible]
                if family == "aux_cnn":
                    import optimization_sequence_aux as task
                    targets,masks,observed,gold_counts = task.token_targets(training_bank,references)
                    tt,mt,et = map(torch.from_numpy,(targets,masks,eligible))
                    strength = .1 if name == "token_aux01" else 0.
                    training_ids = all_ids
                    target_info = {"observed_questions":int(observed.sum()),"tokens":int(masks.sum()),
                        "positive_tokens":int((targets*masks).sum()),"usable_gold_values_mean":float(gold_counts.mean())}
                elif family == "transformer":
                    import optimization_transformer as task
                    vt = torch.from_numpy(valid)
            # Imported legacy training modules may reset TMP; point it back to
            # this new output before the single optimizer run.
            os.environ["TMP"] = os.environ["TEMP"] = str(output/"tmp")
            state.update(stage="full_training",fit_calls_started=1,detail=f"{epochs} fixed epochs",elapsed_seconds=time.perf_counter()-start_wall)
            _dump(output/"progress.json",state)
            fit_wall,fit_cpu = time.perf_counter(),time.process_time()
            if family == "role_cnn":
                fitted,curve,used = base.train(ct,nt,structure_t,training_ids,True,epochs)
            elif family == "aux_cnn":
                fitted,curve,used = task.train(ct,nt,structure_t,et,tt,mt,training_ids,strength,epochs)
            elif family == "transformer":
                fitted,curve,used = task.train(ct,nt,structure_t,vt,training_ids,name=="token_set_attention",epochs,time.perf_counter()+7200.)
            else:
                fitted,curve,used = task.train(ct,at,structure_t,et,ot,training_ids,strength,epochs,time.perf_counter()+7200.)
            fit_seconds,fit_cpu_seconds = time.perf_counter()-fit_wall,time.process_time()-fit_cpu
            state.update(completed_fits=1,stage="exporting",updated_utc=datetime.now(timezone.utc).isoformat())
            _dump(output/"progress.json",state)
            if used != epochs or len(curve) != epochs:
                raise RuntimeError("Full fit did not complete its fixed epoch count")
            _dump(output/"curve.json",curve)
            if family == "transformer":
                z = task.native_scores(fitted,ct,nt,vt)
                asset = task.export(fitted,transform)
            elif family == "digit_cnn":
                z = task.native_scores(fitted,ct)
                asset = base.export(fitted,transform)
            else:
                z = base.native_scores(fitted,ct,nt)
                asset = base.export(fitted,transform)
            scale = float(max(np.std((z-z.mean(1,keepdims=True))[valid]),1e-6))
            asset["score_scale"] = np.array(scale)
            if family != "digit_cnn":
                asset["token_recipe"] = np.array("headtail_roles")
            asset_path = output/"model.npz"
            np.savez(asset_path,**asset)
            # Fresh input preprocessing plus NumPy forward on training inputs;
            # no saved logits are returned or used by deployment.
            probe_bank = training_bank[:32]
            probe_emb = None if emb is None else np.asarray(emb)[:32]
            replay = _score_asset(name,probe_bank,probe_emb,asset)
            error = float(np.max(abs(replay-z[:32])))
            from selection import select
            if error >= 1e-4 or select(probe_bank,replay,"mean") != select(probe_bank,z[:32],"mean"):
                raise RuntimeError("Full-fit NumPy export mismatch")
            if plan != epoch_provenance(name):
                raise RuntimeError("Frozen source/evidence changed during full refit")
            manifest = {"format":FORMAT,"branch":name,"family":family,"asset_path":"model.npz",
                "asset_sha256":sha(asset_path),"asset_bytes":asset_path.stat().st_size,"score_scale":scale,
                "raw_scores_default":True,"normalized_scores_option":"score_branch(..., normalize=True)",
                "aggregation":"mean","tokenizer":"digit_headtail" if family=="digit_cnn" else "headtail_roles",
                "inference_sources_sha256":_inference_sources(name),
                "training_sources_sha256":plan["source_files"],"refit_source_sha256":sha(__file__),
                "epoch_provenance":plan,"initialization_seed":7506,"shuffle_seed":7507,"cpu_threads":4,
                "training_questions":len(training_bank),"optimizer_questions":len(training_ids),
                "training_ids_sha256":hashlib.sha256(np.ascontiguousarray(training_ids).tobytes()).hexdigest(),
                "transform":transform_info,"auxiliary_target_summary":target_info,
                "model_fit_calls":1,"vocabulary_fits":1,"numeric_normalizer_fits":0 if family=="digit_cnn" else 1,
                "fullfit_wall_seconds":fit_seconds,"fullfit_process_cpu_seconds":fit_cpu_seconds,
                "total_wall_seconds":time.perf_counter()-start_wall,"total_process_cpu_seconds":time.process_time()-start_cpu,
                "torch_version":torch.__version__,"numpy_version":np.__version__,
                "export_training_probe_questions":len(probe_bank),"export_max_abs_error":error,"export_answer_changes":0,
                "validation_test_access":False,"data_files_read":sorted(accesses),
                "created_utc":datetime.now(timezone.utc).isoformat()}
            if manifest["asset_bytes"] > 16*1024*1024:
                raise RuntimeError("Single branch exceeds total submission asset limit")
            _dump(output/"manifest.json",manifest)
            state.update(stage="complete",completed_fits=1,detail="fresh full-training model exported",
                         elapsed_seconds=time.perf_counter()-start_wall,updated_utc=datetime.now(timezone.utc).isoformat())
            _dump(output/"progress.json",state)
    except Exception as error:
        _dump(output/"failure.json",{"type":type(error).__name__,"message":str(error),
            "fit_calls_started":state["fit_calls_started"],"completed_fits":state["completed_fits"],
            "utc":datetime.now(timezone.utc).isoformat()})
        state.update(stage="failed",detail=str(error),updated_utc=datetime.now(timezone.utc).isoformat())
        _dump(output/"progress.json",state)
        raise
    finally:
        for key,value in previous_env.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key] = value
    return output/"manifest.json"


def self_check(output_directory):
    """Synthetic-input routing/export checks, no training or split data access."""
    import copy
    output = Path(output_directory).resolve()
    output.mkdir(parents=True,exist_ok=True)
    bank = [{"id":"synthetic0","question":"There are 3 boxes of 4 apples. How many total?",
             "candidates":[{"index":i,"text":text} for i,text in enumerate(("3*4=12. #### 12","3+4=7. #### 7","There are 12 apples. #### 12","No final number."))]},
            {"id":"synthetic1","question":"What remains after 2 of 10 items are used?",
             "candidates":[{"index":i,"text":text} for i,text in enumerate(("10-2=8. #### 8","8 remain. #### 8","10+2=12. #### 12","2 remain. #### 2"))]}]
    embeddings = np.random.default_rng(7506).normal(size=(2,4,1024)).astype(np.float32)
    checked = []
    for name,(run,family,key) in BRANCHES.items():
        if family == "digit_cnn":
            asset_path = ROOT/"runs"/run/"readiness_asset.npz"
        else:
            asset_path = ROOT/"runs"/run/f"repeat0_fold0_{name}.npz"
        with np.load(asset_path,allow_pickle=False) as saved:
            asset = {k:saved[k] for k in saved.files}
        manifest = {"format":FORMAT,"branch":name,"asset_path":str(asset_path),"asset_sha256":sha(asset_path),
                    "score_scale":float(asset.get("score_scale",1.)),"inference_sources_sha256":_inference_sources(name)}
        z = score_branch(bank,embeddings,manifest)
        normalized = score_branch(bank,embeddings,manifest,normalize=True)
        assert np.array_equal(normalized,z/manifest["score_scale"])
        perm = np.array([2,0,3,1]); permuted = copy.deepcopy(bank)
        for row in permuted:
            row["id"] = "unused"
            row["candidates"] = [row["candidates"][i] for i in perm]
            for candidate in row["candidates"]:
                candidate["index"] = -1
        zp = score_branch(permuted,embeddings[:,perm],manifest)
        perr = float(np.max(abs(zp-z[:,perm])))
        assert perr < 1e-4
        assert score_branch([],embeddings[:0],manifest).shape == (0,4)
        checked.append({"branch":name,"asset_kind":"synthetic_readiness" if family == "digit_cnn" else "existing_outer_fold",
            "permutation_error":perr,"empty_input":True,"normalization_exact":True})
    # Compare the copied deployment-only digit tokenizer to the frozen original.
    import optimization_reference_augmentation as digit_original
    cases = ["+12 -3.5 1,200 2/3 \\frac{4}{5}","word "*150+"#### 123.45","x"*25000+" = 9.01"]
    for question in cases:
        for text in cases:
            assert _digit_sequence(question,text) == digit_original.input_sequence(question,text)
    assert not any(name in sys.modules for name in ("torch","sklearn","sequence_experiment","optimization_sequence_aux","optimization_transformer"))
    plans = {}
    for name in BRANCHES:
        try:
            plans[name] = epoch_provenance(name)
        except RuntimeError as error:
            plans[name] = {"not_ready":str(error)}
    result = {"status":"passed","checks":checked,"digit_tokenizer_exact_parity":True,
        "training_dependencies_imported":False,"new_model_fits":0,"dataset_reads":0,"full_refits":0,
        "epoch_plans":plans,"source_sha256":sha(__file__),"protocol_sha256":sha(PROTOCOL),
        "numpy_version":np.__version__,"updated_utc":datetime.now(timezone.utc).isoformat()}
    _dump(output/"readiness.json",result)
    print(json.dumps({"status":"passed","branches_checked":len(checked),"new_model_fits":0,"dataset_reads":0},indent=2))


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--fit",action="store_true")
    action.add_argument("--describe",action="store_true")
    action.add_argument("--self-check",action="store_true")
    parser.add_argument("--branch",choices=tuple(BRANCHES))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.describe:
        if not args.branch:
            parser.error("--describe requires --branch")
        print(json.dumps(epoch_provenance(args.branch),indent=2))
    elif args.self_check:
        if args.output is None:
            parser.error("--self-check requires --output")
        self_check(args.output)
    else:
        if not args.branch or args.output is None:
            parser.error("--fit requires --branch and --output")
        # Only the training split is loadable through this CLI.
        from common import DATA, load_inputs, parse_answer, read_jsonl
        bank,arrays = load_inputs("train")
        refs = read_jsonl(DATA/"gsm8k_train.jsonl")
        labels = np.array([[parse_answer(c["text"]) is not None and parse_answer(c["text"]) == parse_answer(ref["answer"])
                          for c in row["candidates"]] for row,ref in zip(bank,refs)],np.float32)
        path = train_branch(args.branch,args.output,bank,arrays["embeddings"],labels,refs)
        print(json.dumps({"manifest":str(path)},indent=2))


if __name__ == "__main__":
    main()
