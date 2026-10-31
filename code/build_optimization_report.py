"""Evidence-bound English review-report builder. No training or evaluation.

--validate-only does not author a PDF. --build is explicit; the caller owns the
PDF operation marker, rendered-page inspection and subsequent human approval.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT/"code/runs"
SCHEMA = "mp2-optimization-report-v1"
STUDIES = {
    "E_CROSS":"optimization_interactions", "E_STOP":"optimization_stopping",
    "E_SUBSET":"optimization_subset", "E_REG":"optimization_regularization",
    "E_PROCESS":"optimization_process", "E_TOKEN":"optimization_sequence_aux",
    "E_CORRUPT":"optimization_reference_augmentation", "E_ATTENTION":"optimization_transformer",
    "E_GRAPH":"optimization_graph", "E_CALIB":"optimization_calibration",
    "E_SEQUENCE":"optimization_sequence", "E_FUSION":"optimization_fusion",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pointer(value, path):
    if path in ("", "/"):
        return value
    if not path.startswith("/"):
        raise ValueError("JSON pointers must start with /")
    for token in path[1:].split("/"):
        token = token.replace("~1","/").replace("~0","~")
        value = value[int(token)] if isinstance(value,list) else value[token]
    return value


def checked_metric(value, hard=False):
    prefix = "hard_" if hard and "hard_correct" in value else ""
    correct,n = value[prefix+"correct"],value[prefix+"n"]
    if type(correct) is not int or type(n) is not int or not 0 <= correct <= n or n < 1:
        raise ValueError("Metric needs valid integer counts and denominator")
    accuracy = 100*correct/n
    if prefix+"accuracy" in value and abs(float(value[prefix+"accuracy"])-accuracy) > 1e-6:
        raise ValueError("Reported percentage disagrees with counts")
    return {"correct":correct,"n":n,"accuracy":accuracy}


def metric_text(metric):
    return f'{metric["correct"]}/{metric["n"]} ({metric["accuracy"]:.2f}%)'


class Evidence:
    def __init__(self, config, base):
        self.records, self.values = {},{}
        for key, record in config.items():
            path = Path(record["path"])
            self.add(key,path if path.is_absolute() else base/path,record["description"],record["sha256"])

    def add(self,key,path,description,expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and expected != actual:
            raise ValueError("Evidence checksum mismatch: "+key)
        if key in self.records:
            if self.records[key]["sha256"] != actual:
                raise ValueError("Evidence ID collision: "+key)
            return
        try:
            display_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            display_path = str(path)
        self.records[key] = {"id":key,"path":display_path,"sha256":actual,"description":description,
            "verification":"programmatic hash and local evidence consistency checks",
            "human_verified":False,"human_review":"pending"}
        self.values[key] = read_json(path) if path.suffix.lower()==".json" else path.read_text(encoding="utf-8")

    def resolve(self, ref):
        if not isinstance(ref,dict) or set(ref) != {"evidence","pointer"}:
            raise ValueError("Evidence references require exactly evidence and pointer")
        return pointer(self.values[ref["evidence"]],ref["pointer"])

    def paragraphs(self, rows):
        result = []
        for row in rows:
            if not row.get("text") or not row.get("evidence"):
                raise ValueError("Narrative paragraphs need text and evidence IDs")
            if re.search(r"\b(TODO|TBD|PLACEHOLDER|lorem ipsum)\b",row["text"],re.I):
                raise ValueError("Unresolved manuscript placeholder")
            if any(key not in self.records for key in row["evidence"]):
                raise ValueError("Unknown narrative evidence ID")
            result.append(escape(row["text"])+" ["+", ".join(row["evidence"])+"]")
        return result


def series(data,name):
    """Normalize observed summary layouts without silently choosing a row."""
    if "results" in data and isinstance(data["results"],list) and data["results"] and "results" in data["results"][0]:
        rows = [row["results"][name] for row in data["results"]]
    else:
        collection = next((data[k] for k in ("configs","recipes","configurations","results") if k in data),None)
        row = next(x for x in collection if x.get("config",x.get("recipe",x.get("name")))==name)
        rows = row["repeats"]
    if len(rows) != 2:
        raise ValueError("Expected exactly two training OOF repeats")
    result=[]
    for row in rows:
        metric = checked_metric(row.get("metrics",row),hard=True)
        if metric["n"] != 2114:
            raise ValueError("Unexpected hard-question denominator")
        ce = row.get("group_ce",row.get("group_ce_hard"))
        if ce is None or not math.isfinite(float(ce)):
            raise ValueError("Missing finite group CE")
        result.append((metric,float(ce)))
    return result


def comparison(data,old,new,label,evidence):
    a,b=series(data,old),series(data,new)
    counts=lambda x:" / ".join(str(m["correct"]) for m,ce in x)
    delta=" / ".join(f'{b[i][0]["correct"]-a[i][0]["correct"]:+d}' for i in range(2))
    ce=sum(x[1] for x in b)/2-sum(x[1] for x in a)/2
    return [label,counts(a),counts(b),delta,f"{ce:+.4f}",evidence]


def components(config, evidence, config_ref):
    source_path=Path(evidence.records[config_ref["evidence"]]["path"])
    directory=(ROOT/source_path).parent if not source_path.is_absolute() else source_path.parent
    rows,kinds=[],set()
    items=config.get("components",config.get("heads",[]))
    for i,item in enumerate(items):
        weight=float(item["weight"])
        if weight<0 or not math.isfinite(weight):
            raise ValueError("Invalid branch weight")
        if "manifest" in item:
            path=directory/item["manifest"];key=f"E_COMPONENT_{i+1}"
            evidence.add(key,path,"Selected component manifest",item["manifest_sha256"])
            manifest=evidence.values[key];name=manifest["branch"]
            members=len(manifest.get("members",[{}]))
            family=manifest.get("family") or manifest.get("members",[{}])[0].get("family") or item["adapter"]
        else:
            name=item.get("name",item["kind"]);family=item["kind"];members=1;key=config_ref["evidence"]
        kinds.add(family)
        descriptions={"group":"Pooled answer-group comparison","kernel":"PCA + antisymmetric RBF comparison",
            "interaction":"PLS + quantity/candidate-comparison trees","pair":"Conditional pair comparison",
            "role_cnn":"Numeric-role token CNN + dense inputs","aux_cnn":"Numeric-role CNN",
            "transformer":"Scratch token attention","digit_cnn":"Digit-preserving text CNN",
            "nonlinear":"Pairwise embedding scorer","residual":"Listwise embedding scorer","trees":"Structured-text trees"}
        rows.append([name,f"{weight:.6g}",str(members),descriptions.get(family,str(family)),key])
    if not rows or abs(sum(float(x["weight"]) for x in items)-1)>1e-8:
        raise ValueError("Selected component weights must sum to one")
    return rows,kinds


def load_inputs(path):
    path=Path(path).resolve();spec=read_json(path)
    if spec.get("schema")!=SCHEMA:
        raise ValueError("Wrong report-input schema")
    if spec.get("human_review")!="pending":
        raise ValueError("This builder produces a local review draft only")
    evidence=Evidence(spec["evidence"],path.parent)
    for key,file,description in (
        ("E_GUIDE","guide/GUIDE.md","Course task, constraints, assessment and attribution"),
        ("E_README","code/README.md","Official input/evaluation contract and split construction"),
        ("E_PROTOCOL","docs/optimization_protocol.md","Prospective renewed-search protocol and prior-history disclosure"),
        ("E_CORE_METHOD","code/optimization_core_refit.py","Fixed selected-branch implementation"),
        ("E_SEQUENCE_METHOD","code/optimization_sequence_refit.py","Sequence refit and NumPy scoring"),
        ("E_SKILLS","docs/optimization_report_reference.json","Current arXiv reference metadata")):
        evidence.add(key,ROOT/file,description)
    studies={}
    for key,run in STUDIES.items():
        evidence.add(key,RUNS/run/"summary.json",run+" completed training-only results")
        evidence.add(key+"_P",RUNS/run/"protocol.json",run+" prospective recipe")
        studies[key]=evidence.values[key]
    final={key:evidence.resolve(spec["final"][key]) for key in
        ("name","validation_full","validation_hard","test","resources","config","selection")}
    final["validation_full"]=checked_metric(final["validation_full"])
    final["validation_hard"]=checked_metric(final["validation_hard"],hard=True)
    final["test"]=checked_metric(final["test"])
    if (final["validation_full"]["n"],final["validation_hard"]["n"],final["test"]["n"])!=(798,355,467):
        raise ValueError("Final evidence has incorrect split denominators")
    if not isinstance(final["name"],str) or not final["name"]:
        raise ValueError("Missing final selected name")
    resources=final["resources"]
    for key in ("prediction_fresh_process_seconds","scoring_fresh_process_seconds","total_fresh_process_seconds",
                "asset_uncompressed_bytes","asset_file_bytes","hardware","config_sha256","score_sha256"):
        if key not in resources:
            raise ValueError("Missing final resource field: "+key)
    if abs(resources["prediction_fresh_process_seconds"]+resources["scoring_fresh_process_seconds"]-resources["total_fresh_process_seconds"])>1e-3:
        raise ValueError("Resource timing components disagree")
    if resources["config_sha256"]!=evidence.records[spec["final"]["config"]["evidence"]]["sha256"]:
        raise ValueError("Resource evidence/config checksum mismatch")
    if resources["score_sha256"]!=evidence.records[spec["final"]["test"]["evidence"]]["sha256"]:
        raise ValueError("Resource evidence/official test score checksum mismatch")
    for key in ("prediction_fresh_process_seconds","scoring_fresh_process_seconds","total_fresh_process_seconds"):
        if not math.isfinite(resources[key]) or resources[key]<=0:
            raise ValueError("Invalid resource timing")
    for key in ("asset_uncompressed_bytes","asset_file_bytes"):
        if type(resources[key]) is not int or resources[key]<=0:
            raise ValueError("Invalid resource asset size: "+key)
    if resources["asset_uncompressed_bytes"]>16*2**20 or not resources["hardware"]:
        raise ValueError("Uncompressed inference assets exceed the course budget, or hardware is missing")
    if resources.get("split")!="test" or resources.get("threads")!=4:
        raise ValueError("Final resources must describe full test prediction/scoring with four threads")
    decision=final["selection"]["selected"]
    if decision["name"]!=final["name"]:
        raise ValueError("Final name differs from recorded selection")
    if decision.get("uncompressed_bytes")!=resources["asset_uncompressed_bytes"]:
        raise ValueError("Final uncompressed asset size differs from the selected checkpoint")
    for prefix,metric in (("validation_",final["validation_full"]),("validation_hard_",final["validation_hard"])):
        for field in ("correct","n"):
            if prefix+field in decision and decision[prefix+field]!=metric[field]:
                raise ValueError("Final metric differs from selected decision record")
    analysis=evidence.resolve(spec["analysis"])
    if analysis.get("status")!="complete" or analysis.get("selected_name")!=final["name"]:
        raise ValueError("Missing complete selected-checkpoint analysis")
    if checked_metric(analysis["validation"]["full"])!=final["validation_full"] or checked_metric(analysis["validation"]["hard"],True)!=final["validation_hard"]:
        raise ValueError("Validation analysis and official metrics disagree")
    for key in ("history","selection_commentary","reliability","failure","limits"):
        if not analysis.get(key):
            raise ValueError("Missing analysis narrative: "+key)
        evidence.paragraphs(analysis[key])
    for key in ("baselines","strata","selection_rows","search_cost_rows"):
        if not analysis.get(key):
            raise ValueError("Missing analysis table: "+key)
    for row in analysis["baselines"]+analysis["strata"]:
        checked_metric(row["metric"])
    for row in analysis["search_cost_rows"]:
        if type(row["model_fits"]) is not int or row["model_fits"]<0 or not row.get("cost_text"):
            raise ValueError("Cost rows require actual nonnegative fit counts and interpretation")
    examples=analysis["examples"]
    if not 1<=len(examples)<=2:
        raise ValueError("Provide one or two real validation examples")
    for row in examples:
        for key in ("question_excerpt","reference_answer","selected_answer","interpretation","evidence"):
            if key not in row:
                raise ValueError("Incomplete validation example")
        if len(row["question_excerpt"])>450 or len(row["interpretation"])>650:
            raise ValueError("Example too long for concise report")
        evidence.paragraphs([{"text":row["interpretation"],"evidence":row["evidence"]}])
    repro=analysis["reproduction"]
    if not repro.get("python") or not repro.get("commands") or not repro.get("evidence"):
        raise ValueError("Missing reproduction evidence")
    if len(repro["commands"])>5:
        raise ValueError("Keep report commands to at most five; link the complete README")
    if any(key not in evidence.records for key in repro["evidence"]):
        raise ValueError("Unknown reproduction evidence ID")
    if studies["E_FUSION"].get("menu_readouts")!=48 or studies["E_FUSION"].get("status")!="complete":
        raise ValueError("Expected completed fixed 48-entry fusion screen")
    component_rows,kinds=components(final["config"],evidence,spec["final"]["config"])
    strong=[comparison(studies["E_CROSS"],a,b,label,"E_CROSS") for a,b,label in (
        ("A_base","A_D1D2","Group network + candidate quantities"),
        ("Q_base","Q_D1D2","Quantity/PLS tree + candidate quantities"),
        ("C_base","C_D1D2","Condition/PLS tree + candidate quantities"))]
    strong.extend([
        comparison(studies["E_REG"],"epoch14","epoch40","Longer group training","E_REG"),
        comparison(studies["E_REG"],"epoch40","epoch40_ema","EMA at the same 40-epoch budget","E_REG"),
        comparison(studies["E_SUBSET"],"original","drop_one_half","Matched candidate-subset augmentation","E_SUBSET"),
        comparison(studies["E_SEQUENCE"],"head64","headtail_roles","Question suffix + numeric-role CNN","E_SEQUENCE")])
    controls=[comparison(studies[e],a,b,label,e) for e,a,b,label in (
        ("E_PROCESS","control_allq","coverage03_allq","Reference coverage, lambda 0.3"),
        ("E_TOKEN","control_allq","token_aux01","Token-membership auxiliary"),
        ("E_CORRUPT","digit_text_control","reference_corruption01","Pure-text reference corruption"),
        ("E_ATTENTION","token_attention","token_set_attention","Candidate-set attention"),
        ("E_GRAPH","node_mlp","equation_gcn","Equation-graph message passing"),
        ("E_CALIB","global_fusion","pattern_shrink_fusion","Support-pattern vs global calibration"),
        ("E_SEQUENCE","headtail","headtail_format","Formatting normalization"))]
    return spec,evidence,final,analysis,studies,component_rows,kinds,strong,controls


def build(inputs,output):
    spec,evidence,final,analysis,studies,component_rows,kinds,strong,controls=load_inputs(inputs)
    output=Path(output).resolve()
    if output==ROOT/"report" or (output/"report.pdf").exists():
        raise FileExistsError("Use a new review directory; never overwrite the historical report")
    output.mkdir(parents=True,exist_ok=True)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom",fontName="Helvetica",fontSize=10,leading=13.7,spaceAfter=7,textColor=colors.HexColor("#233142")))
    styles.add(ParagraphStyle(name="SmallCustom",fontName="Helvetica",fontSize=8.1,leading=10.5,spaceAfter=5))
    styles.add(ParagraphStyle(name="CaptionCustom",fontName="Helvetica-Oblique",fontSize=8.2,leading=10.7,spaceAfter=7,textColor=colors.HexColor("#4d6175")))
    styles.add(ParagraphStyle(name="CodeCustom",fontName="Courier",fontSize=8.1,leading=10.8,spaceAfter=7))
    styles["Title"].fontName="Helvetica-Bold";styles["Title"].fontSize=23;styles["Title"].leading=28
    for name in ("Heading1","Heading2"):
        styles[name].textColor=colors.HexColor("#123d59")
    story,source,claims=[],[],[]
    def p(text,style="BodyCustom"):
        text=text.replace("\u2011","-").replace("\u2013","-").replace("\u2014","-")
        story.append(Paragraph(text,styles[style]));source.append(text)
        ids=re.findall(r"\bE_[A-Z0-9_]+\b",text)
        if any(x not in evidence.records for x in ids):
            raise ValueError("Unknown source in manuscript")
        claims.append({"id":f"C{len(claims)+1:03d}","text_sha256":hashlib.sha256(text.encode()).hexdigest(),
                       "evidence":list(dict.fromkeys(ids)),"human_verified":False})
    def h(text):p(text,"Heading1")
    def sub(text):p(text,"Heading2")
    def page():story.append(PageBreak())
    def paragraphs(key):
        for text in evidence.paragraphs(analysis[key]):p(text)
    def table(headers,rows,widths):
        content=[[Paragraph(escape(str(x)),styles["SmallCustom"]) for x in headers]]
        content += [[Paragraph(escape(str(x)),styles["SmallCustom"]) for x in row] for row in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8f0f5")),("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#527287")),("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#a7b5c0")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        payload=json.dumps({"headers":headers,"rows":rows},ensure_ascii=False)
        story.extend((t,Spacer(1,8)));source.append(payload)
        ids=list(dict.fromkeys(re.findall(r"\bE_[A-Z0-9_]+\b",payload)))
        if not ids:
            ids=list(dict.fromkeys(["E_GUIDE","E_README",spec["analysis"]["evidence"]]+[r["evidence"] for r in spec["final"].values()]))
        claims.append({"id":f"C{len(claims)+1:03d}","kind":"table",
            "text_sha256":hashlib.sha256(payload.encode()).hexdigest(),"evidence":ids,"human_verified":False})
    final_cite=" ["+", ".join(dict.fromkeys(ref["evidence"] for ref in spec["final"].values()))+"]"
    analysis_cite=" ["+spec["analysis"]["evidence"]+"]"

    p("Selecting Reliable LLM Reasoning","Title")
    p("Candidate comparison under a compact inference budget","Heading2")
    p("DASE7506 Mini-project 2 | October 2026","SmallCustom")
    identity=spec.get("student_identity")
    p("<b>LOCAL REVIEW DRAFT - NOT FOR SUBMISSION.</b> "+("Student ID: "+escape(str(identity))+". " if identity else "Student identity is pending. ")+"Human scientific review and final approval are pending.","SmallCustom")
    h("1. Research question and principal result")
    p("Which of four supplied reasoning traces should be trusted when their final answers disagree? This project combines semantic representations with quantities, intermediate calculations and evidence from competing answers, then chooses one supplied candidate. The official target is the parsed final number, not a certified proof. [E_GUIDE, E_README]")
    p("The selected method is <b>"+escape(final["name"])+"</b>. The table reports its validation results and final test evaluation after freezing the implementation and checkpoint."+final_cite)
    table(["Evaluation","Correct / total and accuracy"],[
        ["Validation, all questions",metric_text(final["validation_full"])],
        ["Validation, answer disagreements",metric_text(final["validation_hard"])],
        ["Final test",metric_text(final["test"])]],[265,251])
    p("The main experimental finding concerns candidate comparison. Adding shared/intermediate-quantity features improves both structured-tree variants in both training partitions. Candidate-subset augmentation also improves its matched group-network control in both repeats. These local comparisons provide a clearer design signal than model size alone; graph and candidate-attention controls do not improve matched hard-answer accuracy. [E_CROSS, E_SUBSET, E_GRAPH, E_ATTENTION]")
    p("The final method is evaluated after a fixed fusion screen and a separately recorded selection decision. Question-level training partitions, reused validation evidence and final frozen testing answer different questions about reliability. [E_PROTOCOL, E_FUSION]"+final_cite)
    page()

    h("2. Data, evaluation boundaries and search history")
    table(["Split","Questions","Role"],[
        ["Training","4,755","Fit models, transforms and training-only augmentation"],
        ["Validation","798 (355 disagreements)","Development and the recorded final selection check"],
        ["Test","467","Final frozen evaluation; all groups contain answer disagreement"]],[90,160,266])
    p("Each question has four candidate texts and a fixed 1,024-dimensional representation per candidate. Every group includes a correct parsed answer. IDs align records; they are not predictive inputs. The supplied parser, scorer and benchmark remain unchanged. External data, additional pretrained weights and new generation during evaluation are excluded. [E_GUIDE, E_README]")
    sub("Prospective repeated-fold comparison")
    p("The renewed search uses seeds 20261027 and 20261028, each assigning all training questions to three folds while keeping a question's four candidates together. Learned transforms and statistics fit inside the outer-training boundary. Where early stopping is used, an inner split selects the epoch and a fresh outer refit uses it. Fixed-epoch and checkpoint-readout studies retain their registered recipes. [E_PROTOCOL]")
    p("There are 2,114 disagreement questions in training. Repeating their fold assignment tests sensitivity to the fitting partition; it does not create 4,228 independent observations. OOF estimates remain conditional on model and fusion search. The tables show both repeats rather than treating a favorable repeat as the result. [E_PROTOCOL, E_FUSION]")
    sub("What was already known")
    paragraphs("history")
    p("The new protocol preserves earlier checkpoints and unsuccessful gates. Its later selection stage is prospective relative to the new validation outcomes, while earlier validation reuse remains part of the study history. Final test performance cannot choose a checkpoint or rollback. [E_PROTOCOL]"+final_cite)
    page()

    h("3. Selected method: compare candidates, then answers")
    table(["Component","Weight","Members","Role"],[row[:4] for row in component_rows],[127,47,48,294])
    p("Weights and member counts are read from the selected configuration and hash-matched manifests. Each member computes fresh input scores and applies its fitting-side normalization. Inference reads neither references nor a prediction lookup table."+final_cite+" ["+", ".join(row[4] for row in component_rows)+"]","CaptionCustom")
    p("Let s_j be a weighted candidate score and G(a) the candidates with parsed answer a. The decision compares answer-group means: S(a)=sum(s_j, j in G(a))/|G(a)|, and returns a candidate in the winning group. Duplicate numeric answers therefore do not contribute independent votes to the score. Invalid parses follow the fixed selection implementation. [E_CORE_METHOD, E_SEQUENCE_METHOD]")
    if "group" in kinds:
        p("The group scorer pools candidates sharing an answer and learns correct-versus-incorrect group comparisons. Its training objective is pairwise logistic loss, mean log(1+exp(s_wrong-s_correct)); group cross-entropy is an evaluation quantity, not a description of this training loss. The recipe uses a 128-unit hidden layer, dropout and regularization. [E_CORE_METHOD, E_SUBSET_P]")
        p("The selected group branch averages five fixed seeds trained for 40 epochs with Gaussian input noise of standard deviation 0.05. EMA uses decay 0.99 after epoch 5. These settings were fixed before final validation. [E_CORE_METHOD]", "SmallCustom")
        if any(row[0]=="drop_one_half" for row in component_rows):
            p("Subset training removes one eligible candidate with probability 0.5 while retaining correct and incorrect valid answers. Groups, support and pooled features are recomputed. Both arms match initialization, shuffle/dropout draws and optimizer updates; evaluation uses the original four-candidate view. [E_SUBSET_P]")
    if "kernel" in kinds:
        p("The kernel branch standardizes supplied embeddings, fits PCA64 and learns an RBF support-vector comparison on transformed candidate differences. Its antisymmetric margins against other valid candidates provide a nonlinear semantic signal distinct from the pooled group scorer. [E_CORE_METHOD]")
    if "interaction" in kinds:
        p("The structured branch combines fitting-only PLS8 with text, quantity and cross-candidate features in boosted trees. Added features ask whether derivations contain one another and whether distinct answer groups share derived quantities. These are input-derived clues, not gold reasoning annotations. [E_CORE_METHOD, E_CROSS_P]")
    if "pair" in kinds:
        p("The conditional pair branch fits a training-only PCA64 representation and a 64-unit nonlinear comparator. It scores a candidate against alternatives with an antisymmetric difference, so reversing a pair reverses its comparison score. Its fixed 27-epoch training recipe is carried into the full fit. [E_CORE_METHOD]")
    if kinds.intersection(("role_cnn","aux_cnn","transformer","digit_cnn")):
        p("The sequence branch preserves question context and learns from the trace. Numeric-role tokens distinguish question quantities, derived values and self/rival answers. Its main loss is group cross-entropy: log sum exp(S(a)) over valid groups minus the correct-group score. Parameters start from random initialization; exported NumPy inference reproduces the model. [E_SEQUENCE_P, E_SEQUENCE_METHOD]")
    if kinds.intersection(("nonlinear","residual","trees")):
        p("The retained historical configuration combines pairwise and listwise embedding scores with structured-text trees. Its layer sizes, scales and weights remain those in the selected configuration; the new screens do not retroactively change that recipe."+final_cite)
    page()

    h("4. Matched evidence for useful design choices")
    table(["Comparison","Control r0/r1","Variant r0/r1","Change","Mean CE change","Evidence"],strong,[153,75,75,63,76,74])
    p("Counts use the same 2,114 hard training questions. Positive count changes and negative CE changes favor the variant. Each row is the indicated matched comparison; across rows, architecture, optimizer and fixed seed count can differ. [E_CROSS, E_REG, E_SUBSET, E_SEQUENCE]","CaptionCustom")
    p("Cross-candidate features help the quantity/PLS and condition/PLS trees in both repeats and reduce group CE. Their effect on the group network changes sign. This supports a representation-dependent benefit rather than a universal gain from adding features. [E_CROSS]")
    p("Subset augmentation changes the support pattern the verifier must interpret while preserving the task. Its matched update budget separates the new view from merely training longer. Both repeats improve, and this scorer enters the final fusion screen. [E_SUBSET, E_FUSION]")
    p("Longer training alone is less consistent than the associated EMA readout. EMA is compared at the same 40-epoch budget; a separate noisy trajectory uses the registered Gaussian input noise. Neither readout chooses an epoch from heldout answers. [E_REG_P, E_REG]")
    p("Question-suffix retention plus numeric-role tokens improve the CNN; suffix retention alone does not explain that gain. Preserving distinctions between quantities and competing answers matters in this screen. It does not test every tokenizer or truncation scheme. [E_SEQUENCE]")
    page()

    h("5. What additional mechanisms did and did not add")
    table(["Comparison","Control r0/r1","Variant r0/r1","Change","Mean CE change","Evidence"],controls,[153,75,75,63,76,74])
    p("These mechanisms were implemented and measured. C4 is compared with its own digit-preserving pure-text control, not the stronger joint CNN. [E_PROCESS, E_TOKEN, E_CORRUPT, E_ATTENTION, E_GRAPH, E_CALIB, E_SEQUENCE]","CaptionCustom")
    p("Reference coverage and numeric-token membership are weak supervision: appearing in a reference does not certify a reasoning step. Empty usable evidence is masked. The controls retain final-answer labels and fitting boundaries, and token-level auxiliary training does not improve its matched control on average. [E_PROCESS_P, E_TOKEN_P, E_TOKEN]")
    p("Reference corruption retains digits and changes one initially valid arithmetic RHS by +1. Its mean gain is positive but has opposite signs across repeats; standalone accuracy remains substantially below the joint models. Annotation style and repeated-value consistency may explain some effect. Other corruption mechanisms were not evaluated. [E_CORRUPT_P, E_CORRUPT]")
    p("Candidate-set attention acts after token pooling. The graph control compares message passing against a node MLP with the same node information. Neither improves hard-answer accuracy; the graph's small mean CE decrease does not change that selection result. This does not rule out richer cross-trace attention or different graph representations. [E_ATTENTION_P, E_GRAPH_P, E_GRAPH]")
    p("Support-pattern calibration learns its weights inside training-only boundaries and has mixed repeat effects. Formatting normalization tests deterministic canonicalization, not a full stochastic augmentation family. These distinctions keep conclusions tied to the mechanisms actually tested. [E_CALIB_P, E_SEQUENCE_P]")
    page()

    h("6. Fixed fusion screen and recorded final decision")
    fusion=studies["E_FUSION"]
    names=[x["name"] if isinstance(x,dict) else x for x in fusion["provisional_first_two"]]
    shown=list(dict.fromkeys(["F","G"]+names));fusion_rows=[]
    for name in shown:
        row=next(x for x in fusion["results"] if x["name"]==name)
        fusion_rows.append([name,*[str(r["hard_correct"]) for r in row["repeats"]],f'{row["mean_group_ce"]:.4f}'])
    table(["Training-screen candidate","Repeat 0","Repeat 1","Mean group CE"],fusion_rows,[300,65,65,86])
    p("The menu was fixed at 48 combinations before reading its scores. It combines registered families and limited support weights rather than continuously optimizing a weight vector on the displayed OOF labels. The shortlist is nevertheless selected from these results and remains search-conditional. [E_FUSION, E_FUSION_P]")
    table(["Decision candidate","Hard validation","Resource status","Decision"],
        [[r["name"],r["validation_hard"],r["resource_status"],r["decision"]] for r in analysis["selection_rows"]],[245,90,90,91])
    paragraphs("selection_commentary")
    p("The deployment index adds mean difficult-training OOF correct count to difficult-validation correct count. It is a prespecified engineering ranking, not an accuracy estimate or calibrated probability; the two components retain their own denominators and different selection histories."+final_cite,"CaptionCustom")
    p("The fusion intervals resample fixed searched predictions at question level, keeping each question's repeated outcomes together. They exclude fitting and menu-selection uncertainty and are not confirmatory tests. Final testing cannot revise the selection gate. [E_FUSION, E_PROTOCOL]")
    page()

    h("7. Validation reliability and failure analysis")
    table(["Validation comparison","Correct / denominator"],[[r["name"],metric_text(checked_metric(r["metric"]))] for r in analysis["baselines"]],[300,216])
    table(["Diagnostic stratum","Correct / denominator"],[[r["name"],metric_text(checked_metric(r["metric"]))] for r in analysis["strata"]],[300,216])
    p("These diagnostics use the selected checkpoint's validation predictions. Descriptive stratification does not make the reused validation bank independent."+analysis_cite,"CaptionCustom")
    paragraphs("reliability")
    for i,row in enumerate(analysis["examples"],1):
        sub(f"Illustrative validation case {i}")
        p(escape(row["question_excerpt"]),"SmallCustom")
        p("Reference: "+escape(str(row["reference_answer"]))+"; selected: "+escape(str(row["selected_answer"]))+". "+escape(row["interpretation"])+" ["+", ".join(row["evidence"])+"]","SmallCustom")
    paragraphs("failure")
    page()

    h("8. Runtime, search cost and reproduction")
    resources=final["resources"]
    table(["Resource","Measured final-test value"],[
        ["Fresh prediction process",f'{resources["prediction_fresh_process_seconds"]:.3f} s'],
        ["Fresh scoring process",f'{resources["scoring_fresh_process_seconds"]:.3f} s'],
        ["Complete wall time",f'{resources["total_fresh_process_seconds"]:.3f} s'],
        ["Uncompressed inference assets (budget basis)",f'{resources["asset_uncompressed_bytes"]:,} ({resources["asset_uncompressed_bytes"]/2**20:.3f} MiB)'],
        ["Stored inference-asset files",f'{resources["asset_file_bytes"]:,} ({resources["asset_file_bytes"]/2**20:.3f} MiB)'],
        ["Local CPU",resources["hardware"]],["Thread limit","4 total"]],[255,261])
    p("Timing includes startup, imports, input/asset loading, preprocessing, every member and independent scoring. The limits are 30 seconds and 16 MiB of uncompressed additional inference assets, including arrays, vocabularies, statistics and configuration. Stored file size is reported separately, including archive overhead. Eligibility is determined on the reference Xeon Platinum 8457C; local timing does not certify that machine. [E_GUIDE]"+final_cite,"CaptionCustom")
    table(["Recorded workload","Learner fits","Elapsed / interpretation"],
        [[r["stage"],str(r["model_fits"]),r["cost_text"]] for r in analysis["search_cost_rows"]],[230,80,206])
    p("Learner fits, transforms, checkpoint readouts and fixed fusion evaluations are distinct costs. Parallel wall times must not be summed as calendar time. Useful null controls and implementation repairs remain in the experiment archive."+analysis_cite)
    sub("Reproduce the frozen result")
    repro=analysis["reproduction"]
    p("Environment: "+escape(repro["python"])+". Run from code/; full training instructions retain exact versions, seeds and assets. ["+", ".join(repro["evidence"])+"]")
    p("<br/>".join(escape(command) for command in repro["commands"]),"CodeCustom")
    page()

    h("9. Interpretation, reuse and accountability")
    paragraphs("limits")
    p("The useful positive signal is comparative: quantities and support patterns gain meaning relative to competing solutions. Matched tree and subset controls support that mechanism on the supplied bank. Weak process labels and larger relational architectures show that adding structure does not automatically improve selection. Future experiments should change a mechanism while preserving a new evaluation boundary. [E_CROSS, E_SUBSET, E_PROCESS, E_TOKEN, E_ATTENTION, E_GRAPH]")
    sub("Assistance and reused material")
    p("Codex provided substantive assistance with implementation, experimental execution, debugging, analysis, documentation and this report draft. The user supplied optimization advice produced by Claude; subsequent local comparisons tested that advice. Scientific Agent Skills supported evidence organization and reporting practice. No student contribution percentage or unverified human authorship claim is assigned. Student review and final approval remain pending. [E_SKILLS]")
    p("The task, GSM8K-derived questions/references, fixed Qwen3-0.6B representations and official parser/evaluator come from the supplied package; their attribution files are retained. Training uses the recorded scientific Python libraries and CPU PyTorch where applicable. Exported arrays support inference without training dependencies or additional pretrained weights. [E_GUIDE, E_README, E_CORE_METHOD, E_SEQUENCE_METHOD]")
    sub("Evidence and references")
    p("evidence_manifest.json maps each evidence ID to a local source and SHA256 hash. claims.json binds drafted paragraphs to evidence IDs. Programmatic consistency checks are distinct from human verification and submission approval. Visual review remains pending until the generated pages are inspected.","SmallCustom")
    p("[E_GUIDE, E_README] DASE7506 Mini-project 2 guide and running instructions. https://github.com/xudongwu-0/DASE7506-MP2. Protocol dase7506-mp2-open-hard-verifier-v5.","SmallCustom")
    p("[E_PROTOCOL] Renewed-search protocol; [E_CROSS, E_REG, E_SUBSET, E_SEQUENCE] matched controls; [E_PROCESS, E_TOKEN, E_CORRUPT, E_ATTENTION, E_GRAPH, E_CALIB] additional mechanisms; [E_FUSION] fixed fusion menu. Full local paths and hashes appear in the evidence manifest.","SmallCustom")
    p("[E_SKILLS] Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065. Metadata checked against the current record (v2).","SmallCustom")
    pages=[]
    def footer(canvas,doc):
        pages.append(doc.page);canvas.saveState();canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#607080"))
        canvas.drawString(48,29,"DASE7506 | Selecting Reliable LLM Reasoning | Local review draft")
        canvas.drawRightString(565,29,str(doc.page));canvas.restoreState()
    temporary=output/"report.unverified.pdf"
    document=SimpleDocTemplate(str(temporary),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=47,
        title="Selecting Reliable LLM Reasoning",author="")
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    if max(pages)>10:
        raise RuntimeError("Report exceeds ten pages; unverified PDF retained for layout repair")
    temporary.replace(output/"report.pdf")
    (output/"report_source.txt").write_text("\n\n".join(source)+"\n",encoding="utf-8")
    (output/"evidence_manifest.json").write_text(json.dumps({"sources":list(evidence.records.values()),"human_review":"pending","submission_ready":False},indent=2)+"\n",encoding="utf-8")
    (output/"claims.json").write_text(json.dumps(claims,indent=2)+"\n",encoding="utf-8")
    audit={"status":"programmatic_checks_passed","pages":max(pages),"target_pages":9,"maximum_pages":10,
        "inputs_sha256":sha(inputs),"builder_sha256":sha(__file__),"pdf_sha256":sha(output/"report.pdf"),
        "visual_review":"pending","human_scientific_review":"pending","student_identity_provided":bool(identity),
        "submission_ready":False,"created_utc":datetime.now(timezone.utc).isoformat()}
    (output/"report_build_audit.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only",action="store_true")
    group.add_argument("--build",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.validate_only:
        data=load_inputs(args.inputs)
        print(json.dumps({"status":"inputs_valid","selected_name":data[2]["name"],"pdf_created":False,"human_review":"pending"}))
    else:
        if args.output is None:
            parser.error("--build requires --output")
        build(args.inputs,args.output)


if __name__=="__main__":
    main()
