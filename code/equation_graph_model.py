"""Bounded equation-dependency inputs and NumPy-only small graph inference.

Edges express numeric occurrence dependencies, not verified semantic reasoning.
The same nodes and features are supplied to the no-edge MLP control and GCN.
"""
from __future__ import annotations
import math
import numpy as np

from common import parse_answer
from quantitative_features import _equations,_quantities,_match,_value
from sequence_model import numeric_inputs,transform_numeric

MAX_QUESTION_NODES=8
MAX_EQUATIONS=32
MAX_NODES=40
MAX_TEXT=24000
FEATURE_NAMES=("question_node","equation_node","signed_log_value","integer_value","negative_value",
    "matches_question_value","matches_my_answer","matches_rival_answer","arithmetic_checked","arithmetic_correct",
    "arithmetic_relative_error","operation_count_log","relative_position","operation_add","operation_subtract",
    "operation_multiply","operation_divide","operation_power","operation_other","question_operand_fraction","prior_result_operand_fraction")


def _number_node(value,is_question,qvalues,my_answer,rival_answers,position):
    out=np.zeros(len(FEATURE_NAMES),dtype=np.float32)
    out[0]=is_question;out[1]=not is_question
    out[2]=float(np.clip(math.copysign(math.log1p(abs(value)),value)/30,-1,1))
    out[3]=float(abs(value-round(value))<=1e-8);out[4]=float(value<0)
    out[5]=float(any(_match(value,q) for q in qvalues))
    out[6]=float(my_answer is not None and _match(value,my_answer))
    out[7]=float(any(_match(value,a) for a in rival_answers))
    out[12]=position
    return out


def graph_inputs(bank,return_coverage=False):
    nodes=np.zeros((len(bank),4,MAX_NODES,len(FEATURE_NAMES)),dtype=np.float32)
    adjacency=np.zeros((len(bank),4,MAX_NODES,MAX_NODES),dtype=np.float32)
    valid=np.zeros((len(bank),4,MAX_NODES),dtype=bool);coverage=[]
    for i,row in enumerate(bank):
        if len(row["candidates"])!=4:raise ValueError("Expected four candidates")
        question=_quantities(str(row["question"])[:MAX_TEXT],words=True,maximum=MAX_QUESTION_NODES+1)
        truncated_q=len(question)>MAX_QUESTION_NODES;question=question[:MAX_QUESTION_NODES]
        parsed=[parse_answer(c["text"]) for c in row["candidates"]]
        answers=[_value(a) if a is not None else None for a in parsed]
        for j,c in enumerate(row["candidates"]):
            # Keep boxed values inside genuine equations; a standalone final
            # box has no '=' and therefore creates no equation node.
            body=str(c["text"])[:MAX_TEXT].split("####",1)[0]
            available=_equations(body);equations=available[:MAX_EQUATIONS]
            rivals=sorted({a for k,a in enumerate(answers) if k!=j and a is not None and parsed[k]!=parsed[j]})
            for q,value in enumerate(question):
                nodes[i,j,q]=_number_node(value,True,question,answers[j],rivals,q/max(1,len(question)-1))
                valid[i,j,q]=True;adjacency[i,j,q,q]=1
            results=[];incoming=0
            for e,(lhs,rhs) in enumerate(equations):
                slot=len(question)+e;value=float(rhs[0]);operands=lhs[1];operations=lhs[2]
                feature=_number_node(value,False,question,answers[j],rivals,e/max(1,len(equations)-1))
                feature[8]=1;feature[9]=float(_match(lhs[0],value))
                feature[10]=min(1.,abs(lhs[0]-value)/max(1.,abs(lhs[0]),abs(value)))
                feature[11]=math.log1p(len(operations))/4
                for k,op in enumerate(("Add","Sub","Mult","Div","Pow")):
                    feature[13+k]=operations.count(op)/max(1,len(operations))
                feature[18]=sum(op not in {"Add","Sub","Mult","Div","Pow"} for op in operations)/max(1,len(operations))
                feature[19]=sum(any(_match(v,q) for q in question) for v in operands)/max(1,len(operands))
                feature[20]=sum(any(_match(v,r) for r in results) for v in operands)/max(1,len(operands))
                nodes[i,j,slot]=feature;valid[i,j,slot]=True;adjacency[i,j,slot,slot]=1
                for q,qvalue in enumerate(question):
                    if any(_match(qvalue,v) for v in operands):adjacency[i,j,slot,q]=1;incoming+=1
                for prior,prior_value in enumerate(results):
                    if any(_match(prior_value,v) for v in operands):adjacency[i,j,slot,len(question)+prior]=1;incoming+=1
                results.append(value)
            adjacency[i,j]/=np.maximum(adjacency[i,j].sum(axis=1,keepdims=True),1)
            coverage.append({"question_nodes":len(question),"equation_nodes":len(equations),"dependency_edges":incoming,
                             "question_numbers_truncated":truncated_q,"equations_truncated":len(available)>MAX_EQUATIONS})
    if not np.isfinite(nodes).all() or not np.isfinite(adjacency).all():raise ValueError("Nonfinite graph inputs")
    return (nodes,adjacency,valid,coverage) if return_coverage else (nodes,adjacency,valid)


def numpy_forward(nodes,adjacency,mask,numeric,asset,batch_candidates=128):
    count=len(nodes);nodes=np.asarray(nodes,dtype=np.float32).reshape(-1,MAX_NODES,len(FEATURE_NAMES))
    adjacency=np.asarray(adjacency,dtype=np.float32).reshape(-1,MAX_NODES,MAX_NODES)
    mask=np.asarray(mask,dtype=bool).reshape(-1,MAX_NODES)
    numeric=np.asarray(numeric,dtype=np.float32).reshape(-1,1099)
    out=np.empty(len(nodes),dtype=np.float32)
    for start in range(0,len(nodes),batch_candidates):
        h=nodes[start:start+batch_candidates];a=adjacency[start:start+batch_candidates];m=mask[start:start+batch_candidates]
        for layer in ("node1","node2"):
            if bool(asset["graph"]):h=a@h
            h=np.maximum(h@asset[layer+"_weight"].T+asset[layer+"_bias"],0)*m[...,None]
        maximum=np.where(m[...,None],h,np.float32(-1e9)).max(axis=1)
        maximum=np.where(m.any(axis=1)[:,None],maximum,np.float32(0))
        mean=(h*m[...,None]).sum(axis=1)/np.maximum(m.sum(axis=1,keepdims=True),1).astype(np.float32)
        pooled=np.concatenate((mean,maximum,numeric[start:start+len(h)]),axis=1)
        hidden=np.tanh(pooled@asset["hidden_weight"].T+asset["hidden_bias"])
        out[start:start+len(h)]=(hidden@asset["output_weight"].T+asset["output_bias"]).ravel()
    return out.reshape(count,4)


def score_candidates(bank,embeddings,asset):
    if not bank:return np.empty((0,4),dtype=np.float32)
    nodes,adjacency,mask=graph_inputs(bank)
    numeric=transform_numeric(numeric_inputs(bank,embeddings),asset)
    return numpy_forward(nodes,adjacency,mask,numeric,asset)


def synthetic_checks():
    row={"id":"ignored","question":"A box has 3 and 4 objects. What is the total?","candidates":[
        {"text":"3 + 4 = 7\n7 * 2 = 14\n#### 14"},{"text":"3 + 4 = 8\n#### 8"},
        {"text":"3 + 4 = \\boxed{7}"},{"text":"No equations. #### 7"}]}
    x,a,v,c=graph_inputs([row],True)
    assert c[0]["equation_nodes"]==2 and c[2]["equation_nodes"]==1
    nq=c[0]["question_nodes"]
    assert a[0,0,nq+1,nq]>0 and not a[0,0,nq,nq+1]
    assert np.allclose(a.sum(axis=-1)[v],1.)
    assert x[0,1,nq,9]==0 and x[0,0,nq,9]==1
    perm=[2,0,3,1]
    changed=dict(row,id="different",candidates=[row["candidates"][i] for i in perm])
    px,pa,pv=graph_inputs([changed])
    assert np.array_equal(x[:,perm],px) and np.array_equal(a[:,perm],pa) and np.array_equal(v[:,perm],pv)
    empty=graph_inputs([]);assert empty[0].shape==(0,4,MAX_NODES,len(FEATURE_NAMES))
    blank=dict(row,question="Objects?",candidates=[{"text":""} for _ in range(4)])
    bx,ba,bv=graph_inputs([blank]);assert not bv.any() and np.isfinite(bx).all()
    return {"earlier_operand_dependency_edge":True,"no_future_edges":True,"row_normalized_self_loops":True,
            "arithmetic_pass_flag":True,"boxed_equation_result_preserved":True,"candidate_permutation_and_ids":True,
            "empty_and_no_nodes":True}
