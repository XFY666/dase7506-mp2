"""Bounded question-only arithmetic expression evidence for candidate answers.

The search is deliberately incomplete: an answer not reached is not proved
impossible, and a reached answer is not proved to answer the word problem.
No labels, references, fitted assets, IDs or candidate positions are used.
"""
from fractions import Fraction
import math

import numpy as np

from common import parse_answer
from cross_candidate_features import numeric_mentions, _fraction, _equal

MAX_QUANTITIES=8
CONSTANTS=(Fraction(1),Fraction(2),Fraction(100))
DEPTH_STATES=(160,320,480)
DEPTH_ATTEMPTS=(800,2000,3200)
MAX_ABS=10**9
MAX_DENOMINATOR=10**6
FEATURE_NAMES=["error_depth_"+str(i) for i in range(4)]+["reachable_depth_"+str(i) for i in range(4)]+[
    "minimum_matching_depth","minimum_matching_operations","minimum_matching_constants","matching_question_coverage",
    "matching_root_add","matching_root_subtract","matching_root_multiply","matching_root_divide",
    "matching_all_quantities_fraction","matching_state_count","minimum_error_any",
    "minimum_error_no_constants","minimum_error_all_quantities","minimum_error_at_most3_operations",
    "question_quantity_count","question_quantities_truncated","retained_state_count","operation_budget_fraction",
    "depth1_retention_capped","depth2_retention_capped","depth3_retention_capped"]


def _rank(s):
    value,mask,depth,operations,constants,root=s
    return (operations,constants,-mask.bit_count(),value.denominator,abs(value.numerator),value.numerator,mask,root)


def search(question):
    values=numeric_mentions(question,words=True)
    truncated=len(values)>MAX_QUANTITIES
    quantities=(values[:4]+values[-4:]) if truncated else values
    leaves=[(v,1<<i,0,0,0,"leaf") for i,v in enumerate(quantities)]
    leaves += [(v,0,0,0,1,"constant") for v in CONSTANTS]
    states=list(leaves);counts=[];capped=[];attempts=[]
    for depth,(limit,budget) in enumerate(zip(DEPTH_STATES,DEPTH_ATTEMPTS),1):
        left=[s for s in states if s[2]==depth-1];right=list(states)
        pool={};n_pairs=len(left)*len(right);used=0
        stride=max(1,n_pairs//2+1)
        while n_pairs and math.gcd(stride,n_pairs)!=1:stride+=1
        # Deterministic spread over pair indices avoids exhausting the budget on
        # the first left operand. No answer or learned score ranks expressions.
        for pair_index in range(n_pairs):
            if used+6>budget:break
            k=(pair_index*stride)%n_pairs
            a=left[k//len(right)];b=right[k%len(right)]
            used+=6
            if a[1]&b[1] or not (a[1]|b[1]):continue
            av,bv=a[0],b[0]
            candidates=[(av+bv,"add"),(av-bv,"subtract"),(bv-av,"subtract"),(av*bv,"multiply")]
            if bv:candidates.append((av/bv,"divide"))
            if av:candidates.append((bv/av,"divide"))
            for value,op in candidates:
                if abs(value)>MAX_ABS or value.denominator>MAX_DENOMINATOR:continue
                state=(value,a[1]|b[1],depth,a[3]+b[3]+1,a[4]+b[4],op)
                key=(value,state[1])
                if key not in pool or _rank(state)<_rank(pool[key]):pool[key]=state
        retained=sorted(pool.values(),key=_rank)[:limit]
        states.extend(retained);counts.append(len(retained));capped.append(len(pool)>limit);attempts.append(used)
    assert len(states)<=1000 and sum(attempts)<=6000
    return states,{"quantities":len(quantities),"truncated":truncated,"states_by_depth":[len(leaves)]+counts,
                   "retention_capped":capped,"arithmetic_attempt_slots_by_depth":attempts,
                   "operation_budget_fraction":sum(attempts)/sum(DEPTH_ATTEMPTS)}


def _min_error(answer,states):
    if answer is None:return 1.
    return min((min(1.,abs(float(answer-s[0]))/max(1.,abs(float(answer)),abs(float(s[0])))) for s in states),default=1.)


def row_features(row):
    states,coverage=search(row["question"])
    # Constant-only leaves are not derivations from the question.
    evidence=[s for s in states if s[1]]
    result=[]
    for candidate in row["candidates"]:
        parsed=parse_answer(candidate["text"]);answer=_fraction(parsed) if parsed is not None else None
        matches=[s for s in evidence if _equal(answer,s[0])]
        depth_sets=[[s for s in evidence if s[2]==d] for d in range(4)]
        values=[_min_error(answer,ss) for ss in depth_sets]
        values += [float(any(_equal(answer,s[0]) for s in ss)) for ss in depth_sets]
        values += [min((s[2] for s in matches),default=4)/3,
                   min((s[3] for s in matches),default=8)/7,
                   min((s[4] for s in matches),default=8)/7,
                   max((s[1].bit_count()/max(1,coverage["quantities"]) for s in matches),default=0.)]
        values += [sum(s[5]==op for s in matches)/max(1,len(matches)) for op in ("add","subtract","multiply","divide")]
        values += [sum(s[1].bit_count()==coverage["quantities"] for s in matches)/max(1,len(matches)),math.log1p(len(matches)),
                   _min_error(answer,evidence),_min_error(answer,[s for s in evidence if not s[4]]),
                   _min_error(answer,[s for s in evidence if s[1].bit_count()==coverage["quantities"]]),
                   _min_error(answer,[s for s in evidence if s[3]<=3]),coverage["quantities"]/MAX_QUANTITIES,
                   float(coverage["truncated"]),math.log1p(len(evidence)),coverage["operation_budget_fraction"],
                   *[float(x) for x in coverage["retention_capped"]]]
        result.append(values)
    return np.asarray(result,dtype=np.float32),coverage


def extract_features(bank,return_coverage=False):
    features=np.zeros((len(bank),4,len(FEATURE_NAMES)),dtype=np.float32);coverage=[]
    for i,row in enumerate(bank):
        if len(row["candidates"])!=4:raise ValueError("Expected four candidates")
        features[i],record=row_features(row);coverage.append(record)
    if not np.isfinite(features).all():raise ValueError("Nonfinite expression feature")
    return (features,coverage) if return_coverage else features


def self_checks():
    row={"id":"ignored","question":"There are 7 red beads and 3 blue beads. How many beads are there?",
         "candidates":[{"text":"#### 10"},{"text":"#### 4"},{"text":"#### 21"},{"text":"#### 7/3"}]}
    x,coverage=extract_features([row],True)
    assert all(x[0,:,FEATURE_NAMES.index("reachable_depth_1")]==1)
    assert sum(coverage[0]["states_by_depth"])<=1000
    assert sum(coverage[0]["arithmetic_attempt_slots_by_depth"])<=6000
    perm=[2,0,3,1]
    changed=dict(row,id="different",candidates=[row["candidates"][j] for j in perm])
    assert np.array_equal(x[:,perm],extract_features([changed]))
    assert extract_features([]).shape==(0,4,len(FEATURE_NAMES))
    no_numbers=dict(row,question="How many objects are there?")
    assert np.isfinite(extract_features([no_numbers])).all()
    return {"simple_arithmetic_reached":True,"state_and_attempt_caps":True,"candidate_permutation_ids":True,
            "empty_and_no_quantity_inputs":True,"features":len(FEATURE_NAMES)}
