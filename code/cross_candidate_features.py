"""Reference-free directed numeric relations between four candidate solutions.

Numeric mentions are fallible evidence, not proof of equivalent derivations.
Question quantities and explicit final-answer markers are excluded from the
derived-number pool. Candidate support is counted by distinct parsed-answer
groups, so repeated candidates cannot multiply a group's vote. No IDs, labels,
fitted vocabulary, candidate-position feature or outside data are consulted.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer
from quantitative_features import _equations

MAX_TEXT = 24000
MAX_MENTIONS = 128
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d[\d,]*(?:\.\d+)?)?")
_FRAC = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([-+]?\d[\d,.]*)\}\s*\{([-+]?\d[\d,.]*)\}")
_BOX = re.compile(r"\\boxed\s*\{")
_FINAL = re.compile(r"(?im)(?:^|\n)\s*(?:#{1,3}\s*)?(?:\*\*)?(?:final\s+answer|answer)\s*(?:\*\*)?\s*:")
_WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand half quarter dozen twice thrice".split(), list(range(20))+[20,30,40,50,60,70,80,90,100,1000,Fraction(1,2),Fraction(1,4),12,2,3]))
_WORD_NUMBER = re.compile(r"\b(?:"+"|".join(_WORDS)+r")\b", re.I)
TARGETS = {
    "more": re.compile(r"\b(?:more|less|fewer|extra|increase|decrease|difference|saving|save)\b"),
    "left": re.compile(r"\b(?:left|remain\w*|rest|leftover)\b"),
    "total": re.compile(r"\b(?:total|altogether|combined|together|in all)\b"),
    "year": re.compile(r"\b(?:year|born|birth)\b"),
    "age": re.compile(r"\b(?:age|old|older|younger)\b"),
    "percent": re.compile(r"\b(?:percent|percentage|fraction|proportion)\b|%"),
}
OPERATIONS = ("none", "Add", "Sub", "Mult", "Div", "Pow", "other")
GROUP_NAMES = {
    "D1": ["rival_answer_in_my_derivation", "my_answer_in_rival_derivation",
           "directed_derivation_advantage", "directed_early_stop_warning",
           "containment_asymmetry", "reciprocal_containment",
           "rival_answer_first_position", "rival_answer_last_position",
           "steps_after_rival_answer", "rival_steps_after_my_answer",
           "different_answer_group_count", "derived_number_count"],
    "D2": ["shared_intermediate_count", "shared_intermediate_coverage",
           "shared_intermediate_missing", "my_intermediate_group_support_mean",
           "my_intermediate_multigroup_fraction", "my_intermediate_group_support_max",
           "my_intermediate_exclusive_fraction", "my_last_intermediate_group_support",
           "shared_intermediate_tail_fraction", "shared_prefix_fraction",
           "other_groups_consensus_coverage", "other_groups_consensus_count"],
    "D3": ["target_"+t for t in TARGETS]+["last_operation_"+op for op in OPERATIONS]+
          ["target_"+t+"__last_"+op for t in TARGETS for op in OPERATIONS],
}
FEATURE_NAMES = [name for group in GROUP_NAMES.values() for name in group]
FEATURE_GROUPS = {}
_offset = 0
for _name, _names in GROUP_NAMES.items():
    FEATURE_GROUPS[_name] = list(range(_offset, _offset+len(_names)))
    _offset += len(_names)


def _fraction(value):
    try:
        text = str(value).replace(",", "").replace(" ", "")
        if len(text) > 80:
            return None
        # Fraction permits an integer denominator; decimal ratios are parsed explicitly.
        if "/" in text:
            left, right = text.split("/", 1)
            result = Fraction(left)/Fraction(right)
        else:
            result = Fraction(text)
        return result if abs(result) <= 10**12 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _equal(a, b):
    return a is not None and b is not None and (a == b or abs(float(a-b)) <= 1e-6*max(1.,abs(float(a)),abs(float(b))))


def _contains(values, value):
    return value is not None and any(_equal(value, v) for v in values)


def _unique(values):
    result = []
    for value in values:
        if not _contains(result, value):
            result.append(value)
    return result


def _mean(values):
    return math.fsum(sorted(values))/len(values) if values else 0.


def reasoning_body(text):
    """Remove labelled final blocks and boxed spans, preserving preceding equations."""
    text = str(text)[:MAX_TEXT]
    if "####" in text:
        text = text.split("####", 1)[0]
    marker = _FINAL.search(text)
    if marker:
        text = text[:marker.start()]
    # Remove every balanced boxed span; a box alone is final-answer evidence,
    # not an intermediate derivation. This is independent of the parsed answer.
    spans = []
    for match in _BOX.finditer(text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] == "{": depth += 1
            elif text[end] == "}": depth -= 1
            end += 1
        spans.append((match.start(), end))
    for start, end in reversed(spans):
        text = text[:start]+" "+text[end:]
    # List numbering is formatting, not a derived mathematical quantity.
    return re.sub(r"(?im)^\s*(?:step\s*)?\d+\s*[.):]\s*", " ", text)


def numeric_mentions(text, words=False):
    text = _FRAC.sub(lambda m: m.group(1)+"/"+m.group(2), str(text)[:MAX_TEXT])
    mentions = [(m.start(),_fraction(m.group())) for m in _NUMBER.finditer(text)]
    if words:
        mentions += [(m.start(),Fraction(_WORDS[m.group().lower()])) for m in _WORD_NUMBER.finditer(text)]
    return [v for _,v in sorted(mentions,key=lambda p:p[0]) if v is not None][:MAX_MENTIONS]


def _positions(sequence, value):
    return [(i+1)/max(1,len(sequence)) for i,v in enumerate(sequence) if _equal(v,value)]


def _prefix(a, b):
    n = 0
    for left,right in zip(a,b):
        if not _equal(left,right): break
        n += 1
    return n/max(1,min(len(a),len(b)))


def _target(text):
    text = str(text).lower()[:MAX_TEXT]
    starts = list(re.finditer(r"\b(?:how|what|which)\b",text))
    return text[starts[-1].start():] if starts else text[-400:]


def _last_operation(body):
    equations = _equations(body)
    if not equations:
        return "none"
    expression = equations[-1][0][3]
    try:
        root = ast.parse(expression,mode="eval").body
        if isinstance(root,ast.BinOp):
            op = type(root.op).__name__
            return op if op in OPERATIONS else "other"
    except (ValueError,SyntaxError):
        pass
    return "none"


def extract_features(bank):
    result = np.zeros((len(bank),4,len(FEATURE_NAMES)),dtype=np.float32)
    for row_i,row in enumerate(bank):
        if len(row["candidates"]) != 4:
            raise ValueError("Expected four candidates")
        question_values = numeric_mentions(row["question"],words=True)
        texts = [str(c["text"]) for c in row["candidates"]]
        keys = [parse_answer(t) for t in texts]
        answers = [_fraction(key) if key is not None else None for key in keys]
        bodies = [reasoning_body(t) for t in texts]
        derived = [[v for v in numeric_mentions(t) if not _contains(question_values,v)] for t in bodies]
        intermediates = [_unique([v for v in seq if not _equal(v,answer)]) for seq,answer in zip(derived,answers)]
        groups = {key:[k for k in range(4) if keys[k]==key] for key in sorted({k for k in keys if k is not None},key=lambda x:Fraction(x))}
        group_values = {key:_unique(sorted([v for k in ids for v in intermediates[k]])) for key,ids in groups.items()}
        values = _unique(sorted([v for seq in group_values.values() for v in seq]))
        support = [sum(_contains(seq,v) for seq in group_values.values()) for v in values]
        shared = [v for v,s in zip(values,support) if s >= 2]
        target = [float(rx.search(_target(row["question"])) is not None) for rx in TARGETS.values()]
        for j in range(4):
            rivals = [(key,ids) for key,ids in groups.items() if key != keys[j]]
            outgoing=[]; incoming=[]; first=[];last=[]; after=[];rival_after=[]
            for key,ids in rivals:
                pos = _positions(derived[j],_fraction(key))
                other_pos = [_positions(derived[k],answers[j]) for k in ids]
                outgoing.append(float(bool(pos)))
                incoming.append(_mean([float(bool(p)) for p in other_pos]))
                first.append(min(pos) if pos else 0.)
                last.append(max(pos) if pos else 0.)
                after.append(1-max(pos) if pos else 0.)
                rival_after.append(_mean([1-max(p) if p else 0. for p in other_pos]))
            d1 = [_mean(outgoing),_mean(incoming),
                  _mean([float(o>0 and i==0) for o,i in zip(outgoing,incoming)]),
                  _mean([float(i>0 and o==0) for o,i in zip(outgoing,incoming)]),
                  _mean([o-i for o,i in zip(outgoing,incoming)]),
                  _mean([o*i for o,i in zip(outgoing,incoming)]),
                  _mean(first),_mean(last),_mean(after),_mean(rival_after),
                  len(rivals)/3,math.log1p(len(_unique(derived[j])))]
            mine = intermediates[j]
            mine_support = [sum(_contains(seq,v) for seq in group_values.values()) for v in mine]
            coverage = sum(_contains(mine,v) for v in shared)/max(1,len(shared))
            other_consensus = [v for v in values if sum(_contains(group_values[key],v) for key,_ in rivals)>=2]
            other_coverage = sum(_contains(mine,v) for v in other_consensus)/max(1,len(other_consensus))
            prefix = [_mean([_prefix(mine,intermediates[k]) for k in ids]) for _,ids in rivals]
            tail = mine[-max(1,(len(mine)+2)//3):]
            d2 = [math.log1p(len(shared)),coverage, float(bool(shared))*(1-coverage),
                  _mean([s/max(1,len(groups)) for s in mine_support]),
                  _mean([float(s>=2) for s in mine_support]),
                  max(mine_support,default=0)/max(1,len(groups)),
                  _mean([float(s<=1) for s in mine_support]),
                  (mine_support[-1]/max(1,len(groups))) if mine_support else 0.,
                  sum(_contains(shared,v) for v in tail)/max(1,len(tail)),
                  _mean(prefix),other_coverage,math.log1p(len(other_consensus))]
            operation = _last_operation(bodies[j])
            ops = [float(operation==op) for op in OPERATIONS]
            result[row_i,j] = d1+d2+target+ops+[t*o for t in target for o in ops]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite cross-candidate feature")
    return result


def self_checks():
    import copy
    rows = [{"id":"arbitrary","question":"A shop has 8 and 2 items. How many more items are there?","candidates":[
        {"index":0,"text":"8 - 2 = 6. 6 * 3 = 18. #### 18"},
        {"index":1,"text":"8 - 2 = 6. #### 6"},
        {"index":2,"text":"8 - 2 = 6. 6 * 3 = 18. #### 18"},
        {"index":3,"text":"8 + 2 = 10. #### 10"}]}]
    x = extract_features(rows);names={name:i for i,name in enumerate(FEATURE_NAMES)}
    assert x[0,0,names["directed_derivation_advantage"]] > x[0,1,names["directed_derivation_advantage"]]
    assert x[0,1,names["directed_early_stop_warning"]] > x[0,0,names["directed_early_stop_warning"]]
    assert _equal(Fraction(1,2),_fraction("0.5000000"))
    assert numeric_mentions(r"A value is \frac{1}{2} and 0.5.") == [Fraction(1,2),Fraction(1,2)]
    assert numeric_mentions(reasoning_body(r"3 * 2 = 6. Final text. #### 999"))[-1] == 6
    assert numeric_mentions(reasoning_body(r"3 * 2 = 6. \boxed{999}"))[-1] == 6
    changed = copy.deepcopy(rows);changed[0]["id"]="unrelated"
    for c in changed[0]["candidates"]:c["index"]=999
    assert np.array_equal(x,extract_features(changed))
    perm=[3,1,0,2];changed[0]["candidates"]=[changed[0]["candidates"][k] for k in perm]
    assert np.array_equal(x[:,perm],extract_features(changed))
    batch=copy.deepcopy(rows)*2
    batch[1]=copy.deepcopy(rows[0]);batch[1]["question"]="A basket holds 8 red objects and 2 blue ones. What is the total?"
    assert np.array_equal(extract_features(batch)[::-1],extract_features(batch[::-1]))
    assert extract_features([]).shape==(0,4,len(FEATURE_NAMES))
    changed = copy.deepcopy(rows)
    changed[0]["candidates"]=[{"text":"No numerical answer."} for _ in range(4)]
    assert np.isfinite(extract_features(changed)).all()
    return {"synthetic_directionality":True,"fraction_decimal_equivalence":True,
            "final_markers_excluded":True,"id_and_candidate_index_independent":True,
            "candidate_permutation_exact":True,"empty_and_invalid_inputs":True,
            "question_permutation_exact":True,
            "feature_count":len(FEATURE_NAMES)}
