"""Local question-condition/reasoning correspondences, without fitted resources.

The signals measure lexical binding, not semantic entailment. An "object" is
only a content-word proxy after a fixed small grammatical/measurement lexicon.
Numeric values are used to align contexts; no raw value, ID or position feature
is emitted. Arithmetic correctness and global number counts belong to earlier
feature families and are deliberately not duplicated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer

MAX_TEXT = 24000
MAX_SENTENCES = 80
MAX_QUESTION_ANCHORS = 32
MAX_CANDIDATE_ANCHORS = 256
WINDOW = 6
_TOKEN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?|[a-z]+(?:'[a-z]+)?|[$%]", re.I)
_SPLIT = re.compile(r"(?<!\d)[.!?;]+|[.!?;]+(?!\d)|\n+")
_WORDS_AS_NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety half quarter dozen twice thrice".split(),
    list(range(20)) + [20,30,40,50,60,70,80,90,.5,.25,12,2,3]))
_STOP = frozenset(("a an the is are was were be been being to of and or in on for as by with it its that this these those "
    "how many much what we they their them he she his her him you your i my our us so then let find calculate answer step "
    "given told text frac boxed times therefore thus hence need want would will can could should does do did has have had "
    "get gets got make makes made take takes took use uses used determine solve solution equals equal number amount").split())
_UNITS = {
    "$":"money", "dollar":"money", "cent":"cent", "penny":"cent", "percent":"percent", "%":"percent",
    "second":"second", "minute":"minute", "hour":"hour", "day":"day", "week":"week", "month":"month", "year":"year",
    "meter":"meter", "metre":"meter", "kilometer":"kilometer", "kilometre":"kilometer", "mile":"mile",
    "foot":"foot", "feet":"foot", "inch":"inch", "yard":"yard", "centimeter":"centimeter",
    "gram":"gram", "kilogram":"kilogram", "pound":"pound", "ounce":"ounce", "liter":"liter", "litre":"liter",
    "gallon":"gallon", "cup":"cup", "mph":"mph"}
_RELATIONS = {
    "increase": frozenset("more higher extra additional increase increased increases greater longer older above added gain gains".split()),
    "decrease": frozenset("less fewer lower decrease decreased decreases shorter younger below lost lose loses".split()),
    "before": frozenset("before earlier initially original originally previous previously first".split()),
    "after": frozenset("after later finally current currently now next".split()),
    "rate": frozenset("each per every rate speed".split()),
    "aggregate": frozenset("total altogether together combined sum all".split()),
    "remaining": frozenset("remaining remain remains left rest leftover".split()),
    "exclude": frozenset("not no never except excluding exclude without neither".split()),
    "include": frozenset("including include includes both".split()),
    "exact": frozenset("only exactly equal same".split()),
    "minimum": frozenset("minimum least atleast".split()),
    "maximum": frozenset("maximum most atmost".split()),
}
_OPPOSITES = {"increase":"decrease", "decrease":"increase", "before":"after", "after":"before",
              "exclude":"include", "include":"exclude", "minimum":"maximum", "maximum":"minimum"}
_REL_WORDS = set().union(*_RELATIONS.values())
_OBJECT_STOP = _REL_WORDS | {"cost","price","pay","paid","earn","earned","sell","sold","buy","bought",
                          "spend","spent","receive","received","give","gave","start","started","end","ended"}

# Definitions are exported with the experiment, so a feature's evidence can be
# inspected without inferring its meaning from a short name.
GROUP_DEFINITIONS = {
    "numeric_context": [
        ("number_context_recall_mean", "Mean best content-word recall around the same numeric value, per question occurrence; missing matches contribute zero."),
        ("number_context_recall_min", "Minimum best same-value context recall across question numeric occurrences."),
        ("number_context_jaccard_mean", "Mean best Jaccard overlap of same-value local content-word windows."),
        ("number_context_supported_fraction", "Fraction of question numeric occurrences with same-value context recall at least one half."),
        ("number_present_context_missing_fraction", "Fraction with a literal match but no overlapping content word; distinguishes copied numbers from contextual use."),
        ("object_binding_recall_mean", "Mean best same-value recall of local content-word object proxies, excluding units and relation words."),
        ("object_binding_supported_fraction", "Fraction of object-bearing numeric anchors with same-value object recall at least one half."),
        ("object_binding_wrong_context_fraction", "Fraction of object-bearing numeric anchors present only in contexts without any of their object words."),
        ("object_binding_specificity_margin", "Mean best correct-number object recall minus best different-number object recall."),
        ("number_unit_context_recall", "Mean local unit-word recall at same-value anchors, restricted to unit-bearing question anchors."),
        ("number_unit_conflict_fraction", "Fraction of unit-bearing question anchors whose matched-number contexts contain units but never a shared unit."),
        ("object_bound_quantity_coverage", "Mean fraction of the question quantities associated with each object word recovered with that word in the candidate."),
    ],
    "condition_relations": [
        ("anchor_relation_preservation", "Mean relation-category recall at same-value candidate anchors with best object overlap."),
        ("anchor_opposite_relation_fraction", "Mean opposite-category rate in those locally aligned numeric contexts; a lexical warning, not contradiction proof."),
        ("anchor_condition_omission_fraction", "Fraction of relation-bearing numeric anchors with none of their relation categories preserved locally."),
        ("anchor_time_preservation", "Mean preservation of before/after categories in context-aligned same-value anchors."),
        ("anchor_time_reversal_fraction", "Mean opposite temporal category occurrence at those aligned anchors."),
        ("anchor_rate_preservation", "Fraction of rate-bearing numeric anchors retaining rate language in matched contexts."),
        ("anchor_exclusion_preservation", "Fraction of exclusion-bearing numeric anchors retaining exclusion language locally."),
        ("sentence_relation_preservation", "Relation recall in each question condition sentence's best content-matching candidate sentence."),
        ("sentence_opposite_relation_fraction", "Opposite relation categories in the best content-matching candidate sentence."),
        ("sentence_negation_mismatch_fraction", "Fraction of aligned condition sentences disagreeing on presence of exclusion language."),
    ],
    "target_correspondence": [
        ("target_final_context_recall", "Target-question content-word recall in the last two reasoning sentences, before the final answer marker."),
        ("target_final_context_jaccard", "Jaccard of target-question content words and final reasoning context."),
        ("target_answer_context_recall", "Best target content-word recall around a numeric occurrence equal to the parsed answer, excluding the answer marker."),
        ("target_answer_object_recall", "Best requested-object recall around a reasoning occurrence of the final numeric answer."),
        ("target_last_equation_context_recall", "Requested content-word recall in the last sentence containing an explicit equals sign."),
        ("target_final_unit_recall", "Requested unit-word recall in the final reasoning context."),
        ("target_answer_unit_conflict", "Whether answer-valued reasoning anchors use units but never any requested unit."),
        ("target_final_relation_recall", "Requested relation-category recall in final reasoning sentences."),
        ("target_final_opposite_relation", "Opposite requested relation categories in final reasoning sentences."),
        ("target_final_vs_body_recall", "Final reasoning context target recall minus whole-body target recall."),
        ("target_answer_object_margin", "Best answer-valued object recall minus best non-answer numeric anchor object recall."),
        ("target_late_context_fraction", "Fraction of target content words appearing in the final quarter of reasoning sentences."),
    ],
    "condition_sentences": [
        ("condition_sentence_joint_match_mean", "Mean best average of content Jaccard and numeric-set Jaccard for each numeric question sentence."),
        ("condition_sentence_joint_match_min", "Worst best word-and-number sentence match among numeric question sentences."),
        ("condition_sentence_supported_fraction", "Fraction of numeric question sentences finding a candidate sentence with at least half word recall and half number recall."),
        ("condition_sentence_number_only_fraction", "Fraction with a sentence sharing numbers but no content words."),
        ("condition_sentence_object_recall", "Mean object-word recall in the best joint word/number aligned sentence."),
        ("condition_sentence_order_consistency", "Fraction of adjacent distinct question condition sentences whose strongest aligned sentence positions are nondecreasing."),
        ("condition_sentence_unmatched_tail_fraction", "Fraction of the last half of numeric conditions lacking a half-recall joint support."),
        ("condition_clause_relation_missing", "Fraction of question sentences with relation cues for which best content match preserves no cue."),
    ],
    "derived_context": [
        ("derived_source_object_recall", "Mean question-source object-word recall in equation sentences that mention a source value and a new value."),
        ("derived_target_object_recall", "Best target-object recall in an equation sentence with a question value and a new value."),
        ("derived_shared_object_fraction", "Fraction of such equation sentences retaining an object shared by two distinct question quantities."),
        ("derived_answer_source_object_recall", "Best source-object recall in an equation sentence containing both a question quantity and the parsed answer."),
        ("derived_final_source_object_recall", "Source-object recall in the last equation sentence with both source and new quantities."),
        ("derived_relation_source_recall", "Mean source-anchor relation recall in equation sentences with source and new quantities."),
    ],
}
FEATURE_NAMES = [name for group in GROUP_DEFINITIONS.values() for name, _ in group]
FEATURE_MEANINGS = {name: meaning for group in GROUP_DEFINITIONS.values() for name, meaning in group}
FEATURE_GROUPS = {}
for _group, _definitions in GROUP_DEFINITIONS.items():
    FEATURE_GROUPS[_group] = [FEATURE_NAMES.index(name) for name, _ in _definitions]
_RELATIVE_SOURCE = ["number_context_recall_mean", "object_binding_recall_mean", "anchor_relation_preservation",
                    "target_answer_object_recall", "condition_sentence_joint_match_mean", "derived_source_object_recall"]
FEATURE_GROUPS["within_question_relative"] = []
for _source in _RELATIVE_SOURCE:
    _name = "relative_"+_source
    FEATURE_GROUPS["within_question_relative"].append(len(FEATURE_NAMES))
    FEATURE_NAMES.append(_name)
    FEATURE_MEANINGS[_name] = "Candidate value minus the symmetric float64 mean of the four values of "+_source+"."


def _mean(values):
    return math.fsum(values)/len(values) if values else 0.


def _recall(a, b):
    return len(a & b)/len(a) if a else 0.


def _jaccard(a, b):
    return len(a & b)/len(a | b) if a or b else 0.


def _word(word):
    word = word.lower().removesuffix("'s")
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3]+"y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
        word = word[:-1]
    return _UNITS.get(word, word)


def _number(token):
    if token.lower() in _WORDS_AS_NUMBERS:
        return float(_WORDS_AS_NUMBERS[token.lower()])
    if not token or not any(char.isdigit() for char in token):
        return None
    try:
        value = float(Fraction(token.replace(",", "")))
        return round(value, 6) if math.isfinite(value) and abs(value) <= 1e12 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _sets(tokens):
    raw = {t.lower().removesuffix("'s") for t in tokens if _number(t) is None}
    rel = {key for key, words in _RELATIONS.items() if raw & words}
    words = {_word(t) for t in raw if t not in _STOP}
    units = words & set(_UNITS.values())
    objects = {_word(t) for t in raw if t not in _STOP and t not in _OBJECT_STOP} - units
    return words, objects, units, rel


@dataclass
class Anchor:
    value: float
    words: set
    objects: set
    units: set
    relations: set
    sentence: int


@dataclass
class Sentence:
    text: str
    words: set
    objects: set
    units: set
    relations: set
    values: set


def _profile(text, question=False):
    text = str(text)
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT//2]+"\n"+text[-MAX_TEXT//2:]
    # Keep content inside common LaTeX text wrappers, and stop the final hash
    # marker from providing a meaningless answer occurrence with no context.
    text = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r" \1 ", text)
    parts = [s.strip() for s in _SPLIT.split(text) if s.strip()]
    if len(parts) > MAX_SENTENCES:
        parts = parts[:MAX_SENTENCES//2]+parts[-MAX_SENTENCES//2:]
    sentences, anchors = [], []
    limit = MAX_QUESTION_ANCHORS if question else MAX_CANDIDATE_ANCHORS
    for si, part in enumerate(parts):
        tokens = _TOKEN.findall(part)[:512]
        vals = [_number(t) for t in tokens]
        sets = _sets(tokens)
        sentences.append(Sentence(part, *sets, {v for v in vals if v is not None}))
        for ti, value in enumerate(vals):
            if value is not None and len(anchors) < limit:
                local = _sets(tokens[max(0, ti-WINDOW):ti+WINDOW+1])
                anchors.append(Anchor(value, *local, si))
    return sentences, anchors


def _union(items, attribute):
    return set().union(*(getattr(item, attribute) for item in items)) if items else set()


def _opposite(relations, other):
    comparable = {r for r in relations if r in _OPPOSITES}
    return sum(_OPPOSITES[r] in other for r in comparable)/len(comparable) if comparable else 0.


def _candidate(question_sentences, qa, target, original):
    answer = _number(parse_answer(original) or "")
    body = str(original).rsplit("####", 1)[0] if "####" in str(original) else str(original)
    cs, ca = _profile(body)
    byvalue = {}
    for a in ca: byvalue.setdefault(a.value, []).append(a)
    context_recall=[]; context_jaccard=[]; context_missing=[]; objects=[]; object_supported=[]
    object_missing=[]; binding_margin=[]; unitrec=[]; unitconf=[]
    relations=[]; opposite=[]; omission=[]; timekeep=[]; timeflip=[]; rates=[]; exclusions=[]
    for q in qa:
        matches = byvalue.get(q.value, [])
        recalls = [_recall(q.words, a.words) for a in matches]
        context_recall.append(max(recalls, default=0.))
        context_jaccard.append(max((_jaccard(q.words,a.words) for a in matches),default=0.))
        context_missing.append(float(bool(q.words and matches) and not any(recalls)))
        if q.objects:
            obj=max((_recall(q.objects,a.objects) for a in matches),default=0.)
            other=max((_recall(q.objects,a.objects) for a in ca if a.value!=q.value),default=0.)
            objects.append(obj);object_supported.append(float(obj>=.5))
            object_missing.append(float(bool(matches) and obj==0.));binding_margin.append(obj-other)
        if q.units:
            unitrec.append(max((_recall(q.units,a.units) for a in matches),default=0.))
            unitconf.append(float(any(a.units for a in matches) and not any(q.units&a.units for a in matches)))
        # Keep all ties, union their relation evidence: candidate sentence order
        # must not decide whether a cue is present among equally good matches.
        best=max((_recall(q.objects or q.words,a.objects if q.objects else a.words) for a in matches),default=0.)
        aligned=[a for a in matches if _recall(q.objects or q.words,a.objects if q.objects else a.words)==best]
        cr=_union(aligned,"relations")
        if q.relations:
            relations.append(_recall(q.relations,cr));opposite.append(_opposite(q.relations,cr))
            omission.append(float(not(q.relations&cr)))
        temporal=q.relations&{"before","after"}
        if temporal:timekeep.append(_recall(temporal,cr));timeflip.append(_opposite(temporal,cr))
        if "rate" in q.relations:rates.append(float("rate" in cr))
        if "exclude" in q.relations:exclusions.append(float("exclude" in cr))
    object_values={}
    for q in qa:
        for word in q.objects:object_values.setdefault(word,set()).add(q.value)
    bound=[sum(any(word in a.objects for a in byvalue.get(v,[])) for v in values)/len(values)
           for word,values in sorted(object_values.items())]
    numeric=[_mean(context_recall),min(context_recall,default=0.),_mean(context_jaccard),
             _mean([float(x>=.5) for x in context_recall]),_mean(context_missing),_mean(objects),
             _mean(object_supported),_mean(object_missing),_mean(binding_margin),_mean(unitrec),_mean(unitconf),_mean(bound)]
    sr=[];so=[];sn=[];clause_missing=[]
    for q in question_sentences:
        if not q.relations:continue
        best=max((_jaccard(q.words,c.words) for c in cs),default=0.)
        aligned=[c for c in cs if best>0 and _jaccard(q.words,c.words)==best]
        cr=_union(aligned,"relations")
        sr.append(_recall(q.relations,cr));so.append(_opposite(q.relations,cr))
        sn.append(float(("exclude" in q.relations)!=("exclude" in cr)))
        clause_missing.append(float(not(q.relations&cr)))
    relation=[_mean(relations),_mean(opposite),_mean(omission),_mean(timekeep),_mean(timeflip),
              _mean(rates),_mean(exclusions),_mean(sr),_mean(so),_mean(sn)]
    final=cs[-2:];fw=_union(final,"words");fu=_union(final,"units");fr=_union(final,"relations")
    tw,to,tu,tr=target.words,target.objects,target.units,target.relations
    aa=byvalue.get(answer,[]) if answer is not None else []
    other=[a for a in ca if answer is not None and a.value!=answer]
    equations=[c for c in cs if "=" in c.text]
    target_features=[_recall(tw,fw),_jaccard(tw,fw),max((_recall(tw,a.words) for a in aa),default=0.),
        max((_recall(to,a.objects) for a in aa),default=0.),_recall(tw,equations[-1].words) if equations else 0.,
        _recall(tu,fu),float(bool(tu) and any(a.units for a in aa) and not any(tu&a.units for a in aa)),
        _recall(tr,fr),_opposite(tr,fr),_recall(tw,fw)-_recall(tw,_union(cs,"words")),
        max((_recall(to,a.objects) for a in aa),default=0.)-max((_recall(to,a.objects) for a in other),default=0.),
        _recall(tw,_union(cs[-max(1,math.ceil(len(cs)/4)):],"words"))]
    joint=[];supported=[];numberonly=[];sentenceobjects=[];positions=[]
    for q in question_sentences:
        if not q.values:continue
        scores=[(_jaccard(q.words,c.words)+_jaccard(q.values,c.values))/2 for c in cs]
        best=max(scores,default=0.);joint.append(best)
        supported.append(float(any(_recall(q.words,c.words)>=.5 and _recall(q.values,c.values)>=.5 for c in cs)))
        numberonly.append(float(any(q.values&c.values and not(q.words&c.words) for c in cs)))
        aligned=[c for c,score in zip(cs,scores) if score==best and best>0]
        sentenceobjects.append(max((_recall(q.objects,c.objects) for c in aligned),default=0.))
        positions.append(scores.index(best) if best>0 else -1)
    pairs=[(a,b) for a,b in zip(positions,positions[1:]) if a>=0 and b>=0]
    sentence=[_mean(joint),min(joint,default=0.),_mean(supported),_mean(numberonly),_mean(sentenceobjects),
              _mean([float(a<=b) for a,b in pairs]),_mean([1-x for x in supported[len(supported)//2:]]),_mean(clause_missing)]
    qvalues={a.value for a in qa};dsource=[];dtarget=[];dshared=[];dans=[];drel=[]
    for c in equations:
        sources=[q for q in qa if q.value in c.values]
        if not sources:continue
        source_objects=_union(sources,"objects")
        recall=_recall(source_objects,c.objects)
        if answer is not None and answer in c.values:dans.append(recall)
        if not(c.values-qvalues):continue
        dsource.append(recall);dtarget.append(_recall(to,c.objects));drel.append(_recall(_union(sources,"relations"),c.relations))
        shared={word for word in source_objects if len({q.value for q in sources if word in q.objects})>=2}
        dshared.append(float(bool(shared&c.objects)))
    derived=[_mean(dsource),max(dtarget,default=0.),_mean(dshared),max(dans,default=0.),
             dsource[-1] if dsource else 0.,_mean(drel)]
    return numeric+relation+target_features+sentence+derived


def extract_features(bank):
    """Finite float32 [N,4,F] matrix; all intermediates and relative means float64."""
    output=np.zeros((len(bank),4,len(FEATURE_NAMES)),dtype=np.float64)
    nlocal=len(FEATURE_NAMES)-len(_RELATIVE_SOURCE)
    for i,row in enumerate(bank):
        if len(row["candidates"])!=4:raise ValueError("Expected four candidate texts")
        qs,qa=_profile(row["question"],question=True)
        target=next((s for s in reversed(qs) if re.search(r"\b(?:how|what|which)\b",s.text,re.I)),
                    qs[-1] if qs else Sentence("",set(),set(),set(),set(),set()))
        for j,c in enumerate(row["candidates"]):output[i,j,:nlocal]=_candidate(qs,qa,target,c["text"])
    for k,source in enumerate(_RELATIVE_SOURCE):
        x=output[:,:,FEATURE_NAMES.index(source)]
        # Sorting four scalars makes the reduction exactly candidate-order independent.
        mean=np.sort(x,axis=1).mean(axis=1,keepdims=True)
        output[:,:,nlocal+k]=x-mean
    if not np.isfinite(output).all():raise ValueError("Nonfinite condition feature")
    return output.astype(np.float32)

