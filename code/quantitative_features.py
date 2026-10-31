"""Bounded, generic quantitative consistency features for arbitrary banks.

No answer references, IDs, fitted statistics, or problem templates are used.
These are fallible numeric/linguistic indicators, not a symbolic proof checker.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer
from text_features import _safe_arithmetic

MAX_TEXT = 24000
MAX_QUANTITIES = 20
MAX_EQUALITIES = 64
_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_WORD_VALUES = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand half quarter dozen twice thrice".split(),
    list(range(20)) + [20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, .5, .25, 12, 2, 3]))
_WORD_NUMBER = re.compile(r"\b(?:" + "|".join(_WORD_VALUES) + r")\b", re.I)
_FRAC = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]{1,100})\}\s*\{([^{}]{1,100})\}")
_TEXT = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_BOX = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_LEFT = re.compile(r"([\d.()+*/%\-\s]+)$")
_RIGHT = re.compile(r"^\s*([\d.()+*/%\-\s]+)")
_UNITS = {
    "money": r"\$|\b(?:dollars?|cents?|cost|price|paid|pay|profit|revenue)\b",
    "time": r"\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    "length": r"\b(?:kilometers?|kilometres?|meters?|metres?|miles?|feet|foot|inches?|yards?|centimeters?)\b",
    "mass_volume": r"\b(?:grams?|kilograms?|pounds?|ounces?|liters?|litres?|gallons?|cups?)\b",
    "rate": r"\b(?:per|each|every|rate|speed|mph)\b|/\s*(?:hour|minute|day|week)",
    "change": r"\b(?:left|remain\w*|rest|after|before|more|less|fewer|increase\w*|decrease\w*)\b",
}
_UNIT_RE = {k: re.compile(v, re.I) for k, v in _UNITS.items()}
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
_CONVERSIONS = (("time", (60., 24., 7., 365.)), ("money", (100.,)),
                ("length", (12., 3., 5280., 100., 1000.)),
                ("mass_volume", (16., 1000., 8., 4.)))

GROUP_NAMES = {
    "quantity": [
        "question_quantity_count", "question_word_quantity_count", "answer_nearest_quantity_error",
        "answer_pair_sum_error", "answer_pair_difference_error", "answer_pair_product_error", "answer_pair_ratio_error",
        "answer_pair_sum_matches", "answer_pair_difference_matches", "answer_pair_product_matches", "answer_pair_ratio_matches",
        "answer_sum_all_error", "answer_mean_all_error", "answer_range_fraction", "answer_outside_quantity_range",
        "answer_near_quantity_plus_one", "answer_near_quantity_minus_one",
        "answer_question_fractional", "integer_question_fractional_answer", "question_count_request_fractional_answer",
    ],
    "equation_chain": [
        "checked_chain_count", "chain_correct_count", "chain_incorrect_count", "chain_correct_fraction",
        "chain_tail_correct_fraction", "chain_correct_answer_result", "chain_incorrect_answer_result",
        "chain_correct_final_answer_result", "chain_bad_final_answer_result", "chain_result_answer_min_error",
        "chain_final_result_answer_error", "chain_answer_result_frequency", "chain_reused_result_fraction",
        "chain_question_operand_fraction", "chain_derived_operand_fraction", "chain_unexplained_operand_fraction",
        "chain_question_quantity_coverage", "chain_tail_question_operand_fraction", "chain_tail_unexplained_operand_fraction",
        "chain_add_fraction", "chain_subtract_fraction", "chain_multiply_fraction", "chain_divide_fraction",
        "chain_power_fraction", "chain_max_operation_count", "chain_mean_operation_count",
        "chain_multi_operation_fraction", "chain_repeated_lhs_conflict_count", "chain_adjacent_link_fraction",
        "chain_longest_link_fraction", "answer_prior_occurrences", "answer_tail_number_error",
    ],
    "units_percent": [
        "question_percent_count", "answer_percent_product_error", "answer_percent_remaining_error",
        "answer_percent_inverse_error", "answer_percent_increase_error", "answer_percent_before_increase_error",
        "percent_factor_100_in_reasoning", "percent_factor_1000_in_reasoning", "unit_conversion_answer_error",
        "unit_conversion_reasoning_coverage", "time_conversion_60_seen", "money_conversion_100_seen",
        "question_money", "question_time", "question_length", "question_mass_volume", "question_rate", "question_change",
        "money_word_coverage", "time_word_coverage", "length_word_coverage", "mass_volume_word_coverage",
        "rate_word_coverage", "change_word_coverage", "last_line_requested_unit_match",
        "question_rate_division_evidence", "question_change_subtraction_evidence", "percent_question_division_evidence",
    ],
}
FEATURE_NAMES = [n for group in GROUP_NAMES.values() for n in group]
FEATURE_GROUPS = {}
_offset = 0
for _key, _names in GROUP_NAMES.items():
    FEATURE_GROUPS[_key] = list(range(_offset, _offset + len(_names)))
    _offset += len(_names)


def _value(text):
    try:
        value = float(Fraction(text.replace(",", "")))
        return value if math.isfinite(value) and abs(value) <= 1e12 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _quantities(text, words=False, maximum=MAX_QUANTITIES, unique=True):
    pairs = [(m.start(), _value(m.group())) for m in _NUM.finditer(text)]
    if words:
        pairs += [(m.start(), _WORD_VALUES[m.group().lower()]) for m in _WORD_NUMBER.finditer(text)]
    result = []
    for _, value in sorted(pairs):
        if value is not None and (not unique or value not in result):
            result.append(value)
        if len(result) >= maximum:
            break
    return result


def _error(a, b):
    return min(1., abs(a-b)/max(1., abs(a), abs(b)))


def _min_error(answer, values):
    return min((_error(answer, value) for value in values), default=1.) if answer is not None else 1.


def _match(a, b):
    return _error(a, b) <= 1e-6


def _normalise(text):
    text = re.sub(r"(?<=\d)\*\*(?=[\d(])", "^", text)
    text = text.replace("**", "").replace("$", "").replace(",", "")
    for source, replacement in (("−", "-"), ("×", "*"), ("÷", "/"),
                                (r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
                                (r"\left", ""), (r"\right", ""), (r"\,", " "),
                                (r"\!", ""), (r"\%", "%")):
        text = text.replace(source, replacement)
    for _ in range(3):
        text, count = _FRAC.subn(r"((\1)/(\2))", text)
        if not count:
            break
    text = _BOX.sub(r"(\1)", text)
    text = _TEXT.sub(r" \1 ", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)
    return text.replace("^", "**")


def _expression(text):
    expression = re.sub(r"\s+", " ", text.strip()).strip(" .")
    value = _safe_arithmetic(expression)
    if value is None:
        return None
    root = ast.parse(expression, mode="eval")
    constants = [float(n.value) for n in ast.walk(root) if isinstance(n, ast.Constant)]
    operations = [type(n.op).__name__ for n in ast.walk(root) if isinstance(n, ast.BinOp)]
    return value, constants, operations, expression


def _equations(text):
    normalized = _normalise(text)
    equations = []
    for m in list(re.finditer("=", normalized))[:MAX_EQUALITIES]:
        before = normalized[max(0, m.start()-180):m.start()].split("\n")[-1]
        after = normalized[m.end():m.end()+180].split("\n")[0]
        left, right = _LEFT.search(before), _RIGHT.match(after)
        if left is None or right is None:
            continue
        lhs, rhs = _expression(left.group(1)), _expression(right.group(1))
        if lhs is None or rhs is None:
            continue
        equations.append((lhs, rhs))
    return equations


def _quantity_features(question, quantities, answer):
    candidates = [[], [], [], []]
    for i, a in enumerate(quantities):
        for j, b in enumerate(quantities):
            if i == j:
                continue
            candidates[0].append(a+b)
            candidates[1].append(a-b)
            candidates[2].append(a*b)
            if abs(b) > 1e-12:
                candidates[3].append(a/b)
    finite_answer = answer if answer is not None else 0.
    integer_question = bool(quantities) and all(abs(v-round(v)) < 1e-8 for v in quantities)
    fractional_answer = answer is not None and abs(answer-round(answer)) > 1e-8
    low, high = min(quantities, default=0.), max(quantities, default=1.)
    width = max(1., high-low)
    return [math.log1p(len(quantities)), math.log1p(len(_WORD_NUMBER.findall(question))),
            _min_error(answer, quantities), *[_min_error(answer, group) for group in candidates],
            *[math.log1p(sum(_match(finite_answer, x) for x in group)) if answer is not None else 0. for group in candidates],
            _min_error(answer, [sum(quantities)] if quantities else []),
            _min_error(answer, [sum(quantities)/len(quantities)] if quantities else []),
            float(np.clip((finite_answer-low)/width, -10, 10)), float(answer is not None and (answer < low or answer > high)),
            _min_error(answer, [q+1 for q in quantities]), _min_error(answer, [q-1 for q in quantities]),
            float(any(abs(v-round(v)) > 1e-8 for v in quantities)),
            float(integer_question and fractional_answer),
            float("how many" in question and fractional_answer)]


def _chain_features(text, question_values, answer):
    equations = _equations(text)
    results, correct, operands, sources, operations, conflicts, links = [], [], [], [], [], 0, []
    lhs_results, used_question = {}, set()
    qvalues = np.array(question_values)
    longest = current = 0
    constants = {0., 1., 2., 100., 60., 24., 7., 12., 365.}
    for lhs, rhs in equations:
        value, literals, ops, expression = lhs
        result = rhs[0]
        is_correct = _match(value, result)
        previous = results[:]
        source = [0, 0, 0]
        for literal in literals:
            matched = [float(q) for q in qvalues if _match(literal, float(q))]
            if matched:
                source[0] += 1; used_question.update(matched)
            elif any(_match(literal, v) for v in previous) or literal in constants:
                source[1] += 1
            else:
                source[2] += 1
        link = bool(previous and any(_match(v, previous[-1]) for v in literals))
        links.append(link); current = current+1 if link else 0; longest = max(longest, current)
        key = re.sub(r"\s+", "", expression)
        if key in lhs_results and not _match(lhs_results[key], result):
            conflicts += 1
        lhs_results[key] = result
        results.append(result); correct.append(is_correct); operands.append(literals)
        sources.append(source); operations.append(ops)
    n = len(equations); denom = max(1, n)
    source_total = np.sum(sources, axis=0) if sources else np.zeros(3)
    tail_source = np.sum(sources[-3:], axis=0) if sources else np.zeros(3)
    op_counts = [len(x) for x in operations]
    flat_ops = [v for x in operations for v in x]
    tail_numbers = _quantities(text[-800:], maximum=128, unique=False)
    occurrences = sum(_match(v, answer) for v in _quantities(text[:-100], maximum=256, unique=False)) if answer is not None else 0
    matches = [answer is not None and _match(result, answer) for result in results]
    return [math.log1p(n), math.log1p(sum(correct)), math.log1p(n-sum(correct)), sum(correct)/denom,
            sum(correct[-3:])/max(1, len(correct[-3:])),
            float(any(c and m for c, m in zip(correct, matches))),
            float(any(not c and m for c, m in zip(correct, matches))),
            float(bool(n and correct[-1] and matches[-1])), float(bool(n and not correct[-1] and matches[-1])),
            _min_error(answer, results), _min_error(answer, results[-1:]), sum(matches)/denom,
            sum(any(_match(v, r) for r in results[:i]) for i, vals in enumerate(operands) for v in vals)/max(1, sum(map(len, operands))),
            *list(source_total/max(1, sum(source_total))), len(used_question)/max(1, len(question_values)),
            tail_source[0]/max(1, sum(tail_source)), tail_source[2]/max(1, sum(tail_source)),
            *[flat_ops.count(op)/max(1, len(flat_ops)) for op in ("Add", "Sub", "Mult", "Div", "Pow")],
            math.log1p(max(op_counts, default=0)), sum(op_counts)/denom,
            sum(c > 1 for c in op_counts)/denom, math.log1p(conflicts), sum(links)/denom, longest/denom,
            math.log1p(occurrences), _min_error(answer, tail_numbers[-1:])]


def _unit_features(question, text, quantities, answer, chain):
    percentages = [float(m.group(1))/100 for m in _PERCENT.finditer(question)][:10]
    pool = [[], [], [], [], []]
    for percent in percentages:
        for q in quantities:
            pool[0].append(q*percent); pool[1].append(q*(1-percent)); pool[3].append(q*(1+percent))
            if abs(percent) > 1e-12: pool[2].append(q/percent)
            if abs(1+percent) > 1e-12: pool[4].append(q/(1+percent))
    quantities_reasoning = _quantities(text, maximum=256)
    reasoning_values = set(quantities_reasoning)
    qcounts = {k: len(rx.findall(question)) for k, rx in _UNIT_RE.items()}
    tcounts = {k: len(rx.findall(text)) for k, rx in _UNIT_RE.items()}
    converted, scale_seen, scale_total = [], 0, 0
    for unit, multipliers in _CONVERSIONS:
        if qcounts[unit]:
            for multiplier in multipliers:
                scale_total += 1; scale_seen += multiplier in reasoning_values
                converted.extend(q*multiplier for q in quantities)
                converted.extend(q/multiplier for q in quantities)
    question_end = re.split(r"[.!?]", question)[-2:]
    requested = {k for k, rx in _UNIT_RE.items() if rx.search(" ".join(question_end))}
    text_tail = text.rsplit("####", 1)[0].splitlines()[-3:]
    tail_units = {k for k, rx in _UNIT_RE.items() if rx.search(" ".join(text_tail))}
    index = {name: i for i, name in enumerate(GROUP_NAMES["equation_chain"])}
    division = chain[index["chain_divide_fraction"]]
    subtraction = chain[index["chain_subtract_fraction"]]
    return [math.log1p(len(percentages)), *[_min_error(answer, group) for group in pool],
            float(bool(percentages) and 100 in reasoning_values), float(bool(percentages) and 1000 in reasoning_values),
            _min_error(answer, converted), scale_seen/max(1, scale_total),
            float(qcounts["time"] > 0 and 60 in reasoning_values), float(qcounts["money"] > 0 and 100 in reasoning_values),
            *[math.log1p(qcounts[k]) for k in _UNITS],
            *[min(3., tcounts[k]/max(1, qcounts[k])) if qcounts[k] else 0. for k in _UNITS],
            len(requested & tail_units)/max(1, len(requested)),
            float(qcounts["rate"] > 0)*division, float(qcounts["change"] > 0)*subtraction,
            float(bool(percentages))*division]


def extract_quantitative(bank):
    """Return finite float32 [N,4,80] without inspecting candidate positions/IDs."""
    out = np.empty((len(bank), 4, len(FEATURE_NAMES)), dtype=np.float32)
    for i, row in enumerate(bank):
        question = str(row["question"])[:MAX_TEXT].lower()
        quantities = _quantities(question, words=True)
        if len(row["candidates"]) != 4:
            raise ValueError("Four candidates are required")
        for j, candidate in enumerate(row["candidates"]):
            original = str(candidate["text"])
            parsed = parse_answer(original)
            answer = _value(parsed) if parsed is not None else None
            text = original[:MAX_TEXT].lower()
            chain = _chain_features(text, quantities, answer)
            features = _quantity_features(question, quantities, answer) + chain + _unit_features(question, text, quantities, answer, chain)
            if len(features) != len(FEATURE_NAMES):
                raise AssertionError((len(features), len(FEATURE_NAMES)))
            out[i, j] = features
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite quantitative features")
    return out
