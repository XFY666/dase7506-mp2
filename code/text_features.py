"""Small, deterministic candidate features; no references or fitted assets.

The three groups separate answer agreement, presentation, and evidence from
the reasoning. Arithmetic checks only cover explicit numeric equalities: they
are a useful consistency signal, not a proof that the problem was modelled
correctly. Parsing is bounded and uses a whitelist interpreter, never eval.
"""
from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import math
import re

import numpy as np

from common import parse_answer


_GROUP_NAMES = {
    "answer_consensus": [
        "answer_valid", "answer_log_abs", "answer_negative", "answer_zero",
        "answer_integer", "answer_log_denominator", "answer_frequency",
        "valid_candidate_fraction", "unique_answer_fraction", "modal_frequency",
        "answer_is_mode", "answer_relative_log_abs", "answer_numeric_rank",
        "answer_in_question", "answer_question_max_ratio", "answer_question_sum_ratio",
        "answer_repeated_in_reasoning", "answer_last_number_match",
        "answer_boxes_disagree", "same_answer_text_similarity", "answer_frequency_margin",
    ],
    "style": [
        "log_chars", "log_words", "log_lines", "log_paragraphs", "log_headings",
        "log_bold_markers", "log_math_delimiters", "log_box_markers", "has_hash_answer",
        "log_bullet_lines", "log_numbered_steps", "log_step_words", "log_question_marks",
        "log_exclamations", "log_confident_words", "log_uncertain_words",
        "log_revision_words", "log_final_words", "log_latex_commands", "unique_word_ratio",
        "relative_log_chars", "relative_log_words", "fraction_of_group_chars",
    ],
    "reasoning": [
        "log_numbers", "log_unique_numbers", "number_repetition_ratio", "number_density",
        "question_number_coverage", "new_number_fraction", "question_word_coverage",
        "question_text_jaccard", "log_equalities", "log_arithmetic_operators",
        "log_fraction_commands", "log_percent_signs", "log_checked_equalities",
        "log_correct_equalities", "log_incorrect_equalities", "equality_correct_fraction",
        "equality_error_fraction", "max_equality_relative_error", "mean_equality_relative_error",
        "last_equality_correct", "last_equality_answer_match", "log_distinct_equality_results",
        "log_relation_words", "log_condition_words", "log_impossibility_words",
        "candidate_text_similarity", "candidate_number_similarity",
        "different_answer_text_similarity", "relative_equality_correct_fraction",
        "relative_question_number_coverage", "relative_number_count",
    ],
}
FEATURE_NAMES = [name for names in _GROUP_NAMES.values() for name in names]
FEATURE_GROUPS = {}
_offset = 0
for _group, _names in _GROUP_NAMES.items():
    FEATURE_GROUPS[_group] = list(range(_offset, _offset + len(_names)))
    _offset += len(_names)

_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_WORDS_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CONFIDENT_RE = re.compile(r"\b(?:clearly|obviously|certainly|definitely|correct|therefore|thus|hence)\b")
_UNCERTAIN_RE = re.compile(r"\b(?:perhaps|maybe|might|assuming|assume|approximately|roughly|seems|likely)\b")
_REVISION_RE = re.compile(r"\b(?:however|but|instead|actually|wait|mistake|incorrect|reconsider|alternatively)\b")
_RELATION_RE = re.compile(r"\b(?:total|difference|remaining|left|each|per|average|twice|half|times|ratio|more|less)\b")
_CONDITION_RE = re.compile(r"\b(?:if|only|except|including|excluding|before|after|already|additional|together)\b")
_IMPOSSIBLE_RE = re.compile(r"\b(?:impossible|contradiction|inconsistent|negative|cannot|can't|ambiguous|ambiguity)\b")
_FRAC_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]{1,100})\}\s*\{([^{}]{1,100})\}")
_BOX_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_TEXT_RE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_LEFT_EXPR_RE = re.compile(r"([\d.()+*/%\-\s]+)$")
_RIGHT_EXPR_RE = re.compile(r"^\s*([\d.()+*/%\-\s]+)")
_SPACE_RE = re.compile(r"\s+")
_STOP_WORDS = frozenset("a an the is are was were be been to of and or in on for as by with it its that this these those how many much what we so then let find calculate answer step total given told text frac boxed times".split())
_MAX_TEXT = 24000
_MAX_EQUATIONS = 48


def _numbers(text):
    """Finite numeric literals, canonicalised to six decimal places."""
    result = []
    for match in _NUM_RE.finditer(text):
        try:
            value = float(Fraction(match.group().replace(",", "")))
            if math.isfinite(value) and abs(value) < 1e15:
                result.append(round(value, 6))
        except (ValueError, OverflowError, ZeroDivisionError):
            pass
        if len(result) >= 1024:
            break
    return result


def _safe_arithmetic(expression):
    """Interpret only short literal arithmetic ASTs with bounded exponents."""
    expression = _SPACE_RE.sub(" ", expression.strip()).strip(" .")
    if not expression or len(expression) > 160:
        return None
    try:
        root = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(root)) > 64:
            return None

        def visit(node):
            if isinstance(node, ast.Expression):
                value = visit(node.body)
            elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
                value = float(node.value)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    value = a + b
                elif isinstance(node.op, ast.Sub):
                    value = a - b
                elif isinstance(node.op, ast.Mult):
                    value = a * b
                elif isinstance(node.op, ast.Div):
                    value = a / b
                elif isinstance(node.op, ast.Pow) and abs(b) <= 6 and abs(a) <= 1e6:
                    value = a ** b
                else:
                    raise ValueError("Unsupported operator")
            else:
                raise ValueError("Not literal arithmetic")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e15:
                raise ValueError("Out of arithmetic range")
            return value

        return visit(root)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _arithmetic_evidence(text, answer):
    """Compare bounded numeric expressions on both sides of explicit '='."""
    # Preserve explicit numeric Python-style powers before removing Markdown bold.
    normalized = re.sub(r"(?<=\d)\*\*(?=[\d(])", "^", text)
    normalized = normalized.replace("**", "").replace("$", "").replace(",", "")
    normalized = normalized.replace("−", "-").replace("×", "*").replace("÷", "/")
    for source, replacement in ((r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
                                (r"\left", ""), (r"\right", ""), (r"\,", " "),
                                (r"\!", ""), (r"\%", "%")):
        normalized = normalized.replace(source, replacement)
    for _ in range(3):
        normalized, count = _FRAC_RE.subn(r"((\1)/(\2))", normalized)
        if not count:
            break
    normalized = _BOX_RE.sub(r"(\1)", normalized)
    # Preserve units/words as boundaries instead of joining unrelated numbers.
    normalized = _TEXT_RE.sub(r" \1 ", normalized)
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized)
    normalized = normalized.replace("^", "**")
    errors, results = [], []
    for match in list(re.finditer("=", normalized))[:_MAX_EQUATIONS]:
        # A line boundary is intentional: lists and separate equations must not
        # accidentally become a single arithmetic expression.
        before = normalized[max(0, match.start() - 180):match.start()].split("\n")[-1]
        after = normalized[match.end():match.end() + 180].split("\n")[0]
        left, right = _LEFT_EXPR_RE.search(before), _RIGHT_EXPR_RE.match(after)
        if left is None or right is None:
            continue
        a, b = _safe_arithmetic(left.group(1)), _safe_arithmetic(right.group(1))
        if a is None or b is None:
            continue
        error = min(1., abs(a - b) / max(1., abs(a), abs(b)))
        errors.append(error)
        results.append(b)
    count = len(errors)
    correct = sum(error <= 1e-6 for error in errors)
    last_match = bool(results and answer is not None and
                      math.isclose(results[-1], answer, rel_tol=1e-6, abs_tol=1e-6))
    return [math.log1p(count), math.log1p(correct), math.log1p(count - correct),
            correct / max(1, count), (count - correct) / max(1, count),
            max(errors, default=0.), sum(errors) / max(1, count),
            float(bool(errors) and errors[-1] <= 1e-6), float(last_match),
            math.log1p(len(set(round(value, 6) for value in results)))]


def _jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def extract_features(bank):
    """Return finite float32 [question, four candidates, feature] array.

    IDs and candidate index fields are deliberately never inspected. All
    comparisons are symmetric across candidates and apply to arbitrary banks.
    """
    output = np.zeros((len(bank), 4, len(FEATURE_NAMES)), dtype=np.float32)
    for row_index, row in enumerate(bank):
        if len(row["candidates"]) != 4:
            raise ValueError("Expected four candidate texts per question")
        question = str(row["question"])[:_MAX_TEXT].lower()
        question_numbers = set(_numbers(question))
        question_words = set(_WORDS_RE.findall(question)) - _STOP_WORDS
        q_max = max((abs(x) for x in question_numbers), default=1.)
        q_sum = sum(abs(x) for x in question_numbers)
        texts = [str(c["text"]) for c in row["candidates"]]
        answers = [parse_answer(text) for text in texts]
        values = []
        for answer in answers:
            try:
                value = float(Fraction(answer)) if answer is not None else None
                values.append(value if value is not None and math.isfinite(value) else None)
            except (ValueError, OverflowError, ZeroDivisionError):
                values.append(None)
        counts = Counter(answer for answer in answers if answer is not None)
        frequencies = [counts.get(answer, 0) for answer in answers]
        mode = max(frequencies, default=0)
        answer_logs = [math.log1p(min(abs(value), 1e15)) if value is not None else 0. for value in values]
        valid_logs = [a for a, value in zip(answer_logs, values) if value is not None]
        log_mean = sum(valid_logs) / max(1, len(valid_logs))
        word_sets, number_sets, local_features = [], [], []
        for j, original_text in enumerate(texts):
            text = original_text[:_MAX_TEXT]
            low = text.lower()
            words = _WORDS_RE.findall(low)
            word_set = set(words) - _STOP_WORDS
            numbers = _numbers(text)
            number_set = set(numbers)
            word_sets.append(word_set)
            number_sets.append(number_set)
            value, parsed = values[j], answers[j]
            numeric_value = value if value is not None else 0.
            fraction = Fraction(parsed) if parsed is not None else Fraction(0)
            other_values = [v for k, v in enumerate(values) if k != j and v is not None]
            boxes = [parse_answer("\\boxed{" + b + "}") for b in _BOX_RE.findall(text)]
            # Counts are transformed to avoid numerical domination by long text.
            answer_features = [
                float(parsed is not None), answer_logs[j], float(numeric_value < 0),
                float(value is not None and value == 0), float(parsed is not None and fraction.denominator == 1),
                math.log1p(min(fraction.denominator, 1e15)), frequencies[j] / 4,
                len(valid_logs) / 4, len(counts) / 4, mode / 4,
                float(frequencies[j] == mode and parsed is not None), answer_logs[j] - log_mean,
                sum(numeric_value > v for v in other_values) / max(1, len(other_values)),
                float(value is not None and round(value, 6) in question_numbers),
                math.log1p(min(abs(numeric_value) / max(1., q_max), 1e15)),
                math.log1p(min(abs(numeric_value) / max(1., q_sum), 1e15)),
                math.log1p(max(0, numbers.count(round(numeric_value, 6)) - 1)) if value is not None else 0.,
                float(bool(numbers) and value is not None and math.isclose(numbers[-1], value, rel_tol=1e-6, abs_tol=1e-6)),
                float(len({b for b in boxes if b is not None}) > 1), 0.,
                (frequencies[j] - max((counts.get(a, 0) for a in counts if a != parsed), default=0)) / 4,
            ]
            style_counts = [len(text), len(words), text.count("\n") + 1,
                            len(re.split(r"\n\s*\n", text)), len(re.findall(r"(?m)^\s*#{1,6}", text)),
                            text.count("**"), text.count("$"), text.count(r"\boxed")]
            style_features = [math.log1p(n) for n in style_counts]
            style_features += [float("####" in original_text)]
            style_features += [math.log1p(n) for n in (
                len(re.findall(r"(?m)^\s*[-*•] ", text)), len(re.findall(r"(?m)^\s*\d+[.)] ", text)),
                len(re.findall(r"\bstep\b", low)), text.count("?"), text.count("!"),
                len(_CONFIDENT_RE.findall(low)), len(_UNCERTAIN_RE.findall(low)), len(_REVISION_RE.findall(low)),
                len(re.findall(r"\b(?:final|answer|result)\b", low)), len(re.findall(r"\\[a-z]+", low)))]
            style_features += [len(set(words)) / max(1, len(words)), 0., 0., 0.]
            coverage = len(number_set & question_numbers) / max(1, len(question_numbers))
            reasoning_features = [
                math.log1p(len(numbers)), math.log1p(len(number_set)),
                1 - len(number_set) / max(1, len(numbers)), len(numbers) / max(1, len(words)),
                coverage, len(number_set - question_numbers) / max(1, len(number_set)),
                len(word_set & question_words) / max(1, len(question_words)), _jaccard(word_set, question_words),
                math.log1p(text.count("=")), math.log1p(len(re.findall(r"[+×÷]|\\times|\\div|\\cdot", text))),
                math.log1p(len(re.findall(r"\\(?:dfrac|tfrac|frac)", text))), math.log1p(text.count("%")),
            ]
            reasoning_features += _arithmetic_evidence(text, value)
            reasoning_features += [math.log1p(len(regex.findall(low))) for regex in (_RELATION_RE, _CONDITION_RE, _IMPOSSIBLE_RE)]
            reasoning_features += [0.] * 6
            local_features.append(answer_features + style_features + reasoning_features)

        features = np.asarray(local_features, dtype=np.float32)
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        for j in range(4):
            similarities = [_jaccard(word_sets[j], word_sets[k]) for k in range(4) if k != j]
            same = [_jaccard(word_sets[j], word_sets[k]) for k in range(4)
                    if k != j and answers[j] is not None and answers[j] == answers[k]]
            different = [_jaccard(word_sets[j], word_sets[k]) for k in range(4)
                         if k != j and answers[j] != answers[k]]
            features[j, index["candidate_text_similarity"]] = sum(similarities) / 3
            features[j, index["same_answer_text_similarity"]] = sum(same) / max(1, len(same))
            features[j, index["different_answer_text_similarity"]] = sum(different) / max(1, len(different))
            features[j, index["candidate_number_similarity"]] = sum(_jaccard(number_sets[j], number_sets[k]) for k in range(4) if k != j) / 3
        for target, source in (("relative_log_chars", "log_chars"), ("relative_log_words", "log_words"),
                               ("relative_equality_correct_fraction", "equality_correct_fraction"),
                               ("relative_question_number_coverage", "question_number_coverage"),
                               ("relative_number_count", "log_numbers")):
            features[:, index[target]] = features[:, index[source]] - features[:, index[source]].mean()
        lengths = np.asarray([len(text) for text in texts], dtype=np.float64)
        features[:, index["fraction_of_group_chars"]] = lengths / max(1., lengths.sum())
        output[row_index] = features
    return np.nan_to_num(output, copy=False, nan=0., posinf=30., neginf=-30.)
