"""Input-only token variants and NumPy inference for the sequence study."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from common import parse_answer
from selection import select
from sequence_model import (SPECIAL, MAX_VOCAB, SEQUENCE_LENGTH, tokens,
                            numeric_inputs, transform_numeric, numpy_forward)

RECIPES = ("head64", "headtail", "headtail_roles", "headtail_format")
_ROLE_PATTERN = re.compile(r"####|\\[a-z]+|[a-z]+(?:'[a-z]+)?|\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?|[^\w\s]", re.I)
_FRACTION = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{\s*(\d+(?:\.\d+)?)\s*\}\s*\{\s*(\d+(?:\.\d+)?)\s*\}")
_ROLE_BASES = tuple([f"q{k}" for k in range(32)] + ["qx", "derived"])
ROLE_TOKENS = tuple(f"<num:{base}{':self' if flags & 1 else ''}{':rival' if flags & 2 else ''}>"
                    for base in _ROLE_BASES for flags in range(4))


def number(part):
    try:
        part = part.replace(",", "")
        value = (Fraction(part.split("/")[0]) / Fraction(part.split("/")[1])
                 if part.count("/") == 1 else Fraction(part))
        return value if abs(value) <= 10**12 else None
    except (ValueError, ZeroDivisionError):
        return None


def role_parts(text):
    text = str(text).lower()
    if len(text) > 24000:
        return role_parts(text[:12000]) + ["<gap>"] + role_parts(text[-12000:])
    text = _FRACTION.sub(r"\1/\2", text)
    return _ROLE_PATTERN.findall(text)


def clean_format(text):
    text = str(text).replace("**", "").replace("`", "").replace("$", "")
    text = re.sub(r"(?m)^\s*#{1,3}\s+", "", text)
    for command in (r"\left", r"\right", r"\!", r"\,"):
        text = text.replace(command, " ")
    return re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", text)


def role_sequence(parts, question_values, own, rivals):
    result = []
    for part in parts:
        value = number(part) if part and part[0].isdigit() else None
        if value is None:
            result.append(part[:64])
            continue
        index = question_values.get(value)
        base = "derived" if index is None else (f"q{index}" if index < 32 else "qx")
        token = f"<num:{base}{':self' if value == own else ''}{':rival' if value in rivals else ''}>"
        result.append(token)
    return result


def question_candidate_tokens(row, recipe):
    if recipe not in RECIPES:
        raise ValueError(recipe)
    roles = recipe == "headtail_roles"
    if roles:
        qparts = role_parts(row["question"])
        qvalues = {}
        for part in qparts:
            value = number(part) if part and part[0].isdigit() else None
            if value is not None and value not in qvalues:
                qvalues[value] = len(qvalues)
        answers = [number(answer) if (answer := parse_answer(c["text"])) is not None else None
                   for c in row["candidates"]]
    else:
        transform = clean_format if recipe == "headtail_format" else str
        question = tokens(transform(row["question"]))
    for j, candidate in enumerate(row["candidates"]):
        if roles:
            own = answers[j]
            rivals = {a for k, a in enumerate(answers) if k != j and a is not None and a != own}
            # Question role coding is deliberately independent of which answer is scored.
            question = role_sequence(qparts, qvalues, None, set())
            text = role_sequence(role_parts(candidate["text"]), qvalues, own, rivals)
        else:
            text = tokens(transform(candidate["text"]))
        q = question[:64] if recipe == "head64" or len(question) <= 64 else question[:32] + question[-32:]
        if len(text) > 256:
            text = text[:128] + ["<gap>"] + text[-128:]
        yield q + ["<sep>"] + text


def sequences(bank, recipe):
    return [list(question_candidate_tokens(row, recipe)) for row in bank]


def vocabulary(sequence_bank, fit_ids, recipe):
    special = tuple(SPECIAL) + (ROLE_TOKENS if recipe == "headtail_roles" else ())
    counts = Counter(token for i in fit_ids for seq in sequence_bank[int(i)] for token in seq)
    words = sorted((t for t in counts if t not in special), key=lambda t: (-counts[t], t))
    return np.array(list(special) + words[:MAX_VOCAB-len(special)])


def encode_sequences(sequence_bank, vocab):
    lookup = {word: i for i, word in enumerate(vocab.tolist())}
    codes = np.zeros((len(sequence_bank), 4, SEQUENCE_LENGTH), np.int64)
    unknown = total = 0
    for i, row in enumerate(sequence_bank):
        if len(row) != 4:
            raise ValueError("Expected four candidates")
        for j, seq in enumerate(row):
            if len(seq) > SEQUENCE_LENGTH:
                raise ValueError("Token budget exceeded")
            values = [lookup.get(token, 1) for token in seq]
            codes[i, j, :len(values)] = values
            unknown += values.count(1)
            total += len(values)
    return codes, {"tokens": total, "unknown_tokens": unknown,
                   "unknown_fraction": unknown / max(total, 1)}


def score_candidates(bank, embeddings, asset):
    if not bank:
        return np.empty((0, 4), np.float32)
    recipe = str(asset["token_recipe"].item())
    codes, _ = encode_sequences(sequences(bank, recipe), asset["vocabulary"])
    numeric = transform_numeric(numeric_inputs(bank, embeddings), asset)
    return numpy_forward(codes, numeric, asset)


def predict(bank, arrays, checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_dir():
        path /= "selected_model.npz"
    with np.load(path, allow_pickle=False) as data:
        asset = {key: data[key] for key in data.files}
    return select(bank, score_candidates(bank, arrays["embeddings"], asset), "mean")
