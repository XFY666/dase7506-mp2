"""Answer-level aggregation using only the input bank and candidate scores."""
import numpy as np
from common import parse_answer


def answer_groups(bank):
    return [[parse_answer(c['text']) for c in b['candidates']] for b in bank]


def select(bank, scores, aggregation='max', consensus=0.0, return_details=False):
    choices, margins, support = [], [], []
    for b, row in zip(bank, np.asarray(scores)):
        aa = [parse_answer(c['text']) for c in b['candidates']]
        groups = {}
        for j, a in enumerate(aa):
            if a is not None:
                groups.setdefault(a, []).append(j)
        if not groups:
            choices.append(int(np.argmax(row))); margins.append(0.0); support.append(0)
            continue
        entries = []
        for a, indices in groups.items():
            vals = row[indices]
            value = float(np.mean(vals) if aggregation == 'mean' else np.max(vals))
            value += consensus * np.log(len(indices))
            best = indices[int(np.argmax(vals))]
            entries.append((value, -best, best, len(indices)))
        entries.sort(reverse=True)
        choices.append(int(entries[0][2]))
        margins.append(float(entries[0][0]-entries[1][0]) if len(entries)>1 else float('inf'))
        support.append(entries[0][3])
    if return_details:
        return choices, np.array(margins), np.array(support)
    return choices


def majority(bank):
    return select(bank, np.zeros((len(bank),4)), 'max', 1.0)
