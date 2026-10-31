"""Shared train/validation analysis; never called by inference."""
import json
from pathlib import Path
import numpy as np
from common import DATA, load_inputs, parse_answer, read_jsonl


def development_data(split):
    if split not in ('train', 'validation'):
        raise ValueError('Development code may read training/validation only')
    bank, arrays = load_inputs(split)
    refs = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    assert all(b['id'] == r['id'] and b['question'] == r['question'] for b, r in zip(bank, refs))
    answers = [[parse_answer(c['text']) for c in b['candidates']] for b in bank]
    y = np.array([[a is not None and a == parse_answer(r['answer']) for a in aa]
                  for aa, r in zip(answers, refs)], dtype=np.float32)
    hard = np.array([len(set(a for a in aa if a is not None)) >= 2 for aa in answers])
    return bank, arrays['embeddings'], y, hard


def metrics(choices, y, hard):
    correct = y[np.arange(len(y)), choices].astype(bool)
    return {'correct': int(correct.sum()), 'n': len(y), 'accuracy': float(100*correct.mean()),
            'hard_correct': int(correct[hard].sum()), 'hard_n': int(hard.sum()),
            'hard_accuracy': float(100*correct[hard].mean())}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
