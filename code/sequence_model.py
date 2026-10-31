"""NumPy-only inference for a small word CNN trained from scratch.

Vocabulary and neural parameters come solely from fitting-side course training
questions. The inference module imports neither torch nor training libraries.
"""
from collections import Counter
from pathlib import Path
import re

import numpy as np

from common import parse_answer
from selection import select
from text_features import extract_features, FEATURE_NAMES

SPECIAL = ("<pad>", "<unk>", "<sep>", "<gap>", "<num>")
MAX_VOCAB = 4096
QUESTION_TOKENS = 64
CANDIDATE_TOKENS = 256
SEQUENCE_LENGTH = 322
_PATTERN = re.compile(r"####|\\[a-z]+|[a-z]+(?:'[a-z]+)?|\d[\d,]*(?:\.\d+)?|[^\w\s]", re.I)
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
_CENTERED = (("relative_log_chars", "log_chars"), ("relative_log_words", "log_words"),
             ("relative_equality_correct_fraction", "equality_correct_fraction"),
             ("relative_question_number_coverage", "question_number_coverage"),
             ("relative_number_count", "log_numbers"))


def tokens(text):
    text = str(text).lower()
    if len(text) > 24000:
        return tokens(text[:12000])+["<gap>"]+tokens(text[-12000:])
    return ["<num>" if part[0].isdigit() else part[:64] for part in _PATTERN.findall(text)]


def question_candidate_tokens(row):
    question = tokens(row["question"])[:QUESTION_TOKENS]
    for candidate in row["candidates"]:
        text = tokens(candidate["text"])
        if len(text) > CANDIDATE_TOKENS:
            text = text[:128]+["<gap>"]+text[-128:]
        yield question+["<sep>"]+text


def build_vocabulary(bank, ids):
    counts = Counter()
    for i in ids:
        for sequence in question_candidate_tokens(bank[int(i)]):
            counts.update(sequence)
    words = sorted((token for token in counts if token not in SPECIAL), key=lambda token: (-counts[token], token))
    return np.array(list(SPECIAL)+words[:MAX_VOCAB-len(SPECIAL)])


def encode_bank(bank, vocabulary):
    index = {word: i for i, word in enumerate(vocabulary.tolist())}
    result = np.zeros((len(bank),4,SEQUENCE_LENGTH),dtype=np.int64)
    unknown = total = 0
    for i,row in enumerate(bank):
        if len(row["candidates"]) != 4: raise ValueError("Expected four candidates")
        for j, sequence in enumerate(question_candidate_tokens(row)):
            code = [index.get(token,1) for token in sequence]
            result[i,j,:len(code)] = code
            unknown += code.count(1); total += len(code)
    return result,{"tokens":total,"unknown_tokens":unknown,"unknown_fraction":unknown/max(total,1)}


def stable_text_features(bank):
    x = extract_features(bank)
    for target,source in _CENTERED:
        values = x[..., _INDEX[source]].astype(np.float64)
        x[..., _INDEX[target]] = (values-values.mean(axis=1,keepdims=True)).astype(np.float32)
    return x


def numeric_inputs(bank, embeddings):
    return np.concatenate((np.asarray(embeddings,dtype=np.float32),stable_text_features(bank)),axis=-1)


def transform_numeric(raw, asset):
    return np.clip((raw-asset["numeric_mean"])/asset["numeric_scale"],-8,8).astype(np.float32)


def numpy_forward(codes, numeric, asset, batch_candidates=64):
    flat = np.asarray(codes).reshape(-1,SEQUENCE_LENGTH)
    if len(flat) == 0:
        return np.empty((0,4),dtype=np.float32)
    numeric = np.asarray(numeric,dtype=np.float32).reshape(len(flat),-1)
    output = np.empty(len(flat),dtype=np.float32)
    for start in range(0,len(flat),batch_candidates):
        ids = flat[start:start+batch_candidates]
        embedded = asset["token_embedding"][ids]
        mask = ids != 0
        pooled = []
        for kernel in (3,5):
            weight = asset[f"conv{kernel}_weight"]
            pad = kernel//2
            padded = np.pad(embedded,((0,0),(pad,pad),(0,0)))
            z = np.broadcast_to(asset[f"conv{kernel}_bias"],(len(ids),SEQUENCE_LENGTH,32)).copy()
            for offset in range(kernel):
                z += padded[:,offset:offset+SEQUENCE_LENGTH,:] @ weight[:,:,offset].T
            z = np.maximum(z,0)
            maximum = np.where(mask[...,None],z,np.float32(-1e9)).max(axis=1)
            mean = (z*mask[...,None]).sum(axis=1)/np.maximum(mask.sum(axis=1,keepdims=True),1)
            pooled.extend((maximum,mean.astype(np.float32)))
        features = np.concatenate(pooled+([numeric[start:start+len(ids)]] if bool(asset["joint"]) else []),axis=1)
        hidden = np.tanh(features@asset["hidden_weight"].T+asset["hidden_bias"])
        output[start:start+len(ids)] = (hidden@asset["output_weight"].T+asset["output_bias"]).ravel()
    return output.reshape(-1,4)


def score_candidates(bank, embeddings, asset):
    if len(bank) == 0:
        return np.empty((0,4),dtype=np.float32)
    codes,_ = encode_bank(bank,asset["vocabulary"])
    numeric = transform_numeric(numeric_inputs(bank,embeddings),asset) if bool(asset["joint"]) else np.zeros((len(bank),4,0),dtype=np.float32)
    return numpy_forward(codes,numeric,asset)


def predict(bank, arrays, checkpoint_path):
    path = Path(checkpoint_path)
    if path.is_dir(): path = path/"selected_model.npz"
    with np.load(path,allow_pickle=False) as data:
        asset = {key:data[key] for key in data.files}
    return select(bank,score_candidates(bank,arrays["embeddings"],asset),"mean",0.)
