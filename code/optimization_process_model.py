"""NumPy inference for process-supervised group networks; no reference inputs."""
from pathlib import Path
import numpy as np
from answer_group_model import group_model_scores
from selection import select


def member_scores(bank, embeddings, asset, features=None):
    # Training-only aux_w/aux_b may be present in an audit asset. They are
    # deliberately ignored: inference uses the original final-answer scorer.
    return group_model_scores(bank, embeddings, asset, "group", features)


def ensemble_scores(bank, embeddings, assets, features=None):
    if not assets:
        raise ValueError("At least one fixed member is required")
    if not bank:
        return np.empty((0, 4), np.float64)
    result = np.zeros((len(bank), 4), np.float64)
    for asset in assets:
        result += member_scores(bank, embeddings, asset, features) / float(asset["score_scale"])
    return result / len(assets)


def predict(bank, arrays, checkpoint):
    directory = Path(checkpoint)
    paths = sorted(directory.glob("seed_*.npz"))
    if not paths:
        raise FileNotFoundError("Expected fixed seed_*.npz members")
    assets = []
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            assets.append({key: stored[key] for key in stored.files})
    return select(bank, ensemble_scores(bank, arrays["embeddings"], assets), "mean")
