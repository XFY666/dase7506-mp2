"""Offline answer verifier; NumPy is the only inference dependency."""
import json
from pathlib import Path

from ensemble import text_scores


def _implementation(checkpoint):
    config = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
    if config.get("format") == "mp2-integrated-verifier-v1":
        import optimized_selection
        return optimized_selection
    import ensemble
    return ensemble


def score_candidates(bank, arrays, checkpoint):
    return _implementation(checkpoint).score_candidates(bank, arrays, checkpoint)


def predict(bank, arrays, checkpoint):
    return _implementation(checkpoint).predict(bank, arrays, checkpoint)

__all__ = ["predict", "score_candidates", "text_scores"]
