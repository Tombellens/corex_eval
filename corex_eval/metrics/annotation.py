"""
metrics/annotation.py
=====================
Evaluation metrics for the data annotation task.

The annotation task is a multi-class classification problem: given a
raw extracted string (from the silver standard), predict the correct
codebook category.

Metrics
-------
- Exact-match accuracy
- Macro-averaged F1
- Weighted F1
- Per-class precision, recall, F1, support
- Optional: mean cosine similarity between predicted and gold label
  embeddings using a sentence-transformers BERT model. This captures
  whether wrong predictions are at least semantically close to the
  correct category.

Input contract
--------------
Receives plain Python lists — no pandas. The caller (evaluate.py)
handles dataframe alignment before calling annotation_metrics().

BERT similarity is opt-in (semantic_similarity=False by default)
and requires the [embeddings] optional dependency group.
"""

from __future__ import annotations

import warnings


def annotation_metrics(
    predictions: list[str],
    gold: list[str],
    semantic_similarity: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Compute annotation classification metrics.

    Parameters
    ----------
    predictions         : List of predicted codebook labels.
    gold                : List of gold codebook labels, aligned with predictions.
    semantic_similarity : If True, also compute mean cosine similarity between
                          predicted and gold label embeddings. Requires
                          sentence-transformers to be installed.
    embedding_model     : Sentence-transformers model name. Only used when
                          semantic_similarity=True.

    Returns
    -------
    {
        "accuracy":     float,
        "macro_f1":     float,
        "weighted_f1":  float,
        "per_class": {
            "<label>": {
                "precision": float,
                "recall":    float,
                "f1":        float,
                "support":   int,
            },
            ...
        },
        "semantic_similarity": float | None,
        "n_total":   int,
        "n_classes": int,
    }
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"predictions and gold must be the same length "
            f"(got {len(predictions)} vs {len(gold)})"
        )

    if len(predictions) == 0:
        return {
            "accuracy":            None,
            "macro_f1":            None,
            "weighted_f1":         None,
            "per_class":           {},
            "semantic_similarity": None,
            "n_total":             0,
            "n_classes":           0,
        }

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    accuracy    = accuracy_score(gold, predictions)
    macro_f1    = f1_score(gold, predictions, average="macro",    zero_division=0)
    weighted_f1 = f1_score(gold, predictions, average="weighted", zero_division=0)

    report = classification_report(
        gold,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # Extract per-class metrics, skipping sklearn's aggregate rows
    _skip = {"accuracy", "macro avg", "weighted avg"}
    per_class = {
        label: {
            "precision": round(vals["precision"], 6),
            "recall":    round(vals["recall"],    6),
            "f1":        round(vals["f1-score"],  6),
            "support":   int(vals["support"]),
        }
        for label, vals in report.items()
        if label not in _skip and isinstance(vals, dict)
    }

    sem_sim = None
    if semantic_similarity:
        sem_sim = _compute_semantic_similarity(
            predictions, gold, embedding_model
        )

    return {
        "accuracy":            round(accuracy,    6),
        "macro_f1":            round(macro_f1,    6),
        "weighted_f1":         round(weighted_f1, 6),
        "per_class":           per_class,
        "semantic_similarity": round(sem_sim, 6) if sem_sim is not None else None,
        "n_total":             len(predictions),
        "n_classes":           len(per_class),
    }


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

# Module-level cache so the model is only loaded once per process
_embedder_cache: dict[str, object] = {}


def _compute_semantic_similarity(
    predictions: list[str],
    gold: list[str],
    model_name: str,
) -> float:
    """
    Mean pairwise cosine similarity between predicted and gold label embeddings.

    The model is loaded once and cached for the lifetime of the process.
    Embeddings are deduplicated — if the same label string appears multiple
    times, it is only encoded once.

    Parameters
    ----------
    predictions : List of predicted label strings.
    gold        : List of gold label strings, aligned with predictions.
    model_name  : Sentence-transformers model identifier.

    Returns
    -------
    Mean cosine similarity in [0, 1] (typically; bounded by the model's
    embedding space).

    Raises
    ------
    ImportError if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Semantic similarity requires the [embeddings] extra:\n"
            "    pip install corex_eval[embeddings]\n"
            "or: pip install sentence-transformers"
        ) from exc

    # Load and cache the model
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    embedder = _embedder_cache[model_name]

    # Encode unique strings only to avoid redundant computation
    unique_labels = list(set(predictions + gold))
    embeddings    = embedder.encode(
        unique_labels,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit vectors → dot product = cosine sim
    )
    emb_map = {label: emb for label, emb in zip(unique_labels, embeddings)}

    # Compute pairwise cosine similarities
    sims = []
    for pred, g in zip(predictions, gold):
        ep, eg = emb_map[pred], emb_map[g]
        # Since embeddings are normalised, dot product = cosine similarity
        sims.append(float(np.dot(ep, eg)))

    return float(np.mean(sims))