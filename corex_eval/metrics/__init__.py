"""
metrics/
========
Metric computation modules for each pipeline task.

Each submodule is self-contained and operates on plain Python types
(lists, dicts, sets) — no pandas, no file I/O. This makes them easy
to test in isolation.

    collection.py  → precision, recall, F1 over retrieved URL sets
    extraction.py  → field accuracy, MAE, record-level P/R/F1
    annotation.py  → accuracy, macro/weighted F1, per-class F1,
                     optional BERT cosine similarity

Location in project:
    corex_eval/
    └── corex_eval/
        └── metrics/
            ├── __init__.py    ← this file
            ├── collection.py
            ├── extraction.py
            └── annotation.py
"""

from corex_eval.metrics.collection import collection_metrics
from corex_eval.metrics.extraction import (
    atomic_accuracy,
    atomic_mae,
    composite_metrics,
)
from corex_eval.metrics.annotation import annotation_metrics

__all__ = [
    "collection_metrics",
    "atomic_accuracy",
    "atomic_mae",
    "composite_metrics",
    "annotation_metrics",
]