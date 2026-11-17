"""
metrics/bootstrap.py
====================
Percentile bootstrap confidence intervals, resampling whole persons
(case_id) so that spells or subjects of the same person stay together.
"""

from __future__ import annotations

from typing import Callable, Sequence

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 0


def cluster_bootstrap_ci(
    clusters: Sequence[str],
    statistic: Callable[[list[int]], dict[str, float | None]],
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[float] | None]:
    """
    Parameters
    ----------
    clusters  : cluster id (case_id) for each evaluated item, in item order.
    statistic : maps a list of item indices to {metric_name: value}.
    n_boot    : number of bootstrap replicates; 0 returns {}.

    Returns
    -------
    {metric_name: [lower, upper]} at level 1 - alpha; None for a metric that
    was undefined in every replicate.
    """
    if n_boot <= 0 or len(clusters) == 0:
        return {}
    import numpy as np

    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    groups = list(by_cluster.values())

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        idx = [i for g in picks for i in groups[g]]
        for name, value in statistic(idx).items():
            draws.setdefault(name, [])
            if value is not None:
                draws[name].append(value)

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return _percentiles(draws, lo_q, hi_q)


def mean_of(values: Sequence[float | None]) -> Callable[[list[int]], float | None]:
    """Statistic: mean over the non-None values among the sampled items."""
    def stat(idx: list[int]) -> float | None:
        vals = [values[i] for i in idx if values[i] is not None]
        return sum(vals) / len(vals) if vals else None
    return stat


def classification_stats(
    gold: Sequence[str], pred: Sequence[str]
) -> Callable[[list[int]], dict[str, float | None]]:
    """
    Statistic: accuracy, macro F1 and weighted F1 on the sampled items, with
    the same label set rule as sklearn (labels seen in gold or predictions).
    """
    import numpy as np

    labels = {lab: i for i, lab in enumerate(sorted(set(gold) | set(pred)))}
    g_all = np.array([labels[x] for x in gold])
    p_all = np.array([labels[x] for x in pred])
    k = len(labels)

    def stat(idx: list[int]) -> dict[str, float | None]:
        g, p = g_all[idx], p_all[idx]
        hit = g == p
        tp = np.bincount(g[hit], minlength=k)
        support = np.bincount(g, minlength=k)
        n_pred = np.bincount(p, minlength=k)
        present = (support + n_pred) > 0
        denom = support + n_pred
        f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
        return {
            "accuracy":    float(hit.mean()),
            "macro_f1":    float(f1[present].mean()),
            "weighted_f1": float((f1 * support).sum() / support.sum()),
        }
    return stat


def _percentiles(draws: dict[str, list[float]], lo_q: float, hi_q: float):
    import numpy as np
    return {
        name: (
            [round(float(np.percentile(v, lo_q)), 6), round(float(np.percentile(v, hi_q)), 6)]
            if v else None
        )
        for name, v in draws.items()
    }
