from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, overload

import numpy as np
import pandas as pd

from ..models.anomaly_detector import AnomalyDetector
from .shap_explain import (
    DEFAULT_BACKGROUND_SIZE,
    build_explainer,
    explain_batch,
    subsample_background,
)


def _unwrap_sklearn_estimator(model: Union[AnomalyDetector, Any]) -> Any:
    if hasattr(model, "sklearn_estimator"):
        return model.sklearn_estimator  # type: ignore[return-value]
    return model


def _decision_function(model: Union[AnomalyDetector, Any], X: np.ndarray) -> np.ndarray:
    if isinstance(model, AnomalyDetector):
        return model.decision_function(X)
    return _unwrap_sklearn_estimator(model).decision_function(X)


def _ensure_2d_float(X: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} muss 2D sein.")
    return arr


@dataclass(frozen=True)
class FaithfulnessResult:
    k: int
    n_samples: int
    n_random_repeats: int
    mean_delta_top: float
    mean_delta_random: float
    median_delta_top: float
    median_delta_random: float
    faithfulness_gap: float
    share_top_better: float


@dataclass(frozen=True)
class StabilityResult:
    k: int
    n_repeats: int
    background_size: int
    n_samples: int
    mean_jaccard: float
    median_jaccard: float
    min_jaccard: float
    per_sample_mean_jaccard: np.ndarray


def faithfulness_top_k_ablation(
    model: Union[AnomalyDetector, Any],
    X: np.ndarray,
    shap_values: np.ndarray,
    *,
    baseline: np.ndarray,
    k: int = 5,
    n_random_repeats: int = 5,
    random_state: Optional[int] = 42,
) -> FaithfulnessResult:
    X = _ensure_2d_float(X, name="X")
    sv = np.asarray(shap_values, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64).ravel()
    if sv.shape != X.shape:
        raise ValueError(f"shap_values-Form {sv.shape} passt nicht zu X {X.shape}.")
    if base.shape[0] != X.shape[1]:
        raise ValueError(f"baseline hat {base.shape[0]} Einträge, erwartet {X.shape[1]}.")
    n, p = X.shape
    k = int(max(1, min(k, p)))

    rng = np.random.RandomState(random_state)

    f_base = _decision_function(model, X).astype(np.float64)
    abs_sv = np.abs(sv)
    top_idx = np.argsort(-abs_sv, axis=1)[:, :k]

    X_ablate_top = X.copy()
    for i in range(n):
        X_ablate_top[i, top_idx[i]] = base[top_idx[i]]
    f_top = _decision_function(model, X_ablate_top).astype(np.float64)
    delta_top = f_top - f_base

    delta_random_per_row = np.zeros(n, dtype=np.float64)
    for _ in range(max(1, int(n_random_repeats))):
        X_ablate_rand = X.copy()
        for i in range(n):
            rand_idx = rng.choice(p, size=k, replace=False)
            X_ablate_rand[i, rand_idx] = base[rand_idx]
        f_rand = _decision_function(model, X_ablate_rand).astype(np.float64)
        delta_random_per_row += (f_rand - f_base)
    delta_random_per_row /= max(1, int(n_random_repeats))

    gap = float(np.mean(delta_top - delta_random_per_row))
    return FaithfulnessResult(
        k=k,
        n_samples=int(n),
        n_random_repeats=int(n_random_repeats),
        mean_delta_top=float(np.mean(delta_top)),
        mean_delta_random=float(np.mean(delta_random_per_row)),
        median_delta_top=float(np.median(delta_top)),
        median_delta_random=float(np.median(delta_random_per_row)),
        faithfulness_gap=gap,
        share_top_better=float(np.mean(delta_top > delta_random_per_row)),
    )


def _topk_set(values: np.ndarray, k: int) -> set[int]:
    order = np.argsort(-np.abs(values))
    return set(int(j) for j in order[:k])


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def stability_topk_jaccard(
    model: Union[AnomalyDetector, Any],
    X: np.ndarray,
    background_pool: np.ndarray,
    *,
    k: int = 5,
    n_repeats: int = 10,
    background_size: int = DEFAULT_BACKGROUND_SIZE,
    random_state: Optional[int] = 42,
) -> StabilityResult:
    X = _ensure_2d_float(X, name="X")
    pool = _ensure_2d_float(background_pool, name="background_pool")
    if X.shape[1] != pool.shape[1]:
        raise ValueError(
            f"Feature-Anzahl passt nicht: X hat {X.shape[1]}, background_pool hat {pool.shape[1]}."
        )
    if n_repeats < 2:
        raise ValueError("Für Stabilität werden ≥ 2 Wiederholungen benötigt.")

    n = X.shape[0]
    k = int(max(1, min(k, X.shape[1])))

    shap_runs: List[np.ndarray] = []
    seed_seq = np.random.SeedSequence(random_state)
    child_seeds = seed_seq.generate_state(n_repeats)
    for seed in child_seeds:
        bg = subsample_background(pool, size=background_size, random_state=int(seed))
        explainer = build_explainer(
            model,
            data=bg,
            feature_perturbation="interventional",
            background_size=None,
        )
        sv = explain_batch(model, X, explainer=explainer)
        shap_runs.append(sv)

    per_sample = np.zeros(n, dtype=np.float64)
    for i in range(n):
        topk_sets = [_topk_set(run[i], k) for run in shap_runs]
        pairwise = []
        for a in range(len(topk_sets)):
            for b in range(a + 1, len(topk_sets)):
                pairwise.append(_jaccard(topk_sets[a], topk_sets[b]))
        per_sample[i] = float(np.mean(pairwise)) if pairwise else 1.0

    return StabilityResult(
        k=k,
        n_repeats=int(n_repeats),
        background_size=int(background_size),
        n_samples=int(n),
        mean_jaccard=float(np.mean(per_sample)),
        median_jaccard=float(np.median(per_sample)),
        min_jaccard=float(np.min(per_sample)),
        per_sample_mean_jaccard=per_sample,
    )


def format_xai_evaluation_table(
    faithfulness: FaithfulnessResult,
    stability: Optional[StabilityResult] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"Metrik": "Faithfulness: mean Δ decision_function (Top-k)", "Wert": faithfulness.mean_delta_top},
        {"Metrik": "Faithfulness: mean Δ decision_function (Random-k)", "Wert": faithfulness.mean_delta_random},
        {"Metrik": "Faithfulness Gap (Top-k − Random-k, höhere Werte günstig)", "Wert": faithfulness.faithfulness_gap},
        {"Metrik": "Faithfulness: Anteil Zeilen mit Top > Random", "Wert": faithfulness.share_top_better},
        {"Metrik": "Faithfulness: k", "Wert": float(faithfulness.k)},
        {"Metrik": "Faithfulness: n evaluierte Zeilen", "Wert": float(faithfulness.n_samples)},
    ]
    if stability is not None:
        rows.extend(
            [
                {"Metrik": "Stability: mean Jaccard (Top-k über Runs, höhere Werte günstig)", "Wert": stability.mean_jaccard},
                {"Metrik": "Stability: median Jaccard", "Wert": stability.median_jaccard},
                {"Metrik": "Stability: min Jaccard", "Wert": stability.min_jaccard},
                {"Metrik": "Stability: k", "Wert": float(stability.k)},
                {"Metrik": "Stability: Background-Größe", "Wert": float(stability.background_size)},
                {"Metrik": "Stability: n Wiederholungen", "Wert": float(stability.n_repeats)},
            ]
        )
    return pd.DataFrame(rows)

