from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import shap

from ..models.anomaly_detector import AnomalyDetector

_SHAP_RECON_ATOL = 1e-5
_SHAP_RECON_RTOL = 1e-4

DEFAULT_BACKGROUND_SIZE = 200


def _unwrap_sklearn_estimator(model: Union[AnomalyDetector, Any]) -> Any:
    if hasattr(model, "sklearn_estimator"):
        return model.sklearn_estimator  # type: ignore[return-value]
    return model


def _extract_shap_array(raw: Any) -> np.ndarray:
    if hasattr(raw, "values") and not isinstance(raw, np.ndarray):
        raw = raw.values  # type: ignore[assignment]
    return np.asarray(raw, dtype=np.float64)


def _ensure_2d_float(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} muss eine 2D-Matrix sein.")
    return arr


@dataclass(frozen=True)
class TopFeatureContribution:
    feature_name: str
    feature_value: float
    shap_value: float
    abs_shap: float


@dataclass(frozen=True)
class AnomalyExplanation:
    feature_names: List[str]
    feature_values: np.ndarray
    shap_values: np.ndarray
    expected_value: float
    decision_function: float
    sklearn_predict: int
    predict_anomaly: int
    top_features: List[TopFeatureContribution]
    shap_internal_output: float
    shap_reconstructed_score: float
    shap_reconstruction_abs_error: float
    shap_reconstruction_ok: bool
    sign_convention: str


def subsample_background(
    X: np.ndarray,
    size: int = DEFAULT_BACKGROUND_SIZE,
    *,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    X = _ensure_2d_float(X, name="X")
    n = X.shape[0]
    if size <= 0:
        raise ValueError(f"size muss > 0 sein, erhalten {size}.")
    if n <= size:
        return X
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=size, replace=False)
    return X[idx]


def build_explainer(
    model: Union[AnomalyDetector, Any],
    *,
    data: Optional[np.ndarray] = None,
    feature_perturbation: Optional[str] = None,
    background_size: Optional[int] = None,
    random_state: Optional[int] = 42,
) -> shap.TreeExplainer:
    est = _unwrap_sklearn_estimator(model)
    fp = feature_perturbation if feature_perturbation is not None else (
        "interventional" if data is not None else "tree_path_dependent"
    )
    if data is None:
        return shap.TreeExplainer(est, feature_perturbation=fp)

    bg = np.asarray(data, dtype=np.float64)
    if background_size is not None:
        bg = subsample_background(bg, size=int(background_size), random_state=random_state)
    return shap.TreeExplainer(est, data=bg, feature_perturbation=fp)


def explain_anomaly(
    model: Union[AnomalyDetector, Any],
    X: np.ndarray,
    index: int,
    feature_names: Optional[Sequence[str]] = None,
    *,
    explainer: Optional[shap.TreeExplainer] = None,
    top_k_features: int = 20,
    invert_sign: bool = False,
) -> AnomalyExplanation:
    X = _ensure_2d_float(X, name="X")
    if index < 0 or index >= X.shape[0]:
        raise IndexError(f"index={index} außerhalb von [0, {X.shape[0]}).")

    n_features = X.shape[1]
    if feature_names is None:
        names: List[str] = [f"x{i}" for i in range(n_features)]
    else:
        names = list(feature_names)
        if len(names) != n_features:
            raise ValueError(
                f"feature_names Länge {len(names)} passt nicht zu X mit {n_features} Spalten."
            )

    row = X[index : index + 1]
    exp = explainer if explainer is not None else build_explainer(model)
    shap_raw = exp.shap_values(row)
    shap_vec = _extract_shap_array(shap_raw).reshape(-1)

    ev = exp.expected_value
    ev_arr = np.asarray(ev, dtype=np.float64).ravel()
    base = float(ev_arr[0]) if ev_arr.size else float("nan")

    det_fn = model.decision_function(row) if isinstance(model, AnomalyDetector) else _unwrap_sklearn_estimator(model).decision_function(row)
    score = float(np.asarray(det_fn, dtype=np.float64).ravel()[0])

    sk_pred = model.predict(row) if isinstance(model, AnomalyDetector) else _unwrap_sklearn_estimator(model).predict(row)
    pred_if = int(np.asarray(sk_pred, dtype=np.int64).ravel()[0])

    shap_internal = float(np.asarray(exp.model.predict(row), dtype=np.float64).ravel()[0])
    reconstructed = float(base + float(shap_vec.sum()))
    recon_err = float(abs(reconstructed - shap_internal))
    recon_ok = bool(np.isclose(reconstructed, shap_internal, rtol=_SHAP_RECON_RTOL, atol=_SHAP_RECON_ATOL))

    if invert_sign:
        shap_vec = -shap_vec
        base = -base
        shap_internal = -shap_internal
        reconstructed = -reconstructed

    order = np.argsort(np.abs(shap_vec))[::-1]
    k = max(0, min(int(top_k_features), int(shap_vec.size)))
    top_features: List[TopFeatureContribution] = []
    for j in order[:k]:
        jj = int(j)
        sv = float(shap_vec[jj])
        top_features.append(
            TopFeatureContribution(
                feature_name=names[jj],
                feature_value=float(X[index, jj]),
                shap_value=sv,
                abs_shap=abs(sv),
            )
        )

    predict_anomaly = int(model.predict_anomaly(row)[0]) if isinstance(model, AnomalyDetector) else int(pred_if == -1)
    return AnomalyExplanation(
        feature_names=names,
        feature_values=X[index].copy(),
        shap_values=shap_vec,
        expected_value=base,
        decision_function=score,
        sklearn_predict=pred_if,
        predict_anomaly=predict_anomaly,
        top_features=top_features,
        shap_internal_output=shap_internal,
        shap_reconstructed_score=reconstructed,
        shap_reconstruction_abs_error=recon_err,
        shap_reconstruction_ok=recon_ok,
        sign_convention="anomaly" if invert_sign else "normal",
    )


def explain_batch(
    model: Union[AnomalyDetector, Any],
    X: np.ndarray,
    *,
    explainer: Optional[shap.TreeExplainer] = None,
    invert_sign: bool = False,
) -> np.ndarray:
    X = _ensure_2d_float(X, name="X")
    exp = explainer if explainer is not None else build_explainer(model)
    raw = exp.shap_values(X)
    arr = _extract_shap_array(raw)
    n, p = X.shape[0], X.shape[1]
    if arr.ndim == 1:
        if arr.size != n * p:
            raise ValueError(f"SHAP liefert 1D der Länge {arr.size}, erwartet {n * p} (= {n}×{p}).")
        arr = arr.reshape(n, p)
    elif arr.ndim == 2:
        if arr.shape != (n, p):
            raise ValueError(f"SHAP-Form {arr.shape} passt nicht zu X mit Form ({n}, {p}).")
    else:
        raise ValueError(f"Unerwartete SHAP-Ausgabe mit ndim={arr.ndim}.")
    if invert_sign:
        arr = -arr
    return arr


def rank_anomaly_indices(
    decision_scores: np.ndarray,
    predicted_anomaly: np.ndarray,
    *,
    top_k: int = 20,
) -> np.ndarray:
    s = np.asarray(decision_scores, dtype=np.float64).ravel()
    a = np.asarray(predicted_anomaly, dtype=np.int64).ravel()
    if len(s) != len(a):
        raise ValueError("decision_scores und predicted_anomaly müssen gleiche Länge haben.")
    idx = np.where(a == 1)[0]
    if idx.size == 0:
        return np.array([], dtype=np.int64)
    order = idx[np.argsort(s[idx])]
    return order[: min(top_k, len(order))]

