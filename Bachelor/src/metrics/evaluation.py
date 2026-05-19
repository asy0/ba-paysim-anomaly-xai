from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve

from ..models.anomaly_detector import AnomalyDetector


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    support_fraud: int
    tn: int
    fp: int
    fn: int
    tp: int


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true und y_pred müssen gleiche Länge haben.")

    prec, rec, f1, _sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    roc_auc_val = float("nan")

    if scores is not None:
        s = np.asarray(scores, dtype=np.float64).ravel()
        if len(s) != len(y_true):
            raise ValueError("scores muss dieselbe Länge wie y_true haben.")
        y_bin = (y_true == 1).astype(np.int64)
        if y_bin.sum() > 0 and (1 - y_bin).sum() > 0:
            fpr, tpr, _ = roc_curve(y_bin, -s)
            roc_auc_val = float(auc(fpr, tpr))

    return ClassificationMetrics(
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        roc_auc=float(roc_auc_val),
        support_fraud=int(np.sum(y_true == 1)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def sensitivity_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination_values: Sequence[float],
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for c in contamination_values:
        det = AnomalyDetector(
            contamination=float(c),
            n_estimators=n_estimators,
            random_state=random_state,
        )
        det.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
        X_te = np.asarray(X_test, dtype=np.float64)
        y_te = np.asarray(y_test, dtype=np.int64)
        y_pred = det.predict_anomaly(X_te)
        scores = det.decision_function(X_te)
        m = evaluate_model(y_te, y_pred, scores=scores)
        rows.append({"contamination": float(c), **m.__dict__})
    return pd.DataFrame(rows)


def format_metrics_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    metrics_map: Mapping[str, Any]
    if isinstance(metrics, ClassificationMetrics):
        metrics_map = metrics.__dict__
    else:
        metrics_map = metrics
    keys = ["precision", "recall", "f1", "roc_auc", "tp", "fp", "fn", "tn"]
    row = {k: metrics_map.get(k, np.nan) for k in keys}
    df = pd.DataFrame([row])
    for col in ["precision", "recall", "f1", "roc_auc"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: f"{v:.4f}" if not pd.isna(v) else "—")
    return df

