from __future__ import annotations

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import auc, confusion_matrix, roc_curve

from ..xai.shap_explain import AnomalyExplanation

_COLOR_PRIMARY = "#4575A6"
_COLOR_ACCENT = "#E78A32"
_COLOR_THRESHOLD = "#C94C4C"
_COLOR_MUTED = "#7A8082"
_COLOR_AXIS = "#2E2E2E"

_CM_HEATMAP = sns.light_palette(_COLOR_PRIMARY, as_cmap=True)
_SHAP_FEATURE_CMAP = sns.diverging_palette(255, 28, s=82, l=58, as_cmap=True)


def plot_decision_score_histogram(
    decision_scores: np.ndarray,
    predicted_anomaly: np.ndarray,
    *,
    title: str = "Verteilung des Modell-Scores im Testset",
    figsize: tuple[float, float] = (8.0, 4.0),
    decision_threshold: Optional[float] = None,
) -> plt.Figure:
    s = np.asarray(decision_scores, dtype=np.float64).ravel()
    a = np.asarray(predicted_anomaly, dtype=np.int64).ravel()
    n_total = int(s.size)
    n_anom = int(np.sum(a == 1))
    share = (100.0 * n_anom / n_total) if n_total > 0 else 0.0
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        s,
        bins=80,
        alpha=0.45,
        label=f"alle Transaktionen (n={n_total:,})",
        color=_COLOR_PRIMARY,
        density=True,
    )
    if np.any(a == 1):
        ax.hist(
            s[a == 1],
            bins=40,
            alpha=0.7,
            label=f"als auffällig markiert (n={n_anom:,}, {share:.2f}%)",
            color=_COLOR_ACCENT,
            density=True,
        )
    if decision_threshold is not None:
        thr = float(decision_threshold)
        ax.axvline(
            thr,
            color=_COLOR_THRESHOLD,
            linestyle="--",
            linewidth=1.5,
            label="Schwellwert",
            zorder=5,
        )
    ax.set_xlabel("decision_function (höher → eher normal)")
    ax.set_ylabel("Dichte")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    figsize: tuple[float, float] = (5.2, 4.2),
) -> plt.Figure:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    cm = confusion_counts(y_true, y_pred)
    vals = cm.to_numpy(dtype=np.float64)
    row_sums = vals.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_pct = np.where(row_sums > 0, 100.0 * vals / row_sums, 0.0)
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(vals[i, j])}\n({row_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=_CM_HEATMAP,
        linewidths=0.5,
        linecolor="#E8ECEF",
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ax=ax,
    )
    ax.set_title("Konfusionsmatrix (1 = auffällig/Fraud, Zeilen-% je wahrer Klasse)")
    fig.tight_layout()
    return fig


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame([[tn, fp], [fn, tp]], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])


def plot_roc_figure(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    *,
    figsize: tuple[float, float] = (5.0, 4.0),
) -> plt.Figure:
    y = (np.asarray(y_true, dtype=np.int64).ravel() == 1).astype(np.int64)
    s = -np.asarray(decision_scores, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=figsize)
    if y.sum() > 0 and (1 - y).sum() > 0:
        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=_COLOR_PRIMARY,
            linewidth=2.0,
            label=f"AUC = {roc_auc:.4f}",
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color=_COLOR_MUTED,
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        label="Zufall",
    )
    ax.set_xlabel("False-Positive-Rate")
    ax.set_ylabel("True-Positive-Rate")
    ax.set_title("ROC-Kurve (Score = −decision_function)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _sign_caption(sign_convention: str) -> str:
    if sign_convention == "anomaly":
        return (
            "Vorzeichen: positiv = Beitrag zur Anomalie, negativ = Beitrag zur Normalität (invertiert)."
        )
    return (
        "Vorzeichen: positiv = Beitrag zur Normalität (höhere decision_function), negativ = Beitrag zur Anomalie."
    )


def shap_summary_figure(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_display: int = 20,
    figsize: tuple[float, float] = (10.0, 7.0),
    sign_convention: str = "normal",
) -> plt.Figure:
    plt.close("all")
    shap.summary_plot(
        shap_values,
        features=X,
        feature_names=list(feature_names),
        max_display=max_display,
        axis_color=_COLOR_AXIS,
        cmap=_SHAP_FEATURE_CMAP,
        show=False,
    )
    fig = plt.gcf()
    ax = fig.axes[0] if fig.axes else plt.gca()
    ax.set_xlabel("SHAP-Beitrag (→ auffälliger)" if sign_convention == "anomaly" else "SHAP-Beitrag (→ normaler)")
    fig.set_size_inches(*figsize)
    fig.tight_layout()
    return fig


def shap_bar_figure(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_display: int = 20,
    figsize: tuple[float, float] = (9.0, 6.0),
    sign_convention: str = "normal",
) -> plt.Figure:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]

    fig, ax = plt.subplots(figsize=figsize)
    names_sorted = [feature_names[i] for i in order]
    vals_sorted = mean_abs[order]
    ax.barh(range(len(order)), vals_sorted[::-1], color=_COLOR_PRIMARY)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names_sorted[::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Globale Feature-Wichtigkeit (mean |SHAP|)")
    ax.text(
        0.0,
        -0.12,
        _sign_caption(sign_convention),
        transform=ax.transAxes,
        fontsize=8,
        color=_COLOR_MUTED,
        ha="left",
        va="top",
    )
    fig.tight_layout()
    return fig


def shap_waterfall_figure(
    explanation: AnomalyExplanation,
    *,
    max_display: int = 20,
) -> plt.Figure:
    values = np.asarray(explanation.shap_values, dtype=np.float64).ravel()
    base = float(explanation.expected_value)
    data = np.asarray(explanation.feature_values, dtype=np.float64).ravel()
    names: Sequence[str] = explanation.feature_names
    sign_convention = str(explanation.sign_convention)

    exp = shap.Explanation(values=values, base_values=base, data=data, feature_names=list(names))
    plt.close("all")
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9.0, 6.5)
    fig.suptitle(
        "SHAP-Waterfall" + (" — Beitrag zur Anomalie" if sign_convention == "anomaly" else " — Beitrag zur Normalität"),
        fontsize=11,
        y=0.995,
    )
    fig.text(
        0.02,
        0.01,
        _sign_caption(sign_convention),
        fontsize=8,
        color=_COLOR_MUTED,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig

