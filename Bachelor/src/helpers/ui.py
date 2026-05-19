from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FeaturePercentiles:
    p05: float
    p50: float
    p95: float


def inverse_scaled_value(
    *,
    feature: str,
    scaled_value: float,
    scaler,
    scaled_cols: Sequence[str],
) -> float:
    if scaler is None or feature not in scaled_cols:
        return float(scaled_value)
    j = scaled_cols.index(feature)
    return float(scaled_value) * float(scaler.scale_[j]) + float(scaler.mean_[j])


def compute_percentiles_original_scale(
    *,
    X_train: np.ndarray,
    scaler,
    idx_by_name: Mapping[str, int],
    scaled_cols: Sequence[str],
    percentiles: Iterable[int] = (5, 50, 95),
) -> Dict[str, FeaturePercentiles]:
    X_tr = np.asarray(X_train, dtype=np.float64)
    ps = list(percentiles)
    out: Dict[str, FeaturePercentiles] = {}
    for f in scaled_cols:
        j = idx_by_name[f]
        vals_scaled = X_tr[:, j]
        vals = np.array(
            [
                inverse_scaled_value(
                    feature=f,
                    scaled_value=float(v),
                    scaler=scaler,
                    scaled_cols=scaled_cols,
                )
                for v in vals_scaled
            ],
            dtype=np.float64,
        )
        p05, p50, p95 = (float(np.percentile(vals, p)) for p in ps)
        out[f] = FeaturePercentiles(p05=p05, p50=p50, p95=p95)
    return out


def _level_from_percentiles(value: float, p: FeaturePercentiles) -> str:
    if value >= p.p95:
        return "ungewöhnlich hoch"
    if value <= p.p05:
        return "ungewöhnlich niedrig"
    return "auffällig im Zusammenspiel mit anderen Merkmalen"


def _format_num(x: float) -> str:
    ax = abs(float(x))
    if ax >= 1_000_000:
        return f"{x:,.0f}"
    if ax >= 10_000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.1f}"
    if ax >= 1:
        return f"{x:,.2f}"
    return f"{x:.4f}"


def _in_usual_range(value: float, p: FeaturePercentiles) -> bool:
    return p.p05 <= value <= p.p95


def reason_sentence(
    *,
    feature_display: str,
    feature_raw: str,
    value_orig: float,
    percentiles: Mapping[str, FeaturePercentiles],
) -> str:
    p = percentiles.get(feature_raw)
    if p is None:
        if feature_raw.startswith("type_") and value_orig >= 0.5:
            return f"Transaktionstyp: {feature_display.replace('Typ: ', '')}."
        return "Ungewöhnlicher Beitrag im Modellvergleich."

    level = _level_from_percentiles(value_orig, p)

    if feature_raw == "amount":
        return f"Der Betrag ist {level} ({_format_num(value_orig)})."
    if feature_raw == "balance_error_orig":
        return (
            f"Die Sender‑Bilanzabweichung ist {level} ({_format_num(value_orig)}). "
            "Bei plausiblen Transaktionen liegt dieser Wert meist nahe 0."
        )
    if feature_raw in ("oldbalanceOrg", "newbalanceOrig"):
        return f"Der Sender‑Kontostand ist {level} ({_format_num(value_orig)})."
    if feature_raw in ("oldbalanceDest", "newbalanceDest"):
        return f"Der Empfänger‑Kontostand ist {level} ({_format_num(value_orig)})."
    if feature_raw == "balance_delta_orig":
        return f"Die Änderung beim Sender‑Kontostand ist {level} ({_format_num(value_orig)})."
    if feature_raw == "balance_delta_dest":
        return f"Die Änderung beim Empfänger‑Kontostand ist {level} ({_format_num(value_orig)})."
    if feature_raw == "step":
        return f"Der Zeitpunkt ist {level} (Stunde {int(round(value_orig))})."
    return f"{feature_display} ist {level} ({_format_num(value_orig)})."


def positive_sentence(
    *,
    feature_display: str,
    feature_raw: str,
    value_orig: float,
    percentiles: Mapping[str, FeaturePercentiles],
) -> str:
    p = percentiles.get(feature_raw)
    hint = " (im üblichen Bereich)" if (p is not None and _in_usual_range(value_orig, p)) else ""

    if feature_raw.startswith("type_") and value_orig >= 0.5:
        return f"Der Transaktionstyp {feature_display.replace('Typ: ', '')} ist hier eher unauffällig."
    if feature_raw == "amount":
        return f"Der Betrag wirkt eher unauffällig ({_format_num(value_orig)}){hint}."
    if feature_raw == "balance_error_orig":
        return f"Die Sender‑Bilanzabweichung wirkt unauffällig ({_format_num(value_orig)}){hint}."
    if feature_raw in ("oldbalanceOrg", "newbalanceOrig"):
        return f"Der Sender‑Kontostand wirkt unauffällig ({_format_num(value_orig)}){hint}."
    if feature_raw in ("oldbalanceDest", "newbalanceDest"):
        return f"Der Empfänger‑Kontostand wirkt unauffällig ({_format_num(value_orig)}){hint}."
    if feature_raw == "balance_delta_orig":
        return f"Die Sender‑Kontostandsänderung wirkt unauffällig ({_format_num(value_orig)}){hint}."
    if feature_raw == "balance_delta_dest":
        return f"Die Empfänger‑Kontostandsänderung wirkt unauffällig ({_format_num(value_orig)}){hint}."
    if feature_raw == "step":
        return f"Der Zeitpunkt wirkt unauffällig (Stunde {int(round(value_orig))}){hint}."
    return f"{feature_display} wirkt eher unauffällig ({_format_num(value_orig)}){hint}."


def interpret_contribution_row(
    *,
    shap_value: float,
    feature_display: str,
    feature_raw: str,
    value_orig: float,
    percentiles: Mapping[str, FeaturePercentiles],
    threshold: float = 0.001,
) -> str:
    if shap_value < -abs(threshold):
        return reason_sentence(
            feature_display=feature_display,
            feature_raw=feature_raw,
            value_orig=value_orig,
            percentiles=percentiles,
        )
    if shap_value > abs(threshold):
        return positive_sentence(
            feature_display=feature_display,
            feature_raw=feature_raw,
            value_orig=value_orig,
            percentiles=percentiles,
        )
    return "Geringer Einfluss."

