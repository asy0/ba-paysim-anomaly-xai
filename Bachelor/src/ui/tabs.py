from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ..metrics.evaluation import format_metrics_table, sensitivity_analysis
from ..models.anomaly_detector import AnomalyDetector
from ..data.preprocessing import readable_names
from ..viz.plots import (
    plot_confusion_matrix_figure,
    plot_decision_score_histogram,
    plot_roc_figure,
    shap_bar_figure,
    shap_summary_figure,
    shap_waterfall_figure,
)
from ..xai.shap_explain import explain_anomaly, explain_batch, rank_anomaly_indices
from ..helpers.ui import FeaturePercentiles, compute_percentiles_original_scale, interpret_contribution_row, inverse_scaled_value
from .xai import render_xai_section


def render_tab_analysis() -> None:
    st.subheader("Datensatz & Training")
    if not st.session_state.trained:
        st.info("Links Datei/Parameter setzen und dann **trainieren** oder **laden**.")
        return

    st.markdown(
        "PaySim simuliert mobile Finanztransaktionen. Die Merkmale sind gut interpretierbar (z. B. Betrag, Kontostände, Typ) – das hilft bei den Erklärungen."
    )

    df0 = st.session_state.get("df_meta")
    if df0 is not None:
        n_raw = st.session_state.get("n_rows_raw", len(df0))
        st.write(
            f"**Rohdaten (Datei):** {n_raw:,} Transaktionen | "
            "für Training/Tests wird daraus eine Stichprobe gezogen."
        )

    col_tr, col_te = st.columns(2)
    with col_tr:
        st.metric(
            "Training",
            f"{st.session_state.X_train.shape[0]:,} Transaktionen",
            f"{st.session_state.X_train.shape[1]} Merkmale",
        )
    with col_te:
        st.metric(
            "Test",
            f"{st.session_state.X_test.shape[0]:,} Transaktionen",
            f"{st.session_state.X_test.shape[1]} Merkmale",
        )

    det: AnomalyDetector = st.session_state.detector
    y_te = st.session_state.y_test
    pred_te = st.session_state.pred_test
    n_test = len(y_te)
    n_fraud_true = int(np.sum(y_te == 1))
    n_pred = int(np.sum(pred_te == 1))
    tp_among_pred = int(np.sum((pred_te == 1) & (y_te == 1)))
    fp_among_pred = int(np.sum((pred_te == 1) & (y_te == 0)))

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("contamination", f"{det.contamination_used:.6f}")
    with col_b:
        st.metric("Auffällig erkannt", f"{n_pred:,}")
    with col_c:
        st.metric(
            "Tatsächlicher Betrug im Testset",
            f"{n_fraud_true:,}",
            f"{100.0 * n_fraud_true / max(n_test, 1):.4f} %",
        )

    if n_pred > 0:
        st.info(
            f"Von **{n_pred}** als auffällig markierten Transaktionen sind **{tp_among_pred}** echte Betrugsfälle "
            f"und **{fp_among_pred}** Fehlalarme. Details stehen unter **Evaluierung**."
        )

    fig_h = plot_decision_score_histogram(
        st.session_state.scores_test,
        st.session_state.pred_test,
        decision_threshold=0.0,
    )
    st.pyplot(fig_h)
    plt.close(fig_h)


def _get_percentiles_original_cached(*, scaler, scaled_cols, idx_by_name) -> Dict[str, FeaturePercentiles]:
    cache_key = "feature_stats_original_v2"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    perc = compute_percentiles_original_scale(
        X_train=np.asarray(st.session_state.X_train, dtype=np.float64),
        scaler=scaler,
        idx_by_name=idx_by_name,
        scaled_cols=scaled_cols,
    )
    st.session_state[cache_key] = perc
    return perc


def render_tab_anomalies(*, scale_cols: list[str]) -> None:
    st.subheader("Auffällige Transaktionen & Erklärungen")
    if not st.session_state.trained:
        st.info("Bitte zuerst links in der Sidebar ein Modell trainieren oder laden.")
        return

    st.markdown(
        "Hier stehen die auffälligen Transaktionen – plus Erklärung, warum das Modell sie so bewertet (SHAP)."
    )
    st.caption(
        "SHAP erklärt die Modellentscheidung (nicht die echte Ursache). Das Fraud‑Label wird nur für die Auswertung genutzt."
    )

    X_test = st.session_state.X_test
    scores = st.session_state.scores_test
    pred = st.session_state.pred_test
    feature_names = st.session_state.feature_names
    display = st.session_state.display_names
    det_shap = st.session_state.detector
    explainer = st.session_state.explainer

    top_idx = rank_anomaly_indices(scores, pred, top_k=20)
    if top_idx.size == 0:
        st.warning("Im Testset wurden keine Transaktionen als auffällig markiert.")
        return

    st.markdown("### Welche Merkmale sind insgesamt am wichtigsten?")
    st.markdown("Der Plot zeigt die globale Wichtigkeit (mean |SHAP|) über die erkannten Auffälligkeiten.")

    anomaly_mask = pred == 1
    X_anomalies = X_test[anomaly_mask]
    if X_anomalies.shape[0] > 500:
        rng = np.random.RandomState(42)
        idx_sample = rng.choice(X_anomalies.shape[0], 500, replace=False)
        X_sample = X_anomalies[idx_sample]
    else:
        X_sample = X_anomalies

    shap_vals_global = explain_batch(det_shap, X_sample, explainer=explainer)

    fig_bar = shap_bar_figure(shap_vals_global, display, max_display=15)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    with st.expander("Details anzeigen (Beeswarm-Plot)"):
        st.markdown(
            "Jeder Punkt ist eine Transaktion. Farbe = Merkmalswert. x‑Achse = SHAP‑Beitrag (rechts normaler, links auffälliger)."
        )
        fig_summ = shap_summary_figure(shap_vals_global, X_sample, display, max_display=15)
        st.pyplot(fig_summ)
        plt.close(fig_summ)

    st.divider()
    st.markdown("### Einzelfall-Erklärung")
    st.markdown("Transaktion auswählen und die wichtigsten Treiber ansehen.")

    rows = []
    for i in top_idx:
        rows.append(
            {
                "Rang": int(np.where(top_idx == i)[0][0]) + 1,
                "Test-Index": int(i),
                "Anomalie-Score": f"{float(scores[i]):.4f}",
                "Tatsächlich Betrug": "Ja" if int(st.session_state.y_test[i]) == 1 else "Nein",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")

    choice = st.selectbox(
        "Transaktion auswählen",
        options=list(range(len(top_idx))),
        format_func=lambda j: (
            f"Rang {j+1}: Transaktion {int(top_idx[j])}, "
            f"Score={float(scores[top_idx[j]]):.4f}"
        ),
    )
    idx = int(top_idx[int(choice)])
    ex = explain_anomaly(det_shap, X_test, idx, display, explainer=explainer)

    st.markdown("---")
    st.markdown(f"**Transaktion {idx}** — Detailansicht")

    pred_label = "Auffällig" if ex.predict_anomaly == 1 else "Normal"
    true_label = "Betrug" if int(st.session_state.y_test[idx]) == 1 else "Kein Betrug"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modell-Einschätzung", pred_label)
    with col2:
        st.metric("Anomalie-Score", f"{ex.decision_function:.4f}")
    with col3:
        st.metric("Tatsächliches Label", true_label)

    st.markdown("**Stärkste Einflussfaktoren:**")
    st.markdown(
        "Die Tabelle zeigt die Merkmale mit dem größten Einfluss. **Positive SHAP‑Werte** schieben Richtung **normal**, **negative** Richtung **auffällig**."
    )

    idx_by_name = {n: i for i, n in enumerate(feature_names)}
    scaled_cols = [c for c in scale_cols if c in idx_by_name]
    scaler = st.session_state.get("scaler")
    percentiles_orig = _get_percentiles_original_cached(
        scaler=scaler,
        scaled_cols=scaled_cols,
        idx_by_name=idx_by_name,
    )

    contrib = (
        pd.DataFrame([t.__dict__ for t in ex.top_features])
        .rename(columns={"feature_name": "Merkmal", "shap_value": "Einfluss (SHAP)", "feature_value": "Wert"})
        .drop(columns=["abs_shap"], errors="ignore")
    )

    display_to_raw = dict(zip(readable_names(feature_names), feature_names))

    def _raw_name_from_display(d: str) -> str:
        return display_to_raw.get(d, d)

    contrib["_feature_raw"] = contrib["Merkmal"].astype(str).apply(_raw_name_from_display)
    contrib["Wert (Original)"] = contrib.apply(
        lambda r: inverse_scaled_value(
            feature=str(r["_feature_raw"]),
            scaled_value=float(r["Wert"]),
            scaler=scaler,
            scaled_cols=scaled_cols,
        ),
        axis=1,
    )

    def _interpret_row(row: pd.Series) -> str:
        return interpret_contribution_row(
            shap_value=float(row["Einfluss (SHAP)"]),
            feature_display=str(row["Merkmal"]),
            feature_raw=str(row["_feature_raw"]),
            value_orig=float(row["Wert (Original)"]),
            percentiles=percentiles_orig,
            threshold=0.001,
        )

    contrib["Interpretation"] = contrib.apply(_interpret_row, axis=1)
    st.dataframe(
        contrib[["Merkmal", "Wert (Original)", "Einfluss (SHAP)", "Interpretation"]].head(10),
        width="stretch",
        height=380,
    )

    with st.expander("Waterfall-Diagramm anzeigen"):
        st.markdown("Waterfall: Beitrag der Merkmale vom Baseline‑Wert zum internen Modell‑Score.")
        st.caption(
            "Skalen-Hinweis: Das Waterfall erklärt den internen Modell‑Output `f(x)` "
            "(additiv: `E[f] + Σ SHAP ≈ f(x)`). Die `decision_function` hat zwar die **gleiche Richtung** "
            "(höher = normaler), liegt aber auf **anderer Skala**."
        )
        fig_w = shap_waterfall_figure(ex, max_display=15)
        st.pyplot(fig_w)
        plt.close(fig_w)

    with st.expander("Technische Details (Additivität & interner Output)"):
        ok = ex.shap_reconstruction_ok
        st.markdown(
            f"Additivität für den internen Output: "
            f"interner Output **{ex.shap_internal_output:.6f}**, "
            f"Rekonstruktion **{ex.shap_reconstructed_score:.6f}**, "
            f"Abweichung **{ex.shap_reconstruction_abs_error:.2e}** "
            f"({'OK' if ok else 'Prüfen'})."
        )


def render_tab_evaluation(*, n_estimators: int, random_state: int) -> None:
    st.subheader("Metriken & Sensitivität")
    if not st.session_state.trained:
        st.info("Bitte zuerst ein Modell trainieren oder laden.")
        return

    st.markdown(
        "Metriken auf dem Testset. **Precision** = Anteil echter Treffer unter den Meldungen, **Recall** = Anteil gefundener Betrugsfälle."
    )

    _det_ev: AnomalyDetector = st.session_state.detector
    m = st.session_state.metrics

    st.markdown("### Klassifikationsmetriken (Betrug = positive Klasse)")
    st.dataframe(format_metrics_table(m), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig_cm = plot_confusion_matrix_figure(st.session_state.y_test, st.session_state.pred_test)
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with c2:
        fig_r = plot_roc_figure(st.session_state.y_test, st.session_state.scores_test)
        st.pyplot(fig_r)
        plt.close(fig_r)

    st.divider()
    st.markdown("### Sensitivität: verschiedene contamination-Werte")
    st.caption(
        "`contamination` steuert den Schwellwert. Hier sieht man, wie sich Treffer und Fehlalarme verändern."
    )
    default_cs = sorted(
        {
            round(0.0013, 6),
            0.001,
            0.005,
            0.01,
            0.02,
            float(min(0.05, _det_ev.contamination_used * 3)),
        }
    )
    contam_text = st.text_input("contamination-Liste (Komma-getrennt)", value=", ".join(str(c) for c in default_cs))
    if st.button("Sensitivität berechnen"):
        try:
            vals = [float(x.strip()) for x in contam_text.split(",") if x.strip()]
        except ValueError:
            st.error("Ungültiges Format. Beispiel: `0.001, 0.01, 0.02`.")
        else:
            with st.spinner("Mehrere Modelle trainieren …"):
                sens = sensitivity_analysis(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    st.session_state.X_test,
                    st.session_state.y_test,
                    vals,
                    n_estimators=int(n_estimators),
                    random_state=int(random_state),
                )
            st.dataframe(sens, width="stretch")

    render_xai_section()

