from __future__ import annotations

import numpy as np
import streamlit as st

from ..xai.shap_explain import explain_batch, subsample_background
from ..xai.quality_metrics import (
    FaithfulnessResult,
    StabilityResult,
    faithfulness_top_k_ablation,
    format_xai_evaluation_table,
    stability_topk_jaccard,
)


def render_xai_section() -> None:
    st.divider()
    st.markdown("### XAI-Qualitätsevaluation (optional)")
    st.markdown(
        "Hier werden die **Erklärungen selbst** bewertet (unabhängig von Precision/Recall).\n\n"
        "- **Faithfulness**: Wenn man die Top‑k‑Merkmale (laut SHAP) „neutralisiert“, sollte sich der Score stärker "
        "Richtung **normal** verschieben als bei zufälligen Merkmalen.\n"
        "- **Stability**: Bleiben die Top‑k‑Merkmale ähnlich, wenn man den SHAP‑Hintergrund variiert "
        "(Jaccard‑Index)."
    )
    st.caption(
        "Das kann je nach Datenmenge 30–120 Sekunden dauern (mehrere SHAP‑Läufe). Ergebnisse werden zwischengespeichert."
    )

    with st.expander("Parameter der XAI-Evaluation"):
        xai_k = st.number_input(
            "k (Top-k Features für Ablation / Jaccard)",
            min_value=1,
            max_value=10,
            value=5,
            help="Anzahl der Features, die für Faithfulness abgetauscht bzw. für Stability verglichen werden.",
        )
        xai_n_samples = st.number_input(
            "Anzahl ausgewerteter Anomalien",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Wie viele vorhergesagte Anomalien für die Evaluation verwendet werden.",
        )
        xai_n_repeats = st.number_input(
            "Stability: Anzahl Wiederholungen",
            min_value=2,
            max_value=20,
            value=5,
            step=1,
            help="Mehr Wiederholungen = robusterer Jaccard-Wert, aber längere Laufzeit.",
        )
        xai_bg_size = st.number_input(
            "Stability: Hintergrund-Stichprobengröße",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Größe jeder Hintergrundsam­ple für den interventionellen SHAP-Modus.",
        )

    if st.button("XAI-Evaluation berechnen", type="primary"):
        pred_te = st.session_state.pred_test
        X_test_ev = st.session_state.X_test
        scores_ev = st.session_state.scores_test
        X_train_ev = st.session_state.X_train
        det_xai = st.session_state.detector
        exp_xai = st.session_state.explainer

        anomaly_mask = pred_te == 1
        X_anom = X_test_ev[anomaly_mask]

        if X_anom.shape[0] == 0:
            st.warning(
                "Im Testset gibt es keine vorhergesagten Anomalien. Stattdessen werden die Transaktionen mit den schlechtesten Scores verwendet."
            )
            worst = np.argsort(scores_ev)[: int(xai_n_samples)]
            X_eval = X_test_ev[worst]
        else:
            X_eval = X_anom

        if X_eval.shape[0] > int(xai_n_samples):
            rng_xai = np.random.RandomState(42)
            idx_ev = rng_xai.choice(X_eval.shape[0], int(xai_n_samples), replace=False)
            X_eval = X_eval[idx_ev]

        with st.spinner("Faithfulness berechnen …"):
            sv_eval = explain_batch(det_xai, X_eval, explainer=exp_xai)
            baseline_mean = X_train_ev.mean(axis=0)
            faith = faithfulness_top_k_ablation(
                det_xai,
                X_eval,
                sv_eval,
                baseline=baseline_mean,
                k=int(xai_k),
                n_random_repeats=5,
                random_state=42,
            )

        stab_samples = min(X_eval.shape[0], 50)
        X_stab = X_eval[:stab_samples]
        bg_pool = subsample_background(
            X_train_ev,
            size=max(int(xai_bg_size) * 10, 2000),
            random_state=42,
        )
        with st.spinner(f"Stability berechnen ({int(xai_n_repeats)} Wiederholungen) …"):
            stab = stability_topk_jaccard(
                det_xai,
                X_stab,
                background_pool=bg_pool,
                k=int(xai_k),
                n_repeats=int(xai_n_repeats),
                background_size=int(xai_bg_size),
                random_state=42,
            )

        st.session_state["xai_eval_faith"] = faith
        st.session_state["xai_eval_stab"] = stab

    if "xai_eval_faith" in st.session_state:
        faith: FaithfulnessResult = st.session_state["xai_eval_faith"]
        stab: StabilityResult = st.session_state["xai_eval_stab"]

        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            gap = faith.faithfulness_gap
            st.metric(
                "Faithfulness Gap (höhere Werte günstig)",
                f"{gap:+.4f}",
                help="Differenz: Δ score bei Top-k-Ablation minus Δ score bei Zufalls-Ablation. "
                "Positiv = SHAP identifiziert echte Anomalietreiber.",
            )
        with c_f2:
            st.metric(
                "Anteil Top > Random",
                f"{faith.share_top_better:.1%}",
                help="Anteil der Zeilen, bei denen Top-k-Ablation den Score stärker verschiebt.",
            )
        with c_f3:
            st.metric(
                "Stability (mean Jaccard, höhere Werte günstig)",
                f"{stab.mean_jaccard:.3f}",
                help="Mittlerer Jaccard-Index der Top-k-Feature-Mengen über verschiedene "
                "Hintergrundsam­ples. 1,0 = vollständig stabil.",
            )

        st.markdown("**Interpretation:**")
        gap_val = faith.faithfulness_gap
        jac_val = stab.mean_jaccard

        if gap_val > 0:
            faith_txt = (
                f"**Faithfulness Gap positiv ({gap_val:+.4f})**: Die Top‑k‑Merkmale sind konsistent mit der Score‑Änderung."
            )
        else:
            faith_txt = (
                f"**Faithfulness Gap negativ ({gap_val:+.4f})**: Die Erklärungen sollten kritisch geprüft werden."
            )

        if jac_val >= 0.8:
            stab_txt = f"**Stability**: Jaccard **{jac_val:.3f}** (sehr stabil)."
        elif jac_val >= 0.5:
            stab_txt = f"**Stability**: Jaccard **{jac_val:.3f}** (mäßig stabil)."
        else:
            stab_txt = f"**Stability**: Jaccard **{jac_val:.3f}** (stark variierend)."

        st.markdown(faith_txt)
        st.markdown(stab_txt)

        with st.expander("Vollständige XAI-Evaluationstabelle"):
            st.dataframe(format_xai_evaluation_table(faith, stab), width="stretch")

