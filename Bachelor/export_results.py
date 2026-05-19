"""
Exportiert alle Ergebnisse der Pipeline als Dateien:
- Metriken-Tabellen (CSV)
- Sensitivitätsanalyse (CSV)
- XAI-Qualität: Faithfulness + Stability (CSV)
- Plots (PNG): Histogramm, Konfusionsmatrix, ROC, SHAP-Waterfalls
- Datenstatistiken (TXT)

Aufruf:
    python export_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import load_data, prepare_data, readable_names
from src.models.anomaly_detector import AnomalyDetector
from src.metrics.evaluation import evaluate_model, sensitivity_analysis
from src.xai.shap_explain import (
    DEFAULT_BACKGROUND_SIZE,
    AnomalyExplanation,
    build_explainer,
    explain_anomaly,
    explain_batch,
    rank_anomaly_indices,
    subsample_background,
)
from src.xai.quality_metrics import (
    FaithfulnessResult,
    StabilityResult,
    faithfulness_top_k_ablation,
    format_xai_evaluation_table,
    stability_topk_jaccard,
)
from src.viz.plots import (
    plot_confusion_matrix_figure,
    plot_decision_score_histogram,
    plot_roc_figure,
    shap_bar_figure,
    shap_summary_figure,
    shap_waterfall_figure,
)

OUT_DIR = Path("results")
PLOT_DIR = OUT_DIR / "plots"
TABLE_DIR = OUT_DIR / "tables"
MODEL_DIR = Path("models")

# Dateinamen-Konvention für die Arbeit:
# - Plots: abb_<nummer>_<kurztitel>.png
# - Tabellen: tab_<nummer>_<kurztitel>.csv
# - Sonstiges: txt_<kurztitel>.txt

ABB = {
    "score_hist": "abb_01_score_verteilung_schwelle.png",
    "confusion": "abb_02_konfusionsmatrix.png",
    "roc": "abb_03_roc_kurve_auc.png",
    "shap_global": "abb_04_shap_globale_importance.png",
    "shap_beeswarm": "abb_05_shap_beeswarm.png",
    "waterfall_tpl": "abb_06_shap_waterfall_rang{rank}.png",
}

TAB = {
    "metrics_main": "tab_01_metriken_hauptmodell.csv",
    "sensitivity": "tab_02_sensitivitaet_contamination.csv",
    "top_anom": "tab_03_top_anomalien_shap.csv",
    "xai_eval": "tab_04_xai_evaluation.csv",
}

# Feste Lauf-Parameter (entsprechen den Werten in der Arbeit / Reproduzierbarkeit)
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_ROWS = 200_000

BACKGROUND_SIZE = DEFAULT_BACKGROUND_SIZE

FAITHFULNESS_K = 5
FAITHFULNESS_SAMPLES = 200
FAITHFULNESS_RANDOM_REPEATS = 5

STABILITY_K = 5
STABILITY_N_REPEATS = 10
STABILITY_SAMPLES = 50


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPORT: Ergebnisse für die Bachelorarbeit")
    print("=" * 60)

    print("\n[1/9] Daten laden und vorbereiten …")
    df_raw = load_data()
    n_raw = len(df_raw)
    n_fraud_raw = int(df_raw["isFraud"].sum())

    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        random_state=RANDOM_STATE, drop_step=False, max_rows=MAX_ROWS
    )

    display_names = readable_names(feature_names)

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_total = n_train + n_test
    fraud_train = int(y_train.sum())
    fraud_test = int(y_test.sum())

    stats = [
        f"Rohdaten (Datei):            {n_raw:>10,} Transaktionen",
        f"Fraud (Rohdaten):            {n_fraud_raw:>10,}",
        f"Max. Sampling:               {MAX_ROWS:>10,}",
        f"Nach Sampling + Dedup:       {n_total:>10,} Transaktionen",
        f"  davon Training (80%):      {n_train:>10,}",
        f"  davon Test (20%):          {n_test:>10,}",
        f"Features:                    {X_train.shape[1]:>10}",
        f"Fraud gesamt (nach Prep):    {fraud_train + fraud_test:>10}",
        f"  davon Training:            {fraud_train:>10}  ({100*fraud_train/n_train:.4f}%)",
        f"  davon Test:                {fraud_test:>10}  ({100*fraud_test/n_test:.4f}%)",
        f"Fraud-Rate:                  {100*(fraud_train+fraud_test)/n_total:>10.4f}%",
        f"random_state:                {RANDOM_STATE:>10}",
        f"n_estimators:                {N_ESTIMATORS:>10}",
        f"SHAP background_size:        {BACKGROUND_SIZE:>10}",
    ]
    stats_text = "\n".join(stats)
    print(stats_text)
    (OUT_DIR / "txt_datenstatistik.txt").write_text(stats_text, encoding="utf-8")

    print("\n[2/9] Isolation Forest trainieren …")
    # contamination aus y_train – wie in der App mit Häkchen „aus y_train schätzen“
    detector = AnomalyDetector(
        contamination=None,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    detector.fit(X_train, y_train)
    c_used = detector.contamination_used
    print(f"  contamination (aus y_train geschätzt): {c_used:.6f}")

    scores_test = detector.decision_function(X_test)
    pred_test = detector.predict_anomaly(X_test)
    n_pred_anomaly = int(pred_test.sum())
    print(f"  Vorhergesagte Anomalien (Test): {n_pred_anomaly}")

    print("\n[3/9] Metriken berechnen …")
    metrics = evaluate_model(y_test, pred_test, scores=scores_test)
    metrics_row = {
        "contamination": c_used,
        "n_estimators": N_ESTIMATORS,
        **metrics.__dict__,
    }
    df_metrics = pd.DataFrame([metrics_row])
    df_metrics.to_csv(TABLE_DIR / TAB["metrics_main"], index=False)
    print(df_metrics.to_string(index=False))

    print("\n[4/9] Sensitivitätsanalyse (contamination) …")
    contam_values = sorted({
        round(c_used, 6), 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    })
    sens = sensitivity_analysis(
        X_train, y_train, X_test, y_test,
        contam_values,
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    sens.to_csv(TABLE_DIR / TAB["sensitivity"], index=False)
    print(sens.to_string(index=False))

    print("\n[5/9] Plots exportieren …")

    fig_hist = plot_decision_score_histogram(
        scores_test,
        pred_test,
        decision_threshold=0.0,
    )
    fig_hist.savefig(PLOT_DIR / ABB["score_hist"], dpi=200, bbox_inches="tight")
    print(f"  -> {ABB['score_hist']}")

    fig_cm = plot_confusion_matrix_figure(y_test, pred_test)
    fig_cm.savefig(PLOT_DIR / ABB["confusion"], dpi=200, bbox_inches="tight")
    print(f"  -> {ABB['confusion']}")

    fig_roc = plot_roc_figure(y_test, scores_test)
    fig_roc.savefig(PLOT_DIR / ABB["roc"], dpi=200, bbox_inches="tight")
    print(f"  -> {ABB['roc']}")

    print("\n[6/9] SHAP: globale Feature-Importance …")
    explainer = build_explainer(detector)

    anomaly_mask = pred_test == 1
    X_anomalies = X_test[anomaly_mask]
    if X_anomalies.shape[0] > 500:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(X_anomalies.shape[0], 500, replace=False)
        X_sample = X_anomalies[sample_idx]
    else:
        X_sample = X_anomalies

    shap_global = explain_batch(detector, X_sample, explainer=explainer)

    fig_bar = shap_bar_figure(shap_global, display_names, max_display=15)
    fig_bar.savefig(PLOT_DIR / ABB["shap_global"], dpi=200, bbox_inches="tight")
    print(f"  -> {ABB['shap_global']}")

    fig_summ = shap_summary_figure(shap_global, X_sample, display_names, max_display=15)
    fig_summ.savefig(PLOT_DIR / ABB["shap_beeswarm"], dpi=200, bbox_inches="tight")
    print(f"  -> {ABB['shap_beeswarm']}")

    print("\n[7/9] SHAP: Top-5 lokale Waterfalls …")
    top_idx = rank_anomaly_indices(scores_test, pred_test, top_k=5)

    shap_rows = []
    for rank, idx in enumerate(top_idx, start=1):
        ex: AnomalyExplanation = explain_anomaly(
            detector, X_test, int(idx), display_names, explainer=explainer
        )
        label = "Fraud" if y_test[idx] == 1 else "Normal"
        print(f"  Rang {rank}: Zeile {idx}, score={ex.decision_function:.4f}, Label={label}")

        fig_w = shap_waterfall_figure(ex, max_display=15)
        fig_w.savefig(PLOT_DIR / ABB["waterfall_tpl"].format(rank=rank), dpi=200, bbox_inches="tight")

        top3_abs = np.argsort(np.abs(ex.shap_values))[::-1][:3]
        top3_str = ", ".join(
            f"{ex.feature_names[j]} ({ex.shap_values[j]:+.4f})"
            for j in top3_abs
        )
        shap_rows.append({
            "rang": rank,
            "test_index": int(idx),
            "decision_function": ex.decision_function,
            "true_class": int(y_test[idx]),
            "pred_anomaly": ex.predict_anomaly,
            "top3_features": top3_str,
        })

    df_shap = pd.DataFrame(shap_rows)
    df_shap.to_csv(TABLE_DIR / TAB["top_anom"], index=False)
    print(f"  -> {ABB['waterfall_tpl'].format(rank='1..5')} + {TAB['top_anom']}")

    print("\n[8/9] XAI-Evaluation: Faithfulness (Feature-Ablation) + Stability …")

    if X_anomalies.shape[0] > 0:
        eval_X = X_anomalies
    else:
        worst = np.argsort(scores_test)[: FAITHFULNESS_SAMPLES]
        eval_X = X_test[worst]

    if eval_X.shape[0] > FAITHFULNESS_SAMPLES:
        rng = np.random.RandomState(RANDOM_STATE)
        eval_idx = rng.choice(eval_X.shape[0], FAITHFULNESS_SAMPLES, replace=False)
        eval_X = eval_X[eval_idx]

    eval_shap = explain_batch(detector, eval_X, explainer=explainer)
    baseline_mean = X_train.mean(axis=0)

    faith: FaithfulnessResult = faithfulness_top_k_ablation(
        detector,
        eval_X,
        eval_shap,
        baseline=baseline_mean,
        k=FAITHFULNESS_K,
        n_random_repeats=FAITHFULNESS_RANDOM_REPEATS,
        random_state=RANDOM_STATE,
    )
    print(
        f"  Faithfulness: delta top-k = {faith.mean_delta_top:+.4f} | "
        f"delta random-k = {faith.mean_delta_random:+.4f} | "
        f"gap = {faith.faithfulness_gap:+.4f} | "
        f"Anteil top > random = {faith.share_top_better:.2%}"
    )

    stab_X = eval_X[: STABILITY_SAMPLES]
    bg_pool = subsample_background(
        X_train, size=max(BACKGROUND_SIZE * 10, 2000), random_state=RANDOM_STATE
    )
    stab: StabilityResult = stability_topk_jaccard(
        detector,
        stab_X,
        background_pool=bg_pool,
        k=STABILITY_K,
        n_repeats=STABILITY_N_REPEATS,
        background_size=BACKGROUND_SIZE,
        random_state=RANDOM_STATE,
    )
    print(
        f"  Stability: mean Jaccard = {stab.mean_jaccard:.3f} | "
        f"median = {stab.median_jaccard:.3f} | "
        f"min = {stab.min_jaccard:.3f}"
    )

    df_xai = format_xai_evaluation_table(faith, stab)
    df_xai.to_csv(TABLE_DIR / TAB["xai_eval"], index=False)
    print(f"  -> {TAB['xai_eval']}")

    model_path = MODEL_DIR / "isolation_forest.joblib"
    detector.save(model_path)
    print(f"\n[8/9] Modell gespeichert: {model_path}")

    print("\n[9/9] Reproduzierbarkeits-Info …")
    import sklearn, shap as shap_lib, streamlit, matplotlib, seaborn
    repro = [
        f"Python:        {sys.version}",
        f"numpy:         {np.__version__}",
        f"pandas:        {pd.__version__}",
        f"scikit-learn:  {sklearn.__version__}",
        f"shap:          {shap_lib.__version__}",
        f"streamlit:     {streamlit.__version__}",
        f"matplotlib:    {matplotlib.__version__}",
        f"seaborn:       {seaborn.__version__}",
        f"random_state:  {RANDOM_STATE}",
        f"n_estimators:  {N_ESTIMATORS}",
        f"max_rows:      {MAX_ROWS}",
        f"contamination: {c_used:.6f}",
        f"SHAP bg_size:  {BACKGROUND_SIZE}",
        f"faithfulness_k:     {FAITHFULNESS_K}",
        f"faithfulness_n:     {FAITHFULNESS_SAMPLES}",
        f"stability_k:        {STABILITY_K}",
        f"stability_repeats:  {STABILITY_N_REPEATS}",
        f"stability_samples:  {STABILITY_SAMPLES}",
    ]
    repro_text = "\n".join(repro)
    (OUT_DIR / "txt_reproduzierbarkeit.txt").write_text(repro_text, encoding="utf-8")
    print(repro_text)

    print("\n" + "=" * 60)
    print(f"FERTIG — alle Ergebnisse in: {OUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
