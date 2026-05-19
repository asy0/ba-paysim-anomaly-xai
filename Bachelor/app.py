"""
Streamlit-Prototyp für die Bachelorarbeit (Interaktion + Demo).

Start im Projektroot ``Bachelor/``::

    streamlit run app.py

Tab „Analyse“ = Daten/Training, „Anomalien“ = SHAP-Fälle, „Evaluierung“ = Metriken/Sensitivität.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.preprocessing import DEFAULT_DATA_PATH, SCALE_COLS
from src.ui import render_tab_analysis, render_tab_anomalies, render_tab_evaluation
from src.services import init_session_state, load_pipeline_run, train_pipeline_run

# Einstieg: Parameter links, Ergebnisse in den drei Tabs
st.set_page_config(page_title="Datenqualitäts-Assistent", layout="wide")
st.title("KI-gestützter Datenqualitäts-Assistent")
st.caption(
    "Findet auffällige Transaktionen und zeigt, welche Merkmale im Modell dafür ausschlaggebend sind (Isolation Forest + SHAP)."
)

# Startwert Sidebar; mit Häkchen wird trotzdem aus y_train geschätzt (siehe train_pipeline_run)
_DEFAULT_CONTAMINATION = 0.0013
_MODEL_PATH = Path("models") / "isolation_forest.joblib"

init_session_state()


train_btn = False
load_btn = False

with st.sidebar:
    st.header("Parameter")
    data_path = st.text_input(
        "PaySim-Datei (`paysim.csv`)", value=str(DEFAULT_DATA_PATH)
    )
    max_rows = st.number_input(
        "Stichprobengröße (Sampling)",
        min_value=10_000,
        max_value=6_400_000,
        value=200_000,
        step=50_000,
        help="PaySim ist groß. Für schnelle Läufe wird eine stratifizierte Stichprobe gezogen (Fraud-Anteil bleibt erhalten).",
    )
    contamination_in = st.number_input(
        "contamination",
        min_value=0.00001,
        max_value=0.49,
        value=float(_DEFAULT_CONTAMINATION),
        format="%.6f",
        help="Erwarteter Ausreißeranteil (Schwellwert-Steuerung).",
    )
    use_auto_contamination = st.checkbox(
        "contamination aus y_train schätzen", value=True
    )
    n_estimators = st.number_input(
        "n_estimators", min_value=10, max_value=500, value=100, step=10
    )
    random_state = st.number_input("random_state", value=42, step=1)
    drop_step = st.checkbox("Spalte step (Zeitschritt) weglassen", value=False)

    st.divider()
    if _MODEL_PATH.is_file() and not st.session_state.trained:
        st.info(
            f"Es gibt bereits ein gespeichertes Modell: `{_MODEL_PATH.name}`. Man kann es laden oder neu trainieren."
        )

    train_btn = st.button(
        "Modell trainieren", type="primary", width="stretch"
    )

    st.divider()
    st.caption("Modell sichern / wiederverwenden")
    col_s, col_l = st.columns(2)
    with col_s:
        save_disabled = not st.session_state.trained
        if st.button("Speichern", width="stretch", disabled=save_disabled):
            p = st.session_state.detector.save(_MODEL_PATH)
            st.success(f"Gespeichert: {p.name}")
    with col_l:
        load_label = "Laden" if _MODEL_PATH.is_file() else "Laden (—)"
        load_btn = st.button(
            load_label,
            width="stretch",
            disabled=not _MODEL_PATH.is_file(),
        )

# Training: gleiche prepare_data-Pipeline wie export_results.py (max_rows, random_state, …)
if train_btn:
    path = Path(data_path)
    if not path.is_file():
        st.error(f"Datei nicht gefunden: {path}")
    else:
        with st.spinner("Training läuft …"):
            c_arg = None if use_auto_contamination else float(contamination_in)
            train_pipeline_run(
                data_path=path,
                max_rows=int(max_rows),
                drop_step=drop_step,
                random_state=int(random_state),
                n_estimators=int(n_estimators),
                contamination=c_arg,
            )
        st.success("Training abgeschlossen.")

# Laden: Modell von Disk, Daten müssen dieselben Features liefern (drop_step, max_rows beachten)
if load_btn:
    path = Path(data_path)
    if not path.is_file():
        st.error(f"Datei nicht gefunden: {path}")
    else:
        with st.spinner("Modell wird geladen …"):
            try:
                load_pipeline_run(
                    data_path=path,
                    model_path=_MODEL_PATH,
                    max_rows=int(max_rows),
                    drop_step=drop_step,
                    random_state=int(random_state),
                )
                st.success("Modell geladen – alle Tabs sind aktualisiert.")
                saved_v = st.session_state.get("model_sklearn_saved")
                runtime_v = st.session_state.get("model_sklearn_runtime")
                if saved_v and runtime_v and saved_v != runtime_v:
                    st.warning(
                        f"Version-Hinweis: Das geladene Modell wurde mit scikit-learn **{saved_v}** gespeichert, "
                        f"du verwendest gerade **{runtime_v}**. "
                        "Empfehlung: Modell neu trainieren und erneut speichern."
                    )
            except Exception as exc:
                st.error(f"Laden fehlgeschlagen: {exc}")

tab1, tab2, tab3 = st.tabs(["Analyse", "Anomalien & Erklärungen", "Evaluierung"])

with tab1:
    render_tab_analysis()

with tab2:
    render_tab_anomalies(scale_cols=SCALE_COLS)

with tab3:
    render_tab_evaluation(n_estimators=int(n_estimators), random_state=int(random_state))
