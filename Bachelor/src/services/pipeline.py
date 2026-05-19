from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from ..metrics.evaluation import evaluate_model
from ..xai.shap_explain import build_explainer
from ..models.anomaly_detector import AnomalyDetector
from ..data.preprocessing import load_data, prepare_data, readable_names

# Streamlit-Session: ein Trainingslauf (trainieren oder gespeichertes Modell laden)


@dataclass(frozen=True)
class PipelineRun:
    detector: AnomalyDetector
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: object
    feature_names: list[str]
    data_path: Path


def init_session_state() -> None:
    if "trained" not in st.session_state:
        st.session_state.trained = False


def _store_run_state(run: PipelineRun) -> None:
    scores_test = run.detector.decision_function(run.X_test)
    pred_test = run.detector.predict_anomaly(run.X_test)
    # SHAP-Explainer einmal erzeugen – sonst bei jedem Tab-Klick neu (dauert ewig)
    explainer = build_explainer(run.detector)

    st.session_state.trained = True
    st.session_state.detector = run.detector
    st.session_state.explainer = explainer
    st.session_state.X_train = run.X_train
    st.session_state.X_test = run.X_test
    st.session_state.y_train = run.y_train
    st.session_state.y_test = run.y_test
    st.session_state.scaler = run.scaler
    st.session_state.feature_names = run.feature_names
    st.session_state.display_names = readable_names(run.feature_names)
    st.session_state.scores_test = scores_test
    st.session_state.pred_test = pred_test
    st.session_state.metrics = evaluate_model(run.y_test, pred_test, scores=scores_test)

    df_meta = load_data(run.data_path)
    st.session_state.df_meta = df_meta
    st.session_state.n_rows_raw = len(df_meta)


def train_pipeline_run(
    *,
    data_path: Path,
    max_rows: int,
    drop_step: bool,
    random_state: int,
    n_estimators: int,
    contamination: Optional[float],
) -> None:
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        data_path,
        random_state=int(random_state),
        drop_step=bool(drop_step),
        max_rows=int(max_rows),
    )
    # contamination=None -> AnomalyDetector schätzt aus y_train (Fraud-Rate im Sample)
    detector = AnomalyDetector(
        contamination=contamination,
        n_estimators=int(n_estimators),
        random_state=int(random_state),
    )
    detector.fit(X_train, y_train)
    _store_run_state(
        PipelineRun(
            detector=detector,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            data_path=data_path,
        )
    )


def load_pipeline_run(
    *,
    data_path: Path,
    model_path: Path,
    max_rows: int,
    drop_step: bool,
    random_state: int,
) -> None:
    detector = AnomalyDetector.load(model_path)
    st.session_state["model_sklearn_saved"] = detector.sklearn_version_saved
    st.session_state["model_sklearn_runtime"] = detector.sklearn_version_runtime

    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        data_path,
        random_state=int(random_state),
        drop_step=bool(drop_step),
        max_rows=int(max_rows),
    )

    # gleiche prepare_data-Parameter wie beim Speichern, sonst Feature-Anzahl passt nicht
    model_n_features = int(detector.sklearn_estimator.n_features_in_)
    if model_n_features != X_train.shape[1]:
        raise ValueError(
            "Gespeichertes Modell passt nicht zu den aktuell erzeugten Features "
            f"(Modell: {model_n_features}, Daten: {X_train.shape[1]})."
        )

    _store_run_state(
        PipelineRun(
            detector=detector,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            data_path=data_path,
        )
    )

