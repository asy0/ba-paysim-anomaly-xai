from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Projektroot -> data/paysim.csv (gleicher Pfad wie in der Arbeit beschrieben)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "paysim.csv"

TARGET_COL = "isFraud"

_NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# IDs und isFlaggedFraud nicht als Merkmal: bringen fürs Modell/SHAP nichts Sinnvolles
_DROP_COLS = ["nameOrig", "nameDest", "isFlaggedFraud"]

# Anzeigenamen für Streamlit + Abbildungen (Spaltennamen im Code bleiben PaySim-original)
_READABLE_NAMES = {
    "step": "Zeitschritt (Stunde)",
    "amount": "Betrag",
    "oldbalanceOrg": "Sender-Kontostand vorher",
    "newbalanceOrig": "Sender-Kontostand nachher",
    "oldbalanceDest": "Empfänger-Kontostand vorher",
    "newbalanceDest": "Empfänger-Kontostand nachher",
    "balance_delta_orig": "Sender-Kontostandsänderung",
    "balance_delta_dest": "Empfänger-Kontostandsänderung",
    "balance_error_orig": "Sender-Bilanzabweichung",
    "type_CASH_IN": "Typ: Einzahlung",
    "type_CASH_OUT": "Typ: Auszahlung",
    "type_DEBIT": "Typ: Lastschrift",
    "type_PAYMENT": "Typ: Zahlung",
    "type_TRANSFER": "Typ: Überweisung",
}


def readable_name(col: str) -> str:
    return _READABLE_NAMES.get(col, col)


def readable_names(cols: List[str]) -> List[str]:
    return [readable_name(c) for c in cols]


def load_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    path = Path(file_path) if file_path is not None else DEFAULT_DATA_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Datensatz nicht gefunden: {path}. "
            "Bitte PaySim-CSV von Kaggle (ealaxi/paysim1) nach "
            "data/paysim.csv legen."
        )
    return pd.read_csv(path)


def deduplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # abgeleitete Kontostands-Merkmale (in der Arbeit: Plausibilität / Fraud-Muster)
    out["balance_delta_orig"] = out["newbalanceOrig"] - out["oldbalanceOrg"]
    out["balance_delta_dest"] = out["newbalanceDest"] - out["oldbalanceDest"]
    # PaySim-Regel: oldbalanceOrg - amount = newbalanceOrig; Abweichung = balance_error_orig
    out["balance_error_orig"] = out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]

    # type als One-Hot, weil Isolation Forest nur numerisch verarbeitet
    type_dummies = pd.get_dummies(out["type"], prefix="type", dtype=np.float64)
    out = pd.concat([out, type_dummies], axis=1)
    out.drop(columns=["type"], inplace=True)

    for col in _DROP_COLS:
        if col in out.columns:
            out.drop(columns=[col], inplace=True)

    return out


def _features_and_target(
    df: pd.DataFrame,
    *,
    drop_step: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Spalte '{TARGET_COL}' fehlt.")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    # drop_step: optional für Sensitivität; Standard in App/Export ist step drin
    if drop_step and "step" in feature_cols:
        feature_cols = [c for c in feature_cols if c != "step"]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].to_numpy(dtype=np.int64)
    return X, y, feature_cols


def _scale_numeric(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    cols_present = [c for c in numeric_cols if c in X_train.columns]
    if not cols_present:
        return X_train, X_test, StandardScaler()

    scaler = StandardScaler()
    X_tr = X_train.copy()
    X_te = X_test.copy()

    scaler.fit(X_tr[cols_present].to_numpy(dtype=np.float64))
    X_tr[cols_present] = scaler.transform(X_tr[cols_present].to_numpy(dtype=np.float64))
    X_te[cols_present] = scaler.transform(X_te[cols_present].to_numpy(dtype=np.float64))
    return X_tr, X_te, scaler


def _subsample(
    df: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    # PaySim hat ~6,3 Mio Zeilen; 200k reicht mir für Training + SHAP-Laufzeit
    # stratifiziert, damit der Fraud-Anteil in der Stichprobe erhalten bleibt
    if len(df) <= max_rows:
        return df
    fraud = df[df[TARGET_COL] == 1]
    normal = df[df[TARGET_COL] == 0]

    fraud_ratio = len(fraud) / len(df)
    n_fraud = max(1, int(max_rows * fraud_ratio))
    n_normal = max_rows - n_fraud

    n_fraud = min(n_fraud, len(fraud))
    n_normal = min(n_normal, len(normal))

    rng = np.random.RandomState(random_state)
    fraud_sample = fraud.iloc[rng.choice(len(fraud), n_fraud, replace=False)]
    normal_sample = normal.iloc[rng.choice(len(normal), n_normal, replace=False)]

    return pd.concat([fraud_sample, normal_sample], ignore_index=True)


_SCALE_COLS = _NUMERIC_FEATURES + ["balance_delta_orig", "balance_delta_dest", "balance_error_orig"]
SCALE_COLS = list(_SCALE_COLS)


def prepare_data(
    file_path: Optional[str | Path] = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    deduplicate: bool = True,
    drop_step: bool = False,
    stratify: bool = True,
    max_rows: Optional[int] = 200_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Hauptpipeline: laden -> ggf. Stichprobe -> Dedup -> Features -> Split -> Skalierung."""
    df = load_data(file_path)

    if max_rows is not None:
        df = _subsample(df, max_rows, random_state)

    if deduplicate:
        df = deduplicate_transactions(df)

    df = _engineer_features(df)

    X, y, feature_names = _features_and_target(df, drop_step=drop_step)

    # 80/20 mit stratify, weil Fraud sonst im Testset zu selten wäre
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    X_train, X_test, scaler = _scale_numeric(X_train, X_test, _SCALE_COLS)

    X_train_arr = X_train.to_numpy(dtype=np.float64)
    X_test_arr = X_test.to_numpy(dtype=np.float64)

    return X_train_arr, X_test_arr, y_train, y_test, scaler, feature_names


def get_dataset_stats(
    file_path: Optional[str | Path] = None,
    *,
    deduplicate: bool = True,
    max_rows: Optional[int] = 200_000,
    random_state: int = 42,
) -> dict:
    df = load_data(file_path)
    n_raw = len(df)
    n_fraud_raw = int(df[TARGET_COL].sum())

    if max_rows is not None:
        df = _subsample(df, max_rows, random_state)

    n_dup = int(df.duplicated().sum())
    if deduplicate:
        df = deduplicate_transactions(df)
    n_after = len(df)
    n_fraud = int(df[TARGET_COL].sum())
    return {
        "n_raw": n_raw,
        "n_fraud_raw": n_fraud_raw,
        "n_sampled": len(df) + n_dup if deduplicate else len(df),
        "n_duplicates": n_dup,
        "n_after_dedup": n_after,
        "n_fraud": n_fraud,
        "fraud_rate": n_fraud / max(n_after, 1),
    }

