from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest

# Fallback ~0,13 % – ungefähr Fraud-Rate in PaySim, falls kein y beim fit() da ist
_DEFAULT_CONTAMINATION = 0.0013


@dataclass(frozen=True)
class SavedModelArtifact:
    forest: IsolationForest
    contamination: Optional[float]
    contamination_used: float
    n_estimators: int
    random_state: int
    max_samples: str | float
    sklearn_version: str


class AnomalyDetector:
    """
    Isolation Forest für die Arbeit; sklearn liefert -1/1, wir vergleichen mit isFraud (0/1).

    sklearn predict: 1 = Inlier (normal), -1 = Outlier (Anomalie).
    predict_anomaly mappt das auf 0/1 wie in den Metriken und Tabellen.
    """

    def __init__(
        self,
        contamination: Optional[float] = None,
        n_estimators: int = 100,
        random_state: int = 42,
        max_samples: str | float = "auto",
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self._forest: Optional[IsolationForest] = None
        self._contamination_used: Optional[float] = None
        self._sklearn_version_saved: Optional[str] = None

    def _resolve_contamination(self, y: Optional[np.ndarray]) -> float:
        # Standard in App/Export: Anteil aus y_train, sonst manueller Wert aus der Sidebar
        if self.contamination is not None:
            c = float(self.contamination)
        elif y is not None and len(y) > 0:
            c = float(np.sum(y) / len(y))
        else:
            c = float(_DEFAULT_CONTAMINATION)
        if c <= 0.0:
            c = float(_DEFAULT_CONTAMINATION)
        if c > 0.5:
            raise ValueError(
                f"contamination liegt außerhalb des gültigen Bereichs (0, 0.5] (sklearn): {c}. "
                "Bitte einen kleineren Wert setzen oder die Eingabedaten prüfen."
            )
        return c

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AnomalyDetector":
        X = np.asarray(X, dtype=np.float64)
        c = self._resolve_contamination(y)
        self._contamination_used = c

        self._forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=c,
            random_state=self.random_state,
            max_samples=self.max_samples,
            n_jobs=-1,
        )
        self._forest.fit(X)
        return self

    def _check_fitted(self) -> IsolationForest:
        if self._forest is None:
            raise RuntimeError("Modell ist nicht trainiert — zuerst fit() aufrufen.")
        return self._forest

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._check_fitted().predict(X)

    def predict_anomaly(self, X: np.ndarray) -> np.ndarray:
        # 1 = als Anomalie erkannt (entspricht isFraud==1 in Auswertung)
        return (self.predict(X) == -1).astype(np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._check_fitted().decision_function(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._check_fitted().score_samples(X)

    @property
    def sklearn_estimator(self) -> IsolationForest:
        return self._check_fitted()

    @property
    def contamination_used(self) -> float:
        if self._contamination_used is None:
            raise RuntimeError("contamination_used erst nach fit() verfügbar.")
        return self._contamination_used

    def save(self, path: str | Path) -> Path:
        # joblib inkl. sklearn-Version – für export_results und Reproduzierbarkeit
        self._check_fitted()
        if self._contamination_used is None:
            raise RuntimeError("contamination_used erst nach fit() verfügbar.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        artifact = SavedModelArtifact(
            forest=cast(IsolationForest, self._forest),
            contamination=self.contamination,
            contamination_used=float(self._contamination_used),
            n_estimators=int(self.n_estimators),
            random_state=int(self.random_state),
            max_samples=self.max_samples,
            sklearn_version=str(sklearn.__version__),
        )
        joblib.dump(artifact, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        # ältere .joblib-Dateien ohne Dataclass werden noch als dict geladen
        raw = joblib.load(Path(path))
        if isinstance(raw, SavedModelArtifact):
            artifact = raw
        else:
            data = cast(dict, raw)
            artifact = SavedModelArtifact(
                forest=data["forest"],
                contamination=data.get("contamination"),
                contamination_used=float(data["contamination_used"]),
                n_estimators=int(data["n_estimators"]),
                random_state=int(data["random_state"]),
                max_samples=data["max_samples"],
                sklearn_version=str(data.get("sklearn_version", "")),
            )
        obj = cls(
            contamination=artifact.contamination,
            n_estimators=artifact.n_estimators,
            random_state=artifact.random_state,
            max_samples=artifact.max_samples,
        )
        obj._forest = artifact.forest
        obj._contamination_used = artifact.contamination_used
        obj._sklearn_version_saved = artifact.sklearn_version or None
        return obj

    @property
    def sklearn_version_saved(self) -> Optional[str]:
        """scikit-learn Version, mit der das Modell gespeichert wurde (falls bekannt)."""
        return self._sklearn_version_saved

    @property
    def sklearn_version_runtime(self) -> str:
        """scikit-learn Version der aktuellen Laufzeit."""
        return sklearn.__version__

