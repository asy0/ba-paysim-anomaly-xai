import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import prepare_data, readable_name, readable_names
from src.metrics.evaluation import ClassificationMetrics, evaluate_model
from src.models.anomaly_detector import AnomalyDetector
from src.xai.quality_metrics import FaithfulnessResult, StabilityResult, faithfulness_top_k_ablation, stability_topk_jaccard
from src.xai.shap_explain import (
    AnomalyExplanation,
    build_explainer,
    explain_anomaly,
    explain_batch,
    rank_anomaly_indices,
    subsample_background,
)


def _make_paysim_toy(n: int = 60, rng_seed: int = 7) -> pd.DataFrame:
    """Erzeugt einen minimalen PaySim-artigen DataFrame für Tests."""
    rng = np.random.RandomState(rng_seed)
    types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    n_fraud = max(4, n // 10)
    y = np.zeros(n, dtype=np.int64)
    y[:n_fraud] = 1
    rng.shuffle(y)

    df = pd.DataFrame({
        "step": rng.randint(0, 744, size=n),
        "type": rng.choice(types, size=n),
        "amount": rng.lognormal(mean=8.0, sigma=1.5, size=n),
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": rng.lognormal(mean=9.0, sigma=1.2, size=n),
        "newbalanceOrig": rng.lognormal(mean=8.5, sigma=1.3, size=n),
        "nameDest": [f"M{i}" for i in range(n)],
        "oldbalanceDest": rng.lognormal(mean=9.0, sigma=1.0, size=n),
        "newbalanceDest": rng.lognormal(mean=9.2, sigma=1.1, size=n),
        "isFraud": y,
        "isFlaggedFraud": np.zeros(n, dtype=np.int64),
    })
    return df


class TestPreprocessingSmoke(unittest.TestCase):
    def test_prepare_data_with_temp_csv(self) -> None:
        df = _make_paysim_toy(n=60)
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "paysim_toy.csv"
            df.to_csv(csv_path, index=False)

            X_train, X_test, y_train, y_test, _scaler, feature_names = prepare_data(
                file_path=csv_path,
                test_size=0.25,
                random_state=7,
                deduplicate=True,
                drop_step=False,
                stratify=True,
                max_rows=None,
            )

        total = len(y_train) + len(y_test)
        self.assertGreater(total, 0)
        self.assertEqual(X_train.shape[1], len(feature_names))
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())
        self.assertIn("amount", feature_names)
        self.assertIn("balance_delta_orig", feature_names)
        self.assertIn("type_TRANSFER", feature_names)

    def test_readable_names(self) -> None:
        cols = ["amount", "oldbalanceOrg", "type_TRANSFER"]
        names = readable_names(cols)
        self.assertEqual(names[0], "Betrag")
        self.assertEqual(names[1], "Sender-Kontostand vorher")
        self.assertEqual(names[2], "Typ: Überweisung")
        self.assertEqual(readable_name("unknown_col"), "unknown_col")


class TestModelExplainEvaluateSmoke(unittest.TestCase):
    def test_contamination_above_half_raises(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(20, 3))
        det = AnomalyDetector(contamination=0.51, n_estimators=10, random_state=0)
        with self.assertRaises(ValueError):
            det.fit(X)

    def test_model_explain_evaluate_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.normal(size=(80, 5))
        y = np.zeros(80, dtype=np.int64)
        y[:8] = 1

        detector = AnomalyDetector(
            contamination=0.1,
            n_estimators=40,
            random_state=42,
        )
        detector.fit(X, y)

        pred = detector.predict_anomaly(X)
        scores = detector.decision_function(X)
        self.assertEqual(pred.shape[0], X.shape[0])
        self.assertEqual(scores.shape[0], X.shape[0])

        metrics = evaluate_model(y, pred, scores=scores)
        self.assertIsInstance(metrics, ClassificationMetrics)
        self.assertTrue(hasattr(metrics, "precision"))
        self.assertTrue(hasattr(metrics, "recall"))
        self.assertTrue(hasattr(metrics, "f1"))
        self.assertTrue(hasattr(metrics, "roc_auc"))

        explainer = build_explainer(detector)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        single: AnomalyExplanation = explain_anomaly(
            detector,
            X,
            index=0,
            feature_names=feature_names,
            explainer=explainer,
        )
        self.assertEqual(len(single.shap_values), X.shape[1])
        self.assertTrue(single.shap_reconstruction_ok)
        self.assertLessEqual(len(single.top_features), 20)
        for row in single.top_features:
            self.assertTrue(hasattr(row, "feature_name"))
            self.assertTrue(hasattr(row, "abs_shap"))
        if len(single.top_features) >= 2:
            self.assertGreaterEqual(
                single.top_features[0].abs_shap,
                single.top_features[1].abs_shap,
            )

        batch = explain_batch(detector, X[:10], explainer=explainer)
        self.assertEqual(batch.shape, (10, X.shape[1]))

        ranked = rank_anomaly_indices(scores, pred, top_k=10)
        if ranked.size > 1:
            ranked_scores = scores[ranked]
            self.assertTrue(np.all(ranked_scores[:-1] <= ranked_scores[1:]))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "iforest.joblib"
            detector.save(model_path)
            loaded = AnomalyDetector.load(model_path)
            np.testing.assert_allclose(
                detector.decision_function(X[:20]),
                loaded.decision_function(X[:20]),
                rtol=1e-12,
                atol=1e-12,
            )


class TestInvertSignAndBackground(unittest.TestCase):
    def test_invert_sign_preserves_additivity(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(60, 4))
        y = np.zeros(60, dtype=np.int64)
        y[:6] = 1
        detector = AnomalyDetector(contamination=0.1, n_estimators=30, random_state=0)
        detector.fit(X, y)
        explainer = build_explainer(detector)

        ex_normal: AnomalyExplanation = explain_anomaly(detector, X, 0, explainer=explainer)
        ex_inv: AnomalyExplanation = explain_anomaly(
            detector, X, 0, explainer=explainer, invert_sign=True
        )

        self.assertEqual(ex_normal.sign_convention, "normal")
        self.assertEqual(ex_inv.sign_convention, "anomaly")
        self.assertTrue(ex_normal.shap_reconstruction_ok)
        self.assertTrue(ex_inv.shap_reconstruction_ok)

        np.testing.assert_allclose(
            ex_inv.shap_values,
            -np.asarray(ex_normal.shap_values, dtype=np.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertAlmostEqual(
            ex_inv.expected_value, -ex_normal.expected_value, places=10
        )
        names_normal = [f.feature_name for f in ex_normal.top_features]
        names_inv = [f.feature_name for f in ex_inv.top_features]
        self.assertEqual(names_normal, names_inv)

    def test_explain_batch_invert_sign(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(40, 3))
        detector = AnomalyDetector(contamination=0.1, n_estimators=20, random_state=0)
        detector.fit(X)
        explainer = build_explainer(detector)
        a = explain_batch(detector, X, explainer=explainer)
        b = explain_batch(detector, X, explainer=explainer, invert_sign=True)
        np.testing.assert_allclose(a, -b, rtol=1e-12, atol=1e-12)

    def test_subsample_background_reproducible_and_size(self) -> None:
        rng = np.random.RandomState(0)
        X = rng.normal(size=(500, 5))
        s1 = subsample_background(X, size=50, random_state=123)
        s2 = subsample_background(X, size=50, random_state=123)
        self.assertEqual(s1.shape, (50, 5))
        np.testing.assert_array_equal(s1, s2)
        small = subsample_background(X[:10], size=50, random_state=0)
        self.assertEqual(small.shape, (10, 5))


class TestXaiEvaluation(unittest.TestCase):
    def _fit(self) -> tuple[AnomalyDetector, np.ndarray]:
        rng = np.random.RandomState(7)
        n_normal, n_anom, p = 180, 20, 4
        X_normal = rng.normal(size=(n_normal, p))
        X_anom = rng.normal(loc=4.0, scale=0.5, size=(n_anom, p))
        X = np.vstack([X_normal, X_anom])
        detector = AnomalyDetector(
            contamination=n_anom / (n_normal + n_anom),
            n_estimators=50,
            random_state=7,
        )
        detector.fit(X)
        return detector, X

    def test_faithfulness_reports_positive_gap_on_clear_signal(self) -> None:
        detector, X = self._fit()
        explainer = build_explainer(detector)
        X_eval = X[-20:]
        sv = explain_batch(detector, X_eval, explainer=explainer)
        res = faithfulness_top_k_ablation(
            detector,
            X_eval,
            sv,
            baseline=X.mean(axis=0),
            k=2,
            n_random_repeats=3,
            random_state=7,
        )
        self.assertIsInstance(res, FaithfulnessResult)
        self.assertEqual(res.k, 2)
        self.assertEqual(res.n_samples, 20)
        self.assertGreater(res.faithfulness_gap, 0.0)
        self.assertTrue(hasattr(res, "mean_delta_top"))
        self.assertTrue(hasattr(res, "share_top_better"))

    def test_stability_jaccard_in_unit_interval(self) -> None:
        detector, X = self._fit()
        X_eval = X[-8:]
        res = stability_topk_jaccard(
            detector,
            X_eval,
            background_pool=X,
            k=2,
            n_repeats=3,
            background_size=30,
            random_state=7,
        )
        self.assertIsInstance(res, StabilityResult)
        self.assertEqual(res.k, 2)
        self.assertEqual(res.n_repeats, 3)
        self.assertEqual(res.background_size, 30)
        self.assertEqual(res.n_samples, X_eval.shape[0])
        self.assertGreaterEqual(res.mean_jaccard, 0.0)
        self.assertLessEqual(res.mean_jaccard, 1.0)


if __name__ == "__main__":
    unittest.main()
