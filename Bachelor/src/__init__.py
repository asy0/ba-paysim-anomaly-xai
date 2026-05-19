"""
Zentrale Exporte der Pipeline-Module.
"""

from .data.preprocessing import get_dataset_stats, prepare_data, readable_name, readable_names
from .metrics.evaluation import ClassificationMetrics, evaluate_model, sensitivity_analysis
from .models.anomaly_detector import AnomalyDetector
from .xai.quality_metrics import (
    FaithfulnessResult,
    StabilityResult,
    faithfulness_top_k_ablation,
    format_xai_evaluation_table,
    stability_topk_jaccard,
)
from .xai.shap_explain import (
    DEFAULT_BACKGROUND_SIZE,
    build_explainer,
    explain_anomaly,
    explain_batch,
    rank_anomaly_indices,
    subsample_background,
)

__all__ = [
    "AnomalyDetector",
    "prepare_data",
    "get_dataset_stats",
    "readable_name",
    "readable_names",
    "build_explainer",
    "explain_anomaly",
    "explain_batch",
    "rank_anomaly_indices",
    "subsample_background",
    "DEFAULT_BACKGROUND_SIZE",
    "evaluate_model",
    "ClassificationMetrics",
    "sensitivity_analysis",
    "faithfulness_top_k_ablation",
    "FaithfulnessResult",
    "stability_topk_jaccard",
    "StabilityResult",
    "format_xai_evaluation_table",
]
