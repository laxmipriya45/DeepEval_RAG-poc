"""Centralized pass/fail and regression thresholds."""

HALLUCINATION_THRESHOLD = 0.20
ANSWER_RELEVANCY_THRESHOLD = 0.70
FAITHFULNESS_THRESHOLD = 0.80
CONTEXTUAL_PRECISION_THRESHOLD = 0.70
CONTEXTUAL_RECALL_THRESHOLD = 0.80
REGRESSION_TOLERANCE = 0.05

METRIC_THRESHOLDS = {
    "hallucination": {"threshold": HALLUCINATION_THRESHOLD, "higher_is_better": False},
    "answer_relevancy": {"threshold": ANSWER_RELEVANCY_THRESHOLD, "higher_is_better": True},
    "faithfulness": {"threshold": FAITHFULNESS_THRESHOLD, "higher_is_better": True},
    "contextual_precision": {"threshold": CONTEXTUAL_PRECISION_THRESHOLD, "higher_is_better": True},
    "contextual_recall": {"threshold": CONTEXTUAL_RECALL_THRESHOLD, "higher_is_better": True},
}
