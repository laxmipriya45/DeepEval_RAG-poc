"""Threshold-based failure analysis."""

from config.metric_thresholds import METRIC_THRESHOLDS


def get_failure_reasons(scores: dict[str, float]) -> list[str]:
    reasons = []
    for metric_name, rule in METRIC_THRESHOLDS.items():
        score = float(scores[metric_name])
        threshold = rule["threshold"]
        if rule["higher_is_better"] and score < threshold:
            reasons.append(f"{metric_name} score {score:.2f} is below the minimum {threshold:.2f}.")
        elif not rule["higher_is_better"] and score > threshold:
            reasons.append(f"{metric_name} score {score:.2f} is above the maximum {threshold:.2f}.")
    return reasons


def is_passing(scores: dict[str, float]) -> bool:
    return not get_failure_reasons(scores)
