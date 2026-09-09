"""DeepEval metric construction and execution."""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)


def evaluate_all_metrics(test_case, evaluation_model):
    metrics = {
        "hallucination": HallucinationMetric(model=evaluation_model),
        "answer_relevancy": AnswerRelevancyMetric(model=evaluation_model),
        "faithfulness": FaithfulnessMetric(model=evaluation_model),
        "contextual_precision": ContextualPrecisionMetric(model=evaluation_model),
        "contextual_recall": ContextualRecallMetric(model=evaluation_model),
    }
    results = {}
    for name, metric in metrics.items():
        metric.measure(test_case)
        results[name] = metric
    return results
