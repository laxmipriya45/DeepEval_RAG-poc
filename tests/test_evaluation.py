"""Unit tests for evaluation threshold/failure logic."""

from evaluation.failure_analysis import get_failure_reasons, is_passing


def test_passing_scores():
    scores = {
        "hallucination": 0.10,
        "answer_relevancy": 0.90,
        "faithfulness": 0.90,
        "contextual_precision": 0.90,
        "contextual_recall": 0.90,
    }
    assert is_passing(scores)
    assert get_failure_reasons(scores) == []


def test_failing_scores_report_all_violations():
    scores = {
        "hallucination": 0.50,
        "answer_relevancy": 0.50,
        "faithfulness": 0.50,
        "contextual_precision": 0.50,
        "contextual_recall": 0.50,
    }
    reasons = get_failure_reasons(scores)
    assert len(reasons) == 5
