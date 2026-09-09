"""Baseline creation and regression comparison."""

from pathlib import Path
import pandas as pd

from config.metric_thresholds import REGRESSION_TOLERANCE

SUMMARY_FIELDS = [
    "Pass_Rate_Percent",
    "Average_hallucination",
    "Average_answer_relevancy",
    "Average_faithfulness",
    "Average_contextual_precision",
    "Average_contextual_recall",
]
LOWER_IS_BETTER = {"Average_hallucination"}


def create_or_compare_baseline(current_summary: dict, baseline_path):
    baseline_path = Path(baseline_path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        pd.DataFrame([current_summary]).to_csv(baseline_path, index=False)
        return None

    baseline_df = pd.read_csv(baseline_path)
    if baseline_df.empty:
        pd.DataFrame([current_summary]).to_csv(baseline_path, index=False)
        return None
    baseline_missing = set(SUMMARY_FIELDS) - set(baseline_df.columns)
    current_missing = set(SUMMARY_FIELDS) - set(current_summary)
    if baseline_missing:
        raise ValueError(f"Baseline is missing required field(s): {', '.join(sorted(baseline_missing))}")
    if current_missing:
        raise ValueError(f"Current summary is missing required field(s): {', '.join(sorted(current_missing))}")

    baseline_summary = baseline_df.iloc[0].to_dict()
    rows = []
    for field in SUMMARY_FIELDS:
        baseline_value = float(baseline_summary[field])
        current_value = float(current_summary[field])
        raw_difference = current_value - baseline_value
        regression = raw_difference > REGRESSION_TOLERANCE if field in LOWER_IS_BETTER else raw_difference < -REGRESSION_TOLERANCE
        improvement = raw_difference < -REGRESSION_TOLERANCE if field in LOWER_IS_BETTER else raw_difference > REGRESSION_TOLERANCE
        rows.append({
            "Metric": field.replace("Average_", "").replace("_", " "),
            "Baseline": baseline_value,
            "Current": current_value,
            "Difference": round(raw_difference, 4),
            "Result": "REGRESSION" if regression else "IMPROVED" if improvement else "STABLE",
        })
    return pd.DataFrame(rows)


def print_regression_result(comparison):
    print("\n========================")
    if comparison is None:
        print("REGRESSION BASELINE CREATED")
        print("Run again after a prompt, dataset, or model change to compare results.")
        return
    print("PROMPT & DATASET REGRESSION RESULTS")
    for _, row in comparison.iterrows():
        print(f"{row['Metric']}: {row['Result']} (baseline {row['Baseline']}, current {row['Current']}, difference {row['Difference']})")
