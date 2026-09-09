"""Create a batch summary from detailed evaluation results."""

import pandas as pd

from config.metric_thresholds import METRIC_THRESHOLDS

METRIC_COLUMNS = list(METRIC_THRESHOLDS)


def create_batch_summary(report_dataframe: pd.DataFrame) -> dict:
    required = set(METRIC_COLUMNS) | {"Status"}
    missing = required - set(report_dataframe.columns)
    if missing:
        raise ValueError(f"Evaluation report is missing required column(s): {', '.join(sorted(missing))}")

    total_tests = len(report_dataframe)
    passed_tests = int((report_dataframe["Status"] == "PASS").sum())
    summary = {
        "Total_Tests": total_tests,
        "Passed_Tests": passed_tests,
        "Failed_Tests": total_tests - passed_tests,
        "Pass_Rate_Percent": round((passed_tests / total_tests * 100) if total_tests else 0.0, 2),
    }
    for column in METRIC_COLUMNS:
        summary[f"Average_{column}"] = round(float(report_dataframe[column].mean()) if total_tests else 0.0, 4)
    return summary


def print_batch_summary(summary: dict):
    print("\n========================")
    print("BATCH EVALUATION SUMMARY")
    for name, value in summary.items():
        print(f"{name.replace('_', ' ')}: {value}")
