"""Single evaluation engine used by standard and robustness runs."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib

import pandas as pd
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from config.metric_thresholds import METRIC_THRESHOLDS
from config.model_config import EMBEDDING_MODEL, GENERATION_MODEL, JUDGE_MODEL, PROMPT_VERSION, TOP_K
from evaluation.batch_summary import create_batch_summary, print_batch_summary
from evaluation.failure_analysis import get_failure_reasons
from evaluation.metrics import evaluate_all_metrics
from evaluation.regression_analysis import create_or_compare_baseline, print_regression_result
from models.ollama_client import generate_response
from retrieval.retriever import retrieve_context
from retrieval.vector_store import vector_store_exists
from retrieval.vector_store import build_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "customer_support_dataset.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation_report.csv"
DEFAULT_BATCH_SUMMARY_PATH = PROJECT_ROOT / "reports" / "batch_summary.csv"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "reports" / "baseline_batch_summary.csv"
DEFAULT_REGRESSION_PATH = PROJECT_ROOT / "reports" / "regression_report.csv"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "datasets" / "knowledge_base.csv"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataset(df: pd.DataFrame):
    required = {"input", "expected_output", "context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Evaluation dataset is empty.")
    for column in ["input", "expected_output", "context"]:
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Evaluation dataset contains empty values in '{column}'.")


def _ensure_vector_store():
    if not vector_store_exists():
        print("Knowledge-base vector store not found. Building it now...")
        count = build_vector_store(KNOWLEDGE_BASE_PATH)
        print(f"Vector store ready with {count} chunks.")


def run_evaluation(
    dataset_path=DEFAULT_DATASET_PATH,
    report_path=DEFAULT_REPORT_PATH,
    batch_summary_path=DEFAULT_BATCH_SUMMARY_PATH,
    baseline_summary_path=DEFAULT_BASELINE_PATH,
    regression_report_path=DEFAULT_REGRESSION_PATH,
):
    dataset_path = Path(dataset_path)
    report_path = Path(report_path)
    batch_summary_path = Path(batch_summary_path)
    baseline_summary_path = Path(baseline_summary_path)
    regression_report_path = Path(regression_report_path)
    for path in [report_path, batch_summary_path, baseline_summary_path, regression_report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    _validate_dataset(df)
    _ensure_vector_store()

    evaluation_model = OllamaModel(model=JUDGE_MODEL)
    report_data = []

    for _, row in df.iterrows():
        input_text = str(row["input"]).strip()
        expected_output = str(row["expected_output"]).strip()
        trusted_context = str(row["context"]).strip()
        scenario = str(row.get("scenario", "Standard")).strip()

        print("\n========================")
        print("INPUT:", input_text)

        retrieval_context = retrieve_context(input_text, top_k=TOP_K)
        if not retrieval_context:
            print("WARNING: No retrieval context found.")

        actual_output = generate_response(input_text, retrieval_context)
        print("ACTUAL OUTPUT:", actual_output)
        print("RETRIEVED CONTEXT:", retrieval_context)

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=[trusted_context],
            retrieval_context=retrieval_context,
        )

        metrics = evaluate_all_metrics(test_case, evaluation_model)
        scores = {name: float(metric.score) for name, metric in metrics.items()}
        failure_reasons = get_failure_reasons(scores)
        status = "FAIL" if failure_reasons else "PASS"
        failure_reason = " | ".join(failure_reasons) or "All metric thresholds met."

        for name, metric in metrics.items():
            print(f"{name.upper()} SCORE: {metric.score}")
            print(f"{name.upper()} REASON: {metric.reason}")
        print("FAILURE ANALYSIS:", failure_reason)
        print("STATUS:", status)

        report_data.append({
            "Input": input_text,
            "Scenario": scenario,
            "Context": trusted_context,
            "Retrieved_Context": " | ".join(retrieval_context),
            "Actual_Output": actual_output,
            "hallucination": scores["hallucination"],
            "answer_relevancy": scores["answer_relevancy"],
            "faithfulness": scores["faithfulness"],
            "contextual_precision": scores["contextual_precision"],
            "contextual_recall": scores["contextual_recall"],
            "Hallucination_Reason": metrics["hallucination"].reason,
            "Failure_Reason": failure_reason,
            "Status": status,
        })

    report_dataframe = pd.DataFrame(report_data)
    report_dataframe.to_csv(report_path, index=False)

    batch_summary = create_batch_summary(report_dataframe)
    metadata = {
        "Generation_Model": GENERATION_MODEL,
        "Judge_Model": JUDGE_MODEL,
        "Embedding_Model": EMBEDDING_MODEL,
        "Top_K": TOP_K,
        "Prompt_Version": PROMPT_VERSION,
        "Dataset": dataset_path.name,
        "Dataset_SHA256": _file_hash(dataset_path),
        "Generated_At_UTC": datetime.now(timezone.utc).isoformat(),
    }
    batch_summary.update(metadata)
    pd.DataFrame([batch_summary]).to_csv(batch_summary_path, index=False)
    print_batch_summary(batch_summary)

    comparison = create_or_compare_baseline(batch_summary, baseline_summary_path)
    if comparison is not None:
        comparison.to_csv(regression_report_path, index=False)
    print_regression_result(comparison)
    print(f"\nEvaluation report generated: {report_path}")
    return report_dataframe, batch_summary, comparison
