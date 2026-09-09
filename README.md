# AI Hallucination Evaluation Suite

A local RAG evaluation project using Ollama for answer generation, ChromaDB + Sentence Transformers for retrieval, and DeepEval for response/retrieval quality metrics.

## Fixed architecture

```text
project/
│
├── config/
│   ├── metric_thresholds.py
│   └── model_config.py
│
├── datasets/
│   ├── customer_support_dataset.csv
│   ├── knowledge_base.csv
│   └── robustness_test_dataset.csv
│
├── models/
│   ├── embedding_model.py
│   ├── ollama_client.py
│   └── deepeval_ollama.py
│
├── retrieval/
│   ├── retriever.py
│   └── vector_store.py
│
├── evaluation/
│   ├── evaluator.py
│   ├── metrics.py
│   ├── failure_analysis.py
│   ├── batch_summary.py
│   └── regression_analysis.py
│
├── scripts/
│   └── build_vector_store.py
│
├── tests/
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   └── test_evaluation.py
│
├── reports/
│
├── main.py
├── requirements.txt
└── README.md
```

## Workflow

```text
Evaluation CSV
     ↓
Validate dataset
     ↓
Question → Sentence Transformer embedding → ChromaDB top-k retrieval
     ↓
Retrieved context → Ollama grounded generation
     ↓
DeepEval judge metrics
     ↓
Failure analysis → detailed report → batch summary → regression comparison
```

The application answer is explicitly grounded in retrieved context. The prompt instructs Ollama to use only the retrieved context, avoid unsupported facts, and state when the context is insufficient.

## Models

Models are centralized in `config/model_config.py`:

- Generation model: `llama3`
- Judge model: `llama3`
- Embedding model: `all-MiniLM-L6-v2`
- Retrieval top-k: `3`
- Prompt version: `v2-grounded`

Change model names in that file rather than editing multiple modules.

## Prerequisites

1. Install Python dependencies:

```text
pip install -r requirements.txt
```

2. Start Ollama.
3. Make sure the configured model is available, for example:

```text
ollama pull llama3
```

## Build the vector store

The explicit build command is:

```text
python scripts/build_vector_store.py
```

The standard evaluator also checks for the vector store and builds it automatically when it is missing.

The vector store is created under `vector_store/chroma_db` at runtime and is intentionally not packaged in the source ZIP.

## Run the standard evaluation

```text
python main.py
```

Reports:

- `reports/evaluation_report.csv`
- `reports/batch_summary.csv`
- `reports/baseline_batch_summary.csv`
- `reports/regression_report.csv` (created on comparison runs)

## Run robustness evaluation

Use the same evaluation engine with the robustness dataset:

```text
python -c "from pathlib import Path; from evaluation.evaluator import run_evaluation; run_evaluation(Path('datasets/robustness_test_dataset.csv'), Path('reports/robustness_evaluation_report.csv'), Path('reports/robustness_batch_summary.csv'), Path('reports/robustness_baseline_summary.csv'), Path('reports/robustness_regression_report.csv'))"
```

This keeps robustness evaluation separate from standard reports while avoiding duplicate evaluation logic.

## Metrics and thresholds

Thresholds are centralized in `config/metric_thresholds.py` and are the single source of truth:

- Hallucination: `<= 0.20`
- Answer Relevancy: `>= 0.70`
- Faithfulness: `>= 0.80`
- Contextual Precision: `>= 0.70`
- Contextual Recall: `>= 0.80`

A row passes only when all five conditions are met.

## Regression behavior

The first completed batch becomes the baseline in `reports/baseline_batch_summary.csv`. Later runs compare raw, unrounded metric differences against `REGRESSION_TOLERANCE = 0.05`; rounding is applied only for display/reporting.

The baseline also records generation model, judge model, embedding model, top-k, prompt version, dataset name, dataset SHA-256, and UTC timestamp so that changes are traceable.

## Testing

Run unit tests with:

```text
pytest
```

The tests are independent of Ollama and ChromaDB network/service availability. Integration execution is performed through `main.py` and the vector-store build script.

## Error handling and validation

The fixed implementation validates dataset columns and empty values, validates retriever arguments, handles empty retrieval, gives actionable Ollama errors, uses project-root-relative paths, creates report directories automatically, and lazily loads the embedding model.
