"""Generate retrieval-only metrics from saved RAG raw predictions.

The LLM notebooks export metrics for the final labels selected by the LLM
(`selected_labels`). The retrieval-only baselines reported in the paper use the
ranked candidates returned by the retriever before LLM filtering
(`retrieved_competencies`). This script materializes those derived baselines as
CSV artifacts so the reported values can be audited without recomputing them by
hand.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
RAW_PATH = ROOT / "dataset.csv"
K_VALUES = [1, 3, 5, 7, 10]
LABEL_COLUMN = "comp_name_filtered"


BASELINES = [
    {
        "method": "retrieval_only_text_embedding_3_small",
        "results_dir": ROOT / "results/llm_proprietary",
        "raw_predictions": ROOT
        / "results/llm_proprietary/openai_gpt4.1mini_text_embedding_3_small_rag_raw_predictions.csv",
        "timing_source": ROOT
        / "results/llm_proprietary/openai_gpt4.1mini_text_embedding_3_small_rag_timing.csv",
        "embedding_model": "text-embedding-3-small",
        "source_methods": [
            "openai_gpt4.1mini_text_embedding_3_small_rag",
            "openai_gpt5mini_text_embedding_3_small_rag",
        ],
        "note": (
            "Derived from retrieved_competencies. The GPT-4.1-mini and "
            "GPT-5-mini raw files use the same OpenAI retriever configuration."
        ),
    },
    {
        "method": "retrieval_only_nomic_embed_text",
        "results_dir": ROOT / "results/llm_open",
        "raw_predictions": ROOT / "results/llm_open/ollama_gemma3_27b_nomic_embed_rag_raw_predictions.csv",
        "timing_source": ROOT / "results/llm_open/ollama_gemma3_27b_nomic_embed_rag_timing.csv",
        "embedding_model": "nomic-embed-text",
        "source_methods": [
            "ollama_gemma3_27b_nomic_embed_rag",
            "ollama_deepseek_r1_70b_nomic_embed_rag",
        ],
        "note": (
            "Derived from retrieved_competencies. The Gemma and DeepSeek raw "
            "files use the same Nomic retriever configuration."
        ),
    },
]


def load_competency_id_to_name() -> dict[str, str]:
    raw_df = pd.read_csv(RAW_PATH)
    return dict(zip(raw_df["comp_id"].astype(str), raw_df["comp_name"]))


def parse_label_list(value) -> list:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return list(value)


def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    list_columns = [
        "comp_id",
        "comp_name",
        "comp_description",
        "cat_id",
        "cat_name",
        "tax_id",
        "tax_name",
        LABEL_COLUMN,
    ]
    for column in list_columns:
        if column in df.columns:
            df[column] = df[column].apply(parse_label_list)
    return df


def binarize(label_lists: list[list[str]], classes: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.zeros((len(label_lists), len(classes)), dtype=int)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            if label in label_to_idx:
                y[row_idx, label_to_idx[label]] = 1
    return y


def f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    values = []
    for col_idx in range(y_true.shape[1]):
        true_col = y_true[:, col_idx]
        pred_col = y_pred[:, col_idx]
        tp = np.logical_and(true_col == 1, pred_col == 1).sum()
        fp = np.logical_and(true_col == 0, pred_col == 1).sum()
        fn = np.logical_and(true_col == 1, pred_col == 0).sum()
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(values))


def compute_ranked_label_metrics(
    y_true: np.ndarray,
    ranked_predictions: list[list[str]],
    classes: list[str],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    metric_rows = []
    prediction_rows = []

    for k in K_VALUES:
        y_pred = np.zeros_like(y_true, dtype=int)
        normalized_rankings = []
        for row_idx, ranking in enumerate(ranked_predictions):
            deduped = []
            for label in ranking:
                if label in label_to_idx and label not in deduped:
                    deduped.append(label)
            selected = deduped[:k]
            normalized_rankings.append(selected)
            for label in selected:
                y_pred[row_idx, label_to_idx[label]] = 1

        hits = np.logical_and(y_true, y_pred).sum(axis=1)
        true_counts = y_true.sum(axis=1)
        precision_at_k = hits.sum() / (len(y_true) * k)
        recall_at_k = np.divide(
            hits,
            true_counts,
            out=np.zeros_like(hits, dtype=float),
            where=true_counts != 0,
        ).mean()
        partial_hit_at_k = (hits > 0).mean()

        metric_rows.append(
            {
                "method": method,
                "k": k,
                "micro_f1": f1_micro(y_true, y_pred),
                "macro_f1": f1_macro(y_true, y_pred),
                "hamming_loss": float((y_true != y_pred).mean()),
                "subset_accuracy": float((y_true == y_pred).all(axis=1).mean()),
                "precision_at_k": float(precision_at_k),
                "recall_at_k": float(recall_at_k),
                "partial_hit_at_k": float(partial_hit_at_k),
                "lrap": np.nan,
                "coverage_error": np.nan,
                "n_samples": int(y_true.shape[0]),
                "n_labels": int(y_true.shape[1]),
            }
        )

        for row_idx, selected in enumerate(normalized_rankings):
            prediction_rows.append(
                {
                    "method": method,
                    "k": k,
                    "sample_index": row_idx,
                    "predicted_labels": selected,
                    "true_labels": [classes[idx] for idx in np.where(y_true[row_idx] == 1)[0]],
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def export_artifacts(
    results_dir: Path,
    method: str,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    timing_row: dict[str, object],
    config: dict[str, object],
) -> dict[str, str]:
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{method}_metrics.csv"
    sensitivity_path = results_dir / f"{method}_sensitivity_by_k.csv"
    predictions_path = results_dir / f"{method}_predictions.csv"
    timing_path = results_dir / f"{method}_timing.csv"
    config_path = results_dir / f"{method}_config.json"

    metrics_df[metrics_df["k"] == 7].to_csv(metrics_path, index=False)
    metrics_df.to_csv(sensitivity_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame([timing_row]).to_csv(timing_path, index=False)

    enriched_config = dict(config)
    enriched_config["run_datetime_utc"] = datetime.now(timezone.utc).isoformat()
    config_path.write_text(json.dumps(enriched_config, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metrics": str(metrics_path),
        "sensitivity": str(sensitivity_path),
        "predictions": str(predictions_path),
        "timing": str(timing_path),
        "config": str(config_path),
    }


def load_retrieved_rankings(raw_predictions_path: Path, comp_id_to_name: dict[str, str]) -> pd.DataFrame:
    raw_predictions = pd.read_csv(raw_predictions_path)
    raw_predictions["courseId"] = raw_predictions["courseId"].astype(str)
    raw_predictions["retrieved_competencies"] = raw_predictions["retrieved_competencies"].apply(parse_label_list)

    def ids_to_labels(comp_ids: list[str]) -> list[str]:
        labels = []
        for comp_id in comp_ids:
            label = comp_id_to_name.get(str(comp_id))
            if label and label not in labels:
                labels.append(label)
        return labels

    raw_predictions["retrieved_labels"] = raw_predictions["retrieved_competencies"].apply(ids_to_labels)
    return raw_predictions[["courseId", "retrieved_labels"]]


def build_timing_row(method: str, timing_source: Path, n_samples: int) -> dict[str, object]:
    source = pd.read_csv(timing_source).iloc[0].to_dict()
    setup_seconds = float(source.get("setup_seconds", 0.0))
    retrieval_seconds = float(source.get("retrieval_seconds", 0.0))
    total_runtime_seconds = setup_seconds + retrieval_seconds

    return {
        "method": method,
        "setup_seconds": setup_seconds,
        "setup_operation": source.get("setup_operation", ""),
        "fit_or_setup_seconds": setup_seconds,
        "vectorstore_reused": source.get("vectorstore_reused", ""),
        "vectorstore_build_seconds": float(source.get("vectorstore_build_seconds", 0.0)),
        "vectorstore_load_seconds": float(source.get("vectorstore_load_seconds", 0.0)),
        "inference_seconds": retrieval_seconds,
        "inference_seconds_per_sample": retrieval_seconds / max(1, n_samples),
        "total_runtime_seconds": total_runtime_seconds,
        "total_runtime_seconds_per_sample": total_runtime_seconds / max(1, n_samples),
        "retrieval_seconds": retrieval_seconds,
        "retrieval_seconds_per_sample": retrieval_seconds / max(1, n_samples),
        "llm_seconds": 0.0,
        "llm_seconds_per_sample": 0.0,
    }


def main() -> None:
    train_df = load_split_csv(TRAIN_PATH)
    test_df = load_split_csv(TEST_PATH)
    classes = sorted({label for labels in train_df[LABEL_COLUMN] for label in labels})
    y_test = binarize(test_df[LABEL_COLUMN].tolist(), classes)
    comp_id_to_name = load_competency_id_to_name()

    for baseline in BASELINES:
        method = baseline["method"]
        retrieved = load_retrieved_rankings(baseline["raw_predictions"], comp_id_to_name)

        test_order = test_df[["courseId"]].copy()
        test_order["courseId"] = test_order["courseId"].astype(str)
        merged = test_order.merge(retrieved, on="courseId", how="left")
        ranked_predictions = merged["retrieved_labels"].apply(lambda value: value if isinstance(value, list) else []).tolist()

        metrics_df, predictions_df = compute_ranked_label_metrics(y_test, ranked_predictions, classes, method)

        timing_row = build_timing_row(method, baseline["timing_source"], n_samples=len(test_df))
        config = {
            "method": method,
            "baseline_type": "retrieval_only",
            "embedding_model": baseline["embedding_model"],
            "source_raw_predictions": str(baseline["raw_predictions"].relative_to(ROOT)),
            "source_timing": str(baseline["timing_source"].relative_to(ROOT)),
            "source_methods_with_same_retriever": baseline["source_methods"],
            "prediction_source_column": "retrieved_competencies",
            "label_space_source": "train.csv",
            "test_set_source": "test.csv",
            "k_values": K_VALUES,
            "n_samples": int(y_test.shape[0]),
            "n_labels": int(y_test.shape[1]),
            "note": baseline["note"],
        }

        paths = export_artifacts(
            results_dir=baseline["results_dir"],
            method=method,
            metrics_df=metrics_df,
            predictions_df=predictions_df,
            timing_row=timing_row,
            config=config,
        )
        print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
