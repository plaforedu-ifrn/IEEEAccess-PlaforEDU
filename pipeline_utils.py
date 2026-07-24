import ast
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    coverage_error,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_SEED = 42
DEFAULT_K_VALUES = [1, 3, 5, 7, 10]
LABEL_COLUMN = "comp_name_filtered"
TEXT_COLUMN = "combinedText"


def _unique_sorted(values):
    return sorted({v for v in values if pd.notna(v)})


def parse_label_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        return list(parsed)
    return list(value)


def serialize_label_list(value):
    return json.dumps(list(value), ensure_ascii=False)


def process_raw_dataset(raw_df, min_course_count=5):
    required = [
        "courseId",
        "courseName",
        "courseDescription",
        "comp_id",
        "comp_name",
        "comp_description",
    ]
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw_df.dropna(subset=["courseName", "courseDescription", "comp_name"]).copy()
    df = df.drop_duplicates()

    grouped = (
        df.groupby("courseId", as_index=False)
        .agg(
            {
                "courseName": "first",
                "courseDescription": "first",
                "courseHours": "first",
                "comp_id": _unique_sorted,
                "comp_name": _unique_sorted,
                "comp_description": _unique_sorted,
                "cat_id": _unique_sorted,
                "cat_name": _unique_sorted,
                "tax_id": _unique_sorted,
                "tax_name": _unique_sorted,
            }
        )
        .sort_values("courseId")
        .reset_index(drop=True)
    )

    counts = grouped["comp_name"].explode().value_counts()
    valid_labels = sorted(counts[counts >= min_course_count].index.tolist())
    removed_labels = sorted(counts[counts < min_course_count].index.tolist())

    grouped[LABEL_COLUMN] = grouped["comp_name"].apply(
        lambda labels: [label for label in labels if label in valid_labels]
    )
    processed = grouped[grouped[LABEL_COLUMN].map(len) > 0].copy()
    processed[TEXT_COLUMN] = (
        processed["courseName"].fillna("").astype(str).str.strip()
        + " "
        + processed["courseDescription"].fillna("").astype(str).str.strip()
    ).str.strip()

    metadata = {
        "min_course_count": min_course_count,
        "n_raw_rows": int(len(raw_df)),
        "n_rows_after_dropna": int(len(df)),
        "n_courses_before_filtering": int(len(grouped)),
        "n_courses_after_filtering": int(len(processed)),
        "n_labels_before_filtering": int(len(counts)),
        "n_labels_after_filtering": int(len(valid_labels)),
        "valid_labels": valid_labels,
        "removed_labels": removed_labels,
    }
    return processed.reset_index(drop=True), metadata


def save_processed_csv(processed_df, path):
    output = processed_df.copy()
    for col in output.columns:
        if output[col].map(lambda x: isinstance(x, list)).any():
            output[col] = output[col].apply(serialize_label_list)
    output.to_csv(path, index=False)


def load_processed_csv(path):
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
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_label_list)
    return df


def split_processed_dataset(processed_df, test_size=0.2, seed=DEFAULT_SEED):
    train_df, test_df = train_test_split(
        processed_df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_df = train_df.sort_values("courseId").reset_index(drop=True)
    test_df = test_df.sort_values("courseId").reset_index(drop=True)
    metadata = {
        "test_size": test_size,
        "seed": seed,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }
    return train_df, test_df, metadata


def save_split_csv(df, path):
    save_processed_csv(df, path)


def load_split_csv(path):
    return load_processed_csv(path)


def prepare_multilabel_targets(train_df, test_df, label_column=LABEL_COLUMN):
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df[label_column])
    y_test = mlb.transform(test_df[label_column])
    return y_train, y_test, list(mlb.classes_), mlb


def labels_to_matrix(label_lists, classes):
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y = np.zeros((len(label_lists), len(classes)), dtype=int)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            if label in class_to_idx:
                y[row_idx, class_to_idx[label]] = 1
    return y


def normalize_score_matrix(y_score, n_labels):
    if isinstance(y_score, list):
        y_score = np.column_stack(
            [score[:, 1] if score.ndim == 2 and score.shape[1] > 1 else score.ravel() for score in y_score]
        )
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)
    if y_score.shape[1] != n_labels:
        raise ValueError(f"Score matrix has {y_score.shape[1]} labels, expected {n_labels}.")
    return y_score


def top_k_prediction_matrix(y_score, k):
    y_score = np.asarray(y_score, dtype=float)
    k = min(k, y_score.shape[1])
    top_indices = np.argsort(y_score, axis=1)[:, -k:]
    y_pred = np.zeros_like(y_score, dtype=int)
    for row_idx, indices in enumerate(top_indices):
        y_pred[row_idx, indices] = 1
    return y_pred, top_indices


def compute_multilabel_metrics(y_true, y_score, labels, method, k_values=DEFAULT_K_VALUES):
    y_true = np.asarray(y_true, dtype=int)
    y_score = normalize_score_matrix(y_score, y_true.shape[1])
    labels = list(labels)

    metric_rows = []
    prediction_rows = []
    for k in k_values:
        if k > y_true.shape[1]:
            continue
        y_pred, top_indices = top_k_prediction_matrix(y_score, k)
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

        try:
            lrap = label_ranking_average_precision_score(y_true, y_score)
        except ValueError:
            lrap = np.nan

        try:
            cov_error = coverage_error(y_true, y_score)
        except ValueError:
            cov_error = np.nan

        metric_rows.append(
            {
                "method": method,
                "k": k,
                "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "hamming_loss": hamming_loss(y_true, y_pred),
                "subset_accuracy": accuracy_score(y_true, y_pred),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "partial_hit_at_k": partial_hit_at_k,
                "lrap": lrap,
                "coverage_error": cov_error,
                "n_samples": int(y_true.shape[0]),
                "n_labels": int(y_true.shape[1]),
            }
        )

        for row_idx, indices in enumerate(top_indices):
            ranked = list(reversed(indices.tolist()))
            prediction_rows.append(
                {
                    "method": method,
                    "k": k,
                    "sample_index": row_idx,
                    "predicted_labels": [labels[idx] for idx in ranked],
                    "true_labels": [labels[idx] for idx in np.where(y_true[row_idx] == 1)[0]],
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def compute_ranked_label_metrics(y_true, ranked_predictions, labels, method, k_values=DEFAULT_K_VALUES):
    y_true = np.asarray(y_true, dtype=int)
    labels = list(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    metric_rows = []
    prediction_rows = []
    for k in k_values:
        if k > len(labels):
            continue

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
                "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "hamming_loss": hamming_loss(y_true, y_pred),
                "subset_accuracy": accuracy_score(y_true, y_pred),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "partial_hit_at_k": partial_hit_at_k,
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
                    "true_labels": [labels[idx] for idx in np.where(y_true[row_idx] == 1)[0]],
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def build_frequency_baseline_scores(y_train, n_samples):
    frequencies = np.asarray(y_train, dtype=float).mean(axis=0)
    return np.tile(frequencies, (n_samples, 1))


def build_random_distribution_baseline_scores(y_train, n_samples, seed=DEFAULT_SEED):
    frequencies = np.asarray(y_train, dtype=float).mean(axis=0)
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, len(frequencies))) * frequencies.reshape(1, -1)


def evaluate_baselines(y_train, y_test, labels, k_values=DEFAULT_K_VALUES, seed=DEFAULT_SEED):
    rows = []
    predictions = []
    timings = []
    baselines = {
        "baseline_frequency_topk": lambda: build_frequency_baseline_scores(y_train, len(y_test)),
        "baseline_random_distribution": lambda: build_random_distribution_baseline_scores(
            y_train, len(y_test), seed=seed
        ),
    }

    for method, builder in baselines.items():
        train_start = time.perf_counter()
        train_time = time.perf_counter() - train_start
        infer_start = time.perf_counter()
        scores = builder()
        infer_time = time.perf_counter() - infer_start
        metric_df, pred_df = compute_multilabel_metrics(y_test, scores, labels, method, k_values)
        rows.append(metric_df)
        predictions.append(pred_df)
        timings.append(
            {
                "method": method,
                "train_seconds": train_time,
                "inference_seconds": infer_time,
                "inference_seconds_per_sample": infer_time / max(1, len(y_test)),
            }
        )

    return pd.concat(rows, ignore_index=True), pd.concat(predictions, ignore_index=True), pd.DataFrame(timings)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def export_experiment_artifacts(
    results_dir,
    method,
    metrics_df,
    predictions_df,
    timing_rows,
    config,
):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / f"{method}_metrics.csv"
    sensitivity_path = results_dir / f"{method}_sensitivity_by_k.csv"
    predictions_path = results_dir / f"{method}_predictions.csv"
    timing_path = results_dir / f"{method}_timing.csv"
    config_path = results_dir / f"{method}_config.json"

    metrics_df[metrics_df["k"] == 7].to_csv(metrics_path, index=False)
    metrics_df.to_csv(sensitivity_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)

    enriched_config = dict(config)
    enriched_config["run_datetime_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(config_path, enriched_config)

    return {
        "metrics": str(metrics_path),
        "sensitivity": str(sensitivity_path),
        "predictions": str(predictions_path),
        "timing": str(timing_path),
        "config": str(config_path),
    }
