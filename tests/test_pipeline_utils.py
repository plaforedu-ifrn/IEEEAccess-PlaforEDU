import tempfile

import numpy as np
import pandas as pd

from pipeline_utils import (
    build_frequency_baseline_scores,
    build_random_distribution_baseline_scores,
    compute_multilabel_metrics,
    compute_ranked_label_metrics,
    load_split_csv,
    process_raw_dataset,
    save_split_csv,
    split_processed_dataset,
)


def _raw_dataset():
    rows = []
    labels_by_course = {
        "c1": ["A", "B"],
        "c2": ["A", "B"],
        "c3": ["A", "B"],
        "c4": ["A", "C"],
        "c5": ["A", "C"],
        "c6": ["A", "D"],
    }
    for course_id, labels in labels_by_course.items():
        for label in labels:
            rows.append(
                {
                    "courseId": course_id,
                    "courseName": f"Course {course_id}",
                    "courseDescription": f"Description {course_id}",
                    "courseHours": 20,
                    "comp_id": label.lower(),
                    "comp_name": label,
                    "comp_description": f"Competency {label}",
                    "cat_id": "cat",
                    "cat_name": "Category",
                    "tax_id": "tax",
                    "tax_name": "Tax",
                }
            )
    return pd.DataFrame(rows)


def test_process_raw_dataset_filters_rare_labels_after_course_aggregation():
    processed, metadata = process_raw_dataset(_raw_dataset(), min_course_count=3)

    assert processed["courseId"].tolist() == ["c1", "c2", "c3", "c4", "c5", "c6"]
    assert processed.loc[processed["courseId"] == "c1", "comp_name_filtered"].iloc[0] == ["A", "B"]
    assert processed.loc[processed["courseId"] == "c4", "comp_name_filtered"].iloc[0] == ["A"]
    assert metadata["removed_labels"] == ["C", "D"]
    assert metadata["n_courses_after_filtering"] == 6


def test_split_processed_dataset_is_reproducible_and_preserves_label_strings():
    processed, _ = process_raw_dataset(_raw_dataset(), min_course_count=3)
    train_df, test_df, metadata = split_processed_dataset(processed, test_size=0.33, seed=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        train_path = f"{tmp_dir}/train.csv"
        save_split_csv(train_df, train_path)
        loaded_train = load_split_csv(train_path)

    assert metadata["n_train"] == len(train_df)
    assert metadata["n_test"] == len(test_df)
    assert isinstance(loaded_train["comp_name_filtered"].iloc[0], list)


def test_compute_multilabel_metrics_includes_strict_and_topk_metrics():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    y_score = np.array([[0.9, 0.1, 0.8], [0.6, 0.7, 0.2]])

    metrics, predictions = compute_multilabel_metrics(
        y_true,
        y_score,
        labels=["A", "B", "C"],
        method="demo",
        k_values=[1, 2],
    )

    assert set(metrics["k"]) == {1, 2}
    assert "micro_f1" in metrics.columns
    assert "macro_f1" in metrics.columns
    assert "hamming_loss" in metrics.columns
    assert "recall_at_k" in metrics.columns
    assert "subset_accuracy" in metrics.columns
    assert predictions.loc[predictions["k"] == 2, "predicted_labels"].iloc[0] == ["A", "C"]


def test_compute_ranked_label_metrics_uses_only_returned_labels_for_each_k():
    y_true = np.array([[1, 0, 1], [0, 1, 0]])
    ranked_predictions = [["A", "C", "B"], ["A"]]

    metrics, predictions = compute_ranked_label_metrics(
        y_true,
        ranked_predictions,
        labels=["A", "B", "C"],
        method="llm",
        k_values=[1, 2],
    )

    assert metrics.loc[metrics["k"] == 1, "partial_hit_at_k"].iloc[0] == 0.5
    assert metrics.loc[metrics["k"] == 2, "recall_at_k"].iloc[0] == 0.5
    assert predictions.loc[predictions["sample_index"] == 1, "predicted_labels"].iloc[0] == ["A"]


def test_baseline_scores_follow_train_label_distribution():
    y_train = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0]])

    freq_scores = build_frequency_baseline_scores(y_train, n_samples=2)
    random_scores_a = build_random_distribution_baseline_scores(y_train, n_samples=2, seed=7)
    random_scores_b = build_random_distribution_baseline_scores(y_train, n_samples=2, seed=7)

    assert freq_scores.shape == (2, 3)
    assert freq_scores[0, 0] > freq_scores[0, 2] > freq_scores[0, 1]
    np.testing.assert_allclose(random_scores_a, random_scores_b)
