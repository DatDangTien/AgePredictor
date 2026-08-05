from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


AGE_INTERVALS: tuple[tuple[str, int, int], ...] = (
    ("0-12", 0, 12),
    ("13-19", 13, 19),
    ("20-29", 20, 29),
    ("30-39", 30, 39),
    ("40-49", 40, 49),
    ("50-59", 50, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80-89", 80, 89),
    ("90-99", 90, 99),
    ("100-109", 100, 109),
    ("110-119", 110, 119),
)
GENDER_LABELS = ("Female", "Male")


def latency_summary(values_ms: Sequence[float]) -> dict[str, Any]:
    if not values_ms:
        return {
            "samples": 0,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "fps_from_mean": None,
        }
    values = np.asarray(values_ms, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "samples": int(values.size),
        "mean_ms": mean_ms,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "fps_from_mean": float(1000.0 / mean_ms) if mean_ms > 0 else None,
    }


def value_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


def age_metrics(
    predictions: Sequence[float],
    labels: Sequence[int],
) -> dict[str, Any]:
    if not predictions:
        return _empty_age_metrics(evaluated_images=0)
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(labels, dtype=np.float64)
    return {
        "evaluated_images": int(predicted.size),
        **_age_error_metrics(predicted, expected),
    }


def age_metrics_by_interval(
    predictions: Sequence[float],
    labels: Sequence[int],
) -> dict[str, dict[str, Any]]:
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = np.asarray(labels, dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}

    for name, start, end in AGE_INTERVALS:
        mask = (expected >= start) & (expected <= end)
        if not np.any(mask):
            result[name] = {"samples": 0, **_empty_age_metrics()}
            continue
        result[name] = {
            "samples": int(mask.sum()),
            **_age_error_metrics(predicted[mask], expected[mask]),
        }
    return result


def _empty_age_metrics(*, evaluated_images: int | None = None) -> dict[str, Any]:
    result = {
        "mae": None,
        "rmse": None,
        "mean_signed_error": None,
        "median_absolute_error": None,
        "p90_absolute_error": None,
        "within_5_years": None,
        "within_10_years": None,
    }
    if evaluated_images is not None:
        return {"evaluated_images": evaluated_images, **result}
    return result


def _age_error_metrics(
    predicted: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float]:
    error = predicted - expected
    absolute_error = np.abs(error)
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_signed_error": float(error.mean()),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.percentile(absolute_error, 90)),
        "within_5_years": float(np.mean(absolute_error <= 5.0)),
        "within_10_years": float(np.mean(absolute_error <= 10.0)),
    }


def gender_metrics(
    predictions: Sequence[str],
    labels: Sequence[str],
    female_probabilities: Sequence[float],
) -> dict[str, Any]:
    if not predictions:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "female_accuracy": None,
            "male_accuracy": None,
            "female_precision": None,
            "female_recall": None,
            "female_f1": None,
            "male_precision": None,
            "male_recall": None,
            "male_f1": None,
            "roc_auc_female": None,
            "pr_auc_female": None,
            "confusion_matrix": _binary_confusion([], []),
        }

    confusion = _binary_confusion(predictions, labels)
    ff = confusion["true_female_pred_female"]
    fm = confusion["true_female_pred_male"]
    mf = confusion["true_male_pred_female"]
    mm = confusion["true_male_pred_male"]

    female_accuracy = _safe_divide(ff, ff + fm)
    male_accuracy = _safe_divide(mm, mm + mf)
    female_precision = _safe_divide(ff, ff + mf)
    female_recall = female_accuracy
    female_f1 = _f1(female_precision, female_recall)
    male_precision = _safe_divide(mm, mm + fm)
    male_recall = male_accuracy
    male_f1 = _f1(male_precision, male_recall)

    return {
        "accuracy": _safe_divide(ff + mm, len(predictions)),
        "balanced_accuracy": float((female_accuracy + male_accuracy) / 2.0),
        "macro_f1": float((female_f1 + male_f1) / 2.0),
        "female_accuracy": female_accuracy,
        "male_accuracy": male_accuracy,
        "female_precision": female_precision,
        "female_recall": female_recall,
        "female_f1": female_f1,
        "male_precision": male_precision,
        "male_recall": male_recall,
        "male_f1": male_f1,
        "roc_auc_female": _roc_auc_binary(labels, female_probabilities),
        "pr_auc_female": _pr_auc_binary(labels, female_probabilities),
        "confusion_matrix": confusion,
    }


def _binary_confusion(
    predictions: Sequence[str],
    labels: Sequence[str],
) -> dict[str, int]:
    return {
        "true_female_pred_female": int(
            sum(true == "Female" and pred == "Female" for true, pred in zip(labels, predictions))
        ),
        "true_female_pred_male": int(
            sum(true == "Female" and pred == "Male" for true, pred in zip(labels, predictions))
        ),
        "true_male_pred_female": int(
            sum(true == "Male" and pred == "Female" for true, pred in zip(labels, predictions))
        ),
        "true_male_pred_male": int(
            sum(true == "Male" and pred == "Male" for true, pred in zip(labels, predictions))
        ),
    }


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return float(2.0 * precision * recall / (precision + recall)) if (
        precision + recall
    ) else 0.0


def _roc_auc_binary(labels: Sequence[str], scores: Sequence[float]) -> float | None:
    positives = np.asarray([label == "Female" for label in labels], dtype=bool)
    if positives.size == 0 or positives.all() or (~positives).all():
        return None

    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        next_index = index + 1
        while (
            next_index < values.size
            and values[order[next_index]] == values[order[index]]
        ):
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        ranks[order[index:next_index]] = average_rank
        index = next_index

    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    positive_rank_sum = float(ranks[positives].sum())
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return float(auc)


def _pr_auc_binary(labels: Sequence[str], scores: Sequence[float]) -> float | None:
    positives = np.asarray([label == "Female" for label in labels], dtype=bool)
    positive_count = int(positives.sum())
    if positive_count == 0:
        return None

    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values)
    sorted_positives = positives[order]
    true_positives = np.cumsum(sorted_positives)
    ranks = np.arange(1, sorted_positives.size + 1)
    precision = true_positives / ranks
    return float(precision[sorted_positives].sum() / positive_count)
