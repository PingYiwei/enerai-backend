from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from app.modules.optimizer.schemas import DeviceType

type Algorithm = Literal["polynomial", "gradient_boosting"]

MODEL_FIELDS: dict[DeviceType, tuple[tuple[str, ...], str]] = {
    "chiller": (("load_pct", "t_chw_ret", "t_chw_sup", "t_cw_sup"), "q_cool"),
    "cooling_tower": (("t_cw_in", "t_wb_wea", "air_water_ratio"), "t_cw_out"),
    "pump": (("flow",), "head"),
}


@dataclass(frozen=True, slots=True)
class TrainingResult:
    artifact: dict[str, Any]
    predictions: list[float]
    metrics: dict[str, float]


def train_model(
    algorithm: Algorithm,
    device_type: DeviceType,
    rows: list[dict[str, float]],
) -> TrainingResult:
    feature_names, target_name = MODEL_FIELDS[device_type]
    features = [[row[name] for name in feature_names] for row in rows]
    targets = [row[target_name] for row in rows]
    if len(rows) < max(5, len(feature_names) + 2):
        raise ValueError("Dataset does not contain enough rows for training")
    if algorithm == "polynomial":
        matrix = [[1.0, *values, *(value * value for value in values)] for values in features]
        coefficients = _least_squares(matrix, targets)
        predictions = [_dot(coefficients, values) for values in matrix]
        artifact: dict[str, Any] = {
            "kind": "polynomial",
            "feature_names": list(feature_names),
            "target_name": target_name,
            "coefficients": coefficients,
        }
    else:
        base, stumps, predictions = _gradient_boost(features, targets)
        artifact = {
            "kind": "gradient_boosting",
            "feature_names": list(feature_names),
            "target_name": target_name,
            "base": base,
            "learning_rate": 0.15,
            "stumps": stumps,
        }
    return TrainingResult(
        artifact=artifact,
        predictions=predictions,
        metrics=_metrics(targets, predictions),
    )


def _least_squares(matrix: list[list[float]], targets: list[float]) -> list[float]:
    width = len(matrix[0])
    normal = [
        [sum(row[left] * row[right] for row in matrix) for right in range(width)]
        for left in range(width)
    ]
    projected = [
        sum(row[index] * target for row, target in zip(matrix, targets, strict=True))
        for index in range(width)
    ]
    for index in range(width):
        normal[index][index] += 1e-8
    return _solve(normal, projected)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    size = len(augmented)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-12:
            raise ValueError("Dataset features are singular and cannot be fitted")
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column], strict=True)
            ]
    return [row[-1] for row in augmented]


def _gradient_boost(
    features: list[list[float]], targets: list[float]
) -> tuple[float, list[dict[str, float | int]], list[float]]:
    learning_rate = 0.15
    base = sum(targets) / len(targets)
    predictions = [base] * len(targets)
    stumps: list[dict[str, float | int]] = []
    for _ in range(30):
        residuals = [
            target - prediction for target, prediction in zip(targets, predictions, strict=True)
        ]
        best: tuple[float, int, float, float, float] | None = None
        for feature_index in range(len(features[0])):
            distinct = sorted({row[feature_index] for row in features})
            candidates = distinct[1 : -1 : max(1, len(distinct) // 16)]
            for threshold in candidates:
                left = [
                    residual
                    for row, residual in zip(features, residuals, strict=True)
                    if row[feature_index] <= threshold
                ]
                right = [
                    residual
                    for row, residual in zip(features, residuals, strict=True)
                    if row[feature_index] > threshold
                ]
                if not left or not right:
                    continue
                left_value = sum(left) / len(left)
                right_value = sum(right) / len(right)
                loss = sum((value - left_value) ** 2 for value in left) + sum(
                    (value - right_value) ** 2 for value in right
                )
                candidate = (loss, feature_index, threshold, left_value, right_value)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            break
        _, feature_index, threshold, left_value, right_value = best
        stump = {
            "feature": feature_index,
            "threshold": threshold,
            "left": left_value,
            "right": right_value,
        }
        stumps.append(stump)
        for index, row in enumerate(features):
            update = left_value if row[feature_index] <= threshold else right_value
            predictions[index] += learning_rate * update
    return base, stumps, predictions


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    errors = [actual - predicted for actual, predicted in zip(targets, predictions, strict=True)]
    mean = sum(targets) / len(targets)
    total = sum((actual - mean) ** 2 for actual in targets)
    residual = sum(error * error for error in errors)
    return {
        "r2": 1.0 - residual / total if total else 1.0,
        "rmse": math.sqrt(residual / len(errors)),
        "mae": sum(abs(error) for error in errors) / len(errors),
    }
