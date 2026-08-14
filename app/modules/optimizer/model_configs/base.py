from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from app.modules.optimizer.schemas import DeviceType

type Algorithm = Literal["polynomial", "gradient_boosting"]
type ModelSeries = dict[str, Any]
type TrainFunction = Callable[[list[dict[str, float]]], "TrainingResult"]
type PredictFunction = Callable[[dict[str, Any], dict[str, float]], dict[str, float]]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    artifact: dict[str, Any]
    series: list[ModelSeries]
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    device_type: DeviceType
    required_fields: tuple[str, ...]
    algorithms: tuple[Algorithm, ...]
    train_polynomial: TrainFunction
    predict: PredictFunction


def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Model evaluation requires matching non-empty values")
    errors = [truth - estimate for truth, estimate in zip(actual, predicted, strict=True)]
    mean = math.fsum(actual) / len(actual)
    total = math.fsum((truth - mean) ** 2 for truth in actual)
    residual = math.fsum(error * error for error in errors)
    percentage_errors = [
        abs(error / truth) for truth, error in zip(actual, errors, strict=True) if truth != 0
    ]
    return {
        "sample_count": float(len(actual)),
        "r2": 1.0 - residual / total if total else 1.0,
        "rmse": math.sqrt(residual / len(errors)),
        "mae": math.fsum(abs(error) for error in errors) / len(errors),
        "mape": (
            math.fsum(percentage_errors) * 100 / len(percentage_errors)
            if percentage_errors
            else 0.0
        ),
    }


def aggregate_metrics(series: list[ModelSeries]) -> dict[str, float]:
    metrics = [item["metrics"] for item in series if item.get("kind") != "submodel"]
    if not metrics:
        metrics = [item["metrics"] for item in series]
    return {
        key: math.fsum(float(item[key]) for item in metrics) / len(metrics)
        for key in ("sample_count", "r2", "rmse", "mae", "mape")
    }


def sample_points(points: list[dict[str, float]], limit: int = 300) -> list[dict[str, float]]:
    if len(points) <= limit:
        return points
    step = (len(points) - 1) / (limit - 1)
    return [points[round(index * step)] for index in range(limit)]


def fit_quadratic(
    rows: list[dict[str, float]], first: str, second: str | None, target: str
) -> dict[str, float]:
    feature_count = 3 if second is None else 6
    minimum = max(feature_count, 5)
    if len(rows) < minimum:
        raise ValueError(f"At least {minimum} valid rows are required for polynomial training")
    matrix = [quadratic_features(row[first], row[second] if second else None) for row in rows]
    coefficients = least_squares(matrix, [row[target] for row in rows])
    names = coefficient_names(first, second)
    return dict(zip(names, coefficients, strict=True))


def quadratic_features(first: float, second: float | None = None) -> list[float]:
    if second is None:
        return [1.0, first, first * first]
    return [1.0, first, second, first * first, first * second, second * second]


def coefficient_names(first: str, second: str | None = None) -> tuple[str, ...]:
    if second is None:
        return ("intercept", first, f"{first}_squared")
    return (
        "intercept",
        first,
        second,
        f"{first}_squared",
        f"{first}_{second}",
        f"{second}_squared",
    )


def predict_quadratic(
    first_value: float,
    second_value: float | None,
    coefficients: dict[str, float],
    first: str,
    second: str | None = None,
) -> float:
    return math.fsum(
        coefficients[name] * feature
        for name, feature in zip(
            coefficient_names(first, second),
            quadratic_features(first_value, second_value),
            strict=True,
        )
    )


def format_formula(
    output: str, coefficients: dict[str, float], first: str, second: str | None = None
) -> str:
    names = coefficient_names(first, second)
    expressions = [first, f"{first}^2"] if second is None else [
        first,
        second,
        f"{first}^2",
        f"{first}*{second}",
        f"{second}^2",
    ]
    formula = f"{output} = {coefficients['intercept']:.8g}"
    for name, expression in zip(names[1:], expressions, strict=True):
        value = coefficients[name]
        formula += f" {'+' if value >= 0 else '-'} {abs(value):.8g}*{expression}"
    return formula


def build_series(
    *,
    key: str,
    name: str,
    kind: Literal["model", "submodel", "evaluation"],
    rows: list[dict[str, float]],
    first: str,
    second: str | None,
    target: str,
) -> tuple[ModelSeries, dict[str, float]]:
    coefficients = fit_quadratic(rows, first, second, target)
    predicted = [
        predict_quadratic(row[first], row[second] if second else None, coefficients, first, second)
        for row in rows
    ]
    points = [
        {
            "actual": row[target],
            "predicted": estimate,
            first: row[first],
            **({second: row[second]} if second else {}),
        }
        for row, estimate in zip(rows, predicted, strict=True)
    ]
    return (
        {
            "key": key,
            "name": name,
            "kind": kind,
            "input_fields": [first, *([second] if second else [])],
            "output_field": target,
            "metrics": regression_metrics([row[target] for row in rows], predicted),
            "formula": format_formula(target, coefficients, first, second),
            "points": sample_points(points),
        },
        coefficients,
    )


def evaluation_series(
    key: str,
    name: str,
    output: str,
    actual: list[float],
    predicted: list[float],
) -> ModelSeries:
    points = [
        {"actual": truth, "predicted": estimate}
        for truth, estimate in zip(actual, predicted, strict=True)
    ]
    return {
        "key": key,
        "name": name,
        "kind": "evaluation",
        "input_fields": [],
        "output_field": output,
        "metrics": regression_metrics(actual, predicted),
        "formula": "",
        "points": sample_points(points),
    }


def least_squares(matrix: list[list[float]], targets: list[float]) -> list[float]:
    width = len(matrix[0])
    normal = [
        [math.fsum(row[left] * row[right] for row in matrix) for right in range(width)]
        for left in range(width)
    ]
    projected = [
        math.fsum(row[index] * target for row, target in zip(matrix, targets, strict=True))
        for index in range(width)
    ]
    diagonal_mean = math.fsum(normal[index][index] for index in range(width)) / width
    ridge = max(diagonal_mean * 1e-12, 1e-12)
    for index in range(width):
        normal[index][index] += ridge
    return solve(normal, projected)


def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector, strict=True)]
    size = len(augmented)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        if abs(divisor) < 1e-14:
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
    coefficients = [row[-1] for row in augmented]
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Polynomial coefficients contain invalid values")
    return coefficients

