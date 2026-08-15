from __future__ import annotations

import math
from typing import Any

from app.modules.optimizer.model_configs.base import (
    ModelConfig,
    TrainingResult,
    aggregate_metrics,
    build_series,
    predict_quadratic,
)

REQUIRED_FIELDS = ("t_cw_in", "t_cw_out", "t_wb_wea", "air_water_ratio")


def train(rows: list[dict[str, float]]) -> TrainingResult:
    training_rows: list[dict[str, float]] = []
    for row in rows:
        denominator = row["t_cw_in"] - row["t_wb_wea"]
        if denominator == 0:
            continue
        eta = (row["t_cw_in"] - row["t_cw_out"]) / denominator
        if math.isfinite(eta):
            training_rows.append({
                "t_wb": row["t_wb_wea"],
                "eta": eta,
                "air_water_ratio": row["air_water_ratio"],
            })
    series, coefficients = build_series(
        key="air_water_ratio",
        name="Air-water ratio model",
        kind="model",
        rows=training_rows,
        first="t_wb",
        second="eta",
        target="air_water_ratio",
    )
    return TrainingResult(
        artifact={
            "kind": "cooling_tower_polynomial",
            "units": {"t_wb": "°C", "eta": "1", "air_water_ratio": "1"},
            "coefficients": coefficients,
        },
        series=[series],
        metrics=aggregate_metrics([series]),
    )


def predict(artifact: dict[str, Any], inputs: dict[str, float]) -> dict[str, float]:
    t_wb = inputs.get("t_wb", inputs.get("t_wb_wea"))
    if t_wb is None:
        raise ValueError("Cooling-tower prediction requires t_wb or t_wb_wea")
    eta = inputs.get("eta")
    if eta is None:
        denominator = inputs["t_cw_in"] - t_wb
        if denominator == 0:
            raise ValueError("Cooling-tower efficiency denominator cannot be zero")
        eta = (inputs["t_cw_in"] - inputs["t_cw_out"]) / denominator
    return {
        "air_water_ratio": predict_quadratic(
            t_wb,
            eta,
            artifact["coefficients"],
            "t_wb",
            "eta",
        )
    }


CONFIG = ModelConfig(
    device_type="cooling_tower",
    required_fields=REQUIRED_FIELDS,
    algorithms=("polynomial",),
    train_polynomial=train,
    predict=predict,
)
