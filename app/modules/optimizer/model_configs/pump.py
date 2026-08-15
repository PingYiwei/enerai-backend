from __future__ import annotations

from typing import Any

from app.modules.optimizer.model_configs.base import (
    ModelConfig,
    TrainingResult,
    aggregate_metrics,
    build_series,
    predict_quadratic,
)

REQUIRED_FIELDS = ("flow", "eff_pump", "head")


def train(rows: list[dict[str, float]]) -> TrainingResult:
    efficiency, efficiency_coefficients = build_series(
        key="flow_efficiency",
        name="Flow-efficiency model",
        kind="model",
        rows=rows,
        first="flow",
        second=None,
        target="eff_pump",
    )
    head, head_coefficients = build_series(
        key="flow_head",
        name="Flow-head model",
        kind="model",
        rows=rows,
        first="flow",
        second=None,
        target="head",
    )
    series = [efficiency, head]
    return TrainingResult(
        artifact={
            "kind": "pump_polynomial",
            "units": {"flow": "m³/h", "eff_pump": "1", "head": "m"},
            "coefficients": {
                "flow_efficiency": efficiency_coefficients,
                "flow_head": head_coefficients,
            },
        },
        series=series,
        metrics=aggregate_metrics(series),
    )


def predict(artifact: dict[str, Any], inputs: dict[str, float]) -> dict[str, float]:
    flow = inputs["flow"]
    coefficients = artifact["coefficients"]
    return {
        "eff_pump": predict_quadratic(
            flow, None, coefficients["flow_efficiency"], "flow"
        ),
        "head": predict_quadratic(flow, None, coefficients["flow_head"], "flow"),
    }


CONFIG = ModelConfig(
    device_type="pump",
    required_fields=REQUIRED_FIELDS,
    algorithms=("polynomial",),
    train_polynomial=train,
    predict=predict,
)

