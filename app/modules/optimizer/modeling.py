from __future__ import annotations

from typing import Any

from app.modules.optimizer.model_configs import MODEL_CONFIGS
from app.modules.optimizer.model_configs.base import Algorithm, TrainingResult
from app.modules.optimizer.schemas import DeviceType

MODEL_FIELDS: dict[DeviceType, tuple[str, ...]] = {
    device_type: config.required_fields for device_type, config in MODEL_CONFIGS.items()
}


def train_model(
    algorithm: Algorithm,
    device_type: DeviceType,
    rows: list[dict[str, float]],
) -> TrainingResult:
    config = MODEL_CONFIGS[device_type]
    if algorithm not in config.algorithms:
        raise ValueError(f"{device_type} models do not support the {algorithm} algorithm")
    return config.train_polynomial(rows)


def predict_model(
    device_type: DeviceType,
    artifact: dict[str, Any],
    inputs: dict[str, float],
) -> dict[str, float]:
    try:
        outputs = MODEL_CONFIGS[device_type].predict(artifact, inputs)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f"Invalid {device_type} model inputs: {error}") from error
    if not outputs:
        raise ValueError("Model prediction did not produce outputs")
    return outputs
