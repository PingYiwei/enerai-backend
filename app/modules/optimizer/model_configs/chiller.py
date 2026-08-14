from __future__ import annotations

import math
from typing import Any

from app.modules.optimizer.model_configs.base import (
    ModelConfig,
    TrainingResult,
    aggregate_metrics,
    build_series,
    evaluation_series,
    predict_quadratic,
)

REQUIRED_FIELDS = (
    "t_chw_ret",
    "t_chw_sup",
    "t_cw_sup",
    "t_cw_ret",
    "flow_chw",
    "flow_cw",
    "q_cool",
    "q_reject",
    "load_pct",
    "t_evap",
    "t_cond",
)


def _lmtd(delta_t1: float, delta_t2: float) -> float:
    if not delta_t1 > delta_t2 > 0:
        raise ValueError("LMTD terminal differences must satisfy delta_t1 > delta_t2 > 0")
    result = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("LMTD is invalid")
    return result


def _temperature_offset(water_delta: float, lmtd: float) -> float:
    if water_delta <= 0 or lmtd <= 0:
        raise ValueError("Water temperature difference and LMTD must be positive")
    denominator = math.expm1(water_delta / lmtd)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("LMTD inverse is invalid")
    return water_delta / denominator


def _actual_values(values: dict[str, float]) -> dict[str, float]:
    lmtd_evap = _lmtd(
        values["t_chw_ret"] - values["t_evap"],
        values["t_chw_sup"] - values["t_evap"],
    )
    lmtd_cond = _lmtd(
        values["t_cond"] - values["t_cw_sup"],
        values["t_cond"] - values["t_cw_ret"],
    )
    tce = values["t_cond"] - values["t_evap"]
    power = values["q_reject"] - values["q_cool"]
    if tce <= 0 or power <= 0:
        raise ValueError("Chiller lift and power must be positive")
    icop = (values["t_evap"] + 273.15) / tce
    cop = values["q_cool"] / power
    actual = {
        "kf_evap": values["q_cool"] / lmtd_evap,
        "kf_cond": values["q_reject"] / lmtd_cond,
        "t_evap": values["t_evap"],
        "t_cond": values["t_cond"],
        "tce": tce,
        "icop": icop,
        "cop": cop,
        "dcop": cop / icop,
        "power": power,
    }
    if not all(math.isfinite(value) and value > 0 for value in actual.values()):
        raise ValueError("Chiller operating point violates model constraints")
    return actual


def _predict_values(
    values: dict[str, float], coefficients: dict[str, dict[str, float]]
) -> dict[str, float]:
    if not (
        values["q_cool"] > 0
        and values["q_reject"] > values["q_cool"]
        and values["flow_chw"] > 0
        and values["flow_cw"] > 0
        and values["load_pct"] > 0
        and values["t_chw_ret"] > values["t_chw_sup"]
        and values["t_cw_ret"] > values["t_cw_sup"]
    ):
        raise ValueError("Chiller prediction inputs violate positive-load constraints")
    kf_evap = predict_quadratic(
        values["q_cool"],
        values["flow_chw"],
        coefficients["evaporator"],
        "q_cool",
        "flow_chw",
    )
    t_evap = values["t_chw_sup"] - _temperature_offset(
        values["t_chw_ret"] - values["t_chw_sup"],
        values["q_cool"] / kf_evap,
    )
    kf_cond = predict_quadratic(
        values["q_reject"],
        values["flow_cw"],
        coefficients["condenser"],
        "q_reject",
        "flow_cw",
    )
    t_cond = values["t_cw_ret"] + _temperature_offset(
        values["t_cw_ret"] - values["t_cw_sup"],
        values["q_reject"] / kf_cond,
    )
    tce = t_cond - t_evap
    dcop = predict_quadratic(
        tce,
        values["load_pct"],
        coefficients["compressor"],
        "tce",
        "load_pct",
    )
    icop = (t_evap + 273.15) / tce
    cop = icop * dcop
    power = values["q_cool"] / cop
    result = {
        "kf_evap": kf_evap,
        "kf_cond": kf_cond,
        "t_evap": t_evap,
        "t_cond": t_cond,
        "tce": tce,
        "icop": icop,
        "cop": cop,
        "dcop": dcop,
        "power": power,
    }
    if not all(math.isfinite(value) and value > 0 for value in result.values()):
        raise ValueError("Chiller prediction produced an invalid physical result")
    return result


def train(rows: list[dict[str, float]]) -> TrainingResult:
    evaporator_rows: list[dict[str, float]] = []
    condenser_rows: list[dict[str, float]] = []
    compressor_rows: list[dict[str, float]] = []
    evaluation_rows: list[dict[str, float]] = []
    for row in rows:
        try:
            actual = _actual_values(row)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        base = {**row, **actual}
        evaporator_rows.append(base)
        condenser_rows.append(base)
        compressor_rows.append(base)
        evaluation_rows.append(base)

    evaporator, evaporator_coefficients = build_series(
        key="evaporator",
        name="Evaporator performance",
        kind="submodel",
        rows=evaporator_rows,
        first="q_cool",
        second="flow_chw",
        target="kf_evap",
    )
    condenser, condenser_coefficients = build_series(
        key="condenser",
        name="Condenser performance",
        kind="submodel",
        rows=condenser_rows,
        first="q_reject",
        second="flow_cw",
        target="kf_cond",
    )
    compressor, compressor_coefficients = build_series(
        key="compressor",
        name="Compressor performance",
        kind="submodel",
        rows=compressor_rows,
        first="tce",
        second="load_pct",
        target="dcop",
    )
    coefficients = {
        "evaporator": evaporator_coefficients,
        "condenser": condenser_coefficients,
        "compressor": compressor_coefficients,
    }

    actual_by_field: dict[str, list[float]] = {
        field: [] for field in ("t_evap", "t_cond", "cop", "power")
    }
    predicted_by_field: dict[str, list[float]] = {
        field: [] for field in actual_by_field
    }
    for row in evaluation_rows:
        try:
            predicted = _predict_values(row, coefficients)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        for field in actual_by_field:
            actual_by_field[field].append(row[field])
            predicted_by_field[field].append(predicted[field])
    if not actual_by_field["t_evap"]:
        raise ValueError("No valid chiller operating points remain for chained evaluation")

    labels = {
        "t_evap": "Evaporating temperature",
        "t_cond": "Condensing temperature",
        "cop": "Coefficient of performance",
        "power": "Compressor power",
    }
    evaluations = [
        evaluation_series(
            f"evaluation_{field}",
            labels[field],
            field,
            actual_by_field[field],
            predicted_by_field[field],
        )
        for field in actual_by_field
    ]
    series = [evaporator, condenser, compressor, *evaluations]
    return TrainingResult(
        artifact={"kind": "chiller_polynomial", "coefficients": coefficients},
        series=series,
        metrics=aggregate_metrics(series),
    )


def predict(artifact: dict[str, Any], inputs: dict[str, float]) -> dict[str, float]:
    return _predict_values(inputs, artifact["coefficients"])


CONFIG = ModelConfig(
    device_type="chiller",
    required_fields=REQUIRED_FIELDS,
    algorithms=("polynomial",),
    train_polynomial=train,
    predict=predict,
)
