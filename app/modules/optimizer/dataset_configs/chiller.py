from __future__ import annotations

import math

from app.modules.optimizer.dataset_configs.base import (
    DatasetConfig,
    NumericRow,
    error_rule,
    warning_rule,
)
from app.modules.optimizer.schemas import ValidationRule

FIELDS = (
    "time",
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


def _minimum_difference(
    rows: list[NumericRow],
    *,
    rule_id: str,
    name: str,
    left: str,
    right: str,
    minimum: float,
) -> ValidationRule:
    invalid_rows = [
        row_number
        for row_number, values in rows
        if values[left] - values[right] < minimum
        and not math.isclose(values[left] - values[right], minimum, abs_tol=1e-9)
    ]
    return warning_rule(
        rule_id=rule_id,
        name=name,
        constraint=f"{left} - {right} >= {minimum:g}",
        invalid_rows=invalid_rows,
    )


def validate_rules(rows: list[NumericRow]) -> list[ValidationRule]:
    return [
        error_rule(
            rule_id="chiller_load_ratio_unit",
            name="Chiller load ratio",
            constraint="0 < load_pct <= 1.2 (dimensionless ratio, not percent)",
            invalid_rows=[
                row_number
                for row_number, values in rows
                if not 0 < values["load_pct"] <= 1.2
            ],
        ),
        _minimum_difference(
            rows,
            rule_id="chiller_condensing_approach",
            name="Condensing approach",
            left="t_cond",
            right="t_cw_ret",
            minimum=0.3,
        ),
        _minimum_difference(
            rows,
            rule_id="chiller_evaporating_approach",
            name="Evaporating approach",
            left="t_chw_sup",
            right="t_evap",
            minimum=0.3,
        ),
        _minimum_difference(
            rows,
            rule_id="chiller_chilled_water_delta",
            name="Chilled-water temperature difference",
            left="t_chw_ret",
            right="t_chw_sup",
            minimum=2.0,
        ),
        _minimum_difference(
            rows,
            rule_id="chiller_cooling_water_delta",
            name="Cooling-water temperature difference",
            left="t_cw_ret",
            right="t_cw_sup",
            minimum=2.0,
        ),
    ]


CONFIG = DatasetConfig(
    device_type="chiller",
    label="Chiller",
    fields=FIELDS,
    field_units={
        "time": "YYYY-MM-DD HH:mm:ss",
        "t_chw_ret": "°C",
        "t_chw_sup": "°C",
        "t_cw_sup": "°C",
        "t_cw_ret": "°C",
        "flow_chw": "m³/h",
        "flow_cw": "m³/h",
        "q_cool": "kW",
        "q_reject": "kW",
        "load_pct": "1",
        "t_evap": "°C",
        "t_cond": "°C",
    },
    validate_rules=validate_rules,
)

