from __future__ import annotations

from app.modules.optimizer.dataset_configs.base import DatasetConfig, NumericRow, error_rule
from app.modules.optimizer.schemas import ValidationRule

FIELDS = ("time", "flow", "eff_pump", "head")


def validate_rules(rows: list[NumericRow]) -> list[ValidationRule]:
    return [
        error_rule(
            rule_id="pump_positive_flow",
            name="Pump flow unit and range",
            constraint="flow must be greater than 0 m³/h",
            invalid_rows=[row_number for row_number, values in rows if values["flow"] <= 0],
        ),
        error_rule(
            rule_id="pump_efficiency_ratio",
            name="Pump efficiency ratio",
            constraint="0 < eff_pump <= 1 (dimensionless ratio, not percent)",
            invalid_rows=[
                row_number
                for row_number, values in rows
                if not 0 < values["eff_pump"] <= 1
            ],
        ),
        error_rule(
            rule_id="pump_positive_head",
            name="Pump head unit and range",
            constraint="head must be greater than 0 m",
            invalid_rows=[row_number for row_number, values in rows if values["head"] <= 0],
        ),
    ]


CONFIG = DatasetConfig(
    device_type="pump",
    label="Pump",
    fields=FIELDS,
    field_units={
        "time": "YYYY-MM-DD HH:mm:ss",
        "flow": "m³/h",
        "eff_pump": "1",
        "head": "m",
    },
    validate_rules=validate_rules,
)

