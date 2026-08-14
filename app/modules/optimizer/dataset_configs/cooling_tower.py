from __future__ import annotations

from collections.abc import Callable

from app.modules.optimizer.dataset_configs.base import DatasetConfig, NumericRow, warning_rule
from app.modules.optimizer.schemas import ValidationRule

FIELDS = ("time", "t_cw_in", "t_cw_out", "t_wb_wea", "air_water_ratio")


def _range_rule(
    rows: list[NumericRow],
    *,
    rule_id: str,
    name: str,
    constraint: str,
    value: Callable[[dict[str, float]], float | None],
    minimum: float,
    maximum: float,
) -> ValidationRule:
    invalid_rows: list[int] = []
    for row_number, values in rows:
        result = value(values)
        if result is None or result < minimum or result > maximum:
            invalid_rows.append(row_number)
    return warning_rule(
        rule_id=rule_id,
        name=name,
        constraint=constraint,
        invalid_rows=invalid_rows,
    )


def validate_rules(rows: list[NumericRow]) -> list[ValidationRule]:
    outlet_values = [values["t_cw_out"] for _, values in rows]
    outlet_passed = bool(outlet_values) and max(outlet_values) - min(outlet_values) <= 3.0
    records = [
        warning_rule(
            rule_id="cooling_tower_outlet_temperature_range",
            name="Outlet temperature range",
            constraint="max(t_cw_out) - min(t_cw_out) <= 3",
            invalid_rows=[] if outlet_passed else [row_number for row_number, _ in rows],
        ),
        _range_rule(
            rows,
            rule_id="cooling_tower_outlet_wet_bulb_difference",
            name="Outlet-to-wet-bulb difference",
            constraint="2 <= t_cw_out - t_wb_wea <= 6",
            value=lambda values: values["t_cw_out"] - values["t_wb_wea"],
            minimum=2.0,
            maximum=6.0,
        ),
        _range_rule(
            rows,
            rule_id="cooling_tower_heat_exchange_efficiency",
            name="Heat-exchange efficiency",
            constraint="0.4 <= (t_cw_in - t_cw_out) / (t_cw_in - t_wb_wea) <= 0.8",
            value=lambda values: (
                (values["t_cw_in"] - values["t_cw_out"])
                / (values["t_cw_in"] - values["t_wb_wea"])
                if values["t_cw_in"] != values["t_wb_wea"]
                else None
            ),
            minimum=0.4,
            maximum=0.8,
        ),
        _range_rule(
            rows,
            rule_id="cooling_tower_air_water_ratio",
            name="Air-water ratio",
            constraint="0.4 <= air_water_ratio <= 2",
            value=lambda values: values["air_water_ratio"],
            minimum=0.4,
            maximum=2.0,
        ),
    ]
    return records


CONFIG = DatasetConfig(
    device_type="cooling_tower",
    label="Cooling tower",
    fields=FIELDS,
    validate_rules=validate_rules,
)
