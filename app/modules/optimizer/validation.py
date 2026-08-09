from __future__ import annotations

import csv
import io
import math
from datetime import datetime

from app.modules.optimizer.schemas import DeviceType, ValidationRule

DEVICE_FIELDS: dict[DeviceType, tuple[str, ...]] = {
    "chiller": (
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
    ),
    "cooling_tower": ("time", "t_cw_in", "t_cw_out", "t_wb_wea", "air_water_ratio"),
    "pump": ("time", "flow", "eff_pump", "head"),
}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SAMPLE_LIMIT = 20


def decode_csv(content: bytes) -> tuple[list[str], list[list[str]]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV must use UTF-8 encoding") from error
    reader = csv.reader(io.StringIO(text, newline=""))
    records = list(reader)
    if not records:
        raise ValueError("CSV is empty")
    header = [column.strip() for column in records[0]]
    return header, records[1:]


def validate_dataset(
    device_type: DeviceType, header: list[str], rows: list[list[str]]
) -> list[ValidationRule]:
    expected = DEVICE_FIELDS[device_type]
    rules: list[ValidationRule] = []
    fields_passed = tuple(header) == expected
    rules.append(
        ValidationRule(
            rule_id="required_fields",
            name="Required fields",
            severity="error",
            passed=fields_passed,
            constraint="Columns must exactly match: " + ", ".join(expected),
            violation_count=0 if fields_passed else 1,
            message="Field set is valid"
            if fields_passed
            else "Missing, extra, or reordered fields",
        )
    )
    if not fields_passed:
        return rules

    wrong_width = [index for index, row in enumerate(rows, start=2) if len(row) != len(expected)]
    rules.append(
        _rows_rule(
            "row_width",
            "Row width",
            f"Every row must contain {len(expected)} fields",
            wrong_width,
        )
    )
    index_by_name = {name: index for index, name in enumerate(expected)}
    invalid_times: list[int] = []
    invalid_numbers: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        time_value = row[0].strip() if row else ""
        try:
            parsed = datetime.strptime(time_value, TIME_FORMAT)
            if parsed.strftime(TIME_FORMAT) != time_value:
                raise ValueError
        except ValueError:
            invalid_times.append(row_number)
        for name, column in index_by_name.items():
            if name == "time":
                continue
            value = row[column].strip() if column < len(row) else ""
            try:
                if not math.isfinite(float(value)):
                    raise ValueError
            except ValueError:
                invalid_numbers.add(row_number)
    rules.extend(
        [
            _rows_rule(
                "time_format",
                "Time format",
                f"time must match {TIME_FORMAT}",
                invalid_times,
            ),
            _rows_rule(
                "finite_numbers",
                "Finite numeric values",
                "Every field except time must be a finite number",
                sorted(invalid_numbers),
            ),
        ]
    )
    if any(not rule.passed for rule in rules):
        return rules
    rules.extend(_physical_rules(device_type, index_by_name, rows))
    return rules


def _rows_rule(rule_id: str, name: str, constraint: str, invalid_rows: list[int]) -> ValidationRule:
    return ValidationRule(
        rule_id=rule_id,
        name=name,
        severity="error",
        passed=not invalid_rows,
        constraint=constraint,
        violation_count=len(invalid_rows),
        invalid_rows=invalid_rows[:SAMPLE_LIMIT],
        message=(
            "Rule passed"
            if not invalid_rows
            else f"{len(invalid_rows)} data rows violate this rule"
        ),
    )


def _physical_rules(
    device_type: DeviceType, columns: dict[str, int], rows: list[list[str]]
) -> list[ValidationRule]:
    values = [
        {name: float(row[index]) for name, index in columns.items() if name != "time"}
        for row in rows
    ]
    if device_type == "chiller":
        return [
            _minimum_rule(
                "condensing_approach", "t_cond - t_cw_ret >= 0.3", values, "t_cond", "t_cw_ret", 0.3
            ),
            _minimum_rule(
                "evaporating_approach",
                "t_chw_sup - t_evap >= 0.3",
                values,
                "t_chw_sup",
                "t_evap",
                0.3,
            ),
            _minimum_rule(
                "chilled_water_delta",
                "t_chw_ret - t_chw_sup >= 2.0",
                values,
                "t_chw_ret",
                "t_chw_sup",
                2.0,
            ),
            _minimum_rule(
                "cooling_water_delta",
                "t_cw_ret - t_cw_sup >= 2.0",
                values,
                "t_cw_ret",
                "t_cw_sup",
                2.0,
            ),
        ]
    if device_type == "cooling_tower":
        return [
            _minimum_rule(
                "tower_range", "t_cw_in - t_cw_out >= 3.0", values, "t_cw_in", "t_cw_out", 3.0
            ),
            _range_rule(
                "air_water_ratio",
                "0.4 <= air_water_ratio <= 2.0",
                values,
                "air_water_ratio",
                0.4,
                2.0,
            ),
        ]
    return [
        _range_rule(
            "pump_efficiency", "0 < eff_pump <= 1", values, "eff_pump", 0.0, 1.0, lower_open=True
        ),
        _minimum_value_rule("pump_flow", "flow >= 0", values, "flow", 0.0),
        _minimum_value_rule("pump_head", "head >= 0", values, "head", 0.0),
    ]


def _minimum_rule(
    rule_id: str,
    constraint: str,
    values: list[dict[str, float]],
    left: str,
    right: str,
    minimum: float,
) -> ValidationRule:
    invalid = [
        index for index, row in enumerate(values, start=2) if row[left] - row[right] < minimum
    ]
    return _rows_rule(rule_id, rule_id.replace("_", " ").title(), constraint, invalid)


def _minimum_value_rule(
    rule_id: str,
    constraint: str,
    values: list[dict[str, float]],
    field: str,
    minimum: float,
) -> ValidationRule:
    invalid = [index for index, row in enumerate(values, start=2) if row[field] < minimum]
    return _rows_rule(rule_id, rule_id.replace("_", " ").title(), constraint, invalid)


def _range_rule(
    rule_id: str,
    constraint: str,
    values: list[dict[str, float]],
    field: str,
    minimum: float,
    maximum: float,
    *,
    lower_open: bool = False,
) -> ValidationRule:
    invalid = [
        index
        for index, row in enumerate(values, start=2)
        if (row[field] <= minimum if lower_open else row[field] < minimum) or row[field] > maximum
    ]
    return _rows_rule(rule_id, rule_id.replace("_", " ").title(), constraint, invalid)
