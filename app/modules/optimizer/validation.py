from __future__ import annotations

import csv
import io
import math
from collections import Counter
from datetime import datetime

from app.modules.optimizer.dataset_configs import DATASET_CONFIGS
from app.modules.optimizer.dataset_configs.base import DatasetConfig, NumericRow
from app.modules.optimizer.schemas import DeviceType, ValidationRule

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
    config = DATASET_CONFIGS[device_type]
    rules = [_fields_rule(config, header)]
    if not rules[0].passed:
        return rules

    indexes = {name: header.index(name) for name in config.fields}
    wrong_width = [index for index, row in enumerate(rows, start=2) if len(row) != len(header)]
    invalid_times: list[int] = []
    invalid_numbers: set[int] = set()
    numeric_rows: list[NumericRow] = []

    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            continue
        time_value = row[indexes["time"]].strip()
        try:
            parsed = datetime.strptime(time_value, TIME_FORMAT)
            if parsed.strftime(TIME_FORMAT) != time_value:
                raise ValueError
        except ValueError:
            invalid_times.append(row_number)

        values: dict[str, float] = {}
        row_has_invalid_number = False
        for name in config.fields:
            if name == "time":
                continue
            try:
                value = float(row[indexes[name]].strip())
                if not math.isfinite(value):
                    raise ValueError
                values[name] = value
            except ValueError:
                invalid_numbers.add(row_number)
                row_has_invalid_number = True
        if row_number not in invalid_times and not row_has_invalid_number:
            numeric_rows.append((row_number, values))

    rules.extend(
        [
            _rows_rule(
                "row_width",
                "Row width",
                f"Every row must contain {len(header)} fields",
                wrong_width,
            ),
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
    if any(rule.severity == "error" and not rule.passed for rule in rules):
        return rules
    return [*rules, *config.validate_rules(numeric_rows)]


def count_valid_rows(
    device_type: DeviceType, header: list[str], rows: list[list[str]]
) -> int:
    config = DATASET_CONFIGS[device_type]
    if Counter(header) != Counter(config.fields):
        return 0
    indexes = {name: header.index(name) for name in config.fields}
    valid = 0
    for row in rows:
        if len(row) != len(header):
            continue
        time_value = row[indexes["time"]].strip()
        try:
            parsed = datetime.strptime(time_value, TIME_FORMAT)
            if parsed.strftime(TIME_FORMAT) != time_value:
                continue
            numbers = [
                float(row[indexes[name]].strip()) for name in config.fields if name != "time"
            ]
            if not all(math.isfinite(number) for number in numbers):
                continue
        except (IndexError, ValueError):
            continue
        valid += 1
    return valid


def _fields_rule(config: DatasetConfig, header: list[str]) -> ValidationRule:
    passed = Counter(header) == Counter(config.fields)
    return ValidationRule(
        rule_id="required_fields",
        name="Required fields",
        severity="error",
        passed=passed,
        constraint="Columns must contain exactly: " + ", ".join(config.fields),
        violation_count=0 if passed else 1,
        message="Field set is valid" if passed else "Missing, extra, or duplicated fields",
    )


def _rows_rule(
    rule_id: str, name: str, constraint: str, invalid_rows: list[int]
) -> ValidationRule:
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
            else f"{len(invalid_rows)} rows violate this rule"
        ),
    )
