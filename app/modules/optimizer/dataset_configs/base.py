from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.modules.optimizer.schemas import DeviceType, ValidationRule

type NumericRow = tuple[int, dict[str, float]]
type DeviceRuleValidator = Callable[[list[NumericRow]], list[ValidationRule]]


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    device_type: DeviceType
    label: str
    fields: tuple[str, ...]
    validate_rules: DeviceRuleValidator


def warning_rule(
    *,
    rule_id: str,
    name: str,
    constraint: str,
    invalid_rows: list[int],
) -> ValidationRule:
    sample_rows = invalid_rows[:20]
    return ValidationRule(
        rule_id=rule_id,
        name=name,
        severity="warning",
        passed=not invalid_rows,
        constraint=constraint,
        violation_count=len(invalid_rows),
        invalid_rows=sample_rows,
        message=(
            "Rule passed"
            if not invalid_rows
            else f"{len(invalid_rows)} rows violate this rule"
        ),
    )
