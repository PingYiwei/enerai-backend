from __future__ import annotations

from app.modules.optimizer.dataset_configs.base import DatasetConfig, NumericRow
from app.modules.optimizer.schemas import ValidationRule

FIELDS = ("time", "flow", "eff_pump", "head")


def validate_rules(rows: list[NumericRow]) -> list[ValidationRule]:
    # Pump datasets only require the shared field, timestamp, and finite-number checks.
    _ = rows
    return []


CONFIG = DatasetConfig(
    device_type="pump",
    label="Pump",
    fields=FIELDS,
    validate_rules=validate_rules,
)

