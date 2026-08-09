from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

type DeviceType = Literal["chiller", "cooling_tower", "pump"]


class ValidationRule(BaseModel):
    rule_id: str
    name: str
    severity: Literal["error", "warning"]
    passed: bool
    constraint: str
    violation_count: int = 0
    invalid_rows: list[int] = Field(default_factory=list)
    message: str


class DatasetSummary(BaseModel):
    id: str
    project_id: str
    name: str
    filename: str
    device_type: DeviceType
    status: Literal["valid", "invalid"]
    row_count: int
    columns: list[str]
    validation: list[ValidationRule]
    created_at: datetime


class DatasetList(BaseModel):
    items: list[DatasetSummary]
    total: int


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total: int


class ModelCreate(BaseModel):
    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    algorithm: Literal["polynomial", "gradient_boosting"] = "polynomial"


class ModelSummary(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    name: str
    device_type: DeviceType
    algorithm: Literal["polynomial", "gradient_boosting"]
    status: Literal["ready"]
    metrics: dict[str, float]
    created_at: datetime


class ModelList(BaseModel):
    items: list[ModelSummary]
    total: int


class ModelPreview(BaseModel):
    model: ModelSummary
    points: list[dict[str, float]]
    artifact: dict[str, Any]
