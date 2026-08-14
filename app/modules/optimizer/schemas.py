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


class DeviceTypeOption(BaseModel):
    value: DeviceType
    label: str
    fields: list[str]


class DeviceTypeList(BaseModel):
    items: list[DeviceTypeOption]


class DatasetSummary(BaseModel):
    id: str
    project_id: str
    name: str
    description: str = ""
    filename: str
    device_type: DeviceType
    status: Literal["valid", "invalid"]
    row_count: int
    valid_row_count: int = 0
    file_size: int = 0
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
    offset: int
    limit: int


class ModelCreate(BaseModel):
    dataset_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=300)
    algorithm: Literal["polynomial", "gradient_boosting"] = "polynomial"


class ModelSummary(BaseModel):
    id: str
    project_id: str
    dataset_id: str
    dataset_name: str = ""
    name: str
    description: str = ""
    device_type: DeviceType
    algorithm: Literal["polynomial", "gradient_boosting"]
    status: Literal["ready"]
    metrics: dict[str, float]
    usage_number: int = 0
    created_at: datetime


class ModelList(BaseModel):
    items: list[ModelSummary]
    total: int


class ModelSeries(BaseModel):
    key: str
    name: str
    kind: Literal["model", "submodel", "evaluation"]
    input_fields: list[str]
    output_field: str
    metrics: dict[str, float]
    formula: str = ""
    points: list[dict[str, float]]


class ModelPreview(BaseModel):
    model: ModelSummary
    series: list[ModelSeries]
    artifact: dict[str, Any]


class ModelPredictRequest(BaseModel):
    inputs: list[dict[str, float]] = Field(min_length=1, max_length=500)


class ModelPredictResponse(BaseModel):
    outputs: list[dict[str, float]]
    usage_number: int
