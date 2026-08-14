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


class EngineeringParameterValue(BaseModel):
    key: str
    label: str
    description: str = ""
    unit: str
    value: float | None = None
    default_value: float | None = None
    source: Literal["default", "user", "missing"]
    editable: bool
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None


class NodeEngineeringParameter(BaseModel):
    key: str
    label: str
    unit: str
    value: float | None = None
    required: bool = False


class NodeEngineeringState(BaseModel):
    node_id: str
    label: str
    device_type: str
    group_id: str | None = None
    group_label: str | None = None
    configured_count: int
    required_count: int
    complete: bool
    parameters: list[NodeEngineeringParameter]


class EngineeringDerivedValue(BaseModel):
    key: str
    label: str
    value: float
    unit: str


class EngineeringModelGroup(BaseModel):
    group_id: str
    label: str
    device_type: str
    member_node_ids: list[str]
    member_labels: list[str]
    complete: bool
    derived_values: list[EngineeringDerivedValue] = Field(default_factory=list)


class EngineeringTopologyConnection(BaseModel):
    node_id: str
    label: str
    related_node_ids: list[str]
    related_labels: list[str]


type EngineeringTopologyMode = Literal["one_to_one", "parallel", "mixed", "unknown"]
type EngineeringTopologySystem = Literal["chilled_water_pumps", "condenser_water_pumps"]


class EngineeringTopology(BaseModel):
    system: EngineeringTopologySystem
    label: str
    mode: EngineeringTopologyMode
    inferred_mode: EngineeringTopologyMode
    source: Literal["reality_model", "llm", "manual"]
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(default="", max_length=2_000)
    inferred_reason: str = Field(default="", max_length=2_000)
    connections: list[EngineeringTopologyConnection] = Field(default_factory=list)


class EngineeringConfigView(BaseModel):
    project_id: str
    graph_revision: int
    physical_properties: list[EngineeringParameterValue]
    water_system_parameters: list[EngineeringParameterValue]
    nodes: list[NodeEngineeringState]
    model_groups: list[EngineeringModelGroup]
    topologies: list[EngineeringTopology]
    updated_at: datetime | None = None


class EngineeringTopologyOverride(BaseModel):
    system: EngineeringTopologySystem
    mode: EngineeringTopologyMode
    source: Literal["llm", "manual"]
    confidence: float = Field(default=1, ge=0, le=1)
    reason: str = Field(default="", max_length=2_000)


class EngineeringConfigUpdate(BaseModel):
    graph_revision: int = Field(ge=0)
    water_system_parameters: dict[str, float] = Field(default_factory=dict)
    topologies: list[EngineeringTopologyOverride] = Field(default_factory=list, max_length=2)


class EngineeringTopologyInference(BaseModel):
    graph_revision: int
    topologies: list[EngineeringTopology]
