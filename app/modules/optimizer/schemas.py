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
    field_units: dict[str, str] = Field(default_factory=dict)


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
    optimization_compatible: bool = False
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


class EngineeringModelBinding(BaseModel):
    node_id: str
    model_id: str
    model_name: str
    device_type: DeviceType
    status: Literal["ready", "missing", "incompatible"]


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
    model_bindings: list[EngineeringModelBinding] = Field(default_factory=list)
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
    model_bindings: dict[str, str] = Field(default_factory=dict)
    topologies: list[EngineeringTopologyOverride] = Field(default_factory=list, max_length=2)


class EngineeringTopologyInference(BaseModel):
    graph_revision: int
    topologies: list[EngineeringTopology]


class OptimizationRange(BaseModel):
    minimum: float
    maximum: float
    step: float = Field(gt=0)


class OptimizationSupplyTemperatureSpace(BaseModel):
    enabled: bool = True
    minimum: float = 5.0
    maximum: float = 9.0
    step: float = Field(default=1.0, gt=0)
    fixed: float = 7.0


class OptimizationSearchSpace(BaseModel):
    wet_bulb: OptimizationRange = Field(
        default_factory=lambda: OptimizationRange(minimum=18, maximum=32, step=2)
    )
    load_ratio: OptimizationRange = Field(
        default_factory=lambda: OptimizationRange(minimum=0.2, maximum=1.0, step=0.1)
    )
    chilled_water_supply: OptimizationSupplyTemperatureSpace = Field(
        default_factory=OptimizationSupplyTemperatureSpace
    )
    condenser_water_delta_t: OptimizationRange = Field(
        default_factory=lambda: OptimizationRange(minimum=3, maximum=6, step=1)
    )
    cooling_tower_approach: OptimizationRange = Field(
        default_factory=lambda: OptimizationRange(minimum=2, maximum=6, step=1)
    )


class OptimizationSolverConfig(BaseModel):
    cop_initial: float = Field(default=5.0, gt=0)
    cop_tolerance: float = Field(default=0.001, gt=0, le=0.1)
    cop_max_iterations: int = Field(default=50, ge=1, le=500)
    maximum_candidates: int = Field(default=2_000_000, ge=1, le=20_000_000)


class OptimizationStrategyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    search_space: OptimizationSearchSpace = Field(default_factory=OptimizationSearchSpace)
    solver: OptimizationSolverConfig = Field(default_factory=OptimizationSolverConfig)


class OptimizationStrategySummary(BaseModel):
    id: str
    project_id: str
    name: str
    description: str = ""
    status: Literal["draft", "ready", "stale"]
    revision: int
    search_space: OptimizationSearchSpace
    solver: OptimizationSolverConfig
    updated_at: datetime
    created_at: datetime


class OptimizationStrategyList(BaseModel):
    items: list[OptimizationStrategySummary]
    total: int


class OptimizationPreflightIssue(BaseModel):
    code: str
    message: str
    target: str = ""


class OptimizationPreflightResult(BaseModel):
    ready: bool
    issues: list[OptimizationPreflightIssue] = Field(default_factory=list)
    warnings: list[OptimizationPreflightIssue] = Field(default_factory=list)
    external_condition_count: int = 0
    estimated_candidate_count: int = 0
    model_group_count: int = 0


class OptimizationRunStage(BaseModel):
    key: str
    label: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    detail: str = ""


class OptimizationRunCounters(BaseModel):
    external_conditions_total: int = 0
    external_conditions_completed: int = 0
    candidates_total: int = 0
    candidates_evaluated: int = 0
    candidates_feasible: int = 0
    candidates_rejected: int = 0


class OptimizationStrategyRow(BaseModel):
    condition_key: str
    wet_bulb: float
    load_ratio: float
    cooling_load: float
    chilled_water_supply_temperature: float
    success: bool
    failure_code: str = ""
    chiller_combination_code: str = ""
    chiller_group_counts: dict[str, int] = Field(default_factory=dict)
    chilled_water_pump_count: int = 0
    chilled_water_pump_frequency: float | None = None
    condenser_water_pump_count: int = 0
    condenser_water_pump_frequency: float | None = None
    cooling_tower_count: int = 0
    cooling_tower_frequency: float | None = None
    condenser_water_delta_t: float | None = None
    cooling_tower_approach: float | None = None
    chiller_power: float | None = None
    chilled_water_pump_power: float | None = None
    condenser_water_pump_power: float | None = None
    cooling_tower_power: float | None = None
    total_power: float | None = None
    eer: float | None = None


class OptimizationRunView(BaseModel):
    id: str
    project_id: str
    strategy_id: str
    strategy_name: str
    strategy_snapshot: OptimizationStrategySummary | None = None
    status: Literal["queued", "running", "completed", "partial", "failed"]
    progress: float = Field(ge=0, le=1)
    current_stage: str = ""
    current_condition: str = ""
    stages: list[OptimizationRunStage] = Field(default_factory=list)
    counters: OptimizationRunCounters = Field(default_factory=OptimizationRunCounters)
    failure_summary: dict[str, int] = Field(default_factory=dict)
    rows: list[OptimizationStrategyRow] = Field(default_factory=list)
    error: str = ""
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class OptimizationRunSummary(BaseModel):
    id: str
    project_id: str
    strategy_id: str
    strategy_name: str
    status: Literal["queued", "running", "completed", "partial", "failed"]
    progress: float = Field(ge=0, le=1)
    counters: OptimizationRunCounters = Field(default_factory=OptimizationRunCounters)
    failure_summary: dict[str, int] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class OptimizationRunList(BaseModel):
    items: list[OptimizationRunSummary]
    total: int
