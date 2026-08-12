from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

type InspectionGrade = Literal["S", "A", "B", "C"]
type InspectionTrigger = Literal["manual", "schedule", "assignment"]
type InspectionStatus = Literal[
    "planning",
    "ready",
    "queued",
    "running",
    "completed",
    "partial",
    "failed",
    "cancelled",
]
type ConclusionStatus = Literal["normal", "warning", "critical", "inconclusive", "skipped"]
type DimensionStatus = Literal[
    "normal", "attention", "critical", "not_assessable", "not_applicable"
]
type CheckId = Literal["graph_integrity", "sensor_coverage", "data_freshness"]


def default_checks() -> list[CheckId]:
    """Compatibility with inspection runs created before the Agent workflow."""
    return ["graph_integrity", "sensor_coverage"]


class InspectionTemplate(BaseModel):
    id: Literal["full_inspection", "critical_equipment"]
    version: int
    name: str
    description: str
    default_minimum_grade: InspectionGrade
    objectives: list[str]


class InspectionTemplateList(BaseModel):
    items: list[InspectionTemplate]


class InspectionPolicyUpdate(BaseModel):
    """Legacy policy request retained while clients migrate to schedules."""

    enabled: bool = False
    interval_minutes: int = Field(default=60, ge=5, le=10_080)
    checks: list[CheckId] = Field(default_factory=default_checks)
    template_id: str = "full_inspection"
    minimum_grade: InspectionGrade = "C"


class InspectionPolicy(InspectionPolicyUpdate):
    project_id: str
    updated_at: datetime


class InspectionScheduleCreate(BaseModel):
    name: str = Field(default="Scheduled inspection", min_length=1, max_length=120)
    enabled: bool = True
    template_id: Literal["full_inspection", "critical_equipment"] = "full_inspection"
    minimum_grade: InspectionGrade | None = None
    interval_minutes: int = Field(default=1_440, ge=5, le=10_080)
    lookback_minutes: int = Field(default=1_440, ge=15, le=43_200)


class InspectionScheduleUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    enabled: bool | None = None
    template_id: Literal["full_inspection", "critical_equipment"] | None = None
    minimum_grade: InspectionGrade | None = None
    interval_minutes: int | None = Field(default=None, ge=5, le=10_080)
    lookback_minutes: int | None = Field(default=None, ge=15, le=43_200)


class InspectionSchedule(InspectionScheduleCreate):
    id: str
    project_id: str
    next_run_at: datetime
    last_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class InspectionScheduleList(BaseModel):
    items: list[InspectionSchedule]
    total: int


class InspectionRunCreate(BaseModel):
    trigger: Literal["manual", "assignment"] = "manual"
    template_id: Literal["full_inspection", "critical_equipment"] | None = None
    minimum_grade: InspectionGrade | None = None
    instruction: str = Field(default="", max_length=20_000)
    lookback_minutes: int = Field(default=1_440, ge=15, le=43_200)
    provider: Literal["openai", "openrouter", "bailian"] | None = None
    api_style: Literal["responses", "chat_completions"] | None = None
    model: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_source(self) -> InspectionRunCreate:
        if self.trigger == "assignment" and not self.instruction.strip():
            raise ValueError("A temporary assignment requires an instruction")
        if self.trigger == "manual" and not self.template_id:
            self.template_id = "full_inspection"
        return self


class InspectionFinding(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str
    category: str = "general"
    node_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class InspectionDimension(BaseModel):
    status: DimensionStatus
    summary: str


class InspectionNodeResult(BaseModel):
    node_id: str
    node_label: str
    grade: InspectionGrade
    status: ConclusionStatus
    summary: str
    dimensions: dict[str, InspectionDimension]
    findings: list[InspectionFinding] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    reviewed_by_agent: bool = True
    review_model: str
    reviewed_at: datetime


class InspectionTaskNode(BaseModel):
    id: str
    kind: Literal["stage", "group", "device", "deep_dive", "summary", "report"]
    title: str
    status: Literal[
        "pending", "ready", "running", "succeeded", "failed", "cancelled", "skipped"
    ] = "pending"
    reality_node_id: str | None = None
    parent_id: str | None = None
    grade: InspectionGrade | None = None
    progress: float = Field(default=0, ge=0, le=1)


class InspectionTaskEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: Literal["flow", "contains", "feeds", "reviews", "produces"]


class InspectionTaskGraph(BaseModel):
    nodes: list[InspectionTaskNode]
    edges: list[InspectionTaskEdge]


class DeviceInspectionManifest(BaseModel):
    node_id: str
    node_label: str
    node_type: str
    grade: InspectionGrade
    parent_id: str | None = None
    related_node_ids: list[str] = Field(default_factory=list)
    declared_properties: list[str] = Field(default_factory=list)
    available_properties: list[str] = Field(default_factory=list)
    selected_properties: list[str] = Field(default_factory=list)
    skipped_properties: list[str] = Field(default_factory=list)
    premises: list[str] = Field(default_factory=list)
    assessable_dimensions: list[str] = Field(default_factory=list)
    unavailable_dimensions: list[str] = Field(default_factory=list)


class InspectionPlanningManifest(BaseModel):
    reality_revision: int
    template_id: str
    template_version: int
    instruction: str = ""
    minimum_grade: InspectionGrade
    window_start: datetime
    window_end: datetime
    data_source_status: Literal["available", "unavailable", "partial"]
    premises: list[str] = Field(default_factory=list)
    devices: list[DeviceInspectionManifest]


class InspectionOverallConclusion(BaseModel):
    status: ConclusionStatus
    executive_summary: str
    operating_assessment: str
    anomaly_assessment: str
    efficiency_assessment: str
    optimization_opportunities: list[str] = Field(default_factory=list)
    data_quality_assessment: str
    coverage_limitations: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    reviewed_by_agent: bool = True
    review_model: str
    reviewed_at: datetime


class InspectionUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


class InspectionReport(BaseModel):
    title: str
    media_type: Literal["text/markdown"] = "text/markdown"
    content: str
    created_at: datetime


class InspectionRun(BaseModel):
    id: str
    project_id: str
    status: InspectionStatus
    trigger: InspectionTrigger
    template_id: str = "legacy_structural"
    template_name: str = "Legacy structural inspection"
    minimum_grade: InspectionGrade = "C"
    instruction: str = ""
    graph_revision: int
    planning_manifest: InspectionPlanningManifest | None = None
    task_graph: InspectionTaskGraph = Field(
        default_factory=lambda: InspectionTaskGraph(nodes=[], edges=[])
    )
    node_results: list[InspectionNodeResult] = Field(default_factory=list)
    overall_conclusion: InspectionOverallConclusion | None = None
    findings: list[InspectionFinding] = Field(default_factory=list)
    checks: list[CheckId] = Field(default_factory=default_checks)
    provider: str | None = None
    model: str | None = None
    usage: InspectionUsage = Field(default_factory=InspectionUsage)
    progress: float = Field(default=0, ge=0, le=1)
    report: InspectionReport | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime
    completed_at: datetime | None = None


class InspectionRunList(BaseModel):
    items: list[InspectionRun]
    total: int


class InspectionEvent(BaseModel):
    seq: int
    type: str
    data: dict[str, Any]
    created_at: datetime
