from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GraphNode(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    type: str = Field(default="equipment", min_length=1, max_length=80)
    position: dict[str, float]
    data: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None

    @model_validator(mode="after")
    def normalize_inspection_grade(self) -> GraphNode:
        if self.type == "group":
            return self
        raw = self.data.get("inspection")
        inspection = dict(raw) if isinstance(raw, dict) else {}
        grade = str(inspection.get("grade") or "B").upper()
        if grade not in {"S", "A", "B", "C"}:
            raise ValueError("Device inspection grade must be S, A, B, or C")
        inspection["grade"] = grade
        inspection.setdefault("enabled", True)
        self.data["inspection"] = inspection
        return self


class GraphEdge(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    type: str = Field(default="connection", min_length=1, max_length=80)
    data: dict[str, Any] = Field(default_factory=dict)


class StudioSensor(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    name: str = Field(min_length=1, max_length=240)
    category: str = Field(min_length=1, max_length=160)
    category_cn: str | None = Field(default=None, max_length=240)
    description: str = Field(default="", max_length=2_000)


class StudioGraph(BaseModel):
    project_id: str
    revision: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    updated_at: datetime


class StudioGraphUpdate(BaseModel):
    revision: int = Field(ge=0)
    nodes: list[GraphNode] = Field(max_length=5_000)
    edges: list[GraphEdge] = Field(max_length=10_000)


class EngineeringParameterDefinition(BaseModel):
    key: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=240)
    label_zh: str | None = Field(default=None, max_length=240)
    description: str = Field(default="", max_length=1_000)
    unit: str = Field(min_length=1, max_length=40)
    value_type: Literal["number"] = "number"
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    less_than_or_equal_to: str | None = Field(default=None, max_length=120)
    precision: int = Field(default=2, ge=0, le=12)
    group: str = Field(default="General", min_length=1, max_length=120)
    sort_order: int = 0

    @model_validator(mode="after")
    def validate_bounds(self) -> EngineeringParameterDefinition:
        lower = self.exclusive_minimum if self.exclusive_minimum is not None else self.minimum
        upper = self.exclusive_maximum if self.exclusive_maximum is not None else self.maximum
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Engineering parameter lower bound cannot exceed its upper bound")
        return self


class EngineeringParameterSchema(BaseModel):
    device_type: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=240)
    label_zh: str | None = Field(default=None, max_length=240)
    version: int = Field(ge=1)
    sort_order: int = 0
    parameters: list[EngineeringParameterDefinition] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_parameter_keys(self) -> EngineeringParameterSchema:
        keys = [parameter.key for parameter in self.parameters]
        if len(keys) != len(set(keys)):
            raise ValueError("Engineering parameter keys must be unique within a device type")
        dangling_constraints = [
            parameter.key
            for parameter in self.parameters
            if parameter.less_than_or_equal_to and parameter.less_than_or_equal_to not in keys
        ]
        if dangling_constraints:
            raise ValueError("Engineering parameter cross-field constraints must reference a field")
        return self


class EngineeringParameterCatalog(BaseModel):
    items: list[EngineeringParameterSchema]


class CatalogItem(BaseModel):
    type: str
    label: str
    category: str
    description: str


class StudioCatalog(BaseModel):
    items: list[CatalogItem]


class CategoryOption(BaseModel):
    label: str
    value: str


class CategoryGroup(BaseModel):
    parent: str
    children: list[CategoryOption]


class StudioCategories(BaseModel):
    devices: list[CategoryGroup]
    sensors: list[CategoryGroup]
