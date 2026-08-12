from __future__ import annotations

from datetime import datetime
from typing import Any

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
