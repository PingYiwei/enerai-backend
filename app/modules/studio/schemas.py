from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    type: str = Field(default="equipment", min_length=1, max_length=80)
    position: dict[str, float]
    data: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None


class GraphEdge(BaseModel):
    id: str = Field(min_length=1, max_length=160)
    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    type: str = Field(default="connection", min_length=1, max_length=80)
    data: dict[str, Any] = Field(default_factory=dict)


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
