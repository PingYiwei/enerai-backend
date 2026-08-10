from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)


class ProjectSummary(BaseModel):
    id: str
    name: str
    description: str
    graph_revision: int
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectSummary):
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class ProjectList(BaseModel):
    items: list[ProjectSummary]
    total: int


class DataSourceUpdate(BaseModel):
    base_url: str = Field(min_length=1, max_length=500)
    properties_path: str = Field(default="/properties", min_length=1, max_length=200)
    query_path: str = Field(default="/query", min_length=1, max_length=200)
    bearer_token: str | None = Field(default=None, max_length=4_000)


class DataSourceView(BaseModel):
    base_url: str
    properties_path: str
    query_path: str
    token_present: bool
    updated_at: datetime


class DataSourceTestResult(BaseModel):
    ok: bool
    status_code: int
    elapsed_ms: int


class PropertyCatalog(BaseModel):
    items: list[dict[str, Any]]
    total: int


class DataQuery(BaseModel):
    property_ids: list[str] = Field(min_length=1, max_length=200)
    start: datetime
    end: datetime
    limit: int = Field(default=10_000, ge=1, le=100_000)


class DataQueryResult(BaseModel):
    data: Any


class PropertyPoint(BaseModel):
    point_name: str
    device_name: str
    property_name: str
    property_name_cn: str = ""
    unit: str = ""
    data_type: str = ""
    range: str = ""


class SensorPoint(BaseModel):
    sensor_name: str
    device_name: str
    category: str = ""
    category_cn: str = ""
    description: str = ""


class PointScheme(BaseModel):
    inherent: list[PropertyPoint]
    calculate: list[PropertyPoint]
    sensor: list[SensorPoint]
    total: int
