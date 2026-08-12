from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

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


class AgentModuleTokenUsage(BaseModel):
    module: Literal["insight", "studio", "inspection"]
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class DailyTokenUsage(BaseModel):
    date: date
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ProjectTokenUsage(BaseModel):
    today: date
    today_total_tokens: int
    by_module: list[AgentModuleTokenUsage]
    daily: list[DailyTokenUsage]


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


class DataSourceTestNode(BaseModel):
    node_id: str
    node_name: str
    device_id: str
    required_properties: list[str]
    provided_properties: list[str]
    missing_properties: list[str]
    status: str
    status_text: str
    message: str


class DataSourceTestResult(BaseModel):
    ok: bool
    status_code: int
    elapsed_ms: int
    project_id: str
    endpoint: str
    overall_status: str
    overall_status_text: str
    node_count: int
    completed_node_count: int
    nodes: list[DataSourceTestNode]
    items: list[dict[str, Any]]


class PropertyCatalog(BaseModel):
    items: list[dict[str, Any]]
    total: int


class DataQuery(BaseModel):
    device_id: str = Field(min_length=1, max_length=500)
    start_time: datetime
    end_time: datetime
    properties: list[str] | None = Field(default=None, max_length=200)


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
