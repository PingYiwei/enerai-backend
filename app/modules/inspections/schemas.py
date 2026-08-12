from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

type CheckId = Literal["graph_integrity", "sensor_coverage", "data_freshness"]


def default_checks() -> list[CheckId]:
    return ["graph_integrity", "sensor_coverage"]


class InspectionPolicyUpdate(BaseModel):
    enabled: bool = False
    interval_minutes: int = Field(default=60, ge=5, le=10_080)
    checks: list[CheckId] = Field(default_factory=default_checks)


class InspectionPolicy(InspectionPolicyUpdate):
    project_id: str
    updated_at: datetime


class InspectionFinding(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    title: str
    detail: str
    node_ids: list[str] = Field(default_factory=list)


class InspectionRun(BaseModel):
    id: str
    project_id: str
    status: Literal["completed", "failed"]
    trigger: Literal["manual", "schedule"]
    checks: list[CheckId] = Field(default_factory=default_checks)
    graph_revision: int
    findings: list[InspectionFinding]
    started_at: datetime
    completed_at: datetime


class InspectionRunList(BaseModel):
    items: list[InspectionRun]
    total: int
