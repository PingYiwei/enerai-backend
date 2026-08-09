from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str = Field(default="New insight", min_length=1, max_length=120)


class SessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ContextReference(BaseModel):
    type: Literal["project", "node", "skill"]
    id: str = Field(min_length=1, max_length=200)
    name: str = Field(min_length=1, max_length=200)


class ContextOption(BaseModel):
    type: Literal["project", "node", "skill"]
    id: str
    name: str
    description: str = ""


class ContextOptions(BaseModel):
    items: list[ContextOption]


class SessionSummary(BaseModel):
    id: str
    project_id: str
    title: str
    surface: Literal["insight", "studio", "inspection"]
    status: Literal["active", "archived"]
    created_at: datetime
    updated_at: datetime


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


class SessionEntry(BaseModel):
    id: str
    seq: int
    parent_id: str | None
    role: Literal["user", "assistant", "tool"]
    content: str
    run_id: str | None = None
    usage: TokenUsage | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_id: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    references: list[ContextReference] = Field(default_factory=list)
    created_at: datetime


class SessionSnapshot(BaseModel):
    session: SessionSummary
    revision: int
    lane: str
    leaf_id: str | None
    active_run_id: str | None
    entries: list[SessionEntry]


class SessionList(BaseModel):
    items: list[SessionSummary]
    total: int


class RunCreate(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    provider: Literal["openai", "openrouter", "bailian"] | None = None
    api_style: Literal["responses", "chat_completions"] | None = None
    model: str | None = Field(default=None, max_length=200)
    attachment_ids: list[str] = Field(default_factory=list, max_length=8)
    references: list[ContextReference] = Field(default_factory=list, max_length=16)


class RunAccepted(BaseModel):
    id: str
    session_id: str
    status: Literal["accepted", "running"]
    accepted_at: datetime
    title_generation: Literal["auxiliary", "primary_fallback", "skipped"] = "skipped"


class RunStatus(BaseModel):
    id: str
    session_id: str
    status: Literal["accepted", "running", "completed", "failed", "cancelled"]
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class AttachmentSummary(BaseModel):
    id: str
    project_id: str
    name: str
    media_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif"]
    size: int
    status: Literal["draft", "committed"]
    created_at: datetime


class ArtifactSummary(BaseModel):
    id: str
    project_id: str
    session_id: str
    run_id: str
    title: str
    file_name: str
    media_type: str
    size: int
    presentation: Literal["download-only", "preview-only", "preview-and-download"]
    created_at: datetime


class ArtifactList(BaseModel):
    items: list[ArtifactSummary]
    total: int
