from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

ProviderId = Literal["openai", "openrouter", "bailian"]


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: Literal["user", "admin"] = "user"
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserResponse


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ApiKeySummary(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None = None


class ApiKeyCreated(ApiKeySummary):
    secret: str


class ApiKeyList(BaseModel):
    items: list[ApiKeySummary]
    total: int


class ModelRoles(BaseModel):
    primary: str = Field(default="", max_length=200)
    text: str = Field(default="", max_length=200)
    multimodal: str = Field(default="", max_length=200)
    auxiliary: str = Field(default="", max_length=200)

    @field_validator("primary", "text", "multimodal", "auxiliary")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        return value.strip()


class ProviderModelConfiguration(BaseModel):
    provider: ProviderId
    api_style: Literal["responses", "chat_completions"]
    base_url: str
    models: ModelRoles = Field(default_factory=ModelRoles)
    api_key_configured: bool = False
    api_key_masked: str | None = None


class ModelSettingsResponse(BaseModel):
    active_provider: ProviderId = "openrouter"
    providers: dict[ProviderId, ProviderModelConfiguration]
    updated_at: datetime | None = None


class ModelSettingsUpdate(BaseModel):
    provider: ProviderId
    api_style: Literal["responses", "chat_completions"] = "responses"
    models: ModelRoles
    api_key: str | None = Field(default=None, max_length=2048)
    clear_api_key: bool = False

    @field_validator("api_key")
    @classmethod
    def normalize_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None
