from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NODEX_",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "Nodex API"
    environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="NODEX_ENV",
    )
    api_prefix: str = "/api/v1"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "nodex"
    jwt_secret: str = "development-only-change-me-32-bytes"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 8 * 60
    agent_context_char_budget: int = Field(default=300_000, ge=10_000)
    cors_origins: list[str] = ["http://localhost:5173"]

    default_provider: Literal["openai", "openrouter", "bailian"] = "openai"
    default_model: str = ""
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_style: Literal["responses", "chat_completions"] = "responses"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_style: Literal["responses", "chat_completions"] = "chat_completions"
    bailian_api_key: SecretStr | None = None
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_api_style: Literal["responses", "chat_completions"] = "responses"

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        if self.environment == "production" and len(self.jwt_secret.encode()) < 32:
            raise ValueError("NODEX_JWT_SECRET must contain at least 32 bytes in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
