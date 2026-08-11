from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64Error
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ENERAI_",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "EnerAI API"
    environment: Literal["development", "test", "production"] = Field(
        default="development",
        validation_alias="ENERAI_ENV",
    )
    api_prefix: str = "/api/v1"
    mongodb_username: str = "deco"
    mongodb_password: str = "pyw88908890"
    mongodb_uri: str = f"mongodb://{mongodb_username}:{mongodb_password}@localhost:27017/"
    mongodb_database: str = "enerai"
    jwt_secret: str = "development-only-change-me-32-bytes"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 8 * 60
    agent_context_char_budget: int = Field(default=300_000, ge=10_000)
    agent_timezone: str = "Asia/Shanghai"
    cors_origins: list[str] = ["http://localhost:5173"]

    minio_endpoint: str = Field(
        default="127.0.0.1:9000",
        validation_alias="MINIO_ENDPOINT",
    )
    minio_secure: bool = Field(default=False, validation_alias="MINIO_SECURE")
    minio_access_key: SecretStr | None = Field(
        default=None,
        validation_alias="MINIO_ACCESS_KEY",
    )
    minio_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias="MINIO_SECRET_KEY",
    )
    minio_bucket: str = Field(default="enerai", validation_alias="MINIO_BUCKET")
    minio_presigned_url_minutes: int = Field(
        default=60,
        ge=1,
        le=7 * 24 * 60,
        validation_alias="MINIO_PRESIGNED_URL_MINUTES",
    )

    provider_secret_key: SecretStr | None = None
    default_provider: Literal["openai", "openrouter", "bailian"] = "openrouter"
    openai_base_url: str = "https://api.openai.com/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        if self.environment == "production" and len(self.jwt_secret.encode()) < 32:
            raise ValueError("ENERAI_JWT_SECRET must contain at least 32 bytes in production")
        if self.environment == "production" and self.provider_secret_key is None:
            raise ValueError("ENERAI_PROVIDER_SECRET_KEY is required in production")
        if self.provider_secret_key is not None:
            value = self.provider_secret_key.get_secret_value()
            try:
                decoded = b64decode(value, altchars=b"-_", validate=True)
            except (Base64Error, ValueError) as error:
                raise ValueError("ENERAI_PROVIDER_SECRET_KEY must be a valid Fernet key") from error
            if len(decoded) != 32:
                raise ValueError("ENERAI_PROVIDER_SECRET_KEY must be a valid Fernet key")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
