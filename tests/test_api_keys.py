from pydantic import SecretStr

from app.core.config import Settings
from app.core.security import (
    create_api_key,
    decrypt_provider_secret,
    encrypt_provider_secret,
    hash_api_key,
)


def test_api_key_is_only_persisted_as_hash() -> None:
    secret, digest = create_api_key()
    assert secret.startswith("ndx_")
    assert len(digest) == 64
    assert secret not in digest
    assert hash_api_key(secret) == digest


def test_provider_api_key_is_encrypted_and_recoverable() -> None:
    settings = Settings(
        provider_secret_key=SecretStr("MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=")
    )
    secret = "sk-provider-secret"

    encrypted = encrypt_provider_secret(secret, settings)

    assert secret not in encrypted
    assert decrypt_provider_secret(encrypted, settings) == secret
