from app.core.security import create_api_key, hash_api_key


def test_api_key_is_only_persisted_as_hash() -> None:
    secret, digest = create_api_key()
    assert secret.startswith("ndx_")
    assert len(digest) == 64
    assert secret not in digest
    assert hash_api_key(secret) == digest
