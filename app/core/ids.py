from __future__ import annotations

from secrets import token_hex


def new_id(prefix: str) -> str:
    """Return an opaque 128-bit identifier with a readable domain prefix."""
    return f"{prefix}_{token_hex(16)}"
