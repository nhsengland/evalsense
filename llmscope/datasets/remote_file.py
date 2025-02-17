from dataclasses import dataclass

from constants import DEFAULT_HASH_TYPE


@dataclass
class RemoteFile:
    """Class for representing remote dataset files."""

    url: str
    requires_auth: bool = False
    filename: str
    expected_size: int | None = None
    expected_hash: str | None = None
    hash_type: str = DEFAULT_HASH_TYPE
