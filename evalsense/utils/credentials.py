"""Keep provider credentials out of serialisable model configuration."""

import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit


class ModelCredentialError(ValueError):
    """Model options contain authentication material or cannot be safely inspected."""


_CREDENTIAL_ERROR = (
    "Model configuration must not contain credentials, authentication headers or "
    "credential-bearing URLs. Configure authentication through the provider's "
    "environment variables or credential chain instead."
)
_VALUE_ERROR = (
    "Model configuration must use plain dictionaries, lists, tuples and scalar "
    "values. Opaque objects are not serialised; keep authentication outside model options."
)
# Normalize separators/case so api_key, api-key and apiKey follow the same rule.
_SECRET_SUFFIXES = (
    "apikey",
    "accesskey",
    "accesskeyid",
    "secret",
    "secretkey",
    "privatekey",
    "password",
    "passwd",
    "credential",
    "credentials",
    "accesstoken",
    "refreshtoken",
    "authtoken",
    "bearertoken",
    "sessiontoken",
    "idtoken",
    "token",
    "subscriptionkey",
    "authorization",
    "auth",
    "cookie",
    "cookies",
)
# Permit public protocol/versioning headers, but not arbitrary authentication names.
_PUBLIC_HEADERS = {
    "accept",
    "contenttype",
    "useragent",
    "apiversion",
    "anthropicversion",
}
_AUTH_KEYS = {"auth", "authentication", "token", "bearer", "clientsecret"}


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _secret_key(value: str) -> bool:
    normalised = _key(value)
    if normalised in {
        "bostoken",
        "eostoken",
        "padtoken",
        "unktoken",
        "septoken",
        "clstoken",
        "masktoken",
    }:
        return False
    return normalised in _AUTH_KEYS or normalised.endswith(_SECRET_SUFFIXES)


def _check_text(value: str, *, url_value: bool = False) -> None:
    if re.fullmatch(r"(?i)\s*(?:bearer|basic|digest)\s+\S+\s*", value):
        raise ModelCredentialError(_CREDENTIAL_ERROR)
    text = value.strip()
    if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text) and not text.startswith("//"):
        if not url_value:
            return
        text = "//" + text
    try:
        parsed = urlsplit(text)
    except ValueError:
        raise ModelCredentialError(_VALUE_ERROR) from None
    if parsed.username is not None or parsed.password is not None:
        raise ModelCredentialError(_CREDENTIAL_ERROR)
    for name, _ in parse_qsl(parsed.query) + parse_qsl(parsed.fragment):
        if (
            _secret_key(name)
            or _key(name) in {"key", "sig"}
            or _key(name).endswith("signature")
        ):
            raise ModelCredentialError(_CREDENTIAL_ERROR)


def validate_public_config(value: Any, *, _depth: int = 0) -> None:
    """Reject common inline authentication fields, including nested dictionaries.

    Only a small allowlist of public protocol/version headers is accepted because
    providers use arbitrary header names for authentication.
    Unknown objects are never stringified: repr/str may contain client secrets.
    This is a configuration boundary, not a detector for secrets hidden in arbitrary
    text, datasets or custom provider implementations.
    """
    if _depth > 50:
        raise ModelCredentialError(_VALUE_ERROR)
    if value is None or type(value) in (bool, int, float):
        return
    if type(value) is str:
        _check_text(value)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) not in (str, int):
                raise ModelCredentialError(_VALUE_ERROR)
            if type(key) is str:
                _check_text(key)
                if type(item) is str and _key(key).endswith(("url", "uri", "endpoint")):
                    _check_text(item, url_value=True)
                if _secret_key(unquote(key)) and item is not None:
                    raise ModelCredentialError(_CREDENTIAL_ERROR)
                if _key(unquote(key)).endswith("headers") and item is not None:
                    if type(item) is not dict or any(
                        type(header) is not str or _key(header) not in _PUBLIC_HEADERS
                        for header in item
                    ):
                        raise ModelCredentialError(
                            "Only documented public HTTP headers are allowed in model configuration. "
                            "Keep authentication and other headers in provider-managed setup."
                        )
            validate_public_config(item, _depth=_depth + 1)
        return
    if type(value) in (list, tuple):
        for item in value:
            validate_public_config(item, _depth=_depth + 1)
        return
    raise ModelCredentialError(_VALUE_ERROR)
