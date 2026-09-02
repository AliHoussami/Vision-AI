"""
secrets.py
----------
Keep camera credentials out of the config file, the logs, and the
database.

A camera ``source`` may carry placeholders that are expanded when the
config loads:

    source: "rtsp://${SECRET:cam-north}@192.168.1.50:554/Streaming/Channels/101"
    source: "rtsp://admin:${SECRET:cam-north-pw}@192.168.1.50:554/..."
    source: "${ENV:NORTH_CAM_URL}"

``${SECRET:name}`` is looked up, in order:

  1. env var  ``FOOTFALL_SECRET_<NAME>``  (name upper-cased, non-alnum -> ``_``)
  2. OS keyring  ``keyring.get_password("footfall", name)`` -- only if the
     optional ``keyring`` package is installed
  3. a secrets file (default ``config/secrets.yaml``, gitignored; override
     with ``$FOOTFALL_SECRETS_FILE``) -- a flat ``name: value`` mapping

The looked-up value is substituted verbatim, so store ``user:pass`` or
just the password, whatever the surrounding URL needs.

``redact()`` masks credentials in any string before it is printed or
stored.
"""

import os
import re
from pathlib import Path

import yaml

from . import ROOT

try:                        # optional; env vars and the file work without it
    import keyring as _keyring
except Exception:           # ImportError, or a backend that raises on import
    _keyring = None


class SecretError(RuntimeError):
    """A ${SECRET:...} / ${ENV:...} placeholder could not be resolved."""


_PLACEHOLDER = re.compile(r"\$\{(SECRET|ENV):([A-Za-z0-9_.\-]+)\}")
_URL_CREDS = re.compile(r"(//)[^/@\s]+@")
_QUERY_PW = re.compile(r"(?i)(pass(?:word)?=)[^&\s]+")


def _env_name(name: str) -> str:
    return "FOOTFALL_SECRET_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _secrets_file() -> Path:
    override = os.environ.get("FOOTFALL_SECRETS_FILE")
    return Path(override) if override else (ROOT / "config" / "secrets.yaml")


def _from_file(name: str):
    path = _secrets_file()
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise SecretError(f"{path} is not valid YAML: {exc}")
    if not isinstance(data, dict):
        raise SecretError(f"{path} must be a flat 'name: value' mapping")
    val = data.get(name)
    return None if val is None else str(val)


def get_secret(name: str) -> str:
    """Resolve one ${SECRET:name} value, or raise SecretError."""
    env = os.environ.get(_env_name(name))
    if env is not None:
        return env
    if _keyring is not None:
        try:
            val = _keyring.get_password("footfall", name)
        except Exception:
            val = None
        if val is not None:
            return val
    val = _from_file(name)
    if val is not None:
        return val
    raise SecretError(
        f"secret {name!r} not found: set {_env_name(name)}, add it to the "
        f"keyring (service 'footfall'), or put it in {_secrets_file()}")


def _get_env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise SecretError(f"environment variable {name!r} is not set")
    return val


def resolve_placeholders(text):
    """Expand ${SECRET:...} / ${ENV:...} in a string. A non-string (an int
    camera index) is returned unchanged. Raises SecretError if a
    placeholder cannot be resolved."""
    if not isinstance(text, str):
        return text

    def _sub(m):
        kind, name = m.group(1), m.group(2)
        return get_secret(name) if kind == "SECRET" else _get_env(name)

    return _PLACEHOLDER.sub(_sub, text)


def redact(text):
    """Mask credentials in a URL-ish string, for logs / storage / display.
    Non-strings pass through. Safe to call on a string with no secrets."""
    if not isinstance(text, str):
        return text
    text = _URL_CREDS.sub(r"\1***@", text)
    text = _QUERY_PW.sub(r"\1***", text)
    return text
