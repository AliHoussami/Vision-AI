"""
Unit tests for footfall.secrets: credential redaction and the
${SECRET:..} / ${ENV:..} resolution chain (env -> keyring -> file).
"""

import pytest

from footfall.secrets import (SecretError, _env_name, get_secret, redact,
                              resolve_placeholders)


@pytest.mark.parametrize("raw,expected", [
    ("rtsp://admin:hunter2@10.0.0.1:554/s", "rtsp://***@10.0.0.1:554/s"),
    ("rtsp://10.0.0.1:554/s", "rtsp://10.0.0.1:554/s"),
    ("rtsp://user@10.0.0.1/s", "rtsp://***@10.0.0.1/s"),
    ("rtsp://${SECRET:cam}@10.0.0.1/s", "rtsp://***@10.0.0.1/s"),
    ("http://h/x?password=abc&y=1", "http://h/x?password=***&y=1"),
    ("http://h/x?pass=abc", "http://h/x?pass=***"),
    ("rtsp://h/live@main", "rtsp://h/live@main"),   # @ in a path, no creds
    (0, 0),
    (None, None),
])
def test_redact(raw, expected):
    assert redact(raw) == expected


def test_env_name_normalisation():
    assert _env_name("cam-north") == "FOOTFALL_SECRET_CAM_NORTH"
    assert _env_name("cam.3_a") == "FOOTFALL_SECRET_CAM_3_A"


def test_no_placeholders_pass_through():
    assert resolve_placeholders("rtsp://10.0.0.1/s") == "rtsp://10.0.0.1/s"
    assert resolve_placeholders(0) == 0


def test_env_placeholder(monkeypatch):
    monkeypatch.setenv("NORTH_URL", "rtsp://10.0.0.1/s")
    assert resolve_placeholders("${ENV:NORTH_URL}") == "rtsp://10.0.0.1/s"


def test_missing_env_placeholder_raises(monkeypatch):
    monkeypatch.delenv("NORTH_URL", raising=False)
    with pytest.raises(SecretError, match="NORTH_URL"):
        resolve_placeholders("${ENV:NORTH_URL}")


def test_secret_from_env(monkeypatch):
    monkeypatch.setenv("FOOTFALL_SECRET_CAM_1", "admin:pw")
    assert resolve_placeholders("rtsp://${SECRET:cam-1}@h/s") \
        == "rtsp://admin:pw@h/s"


def test_secret_from_file(monkeypatch, tmp_path):
    monkeypatch.setattr("footfall.secrets._keyring", None)
    monkeypatch.delenv("FOOTFALL_SECRET_CAM_FILE", raising=False)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text("cam-file: 's3cret'\n")
    monkeypatch.setenv("FOOTFALL_SECRETS_FILE", str(secrets))

    assert get_secret("cam-file") == "s3cret"
    assert resolve_placeholders("x-${SECRET:cam-file}-y") == "x-s3cret-y"


def test_env_beats_file(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text("cam-1: from-file\n")
    monkeypatch.setenv("FOOTFALL_SECRETS_FILE", str(secrets))
    monkeypatch.setenv("FOOTFALL_SECRET_CAM_1", "from-env")
    assert get_secret("cam-1") == "from-env"


def test_missing_secret_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("footfall.secrets._keyring", None)
    monkeypatch.setenv("FOOTFALL_SECRETS_FILE", str(tmp_path / "none.yaml"))
    with pytest.raises(SecretError, match="not found"):
        get_secret("nope")


def test_malformed_secrets_file_raises(monkeypatch, tmp_path):
    monkeypatch.setattr("footfall.secrets._keyring", None)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text("- just\n- a list\n")
    monkeypatch.setenv("FOOTFALL_SECRETS_FILE", str(secrets))
    with pytest.raises(SecretError, match="flat"):
        get_secret("x")
