"""
Unit tests for footfall.logsetup: the JSON formatter and configure().
"""

import io
import json
import logging

import pytest

from footfall.logsetup import JsonFormatter, configure


def _record(**kw):
    defaults = dict(name="footfall.x", level=logging.INFO, pathname=__file__,
                    lineno=1, msg="hello %s", args=("world",), exc_info=None)
    defaults.update(kw)
    return logging.LogRecord(**defaults)


def test_json_formatter_core_fields():
    line = JsonFormatter().format(_record())
    obj = json.loads(line)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "footfall.x"
    assert obj["msg"] == "hello world"
    assert obj["ts"].endswith("Z") and "T" in obj["ts"]


def test_json_formatter_includes_extra_fields():
    rec = _record()
    rec.camera = "door"
    rec.dropped_frames = 4
    obj = json.loads(JsonFormatter().format(rec))
    assert obj["camera"] == "door"
    assert obj["dropped_frames"] == 4


def test_json_formatter_includes_exception_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record(msg="failed", args=(), exc_info=sys.exc_info())
    obj = json.loads(JsonFormatter().format(rec))
    assert "ValueError: boom" in obj["exc"]


def test_configure_json_writes_one_object_per_line():
    stream = io.StringIO()
    configure("INFO", json_output=True, stream=stream)
    log = logging.getLogger("footfall.test.a")
    log.info("first")
    log.warning("second", extra={"n": 2})

    lines = [l for l in stream.getvalue().splitlines() if l]
    assert len(lines) == 2
    assert json.loads(lines[0])["msg"] == "first"
    assert json.loads(lines[1])["n"] == 2

    logging.getLogger("footfall").setLevel(logging.CRITICAL)   # restore quiet


def test_configure_is_idempotent():
    stream = io.StringIO()
    configure("INFO", json_output=True, stream=stream)
    configure("INFO", json_output=True, stream=stream)
    footfall = logging.getLogger("footfall")
    assert sum(getattr(h, "_footfall", False) for h in footfall.handlers) == 1

    logging.getLogger("footfall").setLevel(logging.CRITICAL)


def test_configure_text_format(monkeypatch):
    stream = io.StringIO()
    configure("DEBUG", json_output=False, stream=stream)
    logging.getLogger("footfall.test.b").debug("plain line")
    out = stream.getvalue()
    assert "plain line" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)

    logging.getLogger("footfall").setLevel(logging.CRITICAL)
