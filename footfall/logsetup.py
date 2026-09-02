"""
logsetup.py
-----------
One place to configure logging for the footfall service.

Call ``configure()`` once at process start (``tools/run_site.py`` does).
Library modules only ever do ``logging.getLogger(__name__)`` and never
touch handlers or levels themselves.

Output is line-delimited JSON by default -- one object per line with
``ts``, ``level``, ``logger``, ``msg``, any ``extra=`` fields, and
exception text -- which is what a log collector wants from a container or
a systemd unit. ``FOOTFALL_LOG_FORMAT=text`` switches to a human console
(also the default when stderr is a TTY). ``FOOTFALL_LOG_LEVEL`` sets the
level.
"""

import json
import logging
import os
import sys
import time

# standard LogRecord attributes -- anything else on a record came from an
# extra={...} and should be emitted
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime",
                                                       "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        out = {
            "ts": (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                   + f".{int(record.msecs):03d}Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                out[key] = value
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s  %(message)s"


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _want_json(explicit) -> bool:
    if explicit is not None:
        return bool(explicit)
    env = os.environ.get("FOOTFALL_LOG_FORMAT", "").strip().lower()
    if env in ("json", "text"):
        return env == "json"
    return not _isatty(sys.stderr)


def configure(level=None, *, json_output=None, stream=None):
    """Attach exactly one handler to the ``footfall`` logger. Idempotent --
    a second call replaces the handler rather than stacking another."""
    logger = logging.getLogger("footfall")

    lvl = level or os.environ.get("FOOTFALL_LOG_LEVEL") or "INFO"
    logger.setLevel(lvl.upper() if isinstance(lvl, str) else lvl)

    for handler in list(logger.handlers):
        if getattr(handler, "_footfall", False):
            logger.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler._footfall = True
    handler.setFormatter(JsonFormatter() if _want_json(json_output)
                         else logging.Formatter(_TEXT_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
