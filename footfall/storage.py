"""
storage.py
----------
Where footfall events are written.

The prototype appended rows to a CSV opened in "w" mode, so every run
erased the one before it -- no history, timezone-naive timestamps, and
nothing you could query. This replaces that with an ``EventSink``
interface and a SQLite implementation: still one file (delete it for a
clean slate during testing), but append-only, UTC-stamped, with a
per-run ``run_id`` and idempotent writes keyed on ``(run_id, seq)``.

Postgres + TimescaleDB is planned as a second ``EventSink`` implementation
once there are real multi-site deployments -- see OVERVIEW.md, Phase 3.
Until then this keeps the schema and write path honest without running a
database service.

    with SqliteEventSink("events.db", source="rtsp://cam") as sink:
        sink.emit("line_in", track_id=5)
        sink.emit("zone_exit", track_id=12, value="8.6s")
"""

import abc
import csv
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


def new_run_id() -> str:
    """A sortable, unique id for one FootfallTracker.run(): compact UTC
    timestamp plus a short random tail."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class EventSink(abc.ABC):
    """Append-only store for footfall events.

    The sink assigns a monotonic per-run ``seq`` starting at 0. Writes are
    idempotent on ``(run_id, seq)`` -- re-emitting an already-stored pair
    is a no-op -- so a store-and-forward replay (the Postgres backend,
    later) cannot double-count. Usable as a context manager.
    """

    run_id: str

    @abc.abstractmethod
    def emit(self, event: str, track_id: Optional[int], value=None) -> None:
        ...

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class NullSink(EventSink):
    """Discards everything; counts calls. The default when no store is set."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or new_run_id()
        self.count = 0

    def emit(self, event, track_id, value=None):
        self.count += 1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_utc TEXT NOT NULL,
    ended_utc   TEXT,
    source      TEXT,
    site        TEXT,
    tz          TEXT,
    note        TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    seq       INTEGER NOT NULL,
    ts_utc    TEXT NOT NULL,
    site      TEXT,
    tz        TEXT,
    event     TEXT NOT NULL,
    track_id  INTEGER,
    value     TEXT,
    UNIQUE(run_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_events_run  ON events(run_id);
CREATE INDEX IF NOT EXISTS ix_events_time ON events(ts_utc);
"""


class SqliteEventSink(EventSink):
    def __init__(self, path, *, run_id=None, source=None, site=None, tz=None,
                 note=None, clock=None):
        """
        path    SQLite file; parent dirs are created
        run_id  defaults to new_run_id()
        source  the camera/source string, recorded on the run row
        site    site identifier, stamped on every event (for later
                per-site queries); may be None while prototyping
        tz      IANA site timezone name, stamped alongside the UTC time so
                local wall-clock time is reconstructable
        clock   injectable; returns a tz-aware UTC datetime
        """
        self.path = str(path)
        self.run_id = run_id or new_run_id()
        self.site = site
        self.tz = tz
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._seq = 0
        self.count = 0

        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # emit()/close() are only ever called from one thread at a time,
        # but not necessarily the thread that built the sink (run_site.py
        # --all builds and runs each camera on its own thread)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO runs(run_id, started_utc, source, site, tz, note)"
            " VALUES (?,?,?,?,?,?)",
            (self.run_id, self._now(), source, site, tz, note),
        )
        self._conn.commit()

    def _now(self) -> str:
        return self._clock().isoformat()

    def emit(self, event, track_id, value=None, *, seq=None):
        use_seq = self._seq if seq is None else seq
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO events"
            "(run_id, seq, ts_utc, site, tz, event, track_id, value)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (self.run_id, use_seq, self._now(), self.site, self.tz,
             str(event), track_id, None if value is None else str(value)),
        )
        self._conn.commit()
        if seq is None:
            self._seq += 1
        if cur.rowcount:
            self.count += 1

    def close(self):
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "UPDATE runs SET ended_utc=? WHERE run_id=? AND ended_utc IS NULL",
                (self._now(), self.run_id),
            )
            self._conn.commit()
        finally:
            self._conn.close()
            self._conn = None


class CsvEventSink(EventSink):
    """Legacy CSV output, now append-only and UTC-stamped. Kept for anyone
    passing events_csv=; new code should use SqliteEventSink."""

    _HEADER = ["run_id", "seq", "ts_utc", "event", "track_id", "value"]

    def __init__(self, path, *, run_id=None, clock=None):
        self.path = str(path)
        self.run_id = run_id or new_run_id()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._seq = 0
        self.count = 0

        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fresh = (not os.path.exists(self.path)
                 or os.path.getsize(self.path) == 0)
        self._fh = open(self.path, "a", newline="")
        self._w = csv.writer(self._fh)
        if fresh:
            self._w.writerow(self._HEADER)
            self._fh.flush()

    def emit(self, event, track_id, value=None):
        self._w.writerow([self.run_id, self._seq, self._clock().isoformat(),
                          event, track_id, "" if value is None else value])
        self._fh.flush()
        self._seq += 1
        self.count += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# -- local admin helpers (used by tools/events_db.py) --------------------


def reset(path) -> None:
    """Delete the SQLite file (and its WAL sidecars) so the next run
    starts from an empty store."""
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(path) + suffix)
        if f.exists():
            f.unlink()


def list_runs(path) -> List[Tuple]:
    """(run_id, started_utc, ended_utc, source, event_count) per run."""
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(
            "SELECT r.run_id, r.started_utc, r.ended_utc, r.source,"
            " (SELECT COUNT(*) FROM events e WHERE e.run_id = r.run_id)"
            " FROM runs r ORDER BY r.started_utc"
        ).fetchall()
    finally:
        conn.close()


def drop_run(path, run_id: str) -> int:
    """Delete one run and its events. Returns the number of events removed."""
    conn = sqlite3.connect(str(path))
    try:
        n = conn.execute("SELECT COUNT(*) FROM events WHERE run_id=?",
                         (run_id,)).fetchone()[0]
        conn.execute("DELETE FROM events WHERE run_id=?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        conn.commit()
        return n
    finally:
        conn.close()
