"""
Unit tests for footfall.storage.

Schema and run tracking, append-only across runs, idempotent writes,
UTC timestamps, NULL vs text values, the CSV sink, and the reset /
list_runs / drop_run admin helpers.
"""

import sqlite3
from datetime import datetime, timezone

import pytest

from footfall.storage import (CsvEventSink, NullSink, SqliteEventSink,
                              drop_run, list_runs, new_run_id, reset)

_FIXED = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
_fixed_clock = lambda: _FIXED


def _rows(path, sql, *args):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# -- run tracking --------------------------------------------------------


def test_opening_a_sink_records_the_run(tmp_path):
    db = tmp_path / "events.db"
    sink = SqliteEventSink(db, run_id="r1", source="rtsp://cam",
                           clock=_fixed_clock)
    sink.close()

    (row,) = _rows(db, "SELECT run_id, started_utc, ended_utc, source FROM runs")
    assert row[0] == "r1"
    assert row[1] == "2026-09-02T12:00:00+00:00"
    assert row[2] == "2026-09-02T12:00:00+00:00"      # set on close
    assert row[3] == "rtsp://cam"


def test_events_are_appended_utc_stamped_and_keyed_to_the_run(tmp_path):
    db = tmp_path / "events.db"
    with SqliteEventSink(db, run_id="r1", site="paris-01", tz="Europe/Paris",
                         clock=_fixed_clock) as sink:
        sink.emit("line_in", 5)
        sink.emit("zone_exit", 12, "8.6s")

    rows = _rows(db, "SELECT run_id, seq, ts_utc, site, tz, event, track_id,"
                     " value FROM events ORDER BY seq")
    assert rows == [
        ("r1", 0, "2026-09-02T12:00:00+00:00", "paris-01", "Europe/Paris",
         "line_in", 5, None),
        ("r1", 1, "2026-09-02T12:00:00+00:00", "paris-01", "Europe/Paris",
         "zone_exit", 12, "8.6s"),
    ]


def test_a_second_run_appends_rather_than_overwriting(tmp_path):
    db = tmp_path / "events.db"
    with SqliteEventSink(db, run_id="r1", clock=_fixed_clock) as s1:
        s1.emit("line_in", 1)
        s1.emit("line_in", 2)
    with SqliteEventSink(db, run_id="r2", clock=_fixed_clock) as s2:
        s2.emit("line_in", 3)

    assert _rows(db, "SELECT COUNT(*) FROM events")[0][0] == 3
    assert _rows(db, "SELECT COUNT(*) FROM runs")[0][0] == 2


def test_writes_are_idempotent_on_run_and_seq(tmp_path):
    db = tmp_path / "events.db"
    with SqliteEventSink(db, run_id="r1", clock=_fixed_clock) as sink:
        sink.emit("line_in", 5)                 # seq 0
        sink.emit("line_in", 5, seq=0)          # replay of seq 0 -> ignored
        assert sink.count == 1

    assert _rows(db, "SELECT COUNT(*) FROM events")[0][0] == 1


def test_new_run_id_is_unique():
    assert new_run_id() != new_run_id()


# -- CSV sink ----------------------------------------------------------


def test_csv_sink_writes_one_header_and_appends(tmp_path):
    path = tmp_path / "events.csv"
    with CsvEventSink(path, run_id="r1", clock=_fixed_clock) as sink:
        sink.emit("line_in", 5)
        sink.emit("zone_exit", 12, "8.6s")
    with CsvEventSink(path, run_id="r2", clock=_fixed_clock) as sink:
        sink.emit("line_in", 7)

    lines = path.read_text().splitlines()
    assert lines[0] == "run_id,seq,ts_utc,event,track_id,value"
    assert len(lines) == 4                       # header + 3 events
    assert lines[1] == "r1,0,2026-09-02T12:00:00+00:00,line_in,5,"
    assert lines[3].startswith("r2,0,")


# -- null sink -------------------------------------------------------


def test_null_sink_counts_but_stores_nothing():
    sink = NullSink(run_id="r1")
    sink.emit("line_in", 1)
    sink.emit("line_out", 1)
    assert sink.count == 2
    assert sink.run_id == "r1"


# -- admin helpers -------------------------------------------------


def test_reset_deletes_the_file_and_wal_sidecars(tmp_path):
    db = tmp_path / "events.db"
    SqliteEventSink(db, run_id="r1", clock=_fixed_clock).close()
    assert db.exists()

    reset(db)

    assert not db.exists()
    assert not (tmp_path / "events.db-wal").exists()
    assert not (tmp_path / "events.db-shm").exists()


def test_list_runs_and_drop_run(tmp_path):
    db = tmp_path / "events.db"
    with SqliteEventSink(db, run_id="r1", source="a", clock=_fixed_clock) as s1:
        s1.emit("line_in", 1)
        s1.emit("line_in", 2)
    with SqliteEventSink(db, run_id="r2", source="b", clock=_fixed_clock) as s2:
        s2.emit("line_in", 3)

    runs = list_runs(db)
    assert [(r[0], r[3], r[4]) for r in runs] == [("r1", "a", 2), ("r2", "b", 1)]

    removed = drop_run(db, "r1")
    assert removed == 2
    runs = list_runs(db)
    assert [r[0] for r in runs] == ["r2"]
    assert _rows(db, "SELECT COUNT(*) FROM events")[0][0] == 1
