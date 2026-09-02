"""
Unit tests for the frame-staleness watchdog in footfall.capture.

The freshness bookkeeping (_note_frame / _stale_elapsed) is tested
synchronously with an injected clock. One integration test with real
threads covers the end to end case: a read that hangs -> the watchdog
force-releases the capture -> the reconnect path runs.
"""

import threading

import numpy as np
import pytest

from footfall.capture import ReconnectingCapture

_noop = lambda *_a, **_k: None
_HANG = object()


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeCap:
    """reads entries: an ndarray frame, None (failed read), or _HANG
    (block until release() is called)."""

    def __init__(self, reads, opened=True):
        self._reads = list(reads)
        self._opened = opened
        self._unblock = threading.Event()
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, *_a):
        return True

    def read(self):
        if not self._reads:
            return False, None
        item = self._reads.pop(0)
        if item is _HANG:
            self._unblock.wait(timeout=5.0)
            return False, None
        if item is None:
            return False, None
        return True, item

    def release(self):
        self.released = True
        self._unblock.set()


def _factory_from(seq):
    caps = list(seq)

    def factory(_source):
        assert caps, "cap_factory called more than the test set up"
        nxt = caps.pop(0)
        return FakeCap([], opened=False) if nxt is None else nxt

    return factory


def _take(iterator, n):
    out = []
    for item in iterator:
        out.append(item)
        if len(out) == n:
            break
    return out


def _frame(fill=0):
    return np.full((32, 32, 3), fill, dtype=np.uint8)


# -- freshness bookkeeping (synchronous) ---------------------------------


def test_changed_content_refreshes_the_clock():
    clock = Clock()
    rc = ReconnectingCapture("rtsp://x", monotonic=clock, log=_noop)

    rc._note_frame(_frame(0))
    assert rc._stale_elapsed() == 0.0

    clock.advance(5)
    rc._note_frame(_frame(255))          # different pixels -> fresh
    assert rc._stale_elapsed() == 0.0


def test_repeated_identical_frame_does_not_refresh_the_clock():
    clock = Clock()
    rc = ReconnectingCapture("rtsp://x", monotonic=clock, log=_noop)

    rc._note_frame(_frame(0))            # fresh at t=0
    clock.advance(7)
    rc._note_frame(_frame(0))            # byte-identical -> frozen decoder
    assert rc._stale_elapsed() == 7.0

    clock.advance(5)
    assert rc._stale_elapsed() == 12.0


def test_detect_frozen_disabled_treats_every_read_as_fresh():
    clock = Clock()
    rc = ReconnectingCapture("rtsp://x", detect_frozen=False,
                             monotonic=clock, log=_noop)

    rc._note_frame(_frame(0))
    clock.advance(9)
    rc._note_frame(_frame(0))            # identical, but detection is off
    assert rc._stale_elapsed() == 0.0


def test_watchdog_is_off_for_a_file_or_when_disabled():
    assert ReconnectingCapture("/x/clip.mp4", log=_noop)._watchdog_on() is False
    assert ReconnectingCapture("rtsp://x", stale_after=None,
                               log=_noop)._watchdog_on() is False
    assert ReconnectingCapture("rtsp://x", stale_after=0,
                               log=_noop)._watchdog_on() is False
    assert ReconnectingCapture("rtsp://x", log=_noop)._watchdog_on() is True


def test_file_source_never_starts_a_watchdog_thread():
    cap = FakeCap([_frame(0), _frame(1), None])
    rc = ReconnectingCapture("/x/clip.mp4", cap_factory=_factory_from([cap]),
                             log=_noop)

    list(rc.frames())

    assert rc._monitor is None
    assert rc.stale_trips == 0


# -- integration: a hung read is broken and reconnected -----------------


def test_hung_stream_is_force_reconnected_by_the_watchdog():
    cap1 = FakeCap([_frame(0), _frame(0), _HANG])   # 2 frames, then hang
    cap2 = FakeCap([_frame(1), _frame(1), _frame(1), _frame(1)])

    rc = ReconnectingCapture(
        "rtsp://cam/stream",
        stale_after=0.3,
        backoff_initial=0.05,
        cap_factory=_factory_from([cap1, cap2]),
        log=_noop,
    )

    got = _take(rc.frames(), 4)

    assert len(got) == 4
    assert rc.stale_trips == 1
    assert rc.reconnects == 1
    assert cap1.released is True            # the watchdog let go of the hung one


def test_monitor_thread_stops_when_iteration_ends():
    cap = FakeCap([_frame(i) for i in range(10)])
    rc = ReconnectingCapture("rtsp://x", stale_after=0.3,
                             cap_factory=_factory_from([cap]), log=_noop)

    it = rc.frames()
    next(it)
    next(it)
    monitor = rc._monitor
    assert monitor is not None and monitor.is_alive()

    it.close()

    assert not monitor.is_alive()
    assert rc._monitor is None
