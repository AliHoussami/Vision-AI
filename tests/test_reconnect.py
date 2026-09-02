"""
Unit tests for footfall.capture.ReconnectingCapture.

Covers the backoff schedule, reconnect counting, file-EOF behaviour, and
the max_retries ceiling. No real camera or OpenCV device is touched: the
capture factory and time.sleep are both injected.

    pip install -r requirements-dev.txt
    pytest
"""

import numpy as np
import pytest

from footfall.capture import ReconnectingCapture, _looks_live


class FakeCap:
    """Stand-in for cv2.VideoCapture.

    reads  -- frames to hand out in order. A None entry makes read()
              return (False, None), i.e. a failed read. Once the list is
              exhausted read() keeps reporting failure.
    opened -- what isOpened() returns.
    """

    def __init__(self, reads, opened=True):
        self._reads = list(reads)
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        return True

    def read(self):
        if self._reads:
            item = self._reads.pop(0)
            if item is None:
                return False, None
            return True, item
        return False, None

    def release(self):
        self.released = True


def _frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _factory_from(sequence):
    """A cap_factory yielding the given caps in order.

    A None entry means "open failed": the factory returns a FakeCap whose
    isOpened() is False, which ReconnectingCapture._open treats as a
    failed open.
    """
    caps = list(sequence)

    def factory(source):
        assert caps, "cap_factory called more times than the test set up"
        nxt = caps.pop(0)
        return FakeCap([], opened=False) if nxt is None else nxt

    return factory


def _recorder():
    calls = []
    return calls, calls.append


def _take(iterator, n):
    out = []
    for item in iterator:
        out.append(item)
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------------------


def test_live_stream_reconnects_after_a_dropped_read():
    cap1 = FakeCap([_frame(), _frame(), _frame(), None])   # 3 frames, then drop
    cap2 = FakeCap([_frame(), _frame(), _frame(), _frame()])
    sleeps, sleep = _recorder()

    rc = ReconnectingCapture(
        "rtsp://cam/stream",
        cap_factory=_factory_from([cap1, cap2]),
        sleep=sleep,
        log=lambda *_: None,
    )

    frames = _take(rc.frames(), 5)

    assert len(frames) == 5           # 3 from cap1, 2 from cap2
    assert rc.reconnects == 1
    assert sleeps == [1.0]            # one backoff wait, at the initial delay
    assert rc.waits == [1.0]
    assert cap1.released              # the dead capture was cleaned up


def test_backoff_grows_and_is_capped_during_an_outage():
    cap1 = FakeCap([_frame(), None])                       # 1 frame, then drop
    good = FakeCap([_frame(), _frame()])                   # camera returns
    sleeps, sleep = _recorder()

    rc = ReconnectingCapture(
        "rtsp://cam/stream",
        backoff_initial=1.0,
        backoff_factor=2.0,
        backoff_max=10.0,
        cap_factory=_factory_from([cap1, None, None, None, None, good]),
        sleep=sleep,
        log=lambda *_: None,
    )

    frames = _take(rc.frames(), 3)

    assert len(frames) == 3
    assert rc.reconnects == 1
    # drop -> 1s; then four failed opens -> 2, 4, 8, capped at 10
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 10.0]


def test_file_source_stops_at_eof_without_reconnecting():
    cap = FakeCap([_frame(), _frame(), _frame(), _frame(), None])
    sleeps, sleep = _recorder()

    rc = ReconnectingCapture(
        "/videos/clip.mp4",
        cap_factory=_factory_from([cap]),
        sleep=sleep,
        log=lambda *_: None,
    )

    frames = list(rc.frames())

    assert len(frames) == 4
    assert rc.reconnects == 0
    assert sleeps == []               # a file EOF is not an error


def test_missing_file_raises_immediately_without_backoff():
    sleeps, sleep = _recorder()
    rc = ReconnectingCapture(
        "/videos/missing.mp4",
        cap_factory=_factory_from([None]),
        sleep=sleep,
        log=lambda *_: None,
    )

    with pytest.raises(ConnectionError):
        list(rc.frames())

    assert sleeps == []               # a file that won't open is not retried
    assert rc.reconnects == 0


def test_max_retries_gives_up_with_connectionerror():
    sleeps, sleep = _recorder()
    rc = ReconnectingCapture(
        "rtsp://cam/stream",
        max_retries=3,
        cap_factory=_factory_from([None, None, None, None]),
        sleep=sleep,
        log=lambda *_: None,
    )

    with pytest.raises(ConnectionError):
        list(rc.frames())

    assert sleeps == [1.0, 2.0, 4.0]  # three attempts, then it raises


def test_looks_live_classification():
    assert _looks_live(0) is True
    assert _looks_live("0") is True
    assert _looks_live("rtsp://host/stream") is True
    assert _looks_live("HTTP://host/stream") is True
    assert _looks_live("/home/user/clip.mp4") is False
    assert _looks_live("clip.MP4") is False
