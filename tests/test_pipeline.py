"""
Unit tests for footfall.pipeline.ThreadedFrameSource.

Covers the drop policy for a live source, the lossless-with-backpressure
policy for a file source, ordering, clean shutdown, and exception
propagation from the capture thread.
"""

import threading
import time

import pytest

from footfall.pipeline import ThreadedFrameSource

_noop = lambda *_a, **_k: None


# -- drop policy (white-box, no thread) -------------------------------------


def test_live_source_drops_oldest_when_buffer_is_full():
    src = ThreadedFrameSource(iter([]), drop=True, maxsize=2, log=_noop)

    src._put("a")
    src._put("b")          # buffer full: [a, b]
    src._put("c")          # evict a -> [b, c]
    src._put("d")          # evict b -> [c, d]

    assert src.dropped == 2
    remaining = []
    try:
        while True:
            remaining.append(src._q.get_nowait())
    except Exception:
        pass
    assert remaining == ["c", "d"]     # only the freshest frames survive


# -- lossless file path (real thread, deterministic) ----------------------


def test_file_source_delivers_every_frame_in_order():
    frames = [f"f{i}" for i in range(20)]
    src = ThreadedFrameSource(iter(frames), drop=False, maxsize=3, log=_noop)

    got = []
    for frame in src:
        got.append(frame)
        time.sleep(0.002)     # slow consumer: producer must block, not drop

    assert got == frames
    assert src.dropped == 0


def test_iteration_ends_cleanly_at_end_of_input():
    src = ThreadedFrameSource(iter(["f0", "f1"]), drop=False, maxsize=2,
                              log=_noop)

    assert list(src) == ["f0", "f1"]
    assert not src._thread.is_alive()


# -- shutdown ------------------------------------------------------------


def test_close_stops_the_capture_thread_on_an_endless_source():
    def endless():
        i = 0
        while True:
            yield f"f{i}"
            i += 1

    src = ThreadedFrameSource(endless(), drop=True, maxsize=2, log=_noop)
    it = iter(src)
    next(it)
    next(it)

    it.close()               # generator close -> ThreadedFrameSource.close()

    deadline = time.time() + 3.0
    while src._thread.is_alive() and time.time() < deadline:
        time.sleep(0.02)
    assert not src._thread.is_alive()


def test_generator_close_releases_the_underlying_iterator():
    closed = threading.Event()

    def source():
        try:
            i = 0
            while True:
                yield f"f{i}"
                i += 1
        finally:
            closed.set()

    src = ThreadedFrameSource(source(), drop=True, maxsize=2, log=_noop)
    it = iter(src)
    next(it)
    it.close()

    assert closed.wait(timeout=3.0)   # the capture generator's finally ran


# -- error propagation ---------------------------------------------------


def test_capture_thread_exception_reaches_the_consumer():
    def boom():
        yield "f0"
        raise RuntimeError("capture exploded")

    src = ThreadedFrameSource(boom(), drop=False, maxsize=2, log=_noop)

    got = []
    with pytest.raises(RuntimeError, match="exploded"):
        for frame in src:
            got.append(frame)

    assert got == ["f0"]
