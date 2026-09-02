"""
End-to-end regression test for the counting pipeline.

A tiny video supplies the frames; a fake detector supplies scripted
boxes, so FootfallTracker.run() is driven deterministically -- no YOLO
weights, no randomness. One person walks across the entrance line (one
IN), another enters the queue zone and leaves across the line the other
way (one dwell, one OUT). The asserted counts and event stream are what a
detector or tracker change must not silently move.
"""

import numpy as np
import pytest

from footfall.tracker import FootfallTracker, Point

cv2 = pytest.importorskip("cv2")

W = H = 64
LINE = (Point(32, 0), Point(32, 64))
ZONE = [Point(10, 40), Point(50, 40), Point(50, 60), Point(10, 60)]


def _box(cx, cy):
    return (cx - 4, cy - 10, cx + 4, cy + 10)     # centroid (cx, cy)


# per frame: list of (xyxy, track_id)
SCRIPT = [
    [(_box(55, 32), 1), (_box(30, 10), 2)],       # f0
    [(_box(40, 32), 1), (_box(30, 50), 2)],       # f1  tid2 enters the zone
    [(_box(20, 32), 1), (_box(30, 50), 2)],       # f2  tid1 crosses the line -> IN
    [(_box(10, 32), 1), (_box(50, 70), 2)],       # f3  tid2 crosses -> OUT, leaves zone
]


class _Tensorish:
    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _Boxes:
    def __init__(self, entries):
        self.xyxy = _Tensorish(
            np.array([e[0] for e in entries], dtype=float).reshape(-1, 4))
        self.id = _Tensorish(np.array([e[1] for e in entries], dtype=float))


class _Results:
    def __init__(self, frame, entries):
        self.orig_img = frame
        self.boxes = _Boxes(entries)


class FakeYOLO:
    def __init__(self, script):
        self._script = script
        self._i = 0

    def track(self, frame, **_kw):
        entries = self._script[self._i] if self._i < len(self._script) else []
        self._i += 1
        return [_Results(frame, entries)]


class RecordingSink:
    run_id = "regression"

    def __init__(self):
        self.events = []

    def emit(self, event, track_id, value=None):
        self.events.append((event, int(track_id), value))

    def close(self):
        pass


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             5.0, (W, H))
    assert writer.isOpened()
    for _ in SCRIPT:
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def test_counting_pipeline_regression(clip):
    sink = RecordingSink()
    tracker = FootfallTracker(
        source=clip,
        model=FakeYOLO(SCRIPT),
        line=LINE,
        zone=ZONE,
        event_sink=sink,
        preview=False,
    )

    summary = tracker.run()

    assert summary["frames_processed"] == 4
    assert summary["footfall_in"] == 1
    assert summary["footfall_out"] == 1
    assert summary["dwell_samples"] == 1

    names = [(e, t) for e, t, _v in sink.events]
    assert names == [
        ("zone_enter", 2),
        ("line_in", 1),
        ("line_out", 2),
        ("zone_exit", 2),
    ]
    assert sink.events[-1][2].endswith("s")       # dwell value like "0.0s"
