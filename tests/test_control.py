"""
Unit tests for footfall.control: the LiveControls whitelist and
validation, and the localhost HTTP surface (driven against a fake tracker
so no model is loaded).
"""

import json
import urllib.error
import urllib.request

import pytest

from footfall.control import ControlServer, LiveControls
from footfall.tracker import Point


class FakeTracker:
    def __init__(self):
        self.run_id = "run-xyz"
        self.conf = 0.5
        self.iou = 0.5
        self.min_box_height = 0
        self.max_aspect = None
        self.preview = False
        self.line = (Point(0, 10), Point(20, 10))
        self.zone = [Point(0, 0), Point(5, 0), Point(5, 5)]
        self.ignore_zones = []
        self.geometry_size = (1280, 720)
        self.count_in = 3
        self.count_out = 1
        self._zone_entry_time = {7: 123.0}
        self._capture = None
        self._frame_source = None
        self._geometry_fitted = True
        self._controls = LiveControls(self)


# -- LiveControls.apply ----------------------------------------------


def test_apply_accepts_a_valid_threshold():
    t = FakeTracker()
    assert t._controls.apply({"conf": 0.7}) == {
        "applied": {"conf": 0.7}, "rejected": {}}
    assert t.conf == 0.7


def test_apply_rejects_restart_only_keys():
    t = FakeTracker()
    out = t._controls.apply({"imgsz": 1280, "model": "yolo11m.pt"})
    assert out["applied"] == {}
    assert out["rejected"] == {"imgsz": "requires a restart",
                               "model": "requires a restart"}


def test_apply_rejects_unknown_out_of_range_and_wrong_type():
    t = FakeTracker()
    out = t._controls.apply({"nope": 1, "conf": 1.5, "iou": "high",
                             "min_box_height": True})
    assert out["applied"] == {}
    assert out["rejected"]["nope"] == "unknown or not live-tunable"
    assert out["rejected"]["conf"] == "out of range"
    assert "expected" in out["rejected"]["iou"]
    assert "expected" in out["rejected"]["min_box_height"]   # bool is not int here


def test_apply_partial(t=None):
    t = FakeTracker()
    out = t._controls.apply({"conf": 0.6, "imgsz": 640})
    assert out["applied"] == {"conf": 0.6}
    assert out["rejected"] == {"imgsz": "requires a restart"}
    assert t.conf == 0.6


def test_apply_preview_needs_a_bool():
    t = FakeTracker()
    assert t._controls.apply({"preview": 1})["rejected"]["preview"] == \
        "expected a boolean"
    assert t._controls.apply({"preview": True})["applied"] == {"preview": True}


# -- LiveControls.set_geometry -------------------------------------


def test_set_geometry_replaces_and_marks_unfitted():
    t = FakeTracker()
    geo = t._controls.set_geometry(
        line=[[1, 2], [3, 4]],
        zone=[[0, 0], [10, 0], [10, 10], [0, 10]],
        ignore=[[[1, 1], [2, 1], [2, 2]]],
        size=[640, 480])
    assert [[p.x, p.y] for p in t.line] == [[1, 2], [3, 4]]
    assert len(t.zone) == 4
    assert len(t.ignore_zones) == 1
    assert t.geometry_size == (640, 480)
    assert t._geometry_fitted is False
    assert geo["line"] == [[1, 2], [3, 4]]


def test_set_geometry_leaves_unpassed_fields_alone():
    t = FakeTracker()
    original_zone = t.zone
    t._controls.set_geometry(line=[[9, 9], [8, 8]])
    assert t.zone is original_zone
    assert [[p.x, p.y] for p in t.line] == [[9, 9], [8, 8]]


def test_snapshot_shape():
    snap = FakeTracker()._controls.snapshot()
    assert snap["run_id"] == "run-xyz"
    assert snap["settings"]["conf"] == 0.5
    assert snap["geometry"]["line"] == [[0, 10], [20, 10]]
    assert snap["counters"] == {
        "footfall_in": 3, "footfall_out": 1, "in_zone_now": 1,
        "reconnects": 0, "stale_trips": 0, "dropped_frames": 0,
    }


# -- HTTP surface ---------------------------------------------------


@pytest.fixture
def server():
    tracker = FakeTracker()
    srv = ControlServer(tracker, port=0).start()
    try:
        yield tracker, f"http://127.0.0.1:{srv.port}"
    finally:
        srv.stop()


def _req(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def test_healthz(server):
    _t, base = server
    status, payload = _req(base + "/healthz")
    assert status == 200
    assert payload == {"ok": True, "run_id": "run-xyz"}


def test_status_and_config_endpoints(server):
    _t, base = server
    _, status = _req(base + "/status")
    assert status["counters"]["footfall_in"] == 3
    _, cfg = _req(base + "/config")
    assert set(cfg) == {"settings", "geometry"}


def test_patch_config_applies_and_reports(server):
    tracker, base = server
    status, payload = _req(base + "/config", "PATCH",
                           {"conf": 0.66, "imgsz": 1280})
    assert status == 200
    assert payload["applied"] == {"conf": 0.66}
    assert payload["rejected"] == {"imgsz": "requires a restart"}
    assert tracker.conf == 0.66


def test_put_geometry(server):
    tracker, base = server
    status, payload = _req(base + "/geometry", "PUT",
                           {"line": [[1, 1], [2, 2]], "size": [640, 480]})
    assert status == 200
    assert payload["geometry"]["line"] == [[1, 1], [2, 2]]
    assert tracker._geometry_fitted is False


def test_bad_json_is_400(server):
    _t, base = server
    req = urllib.request.Request(base + "/config", data=b"{nope",
                                 method="PATCH")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req)
    assert excinfo.value.code == 400


def test_unknown_path_is_404(server):
    _t, base = server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(base + "/nope")
    assert excinfo.value.code == 404
