"""
Unit tests for hardware-accelerated decode selection in footfall.capture.

plan_open (which backend + params for a source) and _probe_hw_accel (what
actually engaged) are the two pieces with real logic; the cv2.VideoCapture
call itself is a thin wrapper exercised against bogus paths only, to prove
the hw path never raises.
"""

import cv2
import numpy as np

from footfall.capture import ReconnectingCapture, plan_open

_noop = lambda *_a, **_k: None


# -- plan_open -----------------------------------------------------------


def test_stream_requests_any_hardware_decoder():
    backend, params = plan_open("rtsp://cam/stream", "auto")
    assert backend == cv2.CAP_FFMPEG
    assert params == [int(cv2.CAP_PROP_HW_ACCELERATION),
                      int(cv2.VIDEO_ACCELERATION_ANY)]


def test_file_requests_any_hardware_decoder():
    backend, params = plan_open("/videos/clip.mp4", "auto")
    assert backend == cv2.CAP_FFMPEG
    assert params[0] == int(cv2.CAP_PROP_HW_ACCELERATION)


def test_camera_index_is_left_on_the_default_backend():
    assert plan_open(0, "auto") == (None, None)


def test_none_forces_software_decode():
    assert plan_open("rtsp://cam/stream", "none") == (None, None)


def test_a_gstreamer_pipeline_string_is_detected():
    pipe = ("rtspsrc location=rtsp://cam ! rtph264depay ! "
            "nvv4l2decoder ! nvvidconv ! appsink")
    backend, params = plan_open(pipe, "auto")
    assert backend == cv2.CAP_GSTREAMER
    assert params is None


def test_gstreamer_backend_can_be_forced():
    backend, _ = plan_open("rtsp://cam/stream", "gstreamer")
    assert backend == cv2.CAP_GSTREAMER


def test_plan_open_degrades_when_opencv_lacks_the_property():
    class OldCv2:            # no CAP_PROP_HW_ACCELERATION / CAP_FFMPEG / ...
        pass

    assert plan_open("rtsp://cam/stream", "auto", cv2mod=OldCv2()) == (None, None)


# -- _probe_hw_accel ---------------------------------------------------


class _Reports:
    def __init__(self, value):
        self._value = value

    def get(self, _prop):
        return self._value


def _rc():
    return ReconnectingCapture("rtsp://x", log=_noop)


def test_probe_names_the_active_accelerator():
    rc = _rc()
    rc._probe_hw_accel(_Reports(3.0))          # 3 -> vaapi
    assert rc.hw_accel_active == "vaapi"


def test_probe_reports_none_for_software_decode():
    rc = _rc()
    rc._probe_hw_accel(_Reports(0.0))
    assert rc.hw_accel_active == "none"


def test_probe_keeps_an_unknown_value_visible():
    rc = _rc()
    rc._probe_hw_accel(_Reports(99.0))
    assert rc.hw_accel_active == "type99"


def test_probe_survives_a_capture_that_cannot_report():
    class Bad:
        def get(self, _prop):
            raise RuntimeError("property not supported")

    rc = _rc()
    rc._probe_hw_accel(Bad())
    assert rc.hw_accel_active == "unknown"


# -- probe runs once, from inside frames() ----------------------------


class _FakeCap:
    def __init__(self, frames, accel=0):
        self._frames = list(frames)
        self._accel = accel
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_a):
        return True

    def get(self, _prop):
        return float(self._accel)

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_probe_runs_once_on_first_open():
    cap = _FakeCap([np.zeros((4, 4, 3), np.uint8)] * 3, accel=2)
    rc = ReconnectingCapture("/x/clip.mp4", cap_factory=lambda _s: cap,
                             log=_noop)

    list(rc.frames())

    assert rc._hw_probed is True
    assert rc.hw_accel_active == "d3d11"


# -- the real cv2 wrapper never raises on the hw path ------------------


def test_default_capture_auto_falls_back_without_raising():
    rc = ReconnectingCapture("/no/such/file.mp4", hw_accel="auto", log=_noop)
    cap = rc._default_capture("/no/such/file.mp4")
    assert cap is not None and not cap.isOpened()
    cap.release()


def test_default_capture_none_opens_plainly():
    rc = ReconnectingCapture("/no/such/file.mp4", hw_accel="none", log=_noop)
    cap = rc._default_capture("/no/such/file.mp4")
    assert cap is not None and not cap.isOpened()
    cap.release()
