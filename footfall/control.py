"""
control.py
----------
A small localhost HTTP API for a running FootfallTracker: read the live
status and config, and hot-apply a bounded set of changes -- detection
thresholds, the cheap geometry filters, and the line / zone / ignore
geometry -- without stopping the process.

Anything that would need the model or the capture rebuilt (model, imgsz,
tracker, source, hw_accel) is rejected with "requires a restart".

The server binds to 127.0.0.1 only; that is the whole security boundary
for now. Real authentication is a Phase 5 concern.

    GET   /healthz          -> {"ok": true, "run_id": ...}
    GET   /status           -> counters + settings + geometry
    GET   /config           -> {"settings": {...}, "geometry": {...}}
    PATCH /config   {json}   -> {"applied": {...}, "rejected": {...}}
    PUT   /geometry {json}   -> {"geometry": {...}}   (line/zone/ignore/size)
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# name -> (accepted python types, predicate). bool is handled specially
# because isinstance(True, int) is True.
_TUNABLE = {
    "conf": ((int, float), lambda v: 0.0 < float(v) <= 1.0),
    "iou": ((int, float), lambda v: 0.0 < float(v) <= 1.0),
    "min_box_height": ((int,), lambda v: v >= 0),
    "max_aspect": ((int, float, type(None)), lambda v: v is None or float(v) > 0),
    "preview": ((bool,), lambda v: True),
}
_RESTART_ONLY = {"model", "model_path", "imgsz", "tracker", "device",
                 "source", "hw_accel", "capture_size"}


def _points(seq):
    return [[p.x, p.y] for p in seq] if seq else None


class LiveControls:
    """Owns the lock between the run loop and the control API, and the
    whitelist of what may change while running."""

    def __init__(self, owner):
        self._owner = owner
        self._lock = threading.Lock()

    # -- reads -----------------------------------------------------------

    def snapshot(self) -> dict:
        o = self._owner
        with self._lock:
            cap = getattr(o, "_capture", None)
            src = getattr(o, "_frame_source", None)
            return {
                "run_id": getattr(o, "run_id", None),
                "settings": {k: getattr(o, k, None) for k in _TUNABLE},
                "geometry": {
                    "line": _points(getattr(o, "line", None)),
                    "zone": _points(getattr(o, "zone", None)),
                    "ignore_zones": [_points(z)
                                     for z in (getattr(o, "ignore_zones", None) or [])],
                    "geometry_size": list(o.geometry_size)
                    if getattr(o, "geometry_size", None) else None,
                },
                "counters": {
                    "footfall_in": getattr(o, "count_in", 0),
                    "footfall_out": getattr(o, "count_out", 0),
                    "in_zone_now": len(getattr(o, "_zone_entry_time", {}) or {}),
                    "reconnects": getattr(cap, "reconnects", 0) if cap else 0,
                    "stale_trips": getattr(cap, "stale_trips", 0) if cap else 0,
                    "dropped_frames": getattr(src, "dropped", 0) if src else 0,
                },
            }

    # -- writes ------------------------------------------------------

    def apply(self, changes: dict) -> dict:
        applied, rejected = {}, {}
        with self._lock:
            for key, value in (changes or {}).items():
                reason = self._reject_reason(key, value)
                if reason:
                    rejected[key] = reason
                    continue
                if key in ("conf", "iou", "max_aspect") and value is not None:
                    value = float(value)
                setattr(self._owner, key, value)
                applied[key] = value
        return {"applied": applied, "rejected": rejected}

    def set_geometry(self, line=None, zone=None, ignore=None, size=None) -> dict:
        from .tracker import Point

        with self._lock:
            o = self._owner
            if line is not None:
                o.line = ((Point(*line[0]), Point(*line[1])) if line else None)
            if zone is not None:
                o.zone = [Point(x, y) for x, y in zone] if zone else None
            if ignore is not None:
                o.ignore_zones = [[Point(x, y) for x, y in poly]
                                  for poly in ignore]
            if size is not None:
                o.geometry_size = (int(size[0]), int(size[1]))
            # force _fit_geometry to re-run against the next live frame
            o._geometry_fitted = False
        return self.snapshot()["geometry"]

    # -- validation --------------------------------------------------

    @staticmethod
    def _reject_reason(key, value):
        if key in _RESTART_ONLY:
            return "requires a restart"
        if key not in _TUNABLE:
            return "unknown or not live-tunable"
        types, ok = _TUNABLE[key]
        is_bool = isinstance(value, bool)
        if key == "preview":
            if not is_bool:
                return "expected a boolean"
        elif is_bool or not isinstance(value, types):
            return f"expected {' or '.join(t.__name__ for t in types)}"
        try:
            if not ok(value):
                return "out of range"
        except (TypeError, ValueError):
            return "out of range"
        return None


def _make_handler(tracker):
    controls = tracker._controls

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):        # silence the stderr access log
            pass

        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            return json.loads(raw) if raw else {}

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"ok": True,
                                 "run_id": getattr(tracker, "run_id", None)})
            elif self.path == "/status":
                self._send(200, controls.snapshot())
            elif self.path.rstrip("/") == "/config":
                snap = controls.snapshot()
                self._send(200, {"settings": snap["settings"],
                                 "geometry": snap["geometry"]})
            else:
                self._send(404, {"error": "not found"})

        def do_PATCH(self):
            if self.path.rstrip("/") != "/config":
                return self._send(404, {"error": "not found"})
            try:
                body = self._read_json()
            except ValueError:
                return self._send(400, {"error": "invalid JSON"})
            self._send(200, controls.apply(body))

        def do_PUT(self):
            if self.path.rstrip("/") != "/geometry":
                return self._send(404, {"error": "not found"})
            try:
                body = self._read_json()
            except ValueError:
                return self._send(400, {"error": "invalid JSON"})
            try:
                geo = controls.set_geometry(
                    line=body.get("line"), zone=body.get("zone"),
                    ignore=body.get("ignore"), size=body.get("size"))
            except (TypeError, ValueError, IndexError) as exc:
                return self._send(400, {"error": f"bad geometry: {exc}"})
            self._send(200, {"geometry": geo})

    return Handler


class ControlServer:
    def __init__(self, tracker, host="127.0.0.1", port=0):
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(tracker))
        self.host, self.port = self._httpd.server_address[:2]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="control-api", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)
