"""
footfall_tracker.py
--------------------
Prototype: count people crossing an entrance line (footfall) and measure
dwell time inside a defined zone (e.g. queue area), from any video source
that OpenCV can open — a file, a webcam index, or an RTSP/ONVIF camera URL.

Pipeline:
    video source -> YOLOv8 person detection -> ByteTrack (persistent IDs)
    -> line-crossing counter (in/out) -> polygon zone dwell-time tracker
    -> event store (SQLite) + annotated output video

Usage (see run_demo.py for a runnable example):

    from footfall_tracker import FootfallTracker, Point

    tracker = FootfallTracker(
        source="rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101",
        line=(Point(100, 400), Point(900, 400)),   # entrance line
        zone=[Point(300, 300), Point(700, 300), Point(700, 550), Point(300, 550)],  # queue area
        output_video="annotated_out.mp4",
        events_db="events.db",
    )
    summary = tracker.run()
    print(summary)
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO

from .capture import ReconnectingCapture
from .pipeline import ThreadedFrameSource
from .storage import CsvEventSink, EventSink, NullSink, SqliteEventSink


@dataclass
class Point:
    x: float
    y: float

    def as_tuple(self):
        return (int(self.x), int(self.y))


def _side_of_line(p: Point, a: Point, b: Point) -> float:
    """Signed area trick: >0 means p is on one side of line a->b, <0 the other."""
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)


def _point_in_polygon(p: Point, polygon: List[Point]) -> bool:
    poly_np = [pt.as_tuple() for pt in polygon]
    import numpy as np

    contour = np.array(poly_np, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(p.x), float(p.y)), False)
    return result >= 0



# Common capture modes, widest first. A webcam silently falls back to its
# nearest supported mode, so we set one and read back what we actually got.
COMMON_MODES = [
    (3840, 2160), (2560, 1440), (1920, 1080), (1600, 900),
    (1280, 720), (1024, 576), (960, 540), (800, 600), (640, 480),
]


def max_capture_size(index, modes=None):
    """Return the widest (w, h) this camera really delivers, or None."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    best = None
    for w, h in (modes or COMMON_MODES):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gh, gw = frame.shape[:2]
        if best is None or gw * gh > best[0] * best[1]:
            best = (gw, gh)
        if (gw, gh) == (w, h):
            break
    cap.release()
    return best


class FootfallTracker:
    def __init__(
        self,
        source,
        line: Optional[Tuple[Point, Point]] = None,
        zone: Optional[List[Point]] = None,
        model_path: str = "yolov8n.pt",
        person_class_id: int = 0,
        conf: float = 0.35,
        output_video: Optional[str] = None,
        events_csv: Optional[str] = None,
        events_db: Optional[str] = None,
        event_sink: Optional[EventSink] = None,
        site: Optional[str] = None,
        tz: Optional[str] = None,
        max_frames: Optional[int] = None,
        display_scale: float = 1.0,
        preview: bool = False,
        capture_size: Optional[Tuple[int, int]] = None,
        geometry_size: Optional[Tuple[int, int]] = None,
        imgsz: int = 640,
        iou: float = 0.5,
        tracker: Optional[str] = None,
        device: Optional[str] = None,
        min_box_height: int = 0,
        max_aspect: Optional[float] = None,
        ignore_zones: Optional[List[List[Point]]] = None,
        reconnect_initial: float = 1.0,
        reconnect_factor: float = 2.0,
        reconnect_max: float = 30.0,
        reconnect_retries: Optional[int] = None,
        stale_after: Optional[float] = 10.0,
        detect_frozen: bool = True,
        hw_accel: str = "auto",
        frame_buffer: int = 2,
        drop_stale_frames: Optional[bool] = None,
    ):
        self.source = source
        self.line = line
        self.zone = zone
        self.conf = conf
        self.person_class_id = person_class_id
        self.output_video = output_video
        self.events_csv = events_csv
        self.events_db = events_db
        self.max_frames = max_frames
        self.display_scale = display_scale
        self.preview = preview
        self.capture_size = capture_size
        # frame size the line/zone were authored against, if known
        self.geometry_size = geometry_size
        self._geometry_fitted = False
        self.imgsz = imgsz
        self.iou = iou
        # default to the repo's tuned config, resolved absolutely so it
        # works regardless of the working directory
        if tracker is None:
            from . import DEFAULT_TRACKER
            tracker = str(DEFAULT_TRACKER)
        self.tracker = tracker
        self.device = device
        # geometry sanity filters -- cheap defence against non-person boxes
        self.min_box_height = min_box_height
        self.max_aspect = max_aspect
        self.rejected_boxes = 0
        # detections whose anchor falls in here are discarded entirely:
        # mirrors, screens, posters, windows onto a street
        self.ignore_zones = ignore_zones or []
        self.ignored_detections = 0
        self._preview_window = "Footfall Tracker - press Q to stop"

        # RTSP/webcam reconnection: exponential backoff on a dropped stream.
        # None retries = keep trying forever (the right default for a camera
        # that will come back after a reboot).
        self.reconnect_initial = reconnect_initial
        self.reconnect_factor = reconnect_factor
        self.reconnect_max = reconnect_max
        self.reconnect_retries = reconnect_retries
        # Watchdog: force a reconnect if a live stream stays open but stops
        # delivering new frames (frozen transport or a repeating decoder).
        self.stale_after = stale_after
        self.detect_frozen = detect_frozen
        # "auto" requests a hardware video decoder where the platform has
        # one, with automatic software fallback.
        self.hw_accel = hw_accel
        self._capture = None

        # Capture runs on its own thread with a small bounded buffer, so a
        # slow inference step drops stale frames instead of making the
        # camera fall behind real time. drop_stale_frames=None means "drop
        # on a live source, keep every frame from a file".
        self.frame_buffer = frame_buffer
        self.drop_stale_frames = drop_stale_frames
        self._frame_source = None

        from . import resolve_model

        self.model = YOLO(resolve_model(model_path))

        # track_id -> last known side of the line (float sign)
        self._last_side = {}
        # track_id -> timestamp when it entered the zone (None if not inside)
        self._zone_entry_time = {}
        # completed dwell records: list of (track_id, seconds)
        self._dwell_records = []

        self.count_in = 0
        self.count_out = 0

        # per-minute bucket -> count of "in" events, for peak-hour analysis
        self._in_by_minute = defaultdict(int)

        # Event store. Prefer an injected sink, then a SQLite file, then the
        # legacy CSV, else discard. The sink owns run_id, UTC timestamps,
        # and idempotency -- see footfall/storage.py.
        if event_sink is not None:
            self._sink = event_sink
        elif events_db:
            self._sink = SqliteEventSink(events_db, source=str(source),
                                         site=site, tz=tz)
        elif events_csv:
            self._sink = CsvEventSink(events_csv)
        else:
            self._sink = NullSink()
        self.run_id = self._sink.run_id

    def _log_event(self, event: str, track_id: int, value=""):
        self._sink.emit(event, track_id, value if value != "" else None)

    def _centroid(self, box) -> Point:
        x1, y1, x2, y2 = box
        return Point((x1 + x2) / 2, (y1 + y2) / 2)

    def _in_ignored_region(self, p: Point) -> bool:
        for poly in self.ignore_zones:
            if _point_in_polygon(p, poly):
                return True
        return False

    def _plausible_person(self, box) -> bool:
        """Reject boxes that cannot be a person at this camera geometry.

        The detector already filters to class 0, so anything wrong that gets
        through is a low-quality person detection on a coat, a mannequin, a
        poster. Size and shape catch most of them for near-zero cost.
        """
        x1, y1, x2, y2 = box
        h = y2 - y1
        w = x2 - x1
        if h <= 0 or w <= 0:
            return False
        if self.min_box_height and h < self.min_box_height:
            return False
        if self.max_aspect and (w / h) > self.max_aspect:
            return False
        return True

    def _update_line_crossing(self, track_id: int, centroid: Point):
        if not self.line:
            return
        a, b = self.line
        side = _side_of_line(centroid, a, b)
        prev = self._last_side.get(track_id)
        self._last_side[track_id] = side

        if prev is None:
            return  # first time we see this track, nothing to compare against

        # sign flip => crossed the line
        if prev < 0 and side >= 0:
            self.count_in += 1
            minute_bucket = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._in_by_minute[minute_bucket] += 1
            self._log_event("line_in", track_id)
        elif prev > 0 and side <= 0:
            self.count_out += 1
            self._log_event("line_out", track_id)

    def _update_zone_dwell(self, track_id: int, centroid: Point, now: float):
        if not self.zone:
            return
        inside = _point_in_polygon(centroid, self.zone)
        was_inside = track_id in self._zone_entry_time

        if inside and not was_inside:
            self._zone_entry_time[track_id] = now
            self._log_event("zone_enter", track_id)
        elif not inside and was_inside:
            entered_at = self._zone_entry_time.pop(track_id)
            dwell = now - entered_at
            self._dwell_records.append((track_id, dwell))
            self._log_event("zone_exit", track_id, f"{dwell:.1f}s")

    def _current_queue_length(self) -> int:
        return len(self._zone_entry_time)

    def _draw_overlay(self, frame, results):
        if self.line:
            a, b = self.line
            cv2.line(frame, a.as_tuple(), b.as_tuple(), (0, 255, 255), 2)
        if self.zone:
            pts = [pt.as_tuple() for pt in self.zone]
            import numpy as np

            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (255, 128, 0), 2)

        if self.ignore_zones:
            import numpy as np

            for poly in self.ignore_zones:
                arr = np.array([pt.as_tuple() for pt in poly], dtype=np.int32)
                cv2.polylines(frame, [arr], True, (0, 0, 235), 2)
                cv2.putText(frame, "IGNORED", (arr[0][0] + 6, arr[0][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 235), 1)

        cv2.putText(frame, f"IN: {self.count_in}  OUT: {self.count_out}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        if self.zone:
            cv2.putText(frame, f"In queue zone now: {self._current_queue_length()}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        return frame

    def _iter_results(self):
        """Yield per-frame Results.

        Capture is owned here (through ReconnectingCapture, so a dropped
        RTSP/webcam stream reconnects with backoff) and run on its own
        thread (through ThreadedFrameSource, so this loop dropping behind
        does not stall the camera). Frames reach the tracker one at a time;
        persist=True carries track state across them, and across a
        reconnect.
        """
        track_kw = dict(
            classes=[self.person_class_id],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            persist=True,
            verbose=False,
        )
        if self.device is not None:
            track_kw["device"] = self.device

        self._capture = ReconnectingCapture(
            self.source,
            capture_size=self.capture_size,
            backoff_initial=self.reconnect_initial,
            backoff_factor=self.reconnect_factor,
            backoff_max=self.reconnect_max,
            max_retries=self.reconnect_retries,
            stale_after=self.stale_after,
            detect_frozen=self.detect_frozen,
            hw_accel=self.hw_accel,
        )
        drop = (self.drop_stale_frames if self.drop_stale_frames is not None
                else self._capture.is_live)
        self._frame_source = ThreadedFrameSource(
            self._capture.frames(), drop=drop, maxsize=self.frame_buffer)
        for frame in self._frame_source:
            yield self.model.track(frame, **track_kw)[0]

    def _fit_geometry(self, frame_w: int, frame_h: int):
        """Rescale line/zone if they were drawn on a different frame size.

        Pixel coordinates are meaningless without the resolution they were
        authored at -- a zone drawn on 1280x720 lands nowhere on a 640x480
        frame. Rescaling keeps the geometry over the same part of the scene.
        """
        self._geometry_fitted = True
        if not self.geometry_size:
            return
        gw, gh = self.geometry_size
        if (gw, gh) == (frame_w, frame_h) or not gw or not gh:
            return
        sx, sy = frame_w / float(gw), frame_h / float(gh)
        print(f"[geometry] drawn on {gw}x{gh}, frame is {frame_w}x{frame_h}"
              f" -- rescaling by {sx:.3f}x{sy:.3f}")
        if self.line:
            a, b = self.line
            self.line = (Point(a.x * sx, a.y * sy), Point(b.x * sx, b.y * sy))
        if self.zone:
            self.zone = [Point(p.x * sx, p.y * sy) for p in self.zone]
        if self.ignore_zones:
            self.ignore_zones = [[Point(p.x * sx, p.y * sy) for p in poly]
                                 for poly in self.ignore_zones]

    def run(self) -> dict:
        cap_writer = None
        frame_idx = 0
        started_at = time.time()

        try:
            for result in self._iter_results():
                frame = result.orig_img
                now = time.time()

                if not self._geometry_fitted:
                    self._fit_geometry(frame.shape[1], frame.shape[0])

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu().numpy().astype(int)
                    for box, track_id in zip(boxes, ids):
                        if not self._plausible_person(box):
                            self.rejected_boxes += 1
                            continue
                        centroid = self._centroid(box)
                        if self._in_ignored_region(centroid):
                            self.ignored_detections += 1
                            continue
                        self._update_line_crossing(track_id, centroid)
                        self._update_zone_dwell(track_id, centroid, now)
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        cv2.putText(frame, f"ID {track_id}", (x1, max(0, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

                frame = self._draw_overlay(frame, result)

                if self.output_video:
                    if cap_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        cap_writer = cv2.VideoWriter(self.output_video, fourcc, 20.0, (w, h))
                    cap_writer.write(frame)

                if self.preview:
                    disp = frame
                    if self.display_scale != 1.0:
                        disp = cv2.resize(frame, None, fx=self.display_scale,
                                          fy=self.display_scale)
                    cv2.imshow(self._preview_window, disp)
                    # waitKey is what actually paints the window; Q or ESC quits
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        print("[preview] stop requested")
                        break

                frame_idx += 1
                if self.max_frames and frame_idx >= self.max_frames:
                    break
        except KeyboardInterrupt:
            print("[interrupted] flushing outputs...")
        finally:
            # close out still-open dwell sessions BEFORE the sink closes,
            # so a Ctrl+C on a live camera still leaves usable output
            for track_id, entered_at in self._zone_entry_time.items():
                dwell = time.time() - entered_at
                self._dwell_records.append((track_id, dwell))
                self._log_event("zone_exit", track_id, f"{dwell:.1f}s")
            self._zone_entry_time.clear()
            if cap_writer:
                cap_writer.release()
            self._sink.close()
            if self.preview:
                cv2.destroyAllWindows()

        elapsed = time.time() - started_at
        avg_dwell = (sum(d for _, d in self._dwell_records) / len(self._dwell_records)
                     if self._dwell_records else 0.0)
        peak_minute = max(self._in_by_minute.items(), key=lambda kv: kv[1], default=(None, 0))

        return {
            "run_id": self.run_id,
            "events_logged": getattr(self._sink, "count", None),
            "frames_processed": frame_idx,
            "processing_seconds": round(elapsed, 1),
            "footfall_in": self.count_in,
            "footfall_out": self.count_out,
            "avg_dwell_seconds": round(avg_dwell, 1),
            "dwell_samples": len(self._dwell_records),
            "rejected_boxes": self.rejected_boxes,
            "ignored_detections": self.ignored_detections,
            "peak_minute": peak_minute[0],
            "peak_minute_count": peak_minute[1],
            "reconnects": self._capture.reconnects if self._capture else 0,
            "stale_trips": self._capture.stale_trips if self._capture else 0,
            "hw_decode": (self._capture.hw_accel_active
                          if self._capture else None),
            "dropped_frames": (self._frame_source.dropped
                               if self._frame_source else 0),
        }