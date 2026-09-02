"""
pipeline.py
-----------
Runs frame capture in its own thread so slow downstream work -- inference,
event logic, video encoding -- does not make the camera fall behind real
time.

In the prototype every captured frame was processed in lock step with
inference. When inference is slower than the camera's frame rate (the
normal case on a CPU), unprocessed frames pile up in OpenCV's and the
OS's buffers and the pipeline drifts further and further behind live: the
counts stay right but they arrive late, and latency grows without bound.

``ThreadedFrameSource`` puts a small bounded buffer between capture and
processing:

* a *live* source keeps only the most recent frames -- when the buffer is
  full the oldest frame is dropped (and counted), never blocking the
  capture thread, so processing always works on near-current video;
* a *file* source drops nothing -- the capture thread blocks when the
  buffer is full, so every frame is processed and the counts stay exact.

Only the capture / processing split is handled here. Splitting inference
from the event logic, and hardware-accelerated decode, are separate steps.
"""

import logging
import queue
import threading

_SENTINEL = object()
_log = logging.getLogger(__name__)


class ThreadedFrameSource:
    def __init__(self, frames, *, drop, maxsize=2, log=None):
        """
        frames   iterable of frames, e.g. ReconnectingCapture.frames()
        drop     True  -> live source: discard the oldest buffered frame
                          when the buffer is full; the capture thread never
                          blocks and processing stays near real time
                 False -> file source: block the capture thread when the
                          buffer is full; nothing is lost
        maxsize  buffer depth (frames)
        log      debug sink for drops; defaults to this module's logger
        """
        self._frames = iter(frames)
        self._drop = drop
        self._q = queue.Queue(maxsize=max(1, maxsize))
        self._log = log or _log.debug

        self._stop = threading.Event()
        self._done = False
        self._exc = None
        self.dropped = 0

        self._thread = threading.Thread(
            target=self._produce, name="frame-capture", daemon=True)
        self._started = False

    # -- capture thread -----------------------------------------------------

    def _produce(self):
        try:
            for frame in self._frames:
                if self._stop.is_set():
                    break
                self._put(frame)
        except BaseException as exc:            # surfaced to the consumer
            self._exc = exc
        finally:
            self._done = True
            # let a suspended ReconnectingCapture generator release its
            # cv2 handle instead of waiting for garbage collection
            closer = getattr(self._frames, "close", None)
            if callable(closer):
                try:
                    closer()
                except BaseException:
                    pass
            try:
                self._q.put_nowait(_SENTINEL)
            except queue.Full:
                pass

    def _put(self, frame):
        if not self._drop:
            # file source: apply backpressure, lose nothing
            while not self._stop.is_set():
                try:
                    self._q.put(frame, timeout=0.2)
                    return
                except queue.Full:
                    continue
            return
        # live source: make room by dropping the oldest, never block
        try:
            self._q.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._q.get_nowait()
            self.dropped += 1
            self._log(f"buffer full, dropped a stale frame ({self.dropped} total)")
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    # -- processing thread (the caller) -----------------------------------

    def __iter__(self):
        if not self._started:
            self._started = True
            self._thread.start()
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    if self._done:
                        break
                    continue
                if item is _SENTINEL:
                    break
                yield item
            if self._exc is not None:
                raise self._exc
        finally:
            self.close()

    def close(self):
        self._stop.set()
        # unblock a capture thread parked on a full buffer (file source)
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self._started:
            self._thread.join(timeout=2.0)
