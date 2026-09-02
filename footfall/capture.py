"""
capture.py
----------
A video source that survives the camera going away.

The prototype read frames in a bare ``while True: cap.read()`` loop and
exited the moment a read failed. Real cameras drop constantly -- a
firmware reboot, a PoE renegotiation, a Wi-Fi blip -- and a service that
falls over on the first hiccup is not a service.

``ReconnectingCapture`` wraps ``cv2.VideoCapture`` and, for a *live*
source, reopens it with exponential backoff whenever a read fails or the
stream cannot be opened: wait 1s, then 2s, 4s, ... up to a ceiling, for as
many attempts as allowed (unlimited by default). A *file* source is left
alone -- a failed read there is just the end of the file, so iteration
stops normally.

A live source also runs a staleness watchdog: a stream can stay "open" --
reads keep returning -- while delivering nothing new, because the
transport froze or the decoder is repeating its last frame. A monitor
thread times the gap since the last genuinely new frame and, once it
exceeds ``stale_after``, force-releases the capture so the stuck read
gives up and the reconnect path runs.

Hardware-accelerated decode is a separate step.

    for frame in ReconnectingCapture("rtsp://cam/stream").frames():
        ...
"""

import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

# URL schemes that mean "a live stream", i.e. reconnect forever rather
# than treating a failed read as end-of-input.
_LIVE_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


def _looks_live(source) -> bool:
    """Guess whether a source is a live stream or a finite file.

    Camera indices (``0``) and streaming URLs are live. Anything else is
    assumed to be a file on disk, where a failed read means EOF.
    """
    if isinstance(source, int):
        return True
    s = str(source)
    if s.isdigit():
        return True
    return s.lower().startswith(_LIVE_SCHEMES)


class ReconnectingCapture:
    def __init__(
        self,
        source,
        capture_size: Optional[Tuple[int, int]] = None,
        *,
        is_live: Optional[bool] = None,
        backoff_initial: float = 1.0,
        backoff_factor: float = 2.0,
        backoff_max: float = 30.0,
        max_retries: Optional[int] = None,
        stale_after: Optional[float] = 10.0,
        detect_frozen: bool = True,
        cap_factory: Callable = cv2.VideoCapture,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = print,
    ):
        """
        source          camera index, file path, or stream URL
        capture_size    (w, h) to request from the device, or None for native
        is_live         override the file-vs-stream guess from the source
        backoff_initial seconds to wait before the first retry
        backoff_factor  multiplier applied to the wait after each failed retry
        backoff_max     ceiling on the wait between retries
        max_retries     give up (raise ConnectionError) after this many
                        consecutive failed attempts; None means never give up
        stale_after     seconds without a new frame before a live stream is
                        force-reconnected; None or 0 disables the watchdog
        detect_frozen   also treat a byte-identical repeated frame as "no
                        new frame" (a frozen decoder), not just a stalled read
        cap_factory     injectable for tests; defaults to cv2.VideoCapture
        sleep           injectable for tests; defaults to time.sleep
        monotonic       injectable for tests; defaults to time.monotonic
        log             where status lines go; defaults to print
        """
        # "0" -> 0, so a digit string still opens a device (cv2.VideoCapture
        # needs a real int for a camera index)
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self.source = source
        self.capture_size = capture_size
        self.is_live = _looks_live(source) if is_live is None else is_live
        self.backoff_initial = backoff_initial
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self.max_retries = max_retries
        self.stale_after = stale_after
        self.detect_frozen = detect_frozen
        self._cap_factory = cap_factory
        self._sleep = sleep
        self._monotonic = monotonic
        self._log = log

        # number of times a live stream came back after dropping (does not
        # count the very first open)
        self.reconnects = 0
        # number of times the staleness watchdog forced a reconnect
        self.stale_trips = 0
        # every backoff wait, in order -- handy for the run summary and tests
        self.waits = []

        self._opened_once = False
        self._delay = backoff_initial
        self._retries = 0

        # Shared with the watchdog thread: the live capture handle, the
        # monotonic time of the last genuinely new frame, and a subsample
        # of that frame for the frozen-decoder check.
        self._cap = None
        self._cap_lock = threading.Lock()
        self._last_fresh = None
        self._last_sig = None
        self._stale_tripped = False
        self._monitor = None
        self._monitor_stop = threading.Event()

    # -- backoff bookkeeping -------------------------------------------------

    def _reset_backoff(self):
        self._delay = self.backoff_initial
        self._retries = 0

    def _backoff(self, reason: str):
        self._log(f"[capture] {reason}; retrying in {self._delay:.1f}s "
                  f"(attempt {self._retries + 1})")
        self._sleep(self._delay)
        self.waits.append(self._delay)
        self._retries += 1
        self._delay = min(self._delay * self.backoff_factor, self.backoff_max)

    # -- staleness watchdog ------------------------------------------------

    def _watchdog_on(self) -> bool:
        return bool(self.is_live and self.stale_after and self.stale_after > 0)

    def _frame_signature(self, frame):
        # a cheap strided subsample; comparing raw pixels means real sensor
        # noise reads as "changed", so only a decoder genuinely repeating a
        # frame looks frozen
        try:
            return np.ascontiguousarray(frame[::32, ::32])
        except Exception:
            return None

    def _note_frame(self, frame):
        """Record a freshly read frame. If it is byte-identical to the
        previous one (a frozen decoder) leave the freshness clock running
        down, so the watchdog eventually trips."""
        if not self._watchdog_on():
            return                              # no watchdog -> no bookkeeping
        now = self._monotonic()
        if not self.detect_frozen:
            self._last_fresh = now
            return
        sig = self._frame_signature(frame)
        prev = self._last_sig
        if (sig is not None and prev is not None
                and sig.shape == prev.shape and np.array_equal(sig, prev)):
            return
        self._last_sig = sig
        self._last_fresh = now

    def _stale_elapsed(self, now=None):
        if self._last_fresh is None:
            return None
        return (now if now is not None else self._monotonic()) - self._last_fresh

    def _trip_stale(self):
        with self._cap_lock:
            if self._cap is None:
                return
            self.stale_trips += 1
            self._stale_tripped = True
            self._log(f"[capture] {self.source!r} delivered no new frame for "
                      f"~{self.stale_after:.0f}s; forcing reconnect")
            try:
                self._cap.release()          # unblocks a hung read()
            except Exception:
                pass

    def _monitor_loop(self):
        interval = min(max(self.stale_after / 2.0, 0.05), 1.0)
        while not self._monitor_stop.wait(interval):
            elapsed = self._stale_elapsed()
            if elapsed is not None and elapsed > self.stale_after:
                self._trip_stale()

    def _start_monitor(self):
        if not self._watchdog_on():
            return
        self._monitor_stop.clear()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="capture-watchdog", daemon=True)
        self._monitor.start()

    def _stop_monitor(self):
        self._monitor_stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=2.0)
            self._monitor = None

    def _current_cap(self):
        with self._cap_lock:
            return self._cap

    def _discard_cap(self):
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # -- opening ----------------------------------------------------------------

    def _open(self):
        """Return an opened capture, or None if it could not be opened."""
        cap = self._cap_factory(self.source)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        if self.capture_size is not None:
            w, h = self.capture_size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    # -- iteration ------------------------------------------------------------

    def frames(self):
        """Yield frames forever for a live source, or until EOF for a file.

        Raises ConnectionError if the source cannot be opened -- immediately
        for a file (a missing file will never appear), or after max_retries
        for a live source (None = keep trying forever).

        For a live source a watchdog thread force-reconnects the capture if
        no new frame arrives for stale_after seconds.
        """
        self._reset_backoff()
        self._start_monitor()
        try:
            while True:
                if self._current_cap() is None:
                    cap = self._open()
                    if cap is None:
                        # A file that will not open is a permanent error;
                        # only a live stream is worth waiting on.
                        if not self.is_live:
                            raise ConnectionError(
                                f"could not open {self.source!r}")
                        if (self.max_retries is not None
                                and self._retries >= self.max_retries):
                            raise ConnectionError(
                                f"could not open {self.source!r} after "
                                f"{self._retries} attempts")
                        self._backoff(f"cannot open {self.source!r}")
                        continue
                    with self._cap_lock:
                        self._cap = cap
                        self._stale_tripped = False
                    self._last_fresh = self._monotonic()
                    self._last_sig = None
                    if self._opened_once:
                        self.reconnects += 1
                        self._log(f"[capture] reconnected to {self.source!r} "
                                  f"(reconnect #{self.reconnects})")
                    self._opened_once = True
                    self._reset_backoff()

                cap = self._current_cap()
                if cap is None:
                    continue                  # watchdog released it mid-loop
                try:
                    ok, frame = cap.read()
                except Exception:
                    ok, frame = False, None

                if self._stale_tripped:
                    self._discard_cap()
                    self._backoff(f"stream {self.source!r} went stale")
                    continue

                if ok and frame is not None:
                    self._note_frame(frame)
                    yield frame
                    continue

                # read failed
                self._discard_cap()
                if not self.is_live:
                    return  # end of a file, not an error
                self._backoff(f"stream {self.source!r} dropped")
        finally:
            self._stop_monitor()
            self._discard_cap()
