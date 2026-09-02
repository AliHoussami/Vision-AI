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

Only reconnection is handled here. A watchdog for a stream that stays open
but stops delivering new frames, and hardware-accelerated decode, are
separate steps.

    for frame in ReconnectingCapture("rtsp://cam/stream").frames():
        ...
"""

import time
from typing import Callable, Optional, Tuple

import cv2

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
        cap_factory: Callable = cv2.VideoCapture,
        sleep: Callable[[float], None] = time.sleep,
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
        cap_factory     injectable for tests; defaults to cv2.VideoCapture
        sleep           injectable for tests; defaults to time.sleep
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
        self._cap_factory = cap_factory
        self._sleep = sleep
        self._log = log

        # number of times a live stream came back after dropping (does not
        # count the very first open)
        self.reconnects = 0
        # every backoff wait, in order -- handy for the run summary and tests
        self.waits = []

        self._opened_once = False
        self._delay = backoff_initial
        self._retries = 0

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
        """
        self._reset_backoff()
        cap = None
        try:
            while True:
                if cap is None:
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
                    if self._opened_once:
                        self.reconnects += 1
                        self._log(f"[capture] reconnected to {self.source!r} "
                                  f"(reconnect #{self.reconnects})")
                    self._opened_once = True
                    self._reset_backoff()

                ok, frame = cap.read()
                if ok and frame is not None:
                    yield frame
                    continue

                # read failed
                cap.release()
                cap = None
                if not self.is_live:
                    return  # end of a file, not an error
                self._backoff(f"stream {self.source!r} dropped")
        finally:
            if cap is not None:
                cap.release()
