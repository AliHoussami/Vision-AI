"""
define_zones.py
---------------
Click the entrance line and the queue zone directly onto a still frame
from your camera, and save them to zones.json. Beats guessing pixel
coordinates by hand.

    python define_zones.py --source 0              # webcam
    python define_zones.py --source 1
    python define_zones.py --source clip.mp4       # a saved video
    python define_zones.py --source "rtsp://..."   # a store camera
    python define_zones.py --review                # re-open the saved zones

Controls (in the window):
    LEFT CLICK   place a point
    L            switch to LINE mode      (needs exactly 2 points)
    Z            switch to ZONE mode      (3+ points, auto-closes)
    U            undo last point in the current mode
    R            reset the current mode
    S            save to zones.json
    Q / ESC      quit without saving
"""

import argparse
import json
import math
import os

import cv2

from . import ZONES_FILE

LINE_COLOR = (0, 255, 255)
ZONE_COLOR = (255, 128, 0)
IGNORE_COLOR = (0, 0, 235)
HINT_COLOR = (60, 60, 60)


class ZoneEditor:
    def __init__(self, frame):
        self.base = frame
        self.h, self.w = frame.shape[:2]
        self.line = []
        self.zone = []
        # Regions where detections are thrown away: mirrors, TV screens,
        # posters, windows onto the street. A reflection IS a real person,
        # so no confidence threshold can remove it -- only geometry can.
        self.ignores = []
        self.current_ignore = []
        self.mode = "line"
        # The window may show a shrunk copy of the frame. Clicks arrive in
        # WINDOW space, so we divide by this to get true IMAGE coordinates.
        self.view_scale = 1.0

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.view_scale != 1.0:
            x = int(round(x / self.view_scale))
            y = int(round(y / self.view_scale))
        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))
        if self.mode == "line":
            if len(self.line) < 2:
                self.line.append((x, y))
            else:
                self.line = [(x, y)]
        elif self.mode == "ignore":
            self.current_ignore.append((x, y))
        else:
            self.zone.append((x, y))

    def undo(self):
        if self.mode == "ignore":
            if self.current_ignore:
                self.current_ignore.pop()
            elif self.ignores:
                self.current_ignore = self.ignores.pop()
            return
        target = self.line if self.mode == "line" else self.zone
        if target:
            target.pop()

    def reset(self):
        if self.mode == "line":
            self.line = []
        elif self.mode == "ignore":
            self.current_ignore = []
            self.ignores = []
        else:
            self.zone = []

    def commit_ignore(self):
        """Finish the current exclusion polygon and start another."""
        if len(self.current_ignore) >= 3:
            self.ignores.append(self.current_ignore)
            self.current_ignore = []
            return True
        return False

    def render(self):
        img = self.base.copy()

        # dim banner so the hint text stays readable on any footage
        cv2.rectangle(img, (0, 0), (self.w, 78), (255, 255, 255), -1)
        cv2.addWeighted(img, 0.75, self.base, 0.25, 0, img)

        mode_txt = {"line": "MODE: LINE (2 pts)",
                    "zone": "MODE: ZONE (3+ pts)",
                    "ignore": "MODE: IGNORE (3+ pts, N=next region)"}[self.mode]
        mode_col = {"line": LINE_COLOR, "zone": ZONE_COLOR,
                    "ignore": IGNORE_COLOR}[self.mode]
        cv2.putText(img, mode_txt, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    mode_col, 2)
        cv2.putText(img, "L line  Z zone  X ignore  N next-region  U undo  R reset  S save  Q quit",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, HINT_COLOR, 1)
        cv2.putText(img, f"line {len(self.line)}/2   zone {len(self.zone)}   ignore regions {len(self.ignores)} (+{len(self.current_ignore)} pts)",
                    (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, HINT_COLOR, 1)

        for p in self.line:
            cv2.circle(img, p, 5, LINE_COLOR, -1)
        if len(self.line) == 2:
            (ax, ay), (bx, by) = self.line
            cv2.line(img, self.line[0], self.line[1], LINE_COLOR, 2)
            cv2.putText(img, "A", (ax + 8, ay - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, LINE_COLOR, 2)
            cv2.putText(img, "B", (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, LINE_COLOR, 2)
            # Which side counts as IN follows from the sign convention in
            # footfall_tracker._side_of_line: the positive side of A->B is
            # the direction (-dy, dx). Crossing INTO it increments count_in.
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length > 1:
                nx, ny = -dy / length, dx / length
                mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                tip = (int(mx + nx * 60), int(my + ny * 60))
                cv2.arrowedLine(img, (int(mx), int(my)), tip, LINE_COLOR, 2,
                                tipLength=0.35)
                cv2.putText(img, "IN", (tip[0] + 6, tip[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_COLOR, 2)

        for i, p in enumerate(self.zone):
            cv2.circle(img, p, 5, ZONE_COLOR, -1)
            if i:
                cv2.line(img, self.zone[i - 1], p, ZONE_COLOR, 2)
        if len(self.zone) >= 3:
            cv2.line(img, self.zone[-1], self.zone[0], ZONE_COLOR, 2)

        # exclusion regions, filled so they read as "dead area"
        import numpy as np
        for poly in self.ignores:
            arr = np.array(poly, dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [arr], IGNORE_COLOR)
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [arr], True, IGNORE_COLOR, 2)
            cv2.putText(img, "IGNORE", (poly[0][0] + 6, poly[0][1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, IGNORE_COLOR, 2)
        for i, pt in enumerate(self.current_ignore):
            cv2.circle(img, pt, 5, IGNORE_COLOR, -1)
            if i:
                cv2.line(img, self.current_ignore[i - 1], pt, IGNORE_COLOR, 2)

        return img

    def payload(self):
        return {
            "width": self.w,
            "height": self.h,
            "line": [list(p) for p in self.line] if len(self.line) == 2 else None,
            "zone": [list(p) for p in self.zone] if len(self.zone) >= 3 else None,
            "ignore": [[list(p) for p in poly] for poly in self.ignores] or None,
        }


def grab_frame(source, capture_size=None):
    """Pull one frame. Accepts a camera index, a file path, or an RTSP URL.

    capture_size must match what run_webcam.py will use, or the coordinates
    you click here will not line up with the coordinates it tests against.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None
    if capture_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
    # discard a few frames so webcams finish auto-exposure before we sample
    frame = None
    for _ in range(8):
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame


def load_zones(path=None):
    """Returns (line, zone, size, ignore_regions)."""
    from .tracker import Point
    path = str(path or ZONES_FILE)
    if not os.path.exists(path):
        return None, None, None, None
    with open(path) as f:
        data = json.load(f)
    line = None
    if data.get("line"):
        a, b = data["line"]
        line = (Point(a[0], a[1]), Point(b[0], b[1]))
    zone = None
    if data.get("zone"):
        zone = [Point(x, y) for x, y in data["zone"]]
    size = None
    if data.get("width") and data.get("height"):
        size = (int(data["width"]), int(data["height"]))
    ignore = None
    if data.get("ignore"):
        ignore = [[Point(x, y) for x, y in poly] for poly in data["ignore"]]
    return line, zone, size, ignore


def main():
    ap = argparse.ArgumentParser(description="Draw the entrance line and queue zone")
    ap.add_argument("--source", default="0", help="camera index, video path, or RTSP URL")
    ap.add_argument("--review", action="store_true", help="load zones.json and show it")
    ap.add_argument("--out", default=str(ZONES_FILE))
    ap.add_argument("--res", default="max",
                    help="capture mode: max (default), native, or WxH — must match run_webcam.py")
    args = ap.parse_args()

    capture_size = None
    if args.res == "max" and str(args.source).isdigit():
        from .tracker import max_capture_size
        capture_size = max_capture_size(int(args.source))
    elif args.res not in ("max", "native"):
        w_s, h_s = args.res.lower().split("x")
        capture_size = (int(w_s), int(h_s))

    frame = grab_frame(args.source, capture_size)
    if frame is None:
        print(f"Could not read a frame from source: {args.source}")
        return
    h, w = frame.shape[:2]
    print(f"Frame: {w}x{h}")

    editor = ZoneEditor(frame)

    if args.review and os.path.exists(args.out):
        with open(args.out) as f:
            data = json.load(f)
        if data.get("width") != w or data.get("height") != h:
            print(f"WARNING: zones.json was drawn on {data.get('width')}x{data.get('height')}, "
                  f"this frame is {w}x{h}. Coordinates will not line up.")
        editor.line = [tuple(p) for p in (data.get("line") or [])]
        editor.zone = [tuple(p) for p in (data.get("zone") or [])]
        editor.ignores = [[tuple(p) for p in poly] for poly in (data.get("ignore") or [])]
        print(f"Loaded {args.out}")

    # Fit the frame on screen ourselves rather than letting WINDOW_NORMAL
    # scale it -- otherwise clicks land in window space and the saved
    # coordinates do not match the frame the tracker actually tests.
    MAX_W, MAX_H = 1280, 720
    editor.view_scale = min(1.0, MAX_W / float(w), MAX_H / float(h))
    if editor.view_scale != 1.0:
        print(f"Window shown at {editor.view_scale:.2f}x; clicks mapped back to {w}x{h}.")

    win = "define zones - L line, Z zone, S save, Q quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, editor.on_mouse)

    while True:
        canvas = editor.render()
        if editor.view_scale != 1.0:
            canvas = cv2.resize(canvas, (int(w * editor.view_scale),
                                         int(h * editor.view_scale)))
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("Quit without saving.")
            break
        elif key == ord("l"):
            editor.mode = "line"
        elif key == ord("z"):
            editor.mode = "zone"
        elif key == ord("x"):
            editor.mode = "ignore"
        elif key == ord("n"):
            if editor.commit_ignore():
                print(f"  ignore region {len(editor.ignores)} saved")
            else:
                print("  need 3+ points before starting the next region")
        elif key == ord("u"):
            editor.undo()
        elif key == ord("r"):
            editor.reset()
        elif key == ord("s"):
            editor.commit_ignore()   # do not lose an unfinished region
            data = editor.payload()
            if data["line"] is None and data["zone"] is None and data["ignore"] is None:
                print("Nothing to save - draw a line, zone, or ignore region first.")
                continue
            with open(args.out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved {args.out}")
            print(json.dumps(data, indent=2))
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
