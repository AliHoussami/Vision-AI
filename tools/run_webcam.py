"""
run_webcam.py
-------------
Runs FootfallTracker against a local webcam instead of a video file.

The only change FootfallTracker itself needs is `source=<camera index>`
(0 = default webcam). Everything else in here exists because a live
camera differs from a file in three ways:

  1. It never ends, so the line/zone geometry has to be sized to the
     camera's ACTUAL resolution, not a hardcoded 960x540.
  2. It never ends, so you stop it with Ctrl+C or --seconds.
  3. Its real frame rate is rarely the 20.0 fps the recorder assumes.

Run:
    python run_webcam.py                 # camera 0, until Ctrl+C
    python run_webcam.py --camera 1      # second camera
    python run_webcam.py --seconds 30    # auto-stop after 30s
    python run_webcam.py --list          # show which indices work
"""

import argparse
import sys
from pathlib import Path

# Allow running this file directly (python tools/xxx.py) by putting the
# project root on sys.path before importing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import cv2

from footfall import FootfallTracker, Point, load_zones, max_capture_size, output


def probe(index: int):
    """Open the camera just long enough to learn its resolution + fps."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    return w, h, (fps if 1.0 < fps < 120.0 else 30.0)


def list_cameras(max_index: int = 5):
    print("Scanning camera indices 0..%d" % max_index)
    found = []
    for i in range(max_index + 1):
        info = probe(i)
        if info:
            w, h, fps = info
            print(f"  [{i}] OK  {w}x{h} @ ~{fps:.0f}fps")
            found.append(i)
        else:
            print(f"  [{i}] --  no camera")
    if not found:
        print("\nNo cameras found. On Windows check:")
        print("  Settings > Privacy & security > Camera > 'Let desktop apps access your camera'")
    return found



def identify_cameras(max_index: int = 3):
    """Show a live labelled preview of each camera so you can see which
    index is which. OpenCV exposes no device names, so eyes are the
    only reliable way to tell a phone-as-webcam from a built-in one."""
    shown = 0
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        shown += 1
        print(f"  showing index {i} - SPACE for next, Q to stop")
        win = f"camera index {i} - SPACE=next  Q=quit"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            h, w = frame.shape[:2]
            s = min(1.0, 960.0 / w, 540.0 / h)
            disp = cv2.resize(frame, None, fx=s, fy=s) if s != 1.0 else frame.copy()
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 44), (255, 255, 255), -1)
            cv2.putText(disp, f"INDEX {i}   {w}x{h}", (12, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.imshow(win, disp)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return
            if k == 32:
                break
        cap.release()
        cv2.destroyWindow(win)
    cv2.destroyAllWindows()
    if not shown:
        print("  no cameras opened - is the phone app connected?")


def main():
    ap = argparse.ArgumentParser(description="Run FootfallTracker on a webcam")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--seconds", type=float, default=None, help="auto-stop after N seconds")
    ap.add_argument("--conf", type=float, default=0.50,
                    help="detection confidence (0.50 measured best; 0.35 added junk tracks)")
    ap.add_argument("--model", default=None,
                    help="yolov8n/s/m/l.pt - bigger is more accurate, slower "
                         "(default: s on GPU, n on CPU)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="inference resolution; raise to detect smaller/distant people "
                         "(default: 960 on GPU, 640 on CPU)")
    ap.add_argument("--tracker", default=None,
                    help="tracker config (tracker_people.yaml, botsort.yaml, bytetrack.yaml)")
    ap.add_argument("--device", default=None, help="cuda device e.g. 0, or cpu")
    ap.add_argument("--min-height", type=int, default=0,
                    help="reject person boxes shorter than this many pixels")
    ap.add_argument("--max-aspect", type=float, default=None,
                    help="reject boxes wider than this ratio of their height")
    ap.add_argument("--no-record", action="store_true", help="skip writing the annotated mp4")
    ap.add_argument("--no-preview", action="store_true", help="run headless, no live window")
    ap.add_argument("--scale", type=float, default=1.0, help="preview window size, e.g. 0.75")
    ap.add_argument("--res", default="max",
                    help="capture mode: max (default), native, or WxH e.g. 1280x720")
    ap.add_argument("--list", action="store_true", help="list available cameras and exit")
    ap.add_argument("--identify", action="store_true",
                    help="show each camera live so you can tell which index is which")
    args = ap.parse_args()

    # A big model at high imgsz is the right default on a GPU and unusable
    # on a CPU -- at a few fps the tracker loses people between frames. So
    # the defaults follow the hardware unless the user overrides them.
    import torch
    on_gpu = torch.cuda.is_available()
    if args.model is None:
        args.model = "yolov8s.pt" if on_gpu else "yolov8n.pt"
    if args.imgsz is None:
        args.imgsz = 960 if on_gpu else 640
    print("Device: " + ("CUDA GPU" if on_gpu else "CPU (no CUDA torch installed)"))

    if args.list:
        list_cameras()
        return

    if args.identify:
        identify_cameras()
        return

    info = probe(args.camera)
    if info is None:
        print(f"Could not open camera index {args.camera}.")
        print("Try:  python run_webcam.py --list")
        return
    width, height, fps = info

    # Force the capture mode. Webcams default to a narrow cropped window,
    # so "max" genuinely widens the field of view, not just the pixel count.
    if args.res == "native":
        capture_size = None
    elif args.res == "max":
        capture_size = max_capture_size(args.camera)
    else:
        w_s, h_s = args.res.lower().split("x")
        capture_size = (int(w_s), int(h_s))

    if capture_size:
        width, height = capture_size
        print(f"Camera {args.camera}: forcing {width}x{height} (was {info[0]}x{info[1]}) @ ~{fps:.0f}fps")
    else:
        print(f"Camera {args.camera}: native {width}x{height} @ ~{fps:.0f}fps")

    # Prefer hand-drawn geometry from define_zones.py; fall back to a
    # generic split of the frame so the script still runs out of the box.
    line, zone, geometry_size, ignore_zones = load_zones()
    if line or zone:
        print(f"Using geometry from zones.json (drawn on {geometry_size[0]}x{geometry_size[1]})")
        if ignore_zones:
            print(f"  {len(ignore_zones)} exclusion region(s) active")
    else:
        print("No zones.json found - using default middle-split geometry.")
        print("Draw your own with:  python define_zones.py --source %d" % args.camera)
        line = (Point(width * 0.5, 0), Point(width * 0.5, height))
        zone = [
            Point(width * 0.10, height * 0.35),
            Point(width * 0.45, height * 0.35),
            Point(width * 0.45, height * 0.95),
            Point(width * 0.10, height * 0.95),
        ]

    max_frames = int(fps * args.seconds) if args.seconds else None

    tracker = FootfallTracker(
        source=args.camera,          # <-- the actual "link it to my webcam" bit
        line=line,
        zone=zone,
        conf=args.conf,
        output_video=None if args.no_record else output("webcam_annotated.mp4"),
        events_csv=output("webcam_events.csv"),
        max_frames=max_frames,
        preview=not args.no_preview,
        display_scale=args.scale,
        capture_size=capture_size,
        geometry_size=geometry_size,
        model_path=args.model,
        imgsz=args.imgsz,
        tracker=args.tracker,
        device=args.device,
        min_box_height=args.min_height,
        max_aspect=args.max_aspect,
        ignore_zones=ignore_zones,
    )

    print(f"Model {args.model} @ imgsz {args.imgsz} | tracker {args.tracker}")
    if not args.no_preview:
        print("Live window opens shortly - press Q (or ESC) in it to stop.")
    if max_frames:
        print(f"Recording ~{args.seconds:.0f}s ({max_frames} frames). Ctrl+C also works.")
    else:
        print("Running until you press Q in the window (or Ctrl+C here).")

    summary = tracker.run()

    print("\nSummary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    print("\nOutputs:")
    if not args.no_record:
        print("  - output/webcam_annotated.mp4")
    print("  - output/webcam_events.csv")


if __name__ == "__main__":
    main()
