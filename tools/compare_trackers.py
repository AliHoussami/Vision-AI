"""
compare_trackers.py
-------------------
Measures whether a tracker actually holds onto a person, instead of
guessing from watching the preview.

Live runs cannot be compared to each other - the scene differs every
time. So this records ONE clip, then replays it through each tracker
config, which makes the numbers directly comparable.

    python compare_trackers.py --record 20 --camera 1     # capture a clip
    python compare_trackers.py --people 1                 # score the trackers

While recording, walk the way real customers would: cross the frame,
pause, turn around, walk behind something, come back. ID switches show
up at exactly those moments.

Key metric is ids_per_person. One person walking through should produce
ONE id. Getting 6 means the tracker dropped the lock 5 times.
"""

import argparse
import sys
from pathlib import Path

# Allow running this file directly (python tools/xxx.py) by putting the
# project root on sys.path before importing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

from footfall import DEFAULT_TRACKER, output, resolve_model

CLIP = output("tracker_test.mp4")



def _load_ignore():
    """Exclusion regions from zones.json, so measurements match what
    run_webcam.py actually counts."""
    try:
        from footfall import load_zones
        from footfall.tracker import _point_in_polygon, Point
        _, _, _, ignore = load_zones()
        if not ignore:
            return None
        def inside(cx, cy):
            return any(_point_in_polygon(Point(cx, cy), poly) for poly in ignore)
        print(f"Applying {len(ignore)} exclusion region(s) from zones.json")
        return inside
    except Exception as e:
        print(f"(no exclusion regions: {e})")
        return None


def record(camera, seconds, size=(1280, 720)):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        cap.release()
        print(f"Could not open camera {camera}")
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False
    h, w = frame.shape[:2]

    # Measure the true frame rate rather than assuming one, so the clip
    # plays back at the speed it was captured.
    t0, n = time.time(), 0
    while n < 30:
        ok, _ = cap.read()
        if ok:
            n += 1
    fps = n / max(time.time() - t0, 1e-6)
    print(f"Recording {w}x{h} at ~{fps:.1f}fps for {seconds}s...")

    writer = cv2.VideoWriter(CLIP, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    win = "recording - Q to stop early"
    deadline = time.time() + seconds
    frames = 0
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
        left = deadline - time.time()
        disp = frame.copy()
        cv2.putText(disp, f"REC {left:.0f}s   frames {frames}", (14, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow(win, disp)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Wrote {CLIP}: {frames} frames")
    return frames > 0


def evaluate(model_path, tracker, imgsz, conf, device):
    """Replay the clip and summarise how well identities held together."""
    model = YOLO(resolve_model(model_path))
    ignored = _load_ignore()
    seen = defaultdict(int)        # track id -> frames it appeared in
    first, last = {}, {}
    concurrent = []
    frames = 0

    t0 = time.time()
    for r in model.track(source=CLIP, tracker=tracker, imgsz=imgsz, conf=conf,
                         classes=[0], persist=True, stream=True,
                         verbose=False, **({"device": device} if device else {})):
        frames += 1
        ids = []
        if r.boxes is not None and r.boxes.id is not None:
            raw = r.boxes.id.cpu().numpy().astype(int).tolist()
            if ignored is None:
                ids = raw
            else:
                bx = r.boxes.xyxy.cpu().numpy()
                for b, i in zip(bx, raw):
                    if not ignored((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0):
                        ids.append(i)
        concurrent.append(len(ids))
        for i in ids:
            seen[i] += 1
            first.setdefault(i, frames)
            last[i] = frames
    elapsed = time.time() - t0

    if not seen:
        return {"unique_ids": 0, "frames": frames, "fps": frames / max(elapsed, 1e-6)}

    detected_frames = sum(1 for c in concurrent if c > 0)
    lengths = sorted(seen.values())
    # A track whose id appears over a long span but in few of those frames
    # is flickering: it keeps being lost and re-found.
    coverage = [seen[i] / float(last[i] - first[i] + 1) for i in seen]
    return {
        "frames": frames,
        "fps": frames / max(elapsed, 1e-6),
        "unique_ids": len(seen),
        "shortlived_ids": sum(1 for v in lengths if v <= 5),
        "median_track_frames": lengths[len(lengths) // 2],
        "longest_track_frames": lengths[-1],
        "mean_coverage": sum(coverage) / len(coverage),
        "max_concurrent": max(concurrent) if concurrent else 0,
        "detection_rate": detected_frames / float(max(frames, 1)),
    }



def diagnose(model_path, tracker, imgsz, conf, device, out_video=None):
    """Explain WHERE each id started and ended.

    Six ids for one person is only a failure if the breaks happen in open
    frame. An id that ends at the frame edge and a new one that starts there
    is just the person walking out and back in - correct behaviour. So we
    separate the two rather than tuning against a number that mixes them.
    """
    model = YOLO(resolve_model(model_path))
    ignored = _load_ignore()
    tracks = defaultdict(list)     # id -> [(frame, cx, cy)]
    meta = defaultdict(list)       # id -> [(conf, box_h, box_w)]
    frames = 0
    W = H = 0
    writer = None

    for r in model.track(source=CLIP, tracker=tracker, imgsz=imgsz, conf=conf,
                         classes=[0], persist=True, stream=True,
                         verbose=False, **({"device": device} if device else {})):
        frames += 1
        img = r.orig_img
        H, W = img.shape[:2]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for b, i, c in zip(boxes, ids, confs):
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                if ignored is not None and ignored(cx, cy):
                    continue
                tracks[int(i)].append((frames, cx, cy))
                meta[int(i)].append((float(c), float(b[3] - b[1]), float(b[2] - b[0])))
                if out_video:
                    x1, y1, x2, y2 = b.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    cv2.putText(img, f"ID {int(i)}", (x1, max(0, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if out_video:
            if writer is None:
                writer = cv2.VideoWriter(out_video,
                                         cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
            cv2.putText(img, f"frame {frames}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(img)
    if writer:
        writer.release()

    margin_x, margin_y = W * 0.10, H * 0.10

    def at_edge(x, y):
        return x < margin_x or x > W - margin_x or y < margin_y or y > H - margin_y

    print("")
    print(f"Diagnosis: {tracker}   ({frames} frames, {W}x{H})")
    print(f"{'id':>4}{'frames':>7}{'conf':>6}{'boxh':>6}{'moved':>7}"
          f"{'first':>7}{'last':>6}  {'verdict':<12} where")
    print("-" * 82)
    info = {}
    for i in sorted(tracks, key=lambda k: tracks[k][0][0]):
        pts = tracks[i]
        f0, l0 = pts[0][0], pts[-1][0]
        gaps = [pts[k + 1][0] - pts[k][0] for k in range(len(pts) - 1)]
        maxgap = max(gaps) if gaps else 0
        s, e = pts[0], pts[-1]
        se, ee = at_edge(s[1], s[2]), at_edge(e[1], e[2])
        info[i] = dict(first=f0, last=l0, start=(s[1], s[2]), end=(e[1], e[2]),
                       start_edge=se, end_edge=ee)
        where = ("edge" if se else "OPEN") + "->" + ("edge" if ee else "OPEN")
        m = meta[i]
        mean_conf = sum(x[0] for x in m) / len(m)
        mean_h = sum(x[1] for x in m) / len(m)
        # total path length: furniture does not walk anywhere
        moved = sum((((pts[k + 1][1] - pts[k][1]) ** 2 +
                      (pts[k + 1][2] - pts[k][2]) ** 2) ** 0.5)
                    for k in range(len(pts) - 1))
        verdict = "STATIC?" if moved < W * 0.15 and len(pts) > 60 else "moving"
        print(f"{i:>4}{len(pts):>7}{mean_conf:>6.2f}{mean_h:>6.0f}{moved:>7.0f}"
              f"{f0:>7}{l0:>6}  {verdict:<12} {where}")

    # A break in open frame followed by a new id nearby = a real lost lock.
    print("")
    print("Likely LOST LOCKS (broke mid-frame, not at an edge):")
    found = 0
    for a in info:
        if info[a]["end_edge"]:
            continue
        ax, ay = info[a]["end"]
        for b in info:
            if b == a or info[b]["first"] <= info[a]["last"]:
                continue
            if info[b]["first"] - info[a]["last"] > 120:
                continue
            bx, by = info[b]["start"]
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist < W * 0.25:
                gap = info[b]["first"] - info[a]["last"]
                print(f"  id {a} -> id {b}: {gap} frame gap, "
                      f"{dist:.0f}px apart, at frame {info[a]['last']}")
                found += 1
                break
    if not found:
        print("  none - every break happened at a frame edge (person left/re-entered).")
    return info



def sweep(model_path, tracker, imgsz, device, confs, people):
    """Sweep the detection threshold.

    conf filters detections BEFORE the tracker sees them. ByteTrack is
    built to reuse LOW-confidence detections to continue existing tracks
    (track_low_thresh) while requiring high confidence to start new ones
    (new_track_thresh). A high conf throws that second chance away, which
    is what loses a person in a hard pose -- seated, back turned, occluded.
    """
    print("")
    print(f"conf sweep on {tracker}, model {model_path} @ {imgsz}")
    print(f"{'conf':>6}{'det_rate':>10}{'ids':>6}{'/person':>9}{'median':>8}{'short':>7}")
    print("-" * 46)
    for c in confs:
        r = evaluate(model_path, tracker, imgsz, c, device)
        if not r.get("unique_ids"):
            print(f"{c:>6.2f}{'no detections':>20}")
            continue
        print(f"{c:>6.2f}{r['detection_rate']:>10.2f}{r['unique_ids']:>6}"
              f"{r['unique_ids'] / max(people, 1):>9.1f}"
              f"{r['median_track_frames']:>8}{r['shortlived_ids']:>7}")
    print("")
    print("det_rate: fraction of frames the person was detected at all.")
    print("          A drop here IS the sitting/back-turned failure.")


def main():
    ap = argparse.ArgumentParser(description="Compare tracker configs on one clip")
    ap.add_argument("--record", type=float, default=None, help="record N seconds first")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--people", type=int, default=1,
                    help="how many distinct people appear in the clip")
    ap.add_argument("--model", default=None, help="default: s on GPU, n on CPU")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    ap.add_argument("--trackers",
                    default="bytetrack.yaml,botsort.yaml," + str(DEFAULT_TRACKER))
    ap.add_argument("--diagnose", default=None,
                    help="explain where ids break for one tracker, e.g. tracker_people.yaml")
    ap.add_argument("--sweep", default=None,
                    help="comma-separated conf values to test, e.g. 0.10,0.20,0.35")
    ap.add_argument("--out-video", default=None,
                    help="with --diagnose, write an id-labelled video to watch")
    args = ap.parse_args()

    import torch
    on_gpu = torch.cuda.is_available()
    if args.model is None:
        args.model = "yolov8s.pt" if on_gpu else "yolov8n.pt"
    if args.imgsz is None:
        args.imgsz = 960 if on_gpu else 640
    print("Device: " + ("CUDA GPU" if on_gpu else "CPU"))
    print(f"Model {args.model} @ imgsz {args.imgsz}, conf {args.conf}")

    if args.record:
        if not record(args.camera, args.record):
            return
    if not os.path.exists(CLIP):
        print(f"No {CLIP}. Record one first:  --record 20 --camera {args.camera}")
        return

    if args.sweep:
        vals = [float(x) for x in args.sweep.split(',') if x.strip()]
        tk = args.diagnose or str(DEFAULT_TRACKER)
        sweep(args.model, tk, args.imgsz, args.device, vals, args.people)
        return

    if args.diagnose:
        diagnose(args.model, args.diagnose, args.imgsz, args.conf,
                 args.device, args.out_video)
        return

    results = {}
    for tk in [x.strip() for x in args.trackers.split(",") if x.strip()]:
        print(f"\nEvaluating {tk} ...")
        try:
            results[tk] = evaluate(args.model, tk, args.imgsz, args.conf, args.device)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")

    if not results:
        return

    print("\n" + "=" * 78)
    print(f"{'tracker':<24}{'ids':>5}{'/person':>9}{'short':>7}{'median':>8}{'cover':>7}{'fps':>7}")
    print("-" * 78)
    for tk, r in results.items():
        if not r.get("unique_ids"):
            print(f"{tk:<24}{'no detections':>30}")
            continue
        print(f"{tk:<24}{r['unique_ids']:>5}"
              f"{r['unique_ids'] / max(args.people, 1):>9.1f}"
              f"{r['shortlived_ids']:>7}"
              f"{r['median_track_frames']:>8}"
              f"{r['mean_coverage']:>7.2f}"
              f"{r['fps']:>7.1f}")
    print("=" * 78)
    print("ids/person : 1.0 is perfect. Higher means the lock was dropped and reissued.")
    print("short      : ids lasting <=5 frames - usually false positives or flicker.")
    print("median     : frames a typical id survived. Bigger is better.")
    print("cover      : how solidly an id was present across its own lifespan (1.0 best).")


if __name__ == "__main__":
    main()
