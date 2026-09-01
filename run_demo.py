"""
run_demo.py
-----------
Self-contained smoke test: generates a short synthetic video (moving
person-shaped silhouettes walking across an entrance line and lingering
in a queue zone), then runs FootfallTracker on it end-to-end, proving
the full pipeline (detection -> tracking -> line-crossing -> zone dwell
-> CSV logging -> annotated video output) runs without errors.

Swap `source=` for a real RTSP URL or a real video file to use on an
actual store camera — nothing else in FootfallTracker needs to change.

Run:
    python3 run_demo.py
"""

import cv2
import numpy as np

from footfall_tracker import FootfallTracker, Point

WIDTH, HEIGHT = 960, 540
FPS = 20
DURATION_SEC = 8
SYNTH_VIDEO = "synthetic_input.mp4"


def make_synthetic_video():
    """Draws simple walking silhouettes (blobs with legs) so a person
    detector has a fighting chance of picking something up, and to
    validate the video I/O + geometry logic regardless of detection."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(SYNTH_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

    n_frames = FPS * DURATION_SEC
    # two "walkers" crossing left->right at different heights/speeds,
    # one of which pauses in the middle (simulating someone queueing)
    for i in range(n_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 235, dtype=np.uint8)  # light grey floor

        # entrance line marker area (visual only, tracker draws its own line)
        cv2.putText(frame, "synthetic demo footage", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        # walker 1: steady walk across
        x1 = int(50 + i * (WIDTH - 100) / n_frames)
        y1 = 200
        _draw_person(frame, x1, y1)

        # walker 2: walks in, pauses in the middle third, then continues
        t = i / n_frames
        if t < 0.35:
            x2 = int(50 + (t / 0.35) * (WIDTH * 0.45 - 50))
        elif t < 0.7:
            x2 = int(WIDTH * 0.45)  # paused = dwelling in the zone
        else:
            x2 = int(WIDTH * 0.45 + ((t - 0.7) / 0.3) * (WIDTH - 100 - WIDTH * 0.45))
        y2 = 380
        _draw_person(frame, x2, y2)

        writer.write(frame)

    writer.release()


def _draw_person(frame, x, y):
    # crude silhouette: head + body rectangle, dark on light background
    cv2.circle(frame, (x, y - 30), 12, (40, 40, 40), -1)
    cv2.rectangle(frame, (x - 15, y - 18), (x + 15, y + 40), (40, 40, 40), -1)


def main():
    print("1) Generating synthetic test video...")
    make_synthetic_video()
    print(f"   wrote {SYNTH_VIDEO}")

    print("2) Running FootfallTracker end-to-end on it...")
    tracker = FootfallTracker(
        source=SYNTH_VIDEO,
        # vertical entrance line roughly in the middle of the frame
        line=(Point(WIDTH * 0.5, 0), Point(WIDTH * 0.5, HEIGHT)),
        # a "queue zone" box around where walker 2 pauses
        zone=[
            Point(WIDTH * 0.35, HEIGHT * 0.55),
            Point(WIDTH * 0.55, HEIGHT * 0.55),
            Point(WIDTH * 0.55, HEIGHT * 0.85),
            Point(WIDTH * 0.35, HEIGHT * 0.85),
        ],
        output_video="annotated_out.mp4",
        events_csv="events.csv",
        conf=0.15,  # low threshold since synthetic silhouettes are crude
    )
    summary = tracker.run()

    print("\n3) Run summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")

    print("\nOutputs:")
    print("  - annotated_out.mp4  (video with boxes/line/zone overlay)")
    print("  - events.csv         (raw in/out and zone enter/exit events)")
    print("\nNote: detection counts on this synthetic footage are just a")
    print("pipeline smoke test — real store CCTV footage of actual people")
    print("will give the YOLO model far more to work with.")


if __name__ == "__main__":
    main()