# Footfall & Queue Tracker

Counts people crossing an entrance line and measures dwell time in a zone
(e.g. a checkout queue) from any camera OpenCV can read — an IP/CCTV camera
over RTSP, a USB webcam, a phone used as a webcam, or a video file. No new
hardware needed if a site already has cameras.

No facial recognition and no identity storage: only anonymous detection plus
tracking IDs that reset each run.

## Layout

```
footfall/            importable package - the pipeline itself
  tracker.py         detection, tracking, line crossing, zone dwell, CSV log
  zones.py           geometry editor and zones.json loader
  __init__.py        path resolution, so tools work from any directory
tools/               command-line entry points
  run_webcam.py      run against a local / USB / phone camera
  define_zones.py    draw line, zone, and ignore regions onto a real frame
  run_demo.py        synthetic smoke test, no camera needed
  compare_trackers.py  record a clip and score tracker configs on it
  get_rtsp_url.py    ONVIF stream URL discovery
config/
  tracker_people.yaml  tracker settings, annotated with what was measured
  zones.example.json   documents the geometry file format
models/              downloaded YOLO weights        (gitignored)
output/              recordings, event logs, clips  (gitignored)
zones.json           your camera's geometry         (gitignored)
```

Both `models/` and `output/` are created on first use. Paths resolve against
the project root, not the working directory, so the tools behave identically
wherever you invoke them from.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch. **If you have an NVIDIA
GPU, install the CUDA build instead** — it is the difference between a few
frames per second and real time, and frame rate is the dominant factor in
tracking quality:

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check it took:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"
```

Model weights download automatically on first run.

## Quick start

```bash
# 1. Confirm the pipeline runs end to end (no camera needed)
python tools/run_demo.py

# 2. Find your camera. OpenCV exposes only numeric indices, so this
#    shows each one live with its index labelled on the frame.
python tools/run_webcam.py --identify

# 3. Draw the geometry onto a real frame from that camera
python tools/define_zones.py --source 0

# 4. Run it
python tools/run_webcam.py --camera 0
```

## Drawing the geometry

`tools/define_zones.py` writes `zones.json`. Three things can be drawn:

- **Line** (`L`, 2 clicks) — the entrance. A tracked person crossing it counts
  as in or out. The click order sets which direction is "in"; an arrow labelled
  IN shows you which, and reversing the two clicks flips it.
- **Zone** (`Z`, 3+ clicks) — the queue area. Time inside is logged as dwell.
- **Ignore regions** (`X`, 3+ clicks, `N` to start another) — areas where
  detections are discarded. Essential for mirrors, TV screens, posters, and
  windows onto a street: a reflection is a genuine image of a person, so no
  confidence threshold removes it. Only geometry can.

Keys: `L` line, `Z` zone, `X` ignore, `N` next region, `U` undo, `R` reset,
`S` save, `Q` quit.

Coordinates are pixels tied to one camera at one resolution. `zones.json`
records the frame size it was drawn at, and the tracker rescales the geometry
if the live frame differs. Redraw whenever the camera moves or is re-aimed.

## Tuning detection and tracking

Defaults follow the hardware: `yolov8s` at 960 on a GPU, `yolov8n` at 640 on
a CPU.

```bash
python tools/run_webcam.py --camera 0 --model yolo11m.pt --imgsz 1280 --conf 0.50
```

Do not guess at these — measure them. `compare_trackers.py` records one clip
and replays it through each config, so the comparison is like for like:

```bash
python tools/compare_trackers.py --record 30 --camera 0 --people 2   # capture + score
python tools/compare_trackers.py --diagnose config/tracker_people.yaml  # where IDs break
python tools/compare_trackers.py --sweep "0.35,0.50,0.60"            # threshold sweep
```

`ids/person` is the headline: 1.0 means the identity was never lost.

Findings from that harness are recorded in `config/tracker_people.yaml`, including
two settings that turned out **not** to work — ultralytics' built-in ReID with
`model: auto` does nothing, and `track_buffer` had no measurable effect from 3
to 900. Re-measure on your own footage before trusting either.

## Pointing it at a store camera

Find the RTSP URL via ONVIF:

```bash
python tools/get_rtsp_url.py --ip 192.168.1.50 --user admin --password ****
```

Or use the vendor pattern:

- Hikvision: `rtsp://user:pass@IP:554/Streaming/Channels/101`
- Dahua: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`

Then:

```python
from footfall import FootfallTracker, Point, output

tracker = FootfallTracker(
    source="rtsp://admin:pass@192.168.1.50:554/Streaming/Channels/101",
    line=(Point(100, 400), Point(900, 400)),
    zone=[Point(300, 300), Point(700, 300), Point(700, 550), Point(300, 550)],
    events_csv=output("events.csv"),
)
print(tracker.run())
```

Camera placement affects accuracy more than model choice. An overhead or
high-angle view counts far more reliably than a low wall mount: less
occlusion, and the box centroid lands nearer the person's floor position.

## Output

`output/events.csv` — one row per event, mapping directly onto a
`events(timestamp, event_type, track_id, value)` SQL table:

```
timestamp,event,track_id,value
2026-08-28T00:42:57,line_in,5,
2026-08-28T00:43:08,zone_exit,12,8.6s
```

## Known limitations

- **Dwell times fragment on ID switches.** A lost lock inside the zone ends one
  dwell session and starts another, so `avg_dwell_seconds` under-reports.
  Counting zone occupancy rather than per-identity sessions is the fix.
- **The entrance line is infinite, not a segment.** Someone crossing the line's
  extension, outside the drawn endpoints, still counts. Place the line so its
  extension runs into a wall.
- **The zone tests the box centroid**, roughly mid-torso, not the feet. Draw the
  zone where torsos appear, or mount the camera higher so the two converge.
- **Recordings are written at a fixed 20 fps** regardless of source rate, so
  playback speed is wrong. Counts are unaffected.

## Not built yet

- Edge deployment (mini PC / Jetson) so only aggregate counts leave the site
- SQL Server sink instead of CSV
- Multi-camera merge with de-duplication across shared zones
