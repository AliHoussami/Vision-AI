# Vision AI — Project Overview

## What we are building

Vision AI is software that connects to a client's existing CCTV / camera
system and turns the video into a stream of anonymous events about how people
move through a space. Those events become operational insights: where people
wait, how long they stay, how long a process takes, and how that changes by
hour, day, and location.

The client needs no new hardware. If a site already has cameras, Vision AI
reads from them (RTSP / ONVIF for IP and CCTV systems, plus USB and file
sources for testing). Video is processed into counts and timings; identities
are not stored.

## How it works

1. **Connect** to one or more cameras at a site.
2. **Detect and track** people in each frame, assigning a short-lived tracking
   ID that resets every run.
3. **Apply geometry** drawn onto a real frame from that camera — an entrance
   line to count footfall in and out, and zones that represent meaningful
   areas (a queue, a table, a counter, an aisle).
4. **Emit events** — line crossings and zone entry/exit with timestamps and
   dwell times — to a log that maps directly onto a database table.
5. **Aggregate** those events into metrics and dashboards for the client.

## Privacy stance

- No facial recognition and no biometric matching.
- No identity storage. Tracking IDs are anonymous and reset each run.
- The goal is to measure flows and timings, not to identify individuals.
- Target deployment is at the edge, so that only aggregate counts and timings
  leave the site, not video.

## What the insights look like, by sector

**Restaurant**
- Time from seating to order taken, order to food arrival, food to bill,
  bill to table cleared.
- Table turnover and average sitting time by section and time of day.
- Queue length and wait time at the door or till.

**Supermarket / retail**
- Wait time and queue length per checkout lane.
- Footfall by entrance and by aisle, and dwell time per aisle.
- Peak-hour staffing signals: when queues build before tills are opened.

**Clinic / service counter**
- Time spent in the waiting room before being called.
- Queue length at reception and at each stage of a visit.
- Throughput per hour and where the bottleneck sits.

The common pattern across all of these: define the space as a line plus a set
of zones, then read arrival, wait, and service times off the event stream.

## Why it matters

Operators mostly run on intuition about wait times and busy periods. Vision AI
replaces that with measured numbers from cameras that are already installed,
so decisions about staffing, layout, and process changes can be made from
evidence and then checked against the same metric afterwards.

## Where this repository sits

This repo is the first building block: the detection, tracking, line-crossing,
and zone-dwell pipeline, plus the tooling to point it at a real camera and to
tune it. See `README.md` for how to install and run it. The roadmap below is
the path from that prototype to a product we can sell.

## Roadmap to a market-grade product

"Market-grade" means: a customer plugs in their cameras, the numbers are
accurate enough to act on, the system stays up without us babysitting it, no
video leaves the site, and there is a dashboard and a support process behind
it. This section is an honest account of the distance between the current code
and that bar.

### Where the prototype stands today

| Area | What exists now |
|---|---|
| Ingestion | Opens one source (file / USB / RTSP) via OpenCV. A single dropped frame ends the run — no reconnect. |
| Detection & tracking | YOLOv8n/s + ByteTrack/BoT-SORT, one frame at a time. Tracker config A/B-tested on one clip (`tracker_people.yaml`). |
| Geometry | One entrance line, one polygon zone, ignore regions. Drawn with a local OpenCV GUI (`define_zones.py`), pixel coords tied to one camera at one resolution. |
| Metrics | In/out counts, per-track dwell time, peak-minute bucket held in memory. |
| Output | `events.csv`, reopened in `w` mode every run (previous run is overwritten). Timestamps are naive local time. |
| Runtime | One Python process, one machine, one camera, live `cv2.imshow` preview. CLI flags only — no config file, no service. |
| Ops | None. No packaging, no container, no logging framework, no health checks, no tests, no CI. |
| Security | Camera credentials passed as CLI args / embedded in RTSP URLs. |

### Known technical debt to clear early

- **Dwell time fragments on ID switches.** A lost lock inside a zone closes one
  dwell session and opens another, so average dwell under-reports. The fix is
  to count *zone occupancy* (how many bodies are inside now) and derive wait
  time from occupancy + throughput, rather than trusting a single identity to
  survive the whole wait.
- **The entrance line is infinite, not a segment.** Anyone crossing its
  mathematical extension is counted. Needs real segment-intersection.
- **The zone tests the box centroid (mid-torso), not the feet.** In a
  perspective view this misplaces people by a metre or more. Needs a
  bottom-of-box anchor and, better, a ground-plane homography per camera.
- **`events.csv` is overwritten each run and timestamps are timezone-naive.**
  Not a persistence layer. Being replaced by a SQLite `EventSink` (Phase 1)
  with a per-run `run_id`, UTC + site-zone timestamps, and idempotent
  writes; a Postgres + TimescaleDB backend follows in Phase 3.
- **Single-threaded loop** does decode + inference + drawing + encoding in
  series. Decode, inference, and analytics need to be separate stages.
- **No stream resilience.** Cameras drop, reboot, and change IP; the pipeline
  must reconnect with backoff and report the outage, not exit.

### Phase 1 — A reliable single-camera service

Goal: the current pipeline, but as a service that runs for weeks unattended
and writes to a real database.

- Resilient RTSP ingestion: reconnect with backoff, watchdog on frame
  staleness, hardware-accelerated decode where available.
- Split the pipeline into stages (capture → inference → tracking → event
  logic) with bounded queues, so a slow stage drops frames instead of
  stalling.
- Replace CSV with an event store behind a small `EventSink` interface:
  append-only, timezone-aware (UTC + site zone), idempotent writes keyed on
  `(run_id, event, track_id, monotonic index)`, per-run `run_id`, and a
  one-command reset for local testing. **Implementation now: SQLite** — a
  single file, so a clean slate is still just deleting it, but with a real
  schema and queryable history. Postgres + TimescaleDB is a later, drop-in
  second implementation of the same interface (see Phase 3), added when
  there are real sites and a central store, retention policies, and a
  store-and-forward disk buffer start to matter.
- Configuration as a file instead of CLI flags: one YAML per site — name,
  timezone, a shared `defaults` block, and a list of cameras (source,
  resolution, zones file, per-camera overrides). `footfall/config.py` +
  `tools/run_site.py`.
- A small `localhost` control API on a running tracker (`footfall/control.py`,
  `control_port` in the config): `GET /status` and `/config`, `PATCH
  /config` to retune detection thresholds live, `PUT /geometry` to redraw
  the line / zone / ignore regions — all without restarting. Model, imgsz
  and source are rejected as restart-only. Localhost bind is the only
  boundary for now; real auth is Phase 5. This is the seed of Phase 3
  fleet management and Phase 4 onboarding.
- Secrets handling (`footfall/secrets.py`): a camera `source` carries a
  `${SECRET:name}` / `${ENV:name}` placeholder, resolved at config load
  from an env var, the OS keyring, or a gitignored `config/secrets.yaml`.
  `redact()` masks credentials everywhere a source is printed, stored in
  the event DB, or shown by `--print`; only `cv2.VideoCapture` ever sees
  the real URL. `get_rtsp_url.py` prompts for the password instead of
  taking it on the command line.
- Packaging: `Dockerfile` + `docker-compose.yml` (`restart: unless-stopped`,
  config mounted read-only, models/output as named volumes) and a systemd
  unit in `deploy/` with a crash-loop backoff. Library modules log through
  `logging.getLogger("footfall.*")`; `footfall/logsetup.py` `configure()`
  emits line-delimited JSON (ts, level, logger, msg, extras, exception) by
  default, `FOOTFALL_LOG_FORMAT=text` for a console. The `requirements.txt`
  numpy/opencv pins are fixed so a clean install resolves.
- Test suite (124 tests): `test_geometry.py` covers `_side_of_line`,
  `_point_in_polygon`, the line-crossing state machine, zone dwell,
  `_fit_geometry` rescaling and the box sanity filter (a `model=` hook
  skips the YOLO load); `test_regression.py` drives `FootfallTracker.run()`
  end to end with a scripted fake detector over a generated clip and
  asserts exact counts and the event stream. `.github/workflows/ci.yml`
  runs `pytest` on every push and PR. A real-footage accuracy benchmark is
  Phase 2.

### Phase 2 — Accuracy we can put in a contract

Goal: published accuracy targets (e.g. footfall count within ±5%, wait-time
estimate within ±15%) and a repeatable way to prove we hit them.

- Occupancy-based queue metrics (see debt list) so wait time survives ID
  switches.
- Track stitching / re-identification with a purpose-built person-ReID model
  — the current `model: auto` ReID path is inert, as `tracker_people.yaml`
  documents.
- Line-segment crossing; per-camera ground-plane homography so zones and
  distances are defined on the floor, not in image pixels.
- Occlusion handling for crowded scenes (supermarket aisles, packed waiting
  rooms): tuned detector input size, test-time strategies, possibly a
  crowd-oriented detector.
- An accuracy benchmark harness that runs against **labelled real footage**
  from each vertical and reports MOTA / IDF1 / count error — not the current
  `ids/person` proxy, which only compares trackers to each other.
- Evaluate current vs newer detectors (YOLO11, RT-DETR) on that labelled set;
  fine-tune on domain footage if off-the-shelf accuracy falls short.
- Per-vertical metric definitions as configuration: named zones and the
  derived measures for restaurant (seat→order→food→bill), supermarket
  (per-lane wait, aisle dwell), clinic (waiting-room time, stage queues).

### Phase 3 — Multi-camera sites and edge deployment

Goal: a real site has 4–40 cameras, runs on a box in the back office, and
sends only aggregates to the cloud.

- Edge runtime: mini PC / Jetson image, GPU or accelerator inference, one
  process per camera or a batched multi-stream worker.
- Aggregates-only uplink: counts, occupancy, and timings leave the site;
  frames and tracks stay local. Define the exact payload and retention.
- Promote the event store from SQLite to **Postgres + TimescaleDB** as a
  second implementation of the Phase 1 `EventSink` interface: a central
  multi-site store, TimescaleDB hypertables and continuous aggregates for
  the dashboards, a documented retention policy (auto-drop raw events past
  N days), and a store-and-forward disk buffer on the edge box that flushes
  in order once the uplink returns. SQLite stays the default for local dev
  and single-box installs.
- Multi-camera merge: hand off a person between overlapping views and
  de-duplicate across shared zones so a queue seen by two cameras is counted
  once.
- Fleet management: remote config push, over-the-air updates, per-camera
  health and calibration status, alerting when a camera goes dark or a
  pipeline stalls.
- Observability: metrics (Prometheus-style), dashboards for our ops team,
  paging on site-down.

### Phase 4 — The product surface

Goal: a customer can self-serve most of onboarding and lives in the dashboard,
not in our inbox.

- Browser-based camera onboarding: register a camera, pull a snapshot, draw
  lines and zones on a web canvas (replacing the local OpenCV tool), run a
  calibration wizard for the ground plane.
- Multi-tenant backend: tenant isolation, authentication, role-based access,
  per-client data separation.
- Customer dashboards: live occupancy, historical trends by hour / day /
  location, per-zone and per-vertical views, CSV / API export.
- Alerting for customers: "queue at lane 3 over 5 people for 10 minutes",
  delivered by email / SMS / webhook / Slack.
- A metrics API so customers can pull data into their own BI tools.

### Phase 5 — Compliance and commercial readiness

Goal: legal, security, and support are ready for a paying customer, including
an enterprise one.

- Privacy: a Data Protection Impact Assessment, documented lawful basis, a
  data-minimisation statement (we already process no biometrics and store no
  identities — this needs to be written down and verifiable), signage
  templates for customer sites, and a retention policy.
- Security: TLS everywhere, encryption at rest, secrets management, audit
  logging, dependency scanning, and a third-party penetration test. Plan a
  path to SOC 2 if we target enterprise.
- Data residency options for regions that require them.
- Reliability commitments: a defined uptime target for the cloud side and a
  documented degradation mode for the edge (it keeps counting and buffering
  when the uplink is down).
- Commercial: pricing and packaging, contractual accuracy SLA tied to the
  Phase 2 benchmark, runbooks, an on-call rotation, and a support process.
- Go-to-market proof: paid pilots in each target vertical, each with an
  on-site ground-truth accuracy check against the system's numbers before it
  converts to a rolling contract.

### Definition of done — the market-grade checklist

- [ ] Runs unattended for 30+ days on a real site with automatic recovery from
      camera and network outages.
- [ ] Published accuracy targets, met on labelled footage from every vertical
      we sell into, re-checked every release.
- [ ] No video or personal data leaves the customer site; only aggregates.
- [ ] Customer can onboard a camera and read their metrics without our help.
- [ ] Multi-tenant, authenticated, access-controlled.
- [ ] DPIA, retention policy, and a passed penetration test on file.
- [ ] Versioned releases, CI, regression tests, and an on-call support
      process.
