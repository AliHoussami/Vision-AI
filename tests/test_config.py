"""
Unit tests for footfall.config: loading a site YAML, merging per-camera
overrides onto defaults, validation, path resolution, and flattening a
camera into FootfallTracker kwargs.
"""

import json
import textwrap

import pytest

from footfall import ROOT
from footfall.config import ConfigError, load_config, resolved, tracker_kwargs


def _write(tmp_path, body: str):
    p = tmp_path / "site.yaml"
    p.write_text(textwrap.dedent(body))
    return p


_MINIMAL = """
    site:
      name: Test Site
      timezone: Europe/Paris
    cameras:
      - id: door
        source: 0
    storage:
      events_db: output/events.db
"""


# -- happy path ------------------------------------------------------


def test_loads_minimal_config(tmp_path):
    site = load_config(_write(tmp_path, _MINIMAL))

    assert site.site.name == "Test Site"
    assert site.site.timezone == "Europe/Paris"
    assert [c.id for c in site.cameras] == ["door"]
    assert site.cameras[0].source == 0
    assert site.events_db == str(ROOT / "output/events.db")


def test_camera_inherits_defaults(tmp_path):
    site = load_config(_write(tmp_path, """
        defaults:
          conf: 0.42
          imgsz: 1280
        cameras:
          - id: door
            source: 0
    """))
    s = site.cameras[0].settings
    assert s["conf"] == 0.42
    assert s["imgsz"] == 1280
    assert s["hw_accel"] == "auto"          # untouched default


def test_camera_override_wins_over_default(tmp_path):
    site = load_config(_write(tmp_path, """
        defaults:
          conf: 0.42
        cameras:
          - id: door
            source: 0
            conf: 0.60
    """))
    assert site.cameras[0].settings["conf"] == 0.60


def test_reconnect_block_merges_key_by_key(tmp_path):
    site = load_config(_write(tmp_path, """
        defaults:
          reconnect:
            max: 45
        cameras:
          - id: door
            source: 0
            reconnect:
              retries: 5
    """))
    rc = site.cameras[0].settings["reconnect"]
    assert rc["max"] == 45          # from defaults
    assert rc["retries"] == 5       # from the camera
    assert rc["initial"] == 1.0     # untouched base default


@pytest.mark.parametrize("raw,expected", [
    ("0", 0), (0, 0), ("1", 1), ("rtsp://cam/stream", "rtsp://cam/stream"),
    ("/videos/clip.mp4", "/videos/clip.mp4"),
])
def test_source_normalisation(tmp_path, raw, expected):
    site = load_config(_write(tmp_path, f"""
        cameras:
          - id: door
            source: {json.dumps(raw)}
    """))
    assert site.cameras[0].source == expected


def test_relative_paths_resolve_against_root_absolute_left_alone(tmp_path):
    site = load_config(_write(tmp_path, """
        cameras:
          - id: a
            source: 0
            zones: config/zones/a.json
          - id: b
            source: 1
            zones: /etc/footfall/b.json
    """))
    assert site.cameras[0].zones_path == str(ROOT / "config/zones/a.json")
    assert site.cameras[1].zones_path == "/etc/footfall/b.json"


def test_camera_lookup(tmp_path):
    site = load_config(_write(tmp_path, _MINIMAL))
    assert site.camera("door").id == "door"
    with pytest.raises(ConfigError, match="no camera with id 'missing'"):
        site.camera("missing")


# -- validation ---------------------------------------------------


@pytest.mark.parametrize("body,match", [
    ("cameras: []\n", "no 'cameras'"),
    ("site: {}\n", "no 'cameras'"),
    ("cameras:\n  - source: 0\n", "missing 'id'"),
    ("cameras:\n  - id: door\n", "missing 'source'"),
    ("cameras:\n  - id: door\n    source: 0\n  - id: door\n    source: 1\n",
     "duplicate camera id 'door'"),
    ("wat: 1\ncameras:\n  - id: door\n    source: 0\n", "unknown key.*top level"),
    ("defaults:\n  conf: 0.5\n  bogus: 1\ncameras:\n  - id: d\n    source: 0\n",
     "unknown key.*defaults"),
    ("cameras:\n  - id: d\n    source: 0\n    typo: 1\n", "unknown key.*camera"),
    ("cameras:\n  - id: d\n    source: 0\n    resolution: [1280]\n",
     "resolution.*must be"),
])
def test_invalid_configs_raise_config_error(tmp_path, body, match):
    with pytest.raises(ConfigError, match=match):
        load_config(_write(tmp_path, body))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


# -- flattening to FootfallTracker kwargs -------------------------


def test_tracker_kwargs_without_zones(tmp_path):
    site = load_config(_write(tmp_path, """
        site:
          name: Store 7
          timezone: America/New_York
        defaults:
          conf: 0.55
          reconnect:
            retries: 3
        cameras:
          - id: door
            source: "rtsp://cam/stream"
            resolution: [1920, 1080]
        storage:
          events_db: output/events.db
    """))
    kw = tracker_kwargs(site.cameras[0], site)

    assert kw["source"] == "rtsp://cam/stream"
    assert kw["capture_size"] == (1920, 1080)
    assert kw["conf"] == 0.55
    assert kw["site"] == "Store 7"
    assert kw["tz"] == "America/New_York"
    assert kw["reconnect_retries"] == 3
    assert kw["events_db"] == str(ROOT / "output/events.db")
    assert kw["model_path"] == "yolov8n.pt"     # null model -> a real default
    assert "line" not in kw and "zone" not in kw   # no zones file


def test_tracker_kwargs_overrides_win_last(tmp_path):
    site = load_config(_write(tmp_path, _MINIMAL))
    kw = tracker_kwargs(site.cameras[0], site, overrides={"preview": False,
                                                         "conf": 0.9})
    assert kw["preview"] is False
    assert kw["conf"] == 0.9


def test_tracker_kwargs_loads_zone_geometry(tmp_path):
    zones = tmp_path / "z.json"
    zones.write_text(json.dumps({
        "width": 1280, "height": 720,
        "line": [[100, 400], [900, 400]],
        "zone": [[300, 300], [700, 300], [700, 550], [300, 550]],
        "ignore": [[[0, 0], [50, 0], [50, 50]]],
    }))
    site = load_config(_write(tmp_path, f"""
        cameras:
          - id: door
            source: 0
            zones: {zones}
    """))
    kw = tracker_kwargs(site.cameras[0], site)

    assert kw["geometry_size"] == (1280, 720)
    assert kw["line"] is not None and len(kw["line"]) == 2
    assert len(kw["zone"]) == 4
    assert len(kw["ignore_zones"]) == 1


def test_resolved_is_plain_data(tmp_path):
    site = load_config(_write(tmp_path, _MINIMAL))
    r = resolved(site)
    assert r["site"] == {"name": "Test Site", "timezone": "Europe/Paris"}
    assert r["cameras"][0]["id"] == "door"
    assert r["cameras"][0]["settings"]["conf"] == 0.5


# -- secrets in the source ------------------------------------------


_SECRET_SRC = """
    cameras:
      - id: door
        source: "rtsp://${SECRET:cam1}@10.0.0.1:554/s"
"""


def test_source_secret_is_resolved_display_is_masked(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTFALL_SECRET_CAM1", "admin:pw")
    cam = load_config(_write(tmp_path, _SECRET_SRC)).cameras[0]
    assert cam.source == "rtsp://admin:pw@10.0.0.1:554/s"
    assert cam.source_display == "rtsp://***@10.0.0.1:554/s"


def test_resolve_secrets_false_keeps_the_placeholder(tmp_path):
    cam = load_config(_write(tmp_path, _SECRET_SRC),
                      resolve_secrets=False).cameras[0]
    assert cam.source == "rtsp://${SECRET:cam1}@10.0.0.1:554/s"
    assert cam.source_display == "rtsp://***@10.0.0.1:554/s"


def test_resolved_config_never_carries_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTFALL_SECRET_CAM1", "admin:pw")
    site = load_config(_write(tmp_path, _SECRET_SRC))
    assert "admin:pw" not in json.dumps(resolved(site))
    assert resolved(site)["cameras"][0]["source"] == "rtsp://***@10.0.0.1:554/s"


def test_tracker_kwargs_passes_masked_display(tmp_path, monkeypatch):
    monkeypatch.setenv("FOOTFALL_SECRET_CAM1", "admin:pw")
    site = load_config(_write(tmp_path, _SECRET_SRC))
    kw = tracker_kwargs(site.cameras[0], site)
    assert kw["source"] == "rtsp://admin:pw@10.0.0.1:554/s"
    assert kw["source_display"] == "rtsp://***@10.0.0.1:554/s"
