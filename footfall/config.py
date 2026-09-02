"""
config.py
---------
Load a whole site's setup -- metadata, per-camera sources, geometry, model
and pipeline settings -- from one YAML file, instead of a wall of CLI
flags retyped every run.

    site = load_config("config/site.yaml")
    for cam in site.cameras:
        FootfallTracker(**tracker_kwargs(cam, site)).run()

Per-camera keys override the shared ``defaults`` block. Non-absolute
paths in the file are resolved against the project root. Secrets (camera
passwords) do NOT belong here -- that is the next step; keep them out of
band and reference them from the source URL at deploy time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import ROOT


class ConfigError(ValueError):
    """A site config file is missing something, or has a key we don't know."""


# Keys accepted in `defaults` and in a per-camera override. Each maps to a
# FootfallTracker constructor argument in tracker_kwargs().
_SETTING_KEYS = {
    "model", "imgsz", "conf", "iou", "tracker", "device", "hw_accel",
    "frame_buffer", "stale_after", "detect_frozen", "min_box_height",
    "max_aspect", "preview",
}
_RECONNECT_KEYS = {"initial", "factor", "max", "retries"}

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "model": None,          # None -> caller picks (hardware-dependent)
    "imgsz": None,          # None -> caller picks
    "conf": 0.50,
    "iou": 0.5,
    "tracker": None,        # None -> repo's tuned config
    "device": None,
    "hw_accel": "auto",
    "frame_buffer": 2,
    "stale_after": 10.0,
    "detect_frozen": True,
    "min_box_height": 0,
    "max_aspect": None,
    "preview": False,
    "reconnect": {"initial": 1.0, "factor": 2.0, "max": 30.0, "retries": None},
}

_TOP_KEYS = {"site", "defaults", "cameras", "storage"}
_SITE_KEYS = {"name", "timezone"}
_CAMERA_KEYS = {"id", "source", "resolution", "zones"} | _SETTING_KEYS | {"reconnect"}
_STORAGE_KEYS = {"events_db"}


@dataclass
class SiteMeta:
    name: Optional[str] = None
    timezone: Optional[str] = None


@dataclass
class CameraConfig:
    id: str
    source: Any                                  # int index | path | URL
    resolution: Optional[tuple] = None
    zones_path: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)   # defaults + override


@dataclass
class SiteConfig:
    site: SiteMeta
    cameras: List[CameraConfig]
    events_db: Optional[str] = None
    path: Optional[str] = None

    def camera(self, cam_id: str) -> CameraConfig:
        for c in self.cameras:
            if c.id == cam_id:
                return c
        raise ConfigError(
            f"no camera with id {cam_id!r}; have {[c.id for c in self.cameras]}")


# -- parsing helpers ---------------------------------------------------


def _require_mapping(value, where):
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    return value


def _reject_unknown(d: dict, allowed: set, where: str):
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {sorted(unknown)}")


def _resolve_path(p, base: Path) -> str:
    q = Path(str(p)).expanduser()
    return str(q if q.is_absolute() else (base / q))


def _merge_settings(base: dict, override: dict, where: str) -> dict:
    _require_mapping(override, where)
    _reject_unknown(override, _SETTING_KEYS | {"reconnect"}, where)
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k == "reconnect":
            _require_mapping(v, f"{where}.reconnect")
            _reject_unknown(v, _RECONNECT_KEYS, f"{where}.reconnect")
            merged["reconnect"] = {**merged["reconnect"], **v}
        else:
            merged[k] = v
    return merged


def _parse_site(raw) -> SiteMeta:
    raw = _require_mapping(raw or {}, "'site'")
    _reject_unknown(raw, _SITE_KEYS, "'site'")
    return SiteMeta(name=raw.get("name"), timezone=raw.get("timezone"))


def _parse_camera(raw, index: int, defaults: dict, base: Path) -> CameraConfig:
    raw = _require_mapping(raw, f"cameras[{index}]")
    _reject_unknown(raw, _CAMERA_KEYS, f"cameras[{index}]")

    if not str(raw.get("id", "")).strip():
        raise ConfigError(f"cameras[{index}] is missing 'id'")
    cam_id = str(raw["id"])

    if "source" not in raw:
        raise ConfigError(f"camera {cam_id!r} is missing 'source'")
    source = raw["source"]
    if isinstance(source, str) and source.isdigit():
        source = int(source)                     # "0" -> camera index 0

    resolution = None
    if raw.get("resolution") is not None:
        r = raw["resolution"]
        if (not isinstance(r, (list, tuple)) or len(r) != 2
                or not all(isinstance(x, int) for x in r)):
            raise ConfigError(
                f"camera {cam_id!r}: 'resolution' must be [width, height] ints")
        resolution = (int(r[0]), int(r[1]))

    zones_path = _resolve_path(raw["zones"], base) if raw.get("zones") else None

    override = {k: raw[k] for k in (_SETTING_KEYS | {"reconnect"}) if k in raw}
    settings = _merge_settings(defaults, override, f"camera {cam_id!r}")

    return CameraConfig(id=cam_id, source=source, resolution=resolution,
                        zones_path=zones_path, settings=settings)


# -- public API -------------------------------------------------------


def load_config(path) -> SiteConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    _require_mapping(raw, "the config file")
    _reject_unknown(raw, _TOP_KEYS, "top level")

    site = _parse_site(raw.get("site"))
    defaults = _merge_settings(_DEFAULT_SETTINGS, raw.get("defaults") or {},
                               "defaults")

    cams_raw = raw.get("cameras")
    if not cams_raw:
        raise ConfigError("config has no 'cameras'")
    if not isinstance(cams_raw, list):
        raise ConfigError("'cameras' must be a list")

    cameras: List[CameraConfig] = []
    seen = set()
    for i, c in enumerate(cams_raw):
        cam = _parse_camera(c, i, defaults, ROOT)
        if cam.id in seen:
            raise ConfigError(f"duplicate camera id {cam.id!r}")
        seen.add(cam.id)
        cameras.append(cam)

    storage = _require_mapping(raw.get("storage") or {}, "'storage'")
    _reject_unknown(storage, _STORAGE_KEYS, "'storage'")
    events_db = storage.get("events_db")
    if events_db:
        events_db = _resolve_path(events_db, ROOT)

    return SiteConfig(site=site, cameras=cameras, events_db=events_db,
                      path=str(path))


def tracker_kwargs(cam: CameraConfig, site: SiteConfig,
                   overrides: Optional[dict] = None) -> dict:
    """Flatten a CameraConfig + SiteConfig into FootfallTracker(**kwargs).

    Zone geometry is loaded here when the camera names a zones file.
    `overrides` (e.g. a --no-preview flag) win last.
    """
    s = cam.settings
    kw: Dict[str, Any] = dict(
        source=cam.source,
        capture_size=cam.resolution,
        conf=s["conf"],
        iou=s["iou"],
        imgsz=s["imgsz"] if s["imgsz"] is not None else 640,
        model_path=s["model"] or "yolov8n.pt",
        tracker=s["tracker"],
        device=s["device"],
        hw_accel=s["hw_accel"],
        frame_buffer=s["frame_buffer"],
        stale_after=s["stale_after"],
        detect_frozen=s["detect_frozen"],
        min_box_height=s["min_box_height"],
        max_aspect=s["max_aspect"],
        preview=s["preview"],
        reconnect_initial=s["reconnect"]["initial"],
        reconnect_factor=s["reconnect"]["factor"],
        reconnect_max=s["reconnect"]["max"],
        reconnect_retries=s["reconnect"]["retries"],
        site=site.site.name,
        tz=site.site.timezone,
    )
    if site.events_db:
        kw["events_db"] = site.events_db
    if cam.zones_path:
        from .zones import load_zones
        line, zone, geometry_size, ignore_zones = load_zones(cam.zones_path)
        kw.update(line=line, zone=zone, geometry_size=geometry_size,
                  ignore_zones=ignore_zones)
    if overrides:
        kw.update(overrides)
    return kw


def resolved(site: SiteConfig) -> dict:
    """A plain dict of the fully merged config, for --print / inspection."""
    return {
        "config_file": site.path,
        "site": {"name": site.site.name, "timezone": site.site.timezone},
        "events_db": site.events_db,
        "cameras": [
            {
                "id": c.id,
                "source": c.source,
                "resolution": list(c.resolution) if c.resolution else None,
                "zones": c.zones_path,
                "settings": c.settings,
            }
            for c in site.cameras
        ],
    }
