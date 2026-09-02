"""Footfall & queue tracking package.

Paths are resolved relative to the project root rather than the current
working directory, so the tools behave the same whether you run them from
the repo root, from inside tools/, or from anywhere else.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "output"

# Per-camera geometry drawn by tools/define_zones.py. Kept at the project
# root (and gitignored) because it belongs to one physical camera.
ZONES_FILE = ROOT / "zones.json"

DEFAULT_TRACKER = CONFIG_DIR / "tracker_people.yaml"

# Downloaded YOLO weights. Ultralytics resolves a bare filename against the
# CURRENT WORKING DIRECTORY, so running a tool from elsewhere would not find
# an already-downloaded model and would try to fetch it again into whatever
# directory you happened to be in. Anchoring them here fixes that.
MODELS_DIR = ROOT / "models"


def resolve_model(name: str) -> str:
    """Turn a bare weight name ('yolov8s.pt') into a project-anchored path.

    A name that already carries a directory is passed through untouched, so
    an explicit path still works.
    """
    import os

    if os.path.dirname(str(name)):
        return str(name)
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / name)


def output(name: str) -> str:
    """Absolute path for a generated file, creating output/ on first use.

    Everything the pipeline writes -- recordings, event logs, diagnostic
    clips -- goes here so the project root stays readable.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    return str(OUTPUT_DIR / name)


from .config import (CameraConfig, ConfigError, SiteConfig,  # noqa: E402
                     load_config, resolved, tracker_kwargs)
from .control import ControlServer, LiveControls  # noqa: E402
from .storage import (CsvEventSink, EventSink, NullSink,  # noqa: E402
                      SqliteEventSink, new_run_id)
from .tracker import FootfallTracker, Point, max_capture_size  # noqa: E402
from .zones import load_zones  # noqa: E402

__all__ = [
    "FootfallTracker",
    "Point",
    "max_capture_size",
    "resolve_model",
    "MODELS_DIR",
    "load_zones",
    "output",
    "ROOT",
    "CONFIG_DIR",
    "OUTPUT_DIR",
    "ZONES_FILE",
    "DEFAULT_TRACKER",
    "EventSink",
    "SqliteEventSink",
    "CsvEventSink",
    "NullSink",
    "new_run_id",
    "load_config",
    "tracker_kwargs",
    "resolved",
    "SiteConfig",
    "CameraConfig",
    "ConfigError",
    "ControlServer",
    "LiveControls",
]
