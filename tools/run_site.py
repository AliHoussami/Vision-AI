"""
run_site.py
-----------
Run the footfall pipeline from a site config file instead of CLI flags.

    python tools/run_site.py config/site.yaml                 # the only camera
    python tools/run_site.py config/site.yaml --camera door   # one named camera
    python tools/run_site.py config/site.yaml --all           # every camera, threaded
    python tools/run_site.py config/site.yaml --print         # resolved config, no run

--all runs one thread per camera with the preview forced off (a GUI window
can only live on the main thread). Proper multi-camera orchestration --
one supervised process per camera, clean shutdown -- is Phase 3; --all is
a best-effort convenience for a small single-box site.
"""

import argparse
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from footfall import FootfallTracker
from footfall.config import ConfigError, load_config, resolved, tracker_kwargs


def _apply_hardware_defaults(settings: dict) -> bool:
    """Fill model / imgsz when the config left them null, the way
    run_webcam.py does: a bigger model at a higher resolution on a GPU, a
    small one on a CPU where anything larger runs too slow to track."""
    import torch

    on_gpu = torch.cuda.is_available()
    if settings.get("model") is None:
        settings["model"] = "yolov8s.pt" if on_gpu else "yolov8n.pt"
    if settings.get("imgsz") is None:
        settings["imgsz"] = 960 if on_gpu else 640
    return on_gpu


def _run_camera(cam, site, overrides):
    _apply_hardware_defaults(cam.settings)
    kw = tracker_kwargs(cam, site, overrides=overrides)
    print(f"[{cam.id}] source={cam.source!r} model={kw['model_path']} "
          f"imgsz={kw['imgsz']} preview={kw['preview']}")
    summary = FootfallTracker(**kw).run()
    print(f"[{cam.id}] " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the site YAML")
    ap.add_argument("--camera", help="run just this camera id")
    ap.add_argument("--all", action="store_true",
                    help="run every camera, one thread each (no preview)")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the fully resolved config and exit")
    ap.add_argument("--no-preview", action="store_true",
                    help="force the live preview window off")
    args = ap.parse_args()

    try:
        site = load_config(args.config)
    except ConfigError as exc:
        ap.error(str(exc))

    if args.show:
        print(yaml.safe_dump(resolved(site), sort_keys=False))
        return

    overrides = {"preview": False} if args.no_preview else {}

    if args.camera:
        _run_camera(site.camera(args.camera), site, overrides)
        return

    if args.all:
        overrides = {**overrides, "preview": False}
        threads = [
            threading.Thread(target=_run_camera, args=(cam, site, overrides),
                             name=f"cam-{cam.id}", daemon=True)
            for cam in site.cameras
        ]
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n[interrupted] stopping cameras")
        return

    if len(site.cameras) == 1:
        _run_camera(site.cameras[0], site, overrides)
        return

    ap.error(f"{len(site.cameras)} cameras in the config -- pass --camera <id> "
             f"or --all. ids: {[c.id for c in site.cameras]}")


if __name__ == "__main__":
    main()
