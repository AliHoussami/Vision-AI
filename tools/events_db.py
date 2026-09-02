"""
events_db.py
------------
Manage the local SQLite event store (output/events.db by default).

    python tools/events_db.py runs                  # list runs + event counts
    python tools/events_db.py drop-run <run_id>     # delete one run's events
    python tools/events_db.py reset                 # delete the whole file

`reset` is the "clean slate for testing" button -- the store is a single
file, so this just removes it (and its WAL sidecars).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footfall import OUTPUT_DIR
from footfall.storage import drop_run, list_runs, reset

DEFAULT_DB = str(OUTPUT_DIR / "events.db")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["runs", "drop-run", "reset"])
    ap.add_argument("run_id", nargs="?", help="run id, for drop-run")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"default: {DEFAULT_DB}")
    args = ap.parse_args()

    exists = Path(args.db).exists()

    if args.cmd == "reset":
        if exists:
            reset(args.db)
            print(f"removed {args.db}")
        else:
            print(f"nothing to remove ({args.db} does not exist)")
        return

    if not exists:
        print(f"no event store yet at {args.db}")
        return

    if args.cmd == "runs":
        rows = list_runs(args.db)
        if not rows:
            print("no runs recorded")
            return
        print(f"{'run_id':<28} {'started (UTC)':<20} {'ended (UTC)':<20} "
              f"{'events':>7}  source")
        for run_id, started, ended, source, count in rows:
            print(f"{run_id:<28} {(started or '')[:19]:<20} "
                  f"{(ended or '-')[:19]:<20} {count:>7}  {source or ''}")
        return

    if args.cmd == "drop-run":
        if not args.run_id:
            ap.error("drop-run needs a run_id (see: events_db.py runs)")
        removed = drop_run(args.db, args.run_id)
        print(f"dropped run {args.run_id} ({removed} events)")


if __name__ == "__main__":
    main()
