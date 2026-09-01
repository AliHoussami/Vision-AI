"""Draw the entrance line, queue zone, and ignore regions onto a real frame.

Thin launcher; the editor itself lives in footfall/zones.py so that
run_webcam.py and compare_trackers.py can import load_zones() from the
package without pulling in a command-line tool.

    python tools/define_zones.py --source 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from footfall.zones import main

if __name__ == "__main__":
    main()
