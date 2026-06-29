"""Backward-compatible entry point.

`python main.py "<task>"` (and `index`/`search`/`symbols`/`config check`/`models`)
still work — everything delegates to the same CLI as the `localcli` console script
(see agent/cli.py).
"""

import sys
from agent.cli import main

if __name__ == "__main__":
    sys.exit(main())
