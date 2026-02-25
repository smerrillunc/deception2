#!/usr/bin/env python3
"""Compatibility wrapper for BS deception mining.

This entrypoint now delegates to the universal miner at:
    deception2/src/deception_miner.py
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
UNIVERSAL_SRC = REPO_ROOT / "src"
if str(UNIVERSAL_SRC) not in sys.path:
    sys.path.append(str(UNIVERSAL_SRC))

from deception_miner import main as universal_main


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--game" not in argv:
        argv = ["--game", "bs"] + argv
    return universal_main(argv)


if __name__ == "__main__":
    main()
