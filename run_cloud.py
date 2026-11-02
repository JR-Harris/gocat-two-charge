#!/usr/bin/env python3
'''
The cloud v2 build CLI: one-time generation of the swept-union
candidate cloud; the only entry point that writes the canonical
artifacts.

  python run_cloud.py --build v2              # the gated build (locks on green)
  python run_cloud.py --build v2 --regenerate # explicit regeneration (Jonny)

Exit codes: 0 = fully green build (locked); 1 = gate failure (the gate
reports its exact numbers before the exit).

Run from the repo root (writes only data/derived/candidates_v2*,
cloud_v2_summary.json + the README section).
'''

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloud import build as cloud_build


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    regenerate = "--regenerate" in argv
    if "--regenerate" in argv:
        argv.remove("--regenerate")
    if len(argv) != 2 or argv[0] != "--build" or argv[1] != "v2":
        print(__doc__ or "usage: run_cloud.py --build v2 [--regenerate]")
        return 1
    try:
        cloud_build.build_v2_cloud(derived_root=None,
                                   regenerate=regenerate)
    except ValueError as e:
        print(f"cloud v2: FAILED: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())