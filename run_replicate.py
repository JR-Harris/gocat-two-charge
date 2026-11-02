#!/usr/bin/env python3
'''
The per-replicate pipeline CLI (work item 4b): ONE command per
replicate landscape -- the surface-identity stage, the fresh roll-k
realization, the full per-cloud QC gate set, the vs-canonical
comparison stage, and the enumeration-driver wiring, all isolated
under data/replicates/<NAME>/ with one manifest; the canonical
data/derived/ tree is HARD-FENCED (never written, ever -- even a
`--name canonical` re-derivation is confined to
data/replicates/canonical/).

  python run_replicate.py --name <NAME>                 # cloud + gates + comparison + manifest; STOP
  python run_replicate.py --name <NAME> --stage cloud   # (the same as the default)
  python run_replicate.py --name <NAME> --stage enum [--mode stub|xtb]
                                       [--limit N] [--shards K]
      # the enumeration driver wired to the replicate's cloud + its
      # recorded sha256 (the cloud stage's artifacts + manifest are a
      # prerequisite; resumable via the driver's own records)
  python run_replicate.py --name <NAME> --stage end-to-end-smoke
      # tester mode, throwaway name: cloud + gates + comparison + a
      # 200-row stub subset + a 20-row real-provider subset, all under
      # the replicate root; minutes on the laptop

--name: ^[a-z0-9][a-z0-9_-]{0,40}$; any name matching a canonical
artifact prefix is refused (the name IS the write surface, so the name
fence IS the tree fence). THE 2026-10-14 GENERALIZATION RULING:
`canonical` is a VALID name (the program is name-agnostic;
`--name canonical --roll 0` = the examiner/re-derivation run, confined
to data/replicates/canonical/, always announced: the canonical name
without explicit --roll 0 is refused).

Exit codes: 0 = green; 1 = a stage/gate failure (the gate's numbers
print); 2 = the CLI refused the run (the name fence, a non-empty root,
a missing/mismatching manifest).
'''

# THE PINNED DRAW REGIME — before ANY import (libgomp reads its env at
# load time): the re-derivability band's own measurement is the pinned
# regime (2e-12 A; the unpinned record noted 4.3e-12 A) — every draw of
# this CLI is pinned single-thread.
import os as _os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ[_k] = "1"

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(
    os.path.realpath(__file__))))

from cloud import replicate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="run_replicate.py --name <NAME> [--stage "
                    "{cloud,enum,end-to-end-smoke}] [--mode "
                    "{stub,xtb}] [--limit N] [--shards K]")
    parser.add_argument("--name", required=True,
                        help="the cloud name (^[a-z0-9][a-z0-9_-]{0,40}$; "
                             "canonical = the announced roll-0 "
                             "re-derivation; canonical-artifact prefixes "
                             "refused)")
    parser.add_argument("--stage", choices=("cloud", "enum",
                                            "end-to-end-smoke"),
                        default="cloud",
                        help="cloud (default): realization + gates + "
                             "comparison + manifest, then STOP; enum: "
                             "drive the enumeration against the "
                             "replicate's cloud; end-to-end-smoke: the "
                             "tester's one-command subset run")
    parser.add_argument("--mode", choices=("stub", "xtb"), default="xtb",
                        help="the enum stage's provider (xtb = the real "
                             "oracle; the default)")
    parser.add_argument("--limit", type=int, default=None,
                        help="the enum stage's row limit (an omitted "
                             "limit is the FULL post-floor universe)")
    parser.add_argument("--shards", type=int, default=8,
                        help="the enum stage's concurrent worker count")
    parser.add_argument("--roll", type=int, default=None,
                        help="the cloud stage's stream roll (the "
                             "2026-10-14 draw ruling: canonical = roll "
                             "0; replicate k = roll k by k discarded "
                             "sampler runs; omitted = auto = 1 + the "
                             "existing sibling-root count; a roll "
                             "already recorded by a sibling refuses)")
    args = parser.parse_args(argv)

    try:
        replicate.validate_name(args.name, roll=args.roll)
    except replicate.ReplicateError as e:
        print(f"run_replicate: REFUSED: {e}")
        return 2
    try:
        if args.stage == "cloud":
            replicate.run_cloud_stage(args.name, roll=args.roll)
        elif args.stage == "enum":
            replicate.run_enum_stage(args.name, mode=args.mode,
                                     limit=args.limit,
                                     shards=args.shards)
        elif args.stage == "end-to-end-smoke":
            replicate.run_smoke_stage(args.name)
    except replicate.ReplicateError as e:
        print(f"run_replicate: FAILED ({args.stage}): {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())