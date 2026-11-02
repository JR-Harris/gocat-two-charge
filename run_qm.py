#!/usr/bin/env python3
"""run_qm.py — run one quantum-chemistry stage.

  python run_qm.py --execute endpoints   optimise both endpoints (r2SCAN-3c,
                                         analytic FREQ, the flat-well rule)
  python run_qm.py --execute ts          the coordinate-drive transition-state
                                         chain: relaxed scan -> two-constraint
                                         guess -> OptTS + exact Hessian
  python run_qm.py --execute irc         the IRC from the verified TS, then
                                         re-optimise both endpoints

Each stage renders its ORCA inputs (qm/inputs.py), runs them through the
site-configured launcher (qm/orca_runner.py), parses the outputs
(qm/orca_out.py) and writes its summary JSON under data/derived/. Every stage
is fail-hard: a stage that does not verify does not write.

ORCA 6.1.1 must be installed. qm/orca_runner.py names the sites it knows and
reads GOCAT_ORCA_SITE to pick one.

This driver is trimmed for publication. The working tree's --parse, --gates
and --promote sub-commands checked returned outputs against the project's own
frozen registry numbers and hashes, which are not part of this repository;
see docs/CHANGES.md.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="QM stage orchestrator: run one stage through ORCA.")
    ap.add_argument("--execute", required=True,
                    choices=("endpoints", "ts", "irc"),
                    help="the stage to run (see the module docstring)")
    args = ap.parse_args(argv)

    if args.execute == "endpoints":
        from qm import escape
        escape.execute_endpoints()
    elif args.execute == "ts":
        from qm import ts_chain
        ts_chain.execute_ts()
    else:
        from qm import irc_chain
        irc_chain.execute_irc()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
