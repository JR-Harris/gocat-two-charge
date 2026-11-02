#!/usr/bin/env python3
'''
Run both v2 grips against the canonical data/derived tree, in one run.

Each grip is fail-hard on its own gates; the grips are independent
steps (an R-grip failure does not block the P-grip from building, and
already-written artifacts never block either grip's re-run -- each
write is its own step, overwritten unconditionally by a passing grip).

Writes exactly: data/derived/structures/chorismate_reactive_TSframe.xyz,
data/derived/alignment_summary.json,
data/derived/structures/prephenate_TSframe.xyz,
data/derived/p_registration_summary.json.

Exit codes: 0 = both grips passed; 1 = at least one grip failed (the
grip reports name its breaches before the exit).

Run:  venv/bin/python run_alignment.py
'''

import sys

from alignment import grips


def main(argv=None):
    failures = []
    for label, build in (("R-grip", grips.build_r_grip),
                         ("P-grip", grips.build_p_grip)):
        print(f"=== {label}")
        try:
            build()
        except (ValueError, OSError) as e:
            print(f"=== {label}: FAILED: {e}")
            failures.append(label)
    if failures:
        print("alignment: FAILED for " + ", ".join(failures))
        return 1
    print("alignment: both grips green; the four artifacts are in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())