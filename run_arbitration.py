"""The arbitration stage driver (item 7).

    python run_arbitration.py --landscape canonical --step A \
        --workers 8 --pal 2 --list-k 100

Every step: reads its inputs, checks the resume/cache state, executes
only what is missing, writes records + sidecars, writes the step
summary, exits.  Full cache-resumability at SP, config and step
granularity (see arbitration/steps.py).

Steps: A (top-K re-rank) | B (boundary sweep) | C (fidelity probe --
the stratified 192-configuration sample) | D (certified optimum) |
E (P-side check) | F (relaxation check) | G (wB97M check) | cert
(re-verify + re-emit).

--landscape replicate-1 | replicate-2 runs the same steps against that
replicate's landscape. All three landscapes were arbitrated; their records
are published under runs/arbitration/.

Fail-hard: any step failure halts that step's batch with the config
key + the sp.out tail; nothing under data/ is ever written.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arbitration import steps


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="item 7: DFT arbitration of the certified landscape")
    ap.add_argument("--landscape", default="canonical",
                    choices=("canonical", "replicate-1", "replicate-2"),
                    help="the landscape to arbitrate (canonical, or "
                         "either replicate)")
    ap.add_argument("--step", required=True,
                    choices=tuple(sorted(steps.STEPS)),
                    help="the step to run (A|B|C|D|E|F|G|cert)")
    ap.add_argument("--workers", type=int, default=steps.PRODUCTION_WORKERS,
                    help="concurrent ORCA single-point workers (the "
                         "production cell is 16; the 12-worker fallback "
                         "runs ONLY after an actual failure on (16,1) — "
                         "Ruling 3)")
    ap.add_argument("--pal", type=int, default=steps.PRODUCTION_PAL,
                    help="PAL (OpenMP ranks) per ORCA single point (1 = "
                         "the keyword-free serial input, the production "
                         "cell)")
    ap.add_argument("--list-k", dest="list_k", type=int, default=100,
                    help="K for the top-K certified list (steps A/B)")
    ap.add_argument("--maxcore", type=int,
                    default=None,
                    help="per-rank MaxCore memory in MB (default "
                         + str(steps.orca_prov.BASELINE_MAXCORE_MB)
                         + "; the pricing benchmark may adjust it)")
    args = ap.parse_args(argv)

    maxcore = (args.maxcore if args.maxcore is not None
               else steps.orca_prov.BASELINE_MAXCORE_MB)
    engine = steps.Engine(args.landscape, workers=args.workers,
                          pal=args.pal, list_k=args.list_k,
                          maxcore=maxcore)
    print(f"[arbitration] landscape={args.landscape} step={args.step} "
          f"workers={args.workers} pal={args.pal} list_k={args.list_k} "
          f"maxcore={maxcore}", flush=True)
    summary, path = steps.STEPS[args.step](engine)
    brief = {k: v for k, v in summary.items()
             if k in ("purpose", "list_k", "executed", "from_cache",
                      "f_dft_min_kcal", "f_dft_argmin_row", "k_final",
                      "certified_rows", "row", "f_kcal", "n_quench_true",
                      "n_expected", "relaxed_differential_kcal",
                      "exactly_one_imaginary_gate", "deviation")}
    print(f"[arbitration] summary: {brief}", flush=True)
    print(f"[arbitration] written: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())