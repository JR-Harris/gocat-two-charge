#!/usr/bin/env python3
"""The item-6 scoping harness (battery-owned code; NOT a vendored file).

Composes the vendored, frozen ga/ modules exactly as run_ga.py does
(resolve_mode's oracle semantics: the summary-level cloud gate, a
per-run FRESH provider, the landscape's own fence, compose_record's
provenance, write_record's atomic write + sha256 sidecar) but
constructs GAParams DIRECTLY — the vendored params.py set stays
untouched; this is the declared scoping phase (prereg §7; unreported
tuning, one landscape, budget-capped).

Usage from a working dir whose `data/` resolves to GOCAT_Program/data
(reads only; the provider's init gates sha-assert everything) and whose
`bin/` resolves to GOCAT_Program/bin:

  python experiments/cells.py probe    # 7a: the generation-budget probe
  python experiments/cells.py sweep    # 7b: 36 cells x 10 seeds, G=200
  python experiments/cells.py check    # determinism spot-check

Artifacts: under --out-root (default runs/scoping/replicate-1/armA),
per-cell DIRECTORIES: the runner's record identity is (G, seed,
fitness_id, cloud-sha12), so a sweep collides on names without a
per-cell dir — a scoping-local convention recorded for the addendum;
the reported runs (single frozen param set) need no such convention.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# The snapshot-bearing battery commit; the manifest maps the run record
# provenance back to GOCAT_Program @ 09f967d29bfe9e5ab25793a2fe3d8b8b248c14de.
BATTERY_COMMIT = "98d52ab69b7716058f1dc71a0d72eef8ab68ca9e"
ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(os.environ.get("BAT_OUT_ROOT", ROOT / "runs" / "screen"))
# Two-process sweep partition (nproc=2 in the execution VM): BAT_SLICE
# in {"0","1"} runs the cells whose index is congruent to it mod 2.
SLICE = os.environ.get("BAT_SLICE")

LANDSCAPE_DIR = (ROOT / "data" / "replicates" / "replicate-1"
                  / "derived" / "enumeration_replicate-1")
CLOUD_PATH = ROOT / "data" / "replicates" / "replicate-1" / "candidates_replicate-1.npy"
# The replicate-1 xTB-tier argmin — the FIXED pre-registered target
# (prereg §3; 04 §8b, the float64 record value).
ARGMIN = -125.46892076041331

sys.path.insert(0, str(ROOT))                             # this repository

from ga import ga as ga_mod                      # noqa: E402
from ga import space as ga_space                 # noqa: E402
from ga.params import GAParams                   # noqa: E402
from ga.oracle_provider import landscape_oracle  # noqa: E402
import run_ga as runner                             # noqa: E402

SMOKE = GAParams()  # the landed defaults (7a's operational profile)


def resolved_setup():
    """run_ga.resolve_mode's oracle-semantics replication, one landscape."""
    summary = json.loads((LANDSCAPE_DIR / "enumeration_summary.json")
                         .read_text())
    want = str(summary["cloud_sha256"])
    got = runner.sha256_file(CLOUD_PATH)
    if got != want:
        raise SystemExit(f"cloud sha mismatch: {got} != {want}")
    excl_sha = ((summary.get("artifacts") or {})
                .get("domain_exclusions") or {}).get("sha256")
    tuples, source = runner.read_domain_exclusions(LANDSCAPE_DIR, excl_sha)
    candidates = np.load(CLOUD_PATH)
    space = ga_space.Space(candidates, d_min=1.0, exclusions=tuples,
                           nn_k=8)
    d_min = float((summary.get("universe") or {}).get(
        "d_min", ga_space.D_MIN_A))
    return summary, tuples, source, space, d_min


FITNESS_ID = "enumeration_v2"   # the landscape summaries' fitness_id


def run_one(space, tuples, source, d_min, params, seed, G, outdir,
            cloud_sha):
    """One run: FRESH provider instance, the engine, run_ga.py's records.
    RESUMABLE: an existing record identity is skipped (read + returned,
    nothing rewritten) — the sweep restarts cleanly per execution window."""
    path = Path(runner.record_path(outdir, G, seed,
                                   FITNESS_ID, cloud_sha))
    if path.is_file() and not os.environ.get("BAT_FORCE"):
        rec = json.loads(path.read_text())
        rec["_resumed"] = True
        return path, rec, "<resumed>"
    provider = landscape_oracle(LANDSCAPE_DIR, CLOUD_PATH)  # FRESH per run
    t0 = time.time()
    result = ga_mod.run_ga(provider.eval_batch, space, params, seed, G=G)
    path = runner.record_path(outdir, G, seed,
                              provider.FITNESS_ID, provider.cloud_sha256)
    record = runner.compose_record(
        result, mode="oracle", fitness_id=provider.FITNESS_ID,
        provider_meta=provider.PROVIDER_META, cloud_path=CLOUD_PATH,
        cloud_sha256=provider.cloud_sha256, d_min=d_min,
        exclusions=tuples, exclusion_source=source,
        landscape_dir=str(LANDSCAPE_DIR),
        artifact_sha256s=dict(provider.artifact_sha256s),
        param_set="scoping:" + json.dumps(params.as_dict(),
                                          sort_keys=True),
        git_head=("battery-snapshot:" + BATTERY_COMMIT))
    sha = runner.write_record(record, path)
    record["_wall_seconds_measured"] = time.time() - t0
    return path, record, sha


def cells():
    for pc in (0.7, 0.9):
        for pm in (0.1, 0.2, 0.3):
            for pop in (30, 50, 100):
                for el in (2, 4):
                    if el > pop:
                        continue
                    yield (pc, pm, pop, el)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    summary, tuples, source, space, d_min = resolved_setup()
    cloud_sha = str(summary["cloud_sha256"])
    print(f"resolved: landscape={LANDSCAPE_DIR} "
          f"rows={summary['universe']['n_rows_post_floor']} "
          f"exclusions={len(tuples)} argmin={ARGMIN}", flush=True)

    if mode == "probe":
        outdir = OUT_ROOT / "probe7a"
        for seed in (0, 1, 2):
            path, rec, sha = run_one(space, tuples, source, d_min, SMOKE,
                                     seed, 2000, outdir, cloud_sha)
            ledger = rec["ledger"]
            q = rec["counts"]["queries"]
            firsthit_g = None
            for e in ledger:
                if e[2] <= ARGMIN + 0.5:
                    firsthit_g = e[0]
                    break
            qpg = (q - ledger[0][1]) / max(1, (len(ledger) - 1))
            print(f"probe s={seed}: queries={q} best={ledger[-1][2]:.6f} "
                  f"firsthit_g={firsthit_g} "
                  f"wall={rec.get('_wall_seconds_measured', -1):.2f}s "
                  f"-> {path.name} (sha {sha[:12]})", flush=True)
            print(f"  qpg (ledger last-first): {qpg:.1f}", flush=True)

    elif mode == "sweep":
        total_h = 0
        all_cells = list(cells())
        if SLICE is not None:
            all_cells = [c for k, c in enumerate(all_cells)
                         if k % 2 == int(SLICE)]
        for (pc, pm, pop, el) in all_cells:
            params = GAParams(pc=pc, pm=pm, pop_size=pop, n_elite=el)
            tag = f"c_pc{pc}_pm{pm}_pop{pop}_el{el}"
            outdir = OUT_ROOT / tag
            hits = 0
            qsum = 0
            for seed in range(10):
                path, rec, sha = run_one(space, tuples, source, d_min,
                                         params, seed, 200, outdir,
                                         cloud_sha)
                best_f = rec["ledger"][-1][2]
                hit = best_f <= ARGMIN + 0.5
                hits += hit
                qsum += rec["counts"]["queries"]
                total_h += 1
                print(f"{tag} s={seed}: best={best_f:.4f} "
                      f"queries={rec['counts']['queries']} hit={hit}",
                      flush=True)
            print(f"CELL {tag}: hits {hits}/10 "
                  f"qsum {qsum}", flush=True)
        print(f"SWEEP DONE: {total_h} runs", flush=True)

    elif mode == "check":
        # determinism spot-check: one (cell, seed) run twice — the
        # engine fields must be identical (volatile provenance aside)
        params = GAParams(pc=0.9, pm=0.2, pop_size=50, n_elite=2)
        _, r1, _ = run_one(space, tuples, source, d_min, params, 7, 200,
                           OUT_ROOT / "check", cloud_sha)
        _, r2, _ = run_one(space, tuples, source, d_min, params, 7, 200,
                           OUT_ROOT / "check2", cloud_sha)
        same = (r1["ledger"] == r2["ledger"]
                and r1["trajectory"] == r2["trajectory"]
                and r1["best"] == r2["best"]
                and r1["counts"] == r2["counts"])
        print("determinism spot-check:", "IDENTICAL" if same
              else "MISMATCH", flush=True)
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()