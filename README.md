# gocat-two-charge

The code and the results of a study that asks two questions about a genetic
search over electrostatic catalyst configurations for the chorismate →
prephenate Claisen rearrangement:

1. **Does the search find the answer, and does it cost less?** A genetic
   algorithm searches a discrete space of two-charge configurations. The same
   space is enumerated exhaustively, so "the answer" is known before the search
   starts. Across three independently sampled candidate clouds (three
   landscapes of about 1.7 million configurations each — 1,699,132, 1,699,172
   and 1,699,140 after the separation floor), the search returns the
   enumerated global optimum in **598 of 600 runs**, at **229–391×** fewer
   fitness evaluations than the enumeration.
2. **Does the cheap scoring function rank like the expensive one?** The
   enumeration runs at GFN2-xTB; the best candidates are re-scored at
   r²SCAN-3c. The search is solved and the *scoring tier* is what limits the
   answer: the reference-level optimum sits at xTB ranks 13, 94 and 126 across
   the three landscapes, and the xTB optimum falls to ranks 33, 27 and 23.

The repository is a curated subset of the study's program: every stage that
produced a reported number, with the recorded results it produced. It is not
a general-purpose tool, and it is not the project's full working tree — see
`docs/CHANGES.md` for exactly what was left out and what was edited.

---

## The map

Each stage is a folder with one entry point. The arrows are data, not
requests: every stage reads files and writes files, and the file names are the
interface.

| # | Stage | Folder | Entry point | Consumes | Produces |
|---|---|---|---|---|---|
| 1 | Structure preparation and the reaction path | `qm/` | `run_qm.py` | two SDF structures | optimised endpoints, the transition state, the IRC path, summary JSONs |
| 2 | Registration (the two grips) | `alignment/` | `run_alignment.py` | the stage-1 structures | reactant and product in the transition-state frame |
| 3 | Candidate charge sites | `cloud/` | `run_cloud.py`, `run_replicate.py` | the registration + the path | `candidates_v2.npy`: 928 sites on the path-union surface |
| 4 | The fitness | `fitness/` | (library) | a cloud, two structures | `f = E(TS;q) − E(R;q)` from three interchangeable providers |
| 5 | Exhaustive enumeration | `exhaustive_search/` | `run_exhaustive_search.py` | the cloud + the xTB provider | the landscape: `f.npy` (1.70 M rows per surface) and its summaries |
| 6 | The genetic algorithm | `ga/` | `run_ga.py` | the landscape (oracle) or live xTB | one run's record: ledger, best answer, query counts |
| 7 | The experiments | `experiments/` | `battery.py`, `screen.py` | the three landscapes | the 600-run battery and its statistics; the parameter screen |
| 8 | Reference-level validation | `arbitration/` | `run_arbitration.py` | a landscape + the DFT provider | top-100 re-rank, boundary sweep, fidelity probe, certified optimum |

The three landscapes are the canonical cloud and two independent re-samplings
of the same locked surface (replicates 1 and 2). They live side by side:
`data/derived/` for the canonical chain and `data/replicates/<name>/` for the
replicates, each with its own cloud, landscape and summaries.

## Reader's route

If you are marking the *method* rather than reading a codebase, four files
carry it: `ga/ga.py` (the search), `ga/space.py` (the configuration
arithmetic), `run_ga.py` (one run, end to end) and `experiments/battery.py`
(the reported 600-run experiment). Their shared interface is
`fitness/contract.py`. Everything else is per-stage machinery, with its
assumptions stated at the top of the file.

The code was written inside a working project and still uses that project's
vocabulary — "item 7", "the registry", "the briefing", dated rulings.
`docs/GLOSSARY.md` translates it term by term, and lists the files that were
renamed on the way in.

## Quick start

The search, with no data and no quantum chemistry (a synthetic provider, 2 s):

```
python run_ga.py --mode stub --generations 50 --seed 0
```

The headline run — the search against the published landscape, which is what
the paper's first result is:

```
python run_ga.py --mode oracle --generations 2500 --seed 0
```

It should print `best_f=-125.214630 config=(418, 419, -1, -1) row=1190559`
with `queries=121250`. That single run is the canonical landscape's arm-A
seed 0, and its counters are one line of
`data/results/battery/battery_runs_index.json`.

A partial battery (six runs per group instead of a hundred; ~1 min):

```
BAT_N_SEEDS=6 python experiments/battery.py
```

The test suite for the search (no data beyond the shipped landscape):

```
python ga/test_ga.py
```

## Software

Python 3.13 with the pins in `requirements.txt` is what produced the results.
The search and the experiments need only numpy and scipy; the cloud stage also
needs `trimesh`, `point-cloud-utils` and `manifold3d` (the latter two ship
binary wheels for 3.13 only).

Two external programs are used and are **not** distributed here:

- **ORCA 6.1.1** — all quantum chemistry: endpoint optimisations, the
  transition state, the IRC, and the r²SCAN-3c reference tier. Licensed; the
  input templates are in `qm/inputs.py` and `fitness/orca.py` writes the
  point-charge files.
- **xtb 6.7.1** — the GFN2-xTB search tier. The embedding is done through the
  command line (`bin/xtb-6.7.1/bin/xtb`, OMP=1, one process per single point);
  the in-process `tblite` route cannot carry external charges at all. See
  `fitness/xtb.py`.

What each needs:

| You want to | You need |
|---|---|
| run the search against the published landscapes | nothing beyond numpy/scipy |
| run the search against the real xTB tier, or re-enumerate | the xtb binary |
| rebuild a cloud from the structures | the pinned environment (all five pins) |
| run the reference-level steps, or their tests | ORCA 6.1.1 (the tests patch the launch, so they need the binary installed but do no DFT) |

## Data

`data/` carries the results of the expensive stages and the draws that cannot
be reproduced exactly. Three rules decided what shipped:

- **Results of expensive programs ship.** The ORCA outputs of the QM chain
  (`.out`, their inputs, geometries and Hessians), the three enumerated
  landscapes, and the reference-level records. Wavefunction checkpoints
  (`.gbw`), density grids and property files are restart scratch and do not.
- **Non-reproducible draws ship.** The candidate clouds are drawn by a
  sampler that seeds itself internally and exposes no seed handle; a fresh
  process reproduces a draw to about 1e-12 Å, and a fresh realization is
  obtained by advancing that stream (the "roll": canonical is roll 0,
  replicate *k* is roll *k*). The clouds are therefore recorded artifacts,
  locked by hash, not regenerable files. `docs/cvt_seed_probe.py` re-measures
  the behaviour on your machine.
- **Deterministic runs do not ship.** The search is seeded (PCG64), so the 600
  battery records regenerate from the landscape in minutes. Their digest — the
  accounting counters and final answer of every run — is published as
  `data/results/battery/battery_runs_index.json`, and `experiments/screen.py
  validate` checks new runs against it line for line. The one reported
  statistic that needed a matching rule rather than a lookup — the
  cross-cloud rank correlation — is computed by
  `experiments/matched_correlation.py` (ρ = 0.9851 and 0.9854).

Sizes: about 120 MB of data, dominated by the three landscapes (34 MB each)
and the QM chain's outputs (16 MB). The arbitration records add 13 MB.

## Tests

Three suites, each a script with a `main()`; run them with the pinned
environment.

| Suite | Covers | Notes |
|---|---|---|
| `ga/test_ga.py` | the search: the row-arithmetic master gate against the enumerated universe, the planted-basin recovery bar, accounting, reproducibility, the exclusion fence, the descent bookkeeping, negative gates | 11 tests; the two live-xTB tests skip when the binary is absent |
| `arbitration/test_arbitration.py` | the reference tier: landscape gating and decode, the step engine's semantics, the stratified probe sampler and its pool gate, the rank correlation | runs against the published landscapes with the ORCA launch patched — no DFT is performed |
| `exhaustive_search/test_enumerate.py` | the ground truth: universe counts, row mapping, shard split, kill/resume, accounting, the separation floor, domain exclusions, coverage, value placement | the cloud-identity, promotion and cross-machine band tests of the original tree are not included (they check this project's own registry) |

`fitness/test_xtb.py` carries the empirical pinning that makes the search tier
valid — the Ångström unit check and the hardness convention — and needs the
xtb binary for its live gates.

## Provenance

Every recorded artifact carries a sha256 sidecar, and the summaries record the
hashes they were built from. The battery and parameter-study designs were
frozen **before** the reported runs; the two frozen documents are in `docs/`
(`preregistration.md`, `scoping-addendum.md`), unedited.

One provenance detail deserves stating rather than hiding. The preregistration
hashes to `dda94b58…`, which is the value recorded in every result summary. The
scoping addendum hashes to `172a7095…`, the document's *frozen* sha256, while
the battery records bind `28eb464a…` — the pre-freeze candidate that was
presented for freezing, whose only difference is the §(c) citation line, updated
in the same round for the snapshot manifest's environment-links note. Both
numbers appear in the project's registry; `docs/GLOSSARY.md` explains which is
which, and the records were left untouched because they are the frozen outputs
of the run.

**Division of labour, stated plainly.** The implementation was written by a
coding agent working from the author's specification, in stages, each landing
only after its own tests passed; every stage was then verified independently
from its returned artifacts by a separate review pass that re-derived the
numbers and re-ran the searches (the verification notes are summarised in
`docs/CHANGES.md`). The scientific design, the decisions, and the manuscript
are the author's.

## Licence

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`DATA_LICENSE.md`).
