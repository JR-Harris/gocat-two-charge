# Scoping addendum — the item-6 battery (stage-2 freeze)

*Status: FROZEN candidate — 2026-10-19. This is the stage-2 addendum of
the frozen pre-registration (`docs/2026-10-19_preregistration.md`,
§0/§7): its contents are (a) the frozen `param_set` for both arms;
(b) the frozen `G_battery`; (c) the resolved snapshot manifest citation;
(d) the locked Python environment spec; (e) the resolved item-4
live-record citation. Chosen by the pre-declared scoping criteria — no
tuning happened after the criteria were declared; the scoping ran
UNREPORTED. Once Jonny's freeze is registered (`04` §12 + its sha
below), no edit of this document; corrections as dated registry notes.*

## (a) The frozen param_set (both arms)

| Field | Value | Source |
|---|---|---|
| pop_size | 50 | criterion 7b winner |
| tournament_k | 3 | held at the landed value (sweep not swept, per the pre-registration) |
| pc | 0.9 | criterion 7b winner |
| pm | 0.2 | criterion 7b winner |
| nn_k | 8 | frozen at 8 (params.py assertion) |
| n_elite | 2 | criterion 7b winner |
| acceptance_margin | 0.0 | the landed value |
| use_descent | True (arm A) / **False (arm B)** | the pre-declared ablation, differing in this field ONLY |
| fence_rejection_cap | 1000 | the landed value |
| memo | ON | the landed semantics |

**Arm A = the landed engine defaults, validated by the sweep as the
criterion winner.** This is a scoping FINDING, not an assumption: the
sweep chose the already-frozen default set from 36 cells.

## (b) The frozen G_battery = 2500

Phase 7a (seeds 0–2, arm A at the defaults, G = 2000, replicate-1):
queries 98,990 / 99,670 / 99,478; qpg (ledger-derived) 49.0 / 49.4 /
49.4 — all three BELOW the ≥ 1.2 × 10⁵ margin target at G = 2000, so
the rule's second branch applies (records: `runs/scoping/replicate-1/
armA/probe7a/`, one JSON + sha sidecar per seed). Projection:
smallest G on the 100-grid with min-measured-qpg 49.0 × G ≥ 120,000
→ 2450 → **G_battery = 2500** (projected minimum 122,500 queries =
the 10⁵ checkpoint interior with the 20 % margin). The corner-case
contingency (prereg §7a, audit m2) did NOT fire: every seed at G = 2000
spends ~99k queries; 2500 generations projects ≥ 1.2 × 10⁵ with margin
in every seed.

**Scoping corollary, recorded as an observation (not a gating fact):**
at the defaults, the replicate-1 argmin was already reached (≤ 0.5) by
G = 7 / 40 / 75 in the three probe seeds at G = 2000. The battery's
low-G tail is therefore where the interesting statistics live — the
per-landscape success-vs-G curves (the primary output) resolve it;
no change, just a caution against reading the 10⁵-query end alone.

## (c) The resolved snapshot manifest

`battery/docs/snapshot_manifest.json`, sha256
`6f65bba97ea2f052a3179f92d23f06615844e27068564ec636d3def18f5beed2`
(amended 2026-10-19 evening during the freeze round: ONE note added —
the post-snapshot environment links `vendor/bin` + `vendor/data`
(relative symlinks to the same GOCAT content, needed because
fitness/xtb.py asserts its binary at import time even on the oracle
path; the 11 engine files and their shas are UNCHANGED) — the prior
hash was `0089038df2af436b6492793073d853024fadc12421a33faa8223f67d
3509ee98`) —
source `GOCAT_Program` @ `09f967d29bfe9e5ab25793a2fe3d8b8b248c14de`
(clean tree; 11 files, sha256 set verified identical at copy time;
battery repo commits 8a66b12 → 11a2e49 → 98d52ab; engine files
unchanged since the item-4 landing commits 0bc48c9/0da92e1/4dbfdd0/
926f465/0276b1d/c1f65d4). **Snapshot suite: 11/11 PASS on the
snapshot** (2026-10-19, this VM's sanctioned pattern: test 08a's
live-repeat determinism RAN bitwise with the real xtb binary — 22
configs, 2 fresh providers; test 08b env-skipped as designed — the
identical pattern Claude's item-4 verification recorded; deliverables
+ stdout preserved in `runs/suite_evidence_2026-10-19*`, sha sidecars
on the run JSONs).

## (d) The locked Python environment spec (the REPORTED runs' env)

The battery's reported runs execute in the project's pinned venv per
the standing recipe (`PROJECT_STATE.md` "Environment notes" +
`02` §11): **CPython 3.13** + the locked `GOCAT_Program/requirements.txt`
(sha256
`0ee750294606ac511cc46024f6861efa606771bb538eb5bbd2fc0ca194bce9af`;
relevant members: numpy 2.5.1 pinned; scipy/trimesh/pcu/manifold3d/
tblite present but unused by the oracle-mode battery path — the
provider imports none of them; the xtb 6.7.1 binary at
`bin/xtb-6.7.1/bin/xtb`, binary sha256
`debf27a9e0fa...eaa12a` per `04` §9b/§9 — used only by live mode).

**Honest deviation note (recorded, not hidden):** the SCOPING ran in
this session's execution VM — CPython 3.10.12 + numpy 2.2.6 (suite
green there; engine arithmetic deterministic within the env; the
bitwise-reproducibility guarantee is per-environment). The scoping's
decisions are env-robust to this: the criterion winner differs from
its 10/10 runner-up by 10 of 108,742 queries (a nominal tie broken
only because the tie-break demands an order), and the winner IS the
landed default set — no knife-edge tuning rides on the deviation.
The REPORTED runs run in the pinned 3.13 env; the 600 runs'
bitwise-identical re-execution is guaranteed within that env by
gauge test 5, not across envs — stated plainly.

## (e) The resolved item-4 live-record citation (audit m7)

The ten live-evaluator demonstration restarts (G = 20, real xtb CLI
provider, memo ON) + the transfer pair, in `GOCAT_Program/runs/ga/`:
`g000020_s00011_xtb_084e48a6b526.json` …
`g000020_s00020_xtb_084e48a6b526.json` (seeds 11–20; per-file sha256s
in their `.sha256` sidecars — e.g. s00011
`f4568669757034f1439820bd90f74962b11ce90b5f9a85c60525531c55ddb6f8`;
verify at use), and the memo-OFF transfer record
`runs/ga/transfer_memo_off/g000020_s00011_xtb_084e48a6b526.json`
(sha256 `8d50ca3eae091c937331578c55c061dd1dcfdba3f4b57893d8bf13d79f4f33a9`;
its trajectory + queries are BITWISE-identical to the memo-ON record —
the landed item-4 transfer demonstration, 17/17-gate verified
2026-10-19). The battery binds these by reference; the vendored
snapshot is byte-identical to the code that produced them (manifest),
so no re-run is owed.

## The scoping evidence — the full per-cell table (criterion 7b's input)

36 cells × 10 seeds, G = 200, replicate-1, arm A; hit = final ledger
best ≤ argmin + 0.5. Sorted by the pre-declared ranking
(hits ↓, qsum ↑, lexicographic). Records: `runs/scoping/replicate-1/
armA/c_*` (360 JSONs + sidecars).

| Cell (pc, pm, pop, elite) | Hits | qsum | qmean |
|---|---|---|---|
| **0.9, 0.2, 50, 2 — WINNER** | **10/10** | **108 742** | 10 874 |
| 0.9, 0.3, 50, 2 — runner-up | 10/10 | 108 752 | 10 875 |
| 0.7, 0.2, 100, 2 | 10/10 | 207 520 | 20 752 |
| 0.9, 0.1, 100, 2 | 10/10 | 208 250 | 20 825 |
| 0.7, 0.3, 100, 2 | 10/10 | 208 542 | 20 854 |
| 0.9, 0.3, 100, 2 | 10/10 | 208 556 | 20 856 |
| 0.9, 0.2, 100, 4 | 10/10 | 215 244 | 21 524 |
| 0.9, 0.3, 100, 4 | 10/10 | 216 460 | 21 646 |
| 0.7, 0.3, 100, 4 | 10/10 | 217 368 | 21 737 |
| 0.7, 0.2, 50, 4 | 9/10 | 115 790 | 11 579 |
| 0.7, 0.1, 100, 2 | 9/10 | 206 528 | 20 653 |
| 0.9, 0.2, 100, 2 | 9/10 | 208 694 | 20 869 |
| 0.9, 0.1, 100, 4 | 9/10 | 213 310 | 21 331 |
| 0.7, 0.1, 100, 4 | 9/10 | 214 384 | 21 438 |
| 0.7, 0.3, 30, 4 | 8/10 | 75 552 | 7 555 |
| 0.7, 0.2, 50, 2 | 8/10 | 107 362 | 10 736 |
| 0.7, 0.3, 50, 2 | 8/10 | 108 472 | 10 847 |
| 0.9, 0.1, 50, 2 | 8/10 | 108 724 | 10 872 |
| 0.7, 0.1, 50, 4 | 8/10 | 112 326 | 11 233 |
| 0.7, 0.3, 50, 4 | 8/10 | 113 752 | 11 375 |
| 0.9, 0.3, 50, 4 | 8/10 | 114 174 | 11 417 |
| 0.9, 0.2, 50, 4 | 8/10 | 115 022 | 11 502 |
| 0.7, 0.2, 30, 4 | 7/10 | 73 748 | 7 375 |
| 0.9, 0.3, 30, 4 | 7/10 | 75 544 | 7 554 |
| 0.7, 0.1, 50, 2 | 7/10 | 106 494 | 10 649 |
| 0.9, 0.1, 50, 4 | 7/10 | 113 834 | 11 383 |
| 0.7, 0.2, 100, 4 | 7/10 | 214 554 | 21 455 |
| 0.9, 0.2, 30, 2 | 6/10 | 66 222 | 6 622 |
| 0.9, 0.1, 30, 2 | 6/10 | 67 288 | 6 729 |
| 0.9, 0.3, 30, 2 | 6/10 | 68 238 | 6 824 |
| 0.7, 0.2, 30, 2 | 5/10 | 67 452 | 6 745 |
| 0.9, 0.2, 30, 4 | 5/10 | 73 758 | 7 376 |
| 0.9, 0.1, 30, 4 | 5/10 | 75 244 | 7 524 |
| 0.7, 0.3, 30, 2 | 4/10 | 66 024 | 6 602 |
| 0.7, 0.1, 30, 2 | 4/10 | 67 116 | 6 712 |
| 0.7, 0.1, 30, 4 | 3/10 | 72 916 | 7 292 |

**Zero-hit cells: none** (the pre-registered stop-surface did not
fire). Recorded structure (observations, unreported): pop 50–100
dominates; pop 30 is systematically weaker at every (pc, pm) — the
population-pressure axis carries the sweep's discrimination; elite
count 2 ≥ 4 broadly; the pop-30 cells' cost halving does not buy hits
back.

## Provenance of this addendum

- Frozen prereg: `docs/2026-10-19_preregistration.md`
  (`04` §12, sha `dda94b58…a85292`, never edited post-freeze).
- Harness: `battery/harness_scoping.py` (battery-owned; battery repo
  commit at execution: `ed53f02`; byte-verified into the execution
  scratch at 11/11-vs-manifest; the sweep's second half re-executed
  in a fresh scratch window and agreed BITWISE on all 201 overlapping
  record identities — a free re-determinism proof).
- Scoping records: `battery/runs/scoping/replicate-1/armA/` (365
  JSONs: 360 sweep + 3 probe + 2 determinism checks, sidecars).
- Next, per the frozen prereg: the reported runs (600 = 3 landscapes ×
  2 arms × 100 seeds, G = 2500, seeds 0–99) execute under THIS
  addendum + the prereg's §8 statistics; their artifacts + summaries
  register in `04` at landing.