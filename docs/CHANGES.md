# What is in this repository, what is not, and what was edited

This repository is a curated subset of the study's working program. The rule
was: everything that produced a reported number, nothing that exists only to
service the project's own bookkeeping. This file is the complete record, so a
reader never has to guess whether a file is the one that ran.

## Included

Most of these files are copied verbatim; the ones that are not are named
individually in the *Edited* section below, which is the authoritative list.
This table is an inventory of what is present and what it does.

| Path | Lines | Role |
|---|---:|---|
| `ga/` (ga, space, params, oracle_provider) | 1,378 | the search: engine, space arithmetic, frozen operators, the landscape oracle |
| `fitness/` (contract, stub, xtb, orca) | 1,763 | the evaluation interface and its three providers |
| `exhaustive_search/` (enumerate, worker, proofs) | 1,481 | the landscape: universe arithmetic, resumable shards, the provider proofs |
| `alignment/grips.py` | 423 | registration of the reactant and product into the transition-state frame |
| `cloud/` (machinery, build, replicate) | 2,198 | the path-union surface, CVT sampling, the three clouds |
| `arbitration/` (landscape, probe, steps, heartbeat) | 1,626 | landscape gating, the stratified probe, the reference-tier steps |
| `qm/` (structures, inputs, orca_runner, orca_out, ts_chain, irc_chain, escape) | 2,261 | atom mapping, ORCA inputs, parser, transition state, IRC |
| `run_qm.py`, `run_alignment.py`, `run_cloud.py`, `run_replicate.py`, `run_exhaustive_search.py`, `run_ga.py`, `run_arbitration.py` | 1,680 | one entry point per stage |
| `ga/test_ga.py`, `arbitration/test_arbitration.py`, `exhaustive_search/test_enumerate.py`, `fitness/test_xtb.py`, `fitness/test_orca.py` | 4,012 | the suites described in the README |

## Edited

Sixteen files differ from the working tree. Each change is listed with its
reason; none changes a computed value. (The two frozen documents are listed
last only to record that they were left alone.)

**Load-bearing (the repository would not run without them):**

1. **`fitness/orca.py`, `fitness/xtb.py`** — the module-level provider binding
   was made lazy (PEP 562), and the two static contract names (`FITNESS_ID`,
   `PROVIDER_META`, both class attributes) are bound directly from the class.
   Constructing a provider checks that ORCA / xtb is installed, so the eager
   binding made *every* import of the package fail on a machine without the
   binaries — including the landscape gates, the probe sampler and their tests,
   which never call it. The first use of `eval_batch`/`eval_batch_full` still
   constructs the provider and still fails hard.

2. **`fitness/test_xtb.py`** — three tests (`test_02`, `test_05`, `test_07`)
   construct the real provider and so need the xtb binary; they now skip with a
   message when it is absent, the same convention as the live smoke test. On a
   binary-less machine the suite reports 9/9 (five pass, four skip) rather than
   failing five.

3. **`arbitration/test_arbitration.py`** — `test_09` pinned the exact wording
   of the unknown-landscape refusal ("BLOCKED"). That message was reworded
   directly below (edit 4), so the pin now reads "not enabled".
   The assertion's intent — an unknown name is refused by the fence — is
   unchanged.

4. **`arbitration/steps.py`** — `from arbitration.heartbeat import Heartbeat`
   is a hard import; `heartbeat.py` is therefore included (see below), and the
   runtime message for an unknown landscape no longer reads as a pending
   permission ("execution is not enabled for this name").

**Packaging (import paths, roots, entry points):**

5. **`experiments/battery.py`** (from the project's `harness_battery.py`) —
   import path and data roots repointed to this repository; output to
   `runs/battery/`; the misspelled constant `ADDEDUM_SHA` renamed (its value is
   untouched — it is recorded inside the shipped records); a redundant
   `.__str__()` removed; the per-ε progress line now prints the real
   denominator; and `group_setup()` now checks the pre-registered argmin
   against each landscape's own summary before scoring anything.

6. **`experiments/cells.py`** (from `harness_scoping.py`) — the same
   repointing, plus absolute landscape paths instead of paths relative to a
   working directory, and the usage block renamed.

7. **`experiments/screen.py`** (from the parameter-study driver) — the same
   repointing; the validation command checks new runs against
   `data/results/battery/battery_runs_index.json` instead of the full battery
   records, which are not published; usage block renamed.

8. **`run_qm.py`** — rewritten as a three-command driver (`--execute
   endpoints|ts|irc`). The original also carried `--parse`, `--gates` and
   `--promote`, each of which checks against the project's frozen target
   numbers; those modules are not published. The stage functions it calls are
   unchanged.

9. **`run_arbitration.py`** — the docstring and the `--landscape` help text
   announced that replicate execution was "blocked pending a go-ahead", and the
   step list still marked step C as blocked. Both were stale: all three
   landscapes were arbitrated and step C ran. Corrected.

10. **`run_replicate.py`** — a stray closing quote in the module docstring.

**Text (stale statements, typos, naming):**

11. **`ga/params.py`, `run_ga.py`** — both described the registered parameter
    set as "plumbing only — none of its numbers is a reported result". True
    when written; false since the pre-declared scoping sweep selected exactly
    those values as the frozen set. Both now say so.

12. **`alignment/grips.py`, `cloud/build.py`** — runtime messages that named
    the author by first name ("widening is Jonny's decision") now say "the
    author's decision". Docstrings elsewhere keep the names: they are the
    decision record (see the glossary).

13. **`fitness/xtb.py`** — a missing space in an init-gate message
    ("registeredreactant").

14. **`experiments/battery.py`** — a comment recording why the three
    statistics are written out rather than imported from scipy (the runner is
    kept scipy-free so the reported battery regenerates in a numpy-only
    environment).

15. **`data/results/battery/battery_runs_index.json`** — the `source` field
    pointed at the working project's directory layout; it now names the
    regeneration path in this repository.

16. **`arbitration/probe.py`** — the step-C belt-gate carries a "discrepancy"
    note recording how the spread stratum's pool count was resolved. It named
    the author; it now says "flagged to the author". **Consequence, recorded
    deliberately:** a regenerated step-C summary will differ from the shipped
    one in that single field. The shipped record keeps the original string —
    it is a frozen output and is not rewritten. The pool-count gate itself is
    unaffected (its test asserts on the "288,520" count, not this note).

17. **`docs/preregistration.md`, `docs/scoping-addendum.md`** — *not* edited.
    Their hashes are discussed below.

## Added

| Path | Why |
|---|---|
| `arbitration/heartbeat.py` | imported by `steps.py`; without it stage 8 cannot be imported at all |
| `data/derived/mini_cloud_summary.json` | step C reads it to gate its stratification rules; without it the probe cannot run |
| `exhaustive_search/proof_reference/` (about 10 KB) | the shipped reference that `run_exhaustive_search.py --smoke` diffs against |
| `experiments/matched_correlation.py` + `data/results/matched_correlation.json` | the cross-cloud matched-configuration correlation (ρ = 0.9851 / 0.9854) — previously computed, never shipped as a script |
| `docs/GLOSSARY.md` | the internal vocabulary, translated |
| `README.md`, `LICENSE`, `DATA_LICENSE.md`, `requirements.txt`, `.gitignore` | written for this repository |

## Deliberately excluded

| Path in the working tree | Why it is not here |
|---|---|
| `qm/gates.py`, `qm/ts_gates.py`, `qm/summaries.py`, `qm/promotion.py` | ~1,950 lines of registry bookkeeping: they compare returned outputs against this project's frozen numbers and hashes, which a reader does not hold |
| `hpc/` (deploy zip, PBS wrappers, cluster manager) | the enumeration ran on 16 workers of one cluster node; nothing in the method depends on the scheduler |
| `arbitration/chain.py` | the unattended overnight runner; `run_arbitration.py` drives the same steps one at a time (its watchdog, `heartbeat.py`, *is* included because `steps.py` imports it) |
| `run_arbitration_pricing.py`, `run_arbitration_launchplan.py`, `run_arbitration_cert_close.py`, `run_xtb_benchmark.py` | run planning and measurement scripts; their outputs are quoted in the results, their machinery is not part of the method |
| the project's remaining test suites | they test the excluded bookkeeping, or require the cluster layout |
| analysis scripts for analyses not reported in the manuscript | out of scope for this repository |
| `venv/`, `runs/` except `runs/arbitration/`, `data/hpc_out/`, `data/promotion_fixture/`, `scratch/`, deploy archives | working state |

### References that remain dangling, and why

Some shipped files still point at material from the working project. Each is
either regenerable or irrelevant to a claim:

| Reference | Where | Status |
|---|---|---|
| `runs/arbitration/<landscape>/diagnosis/`, `pricing/` | cited in `steps.py`'s comments | measurement records for sizing the run, not results; not shipped |
| `battery/docs/snapshot_manifest.json`, `battery/runs/scoping/` | cited by the addendum (§c, §e) | the vendoring and scoping records of the working tree |
| `GOCAT_Program/runs/ga/g000020_s00011…` | the addendum's live-record citation | the live-evaluator demonstration; the records are not shipped, and the demonstration is reproducible with `run_ga.py --mode xtb` where xtb is installed |
| `research_docs/xtb_benchmark.md`, `runs/xtb_pcem_units/` | cited by `fitness/xtb.py`'s conventions | the underlying measurements are embedded in `fitness/test_xtb.py`'s `UNIT_GATE` block, which re-derives the unit and hardness conventions from the recorded energies |
| `git_head` fields in run records | every `ga_run_v2` record | they name commits in the working project's history; this repository starts with a single commit, so the field is a record of the source tree, not a link |

## Data: what ships and what does not

Shipped: the QM chain's ORCA outputs, inputs, geometries and Hessians; the three
candidate clouds with their summaries, manifests and comparisons; the three
enumerated landscapes (`f`, `e_ts`, `e_r`, the index arrays, the summaries, the
domain-exclusion records); the reference-tier records for all three landscapes;
the battery's summaries, seed ledgers and aggregation; the parameter study's
cell summaries and analysis; a digest of the 600 battery runs; the matched-cloud
correlation.

Not shipped: ORCA restart scratch (`.gbw`, `.densities`, `.property.txt` —
78 MB of the QM chain's 101 MB is this); the 600 full battery records (81 MB,
regenerated deterministically); the parameter study's 4,800 full records
(639 MB, likewise); the superseded v1 ORCA archive.

## Paths inside the shipped records

Seven JSON records carry absolute paths from the machine that produced them —
the enumeration summary's shard ledger, the two replicate cloud summaries, and
the canonical chain's certificate and step-F records. They are left exactly as
written: each is a provenance field recording *where* a run executed, and
rewriting it would change the file's sha256, which is the value the gates and
the registry verify. Read them as a record of the execution, not as links.

## A note on the frozen documents

`docs/preregistration.md` hashes to `dda94b58…a85292`, the value every result
summary records. `docs/scoping-addendum.md` hashes to `172a7095…` — the
document as frozen — while the battery records bind `28eb464a…`, the pre-freeze
candidate that was presented for freezing. The two differ only in the §(c)
citation line, which was updated in the same round for the snapshot manifest's
environment-links note. The records were left exactly as produced; the README
states both numbers, and no claim depends on their being identical.

## A note on the docstrings

Module docstrings still speak the project's internal language — item numbers,
ruling dates, "the briefing", "the registry". They were left alone
deliberately: they are the record of why each design decision is what it is,
and rewriting them risks quietly changing a claim. Read them as the lab
notebook that accompanies the code; `docs/GLOSSARY.md` is the translation.
