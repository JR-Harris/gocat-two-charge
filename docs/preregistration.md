# Pre-registration — the item-6 statistics battery (GA vs brute-force)

*Status: **DRAFT v2** (2026-10-19) — this revision incorporates the
independent pre-freeze audit **REVIEW-GOCAT-BATTERY-2026-10-19-01**
(verdict: continue-with-changes; 0 Critical / 3 Major / 7 Minor; the
report + artifacts + FIXLIST live in `thesis/audits/`; addressed per
§10 below). Origin of the draft: 2026-10-19, Claude, from
`02_method.md` §6/§10, `03_plan.md` item 5/6, the GA briefing rev 3
(spec of record), and `04_numbers.md` §8/§8b. Freezes at Jonny's
approval: the approved text is finalized, its sha256 recorded, and it
is registered in `04` BEFORE any reported run. The scoping phase below
(§7) is declared NOW and its outcome freezes as a hashed addendum;
reported runs start only after the addendum is frozen. Nothing in this
document may be edited after the freeze — later corrections land as
dated registry notes, never as edits.*

## 0. The two-stage freeze

1. **Stage 1 — this document, frozen at Jonny's approval.** Everything
   except §7's outcomes.
2. **The vendor step** — the snapshot lands, the manifest verifies
   against the item-4-verified tree, and the battery's own suite
   passes on the snapshot (§4). This is an EXPLICIT PREREQUISITE for
   the scoping: nothing runs, scoping included, before it.
3. **Scoping** (declared in this document, §7): unreported,
   budget-capped tuning runs only.
4. **Stage 2 — the scoping addendum**, one page, hashed and
   registered: (a) the frozen `param_set` for both arms; (b) the
   frozen `G_battery` (chosen by the pre-declared rule); (c) THE
   RESOLVED SNAPSHOT MANIFEST (the source commit + per-file sha256s —
   the code-environment + engine-provenance anchor, whose contents are
   future at stage-1 freeze time and bind here); (d) THE LOCKED PYTHON
   ENVIRONMENT SPEC (the pinned interpreter version + the locked
   requirements.txt content + its sha256 — per `02` §11's software
   table); (e) the resolved item-4 live-record citation (§6, m7).
   Reported runs start only after this addendum exists.
5. Post-freeze discipline: any later correction to a frozen quantity
   is a dated note in `04` (+ PROJECT_STATE), never an edit of the
   frozen documents. A change that would alter the battery's reported
   claims is treated as a re-design — Jonny's decision, recorded.

**The audit ordering, stated explicitly:** stage-1 freeze → the vendor
lands + the manifest verifies + the battery suite passes (the scoping
prerequisite) → scoping executes → the stage-2 addendum freezes
(param_set, G_battery, manifest, env spec, citation) → reported runs.

## 1. Purpose and scope

The thesis-critical methods test: **can the GA reproduce the
brute-force answer consistently, and faster, consistently — over
independent landscape realizations?** The battery runs per landscape in
isolation (canonical + the two replicates); AGGREGATION happens here —
at item 6 only — as the cross-landscape consistency statement (§8).
The canonical landscape's DFT-arbitrated optimum (work item 7) remains
the interpretive anchor; the battery never reinterprets any final
answer.

Both searchers use the one frozen fitness view (§2), identical per
landscape. Brute force = the exhaustive oracle enumeration itself
(§5): its answer (the argmin) and its cost (the full query count) are
already-landed artifacts, read from the registry.

## 2. The fitness view — frozen

**f as defined** (`02` §4.1: the differential TS–R barrier,
f = E(TS;q) − E(R;q), kcal/mol, LOWER = BETTER), frozen for the battery
by Jonny's 2026-10-12 decision; both searchers use it. The alternative
views (grip-valid subset, ΔE‡_eff) are report analysis, not pipeline.
One evaluation = one fitness query, in the frozen `02` §6 accounting
(duplicates and memo hits included in `queries`; `submitted` ==
provider n_evals per run; `unique` recorded; "fitness queries" is the
only allowed wording).

## 3. The landscapes and the success targets

Three landscapes, each processed in isolation. Per-landscape success
target = **that landscape's own oracle argmin** (the xTB-tier float64
record value, the authority — bitwise-verified by the 2026-10-19
verification pass; `04` §8/§8b). NaN rows (the pre-registered
SCC domain exclusions, one per landscape) are outside the fitness's
domain of definition — resolved as the FENCE from each landscape's own
`domain_exclusions.json` (fence data only; the engine never sees
fitness values through it; the GA never samples or submits a fenced
config).

| Landscape | Artifact root | Argmin f (kcal/mol, float64 record) | Row | Config | d_min-separated | Evaluated rows (BF cost) | f.npy sha256 |
|---|---|---|---|---|---|---|---|
| canonical | `GOCAT_Program/data/derived/enumeration_v2/` | **−125.21463064719501** | 1,190,559 | (418, 419, −, −) | — | 1,699,131 | `85b5a6c0…b66282` (§8) |
| replicate-1 (roll 1) | `GOCAT_Program/data/replicates/replicate-1/derived/enumeration_replicate-1/` | **−125.46892076041331** | 1,233,787 | (440, 453, −, −) | 1.047 Å | 1,699,171 | `a2102792…a22031` (§8b) |
| replicate-2 (roll 2) | `GOCAT_Program/data/replicates/replicate-2/derived/enumeration_replicate-2/` | **−125.52238044298429** | 1,141,875 | (394, 397, −, −) | 1.046 Å | 1,699,139 | `874ecc85…ac9688` (§8b) |

Full artifact sets (f, e_ts, e_r, pair_indices, sign_index,
enumeration_summary, domain_exclusions) + their sha256s: the tables in
`04` §8 / §8b (the registry's per-file hash blocks are the binding
copy). The provider's own init gates re-assert all of them at run time
(cloud sha vs the summary's record; full-length pairwise row alignment
f≠NaN; NaN mask exactly at the excluded rows).

The three argmin values are FIXED at this freeze. They are never
re-derived, re-rounded, or replaced; per-landscape results are stated
against exactly these constants.

## 4. The searcher — the vendored engine

The GA WILL BE the item-4-verified engine (17/17 gates, 2026-10-19),
**vendored** into `battery/vendor/` as a sha-verified frozen snapshot
per the 2026-10-19 home + vendoring rules (`03_plan` item-6 block;
`battery/README.md`). At freeze time the snapshot does not yet exist
(`battery/vendor/` is empty); the pre-registered requirements it must
satisfy are: (a) the snapshot manifest records the `GOCAT_Program`
source commit + per-file sha256s, verified AT SNAPSHOT TIME against
the item-4-verified tree — the manifest is the code-environment +
engine-provenance anchor of record; (b) nothing runs (scoping
included) until the manifest verifies and the battery's own test
suite passes on the snapshot; (c) the resolved manifest + the locked
Python environment spec enter the stage-2 addendum (§0), where the
citation binds. The engine stays a BLACK BOX as landed: it never reads a
landscape file except through the fitness callable (the oracle provider
serving `eval_batch`); it never imports `exhaustive_search`; a FRESH
provider instance per run. Failure semantics unchanged: a fenced/NaN
config is never sampled or submitted; a provider SCC failure is never
retried, never penalised (fail-hard + status: failed record + nonzero
exit).

Engine facts the battery relies on (unchanged from the landed code):

- One `numpy.random.Generator(PCG64(seed))` per run, documented draw
  order; identical (landscape, params, seed, G) → bitwise-identical
  result record (test 5, landed).
- The mandatory generation ledger: (generation, cumulative queries,
  best_f) and the best-so-far trajectory at the fixed evaluation
  checkpoints [100, 300, 1000, 3000, 10,000, then every 10,000].
- Accounting fields queries / submitted / memo_hits / unique asserted
  in every result (test 4).

## 5. The brute-force side — fixed

BF = the exhaustive oracle enumeration (work item 3 + 5). Per
landscape: answer = the argmin (§3, fixed); cost = the full evaluated
row count (1,699,131 / 1,699,171 / 1,699,139 fitness queries — every
distinct admissible config evaluated once; d_min-excluded configs are
excluded-and-recorded, never evaluated). The comparison is in FITNESS
QUERIES (the only currency). There is no second BF implementation and
no re-run of the enumeration.

## 6. The battery design

**Stop rule — GENERATIONAL, ruled by Jonny 2026-10-19:** the GA runs
to its generation limit and never stops on success. The report-facing
axis is GENERATIONS: "at G generations, the GA's success fraction is
p; 95% confidence requires y generations, 99% requires z". Best-so-far
is recorded at every generation (the mandatory ledger), so
generations-to-correct-answer IS recorded faithfully as a descriptive
statistic — but it is NOT a stopping rule and never influences a run.

**Success:** best-so-far ≤ landscape argmin + ε, with
**ε ∈ {0.25, 0.5, 1.0} kcal/mol — ALL THREE reported** (no
cherry-picking; the ≈×5.4 rate calibration sits behind the 1 kcal/mol
band). Success is evaluated at every generation index from the ledger
(= the success-vs-G curve) AND at the fixed evaluation checkpoint grid
(= the success-vs-queries curve, the GA-vs-BF cost statement).

**Arms (2, pre-declared):**

- **Arm A (main):** use_descent = True (the memetic GA as designed).
- **Arm B (ablation):** use_descent = False — identical in every
  other respect. It exists to measure the memetic layer's contribution
  per landscape; its report role is the ch. 6 descent discussion.

**Operator set:** frozen by the two-stage freeze ONLY — chosen in the
scoping phase (§7) and never tuned afterwards. The smoke defaults are
NOT assumed; they are only the scoping baseline. Any default change
lands via the committed-edit + coding-log rule.

**Restarts:** **100 seeds per (landscape, arm)** — seeds s = 0…99,
PCG64(s), ≥100 per `02` §10. 3 landscapes × 2 arms × 100 seeds =
**600 oracle-mode runs** (table-priced; wall trivial). Seed reuse
across arms/landscapes is intentional (paired seeds aid comparison).
All runs use `G_battery` (§7's frozen value); the budgets are read off
each run's checkpoint-trajectory and generation-ledger.

**Budget coverage — the two grids, stated apart (audit M2):** the
ANALYSIS grid is {10², 3×10², 10³, 3×10³, 10⁴, 10⁵} (`02` §10);
the engine's mandatory ledger/trajectory SAMPLING is [100, 300, 1000,
3000, 10⁴, then every 10⁴] — they share the entries through 10⁴ and
diverge above it, and no verbatim-equality claim is made. Resolution:
the success-vs-queries curve (§8 output 2) is read off the LEDGER's
samples — the union of the analysis grid and the ledger's per-10⁴
samples (2×10⁴, 3×10⁴, …, 10⁵); the 10⁵ analysis entry is the 10th
post-10⁴ ledger sample and is reported for a run iff that run's query
count reaches it. Headline reporting uses the analysis-grid entries;
the fully ledger-sampled values are carried in the archived summaries.
The success-vs-G curve (§8 output 1) is the report's PRIMARY curve,
read at every generation index from the ledgers. A run whose final
query count is below a grid entry reports that entry as
**budget-not-reached** for that run — counted and reported per
(landscape, arm), never dropped silently. The pre-declared
G_battery sizing rule (§7) is designed to make the 10⁵ checkpoint
interior (≥ 1.2 × 10⁵ queries with margin) to every run; if it is
not, that reporting is the finding.

**The live-evaluator demonstration (§6, `02` method):** the item-4
live-restart records (10 restarts through the real xTB CLI provider,
memoisation on; the memo-OFF transfer pair bitwise-identical in
trajectory + queries) stand as landed evidence and are bound BY
REFERENCE (the item-4 records' paths + sha256 citations, resolved at
the briefing). No re-run unless the vendored snapshot's provider
differs from the item-4-verified tree — in which case the transfer
demonstration re-executes against the snapshot. **Citation resolution
rule (audit m7):** the records' paths + sha256s are resolved at the Pi
briefing (post-vendor); until then the by-reference citation is a
declared placeholder, and the stage-2 addendum carries the RESOLVED
citation — that is where it binds.

## 7. The scoping phase (declared here; unreported)

**Scope:** tuning of the operator set + the `G_battery` sizing rule.
Budget-capped, pre-declared, and confined to ONE landscape —
**replicate-1** — so the canonical (the interpretive anchor) is never
the tuning ground and its reported battery numbers stay untuned.
Scoping numbers are entered in no reported statistic; they appear in
the report only as "a scoping sweep on one replicate landscape chose
the operator set" (+ the addendum's citation).

**Phase 7a — sizing probe (3 seeds, canonical excluded):** first
scoping act: seeds 0–2 on replicate-1, arm A, G = 2000: measures
queries-per-generation and confirms the G_battery rule's scaling
formula. If seeds 0–2 at G = 2000 all reach ≥ 1.2 × 10⁵ queries
(a 20 % margin over the 10⁵ checkpoint), **G_battery = 2000**;
otherwise G_battery = the smallest G on the 100-grid whose
projected minimum query count ≥ 1.2 × 10⁵, computed from the
measured queries-per-generation. **Corner-case contingency
(audit m2):** if NO G on the 100-grid projects ≥ 1.2 × 10⁵
at the measured queries-per-generation, the sizing phase STOPS and
surfaces — the target is unmeetable at the measured operational
profile, declared as a finding (never tuned away); G_battery and the
reported runs then await Jonny's decision, recorded as a registry
note.

**Phase 7b — the operator sweep:** grid, all on replicate-1, arm A,
G = 200, seeds 0–9 per cell (table-priced ~minutes):

| Grid axis | Values |
|---|---|
| pc | 0.7, 0.9 |
| pm | 0.1, 0.2, 0.3 |
| pop_size | 30, 50, 100 |
| n_elite | 2, 4 |

= 36 cells × 10 seeds = **360 scoping runs**. (tournament_k is not
swept: held at the landed 3 — its role is dominated by pop_size
pressure and doubling it inflates the grid for little information.)

**Freeze criterion 7b (pre-declared):** the cell with the HIGHEST
argmin-hit rate (best-so-far ≤ argmin + 0.5 at cap G = 200) summed
over its 10 seeds; tied cells broken by lower total queries, then
lexicographically by (pc, pm, pop_size, n_elite). The winner + the
runner-up's identity + the full per-cell table enter the addendum
(they are scoping evidence, citable as such). If the winner's hit
rate is 0 (no cell ever hits), the scoping phase STOPS and surfaces —
a real-engine G = 200 non-hit on a known argmin is a finding, not a
tuning failure to paper over.

**Phase 7c — the addendum:** G_battery (7a's rule result) +
`param_set` = 7b's winner, with tournament_k = 3, nn_k = 8,
acceptance_margin = 0.0, fence_rejection_cap = 1000, memo ON, and
arm B differing only in use_descent = False. Hashed + registered in
`04`; THEN the reported runs execute.

## 8. The reported battery — outputs and statistics

Per (landscape, arm), from the 100 recorded runs:

1. **Success-vs-G curves** at all three ε, with Wilson binomial
   95 % CIs (pre-declared). From the generation ledgers.
2. **Success-vs-queries curves** on the checkpoint grid, all three ε —
   the currency for the GA-vs-BF comparison; read off the LEDGER's
   samples per the §6 two-grid resolution (the analysis headline
   entries {10²…10⁵} carried in full; 10⁵ per run iff reached).
3. **Generations-to-correct-answer distribution** (mean, median, IQR,
   over successful seeds at each ε; the y/z quantile statements at
   95 %/99 % success levels, with the binomial uncertainty of the
   estimate stated — at n = 100 the 99 % level rests on ≈1 success;
   it is reported with that honesty, never smoothed).
4. **ERT-style statistics** (algorithm-behaviour-model form): at each
   ε and each checkpoint budget B,
   ERT(B) = (n_success·mean(T_s) + (n − n_success)·B) / (n_success)
   where T_s = queries-to-success per successful run, B = the
   checkpoint cap, n = 100. ERT is computed on the QUERIES axis (the
   BF comparison) and its generation-axis analogue is the mean
   successful T_g + G_battery·(unsuccessful share) form.
   **The n_success = 0 convention (audit M3, pre-declared):** for a
   (landscape, arm, ε, B) cell with zero successes within the cap,
   ERT(B) is UNDEFINED by this formula and is reported as **"≥ B"**
   — at zero successes inside B, the only justified statement is the
   lower bound; the cell also carries its success count so the table
   stays uniform and auditable. No interpolation, no extrapolation,
   no substitute point estimate is ever made at n_success = 0.
5. **The final-best per seed:** config identity + f at the run's end;
   the distribution of (final best − argmin); the fraction of seeds
   that LAND the argmin config exactly (identity) vs within each ε —
   the canonical-vs-ablation arms compared.
6. **The BF cost anchor:** the landscape's evaluated-row count (§3)
   against which ERT(queries) and queries-to-success are read —
   "faster, consistently" means: the GA's ERT (per ε, per landscape)
   against the full-enumeration cost, and the consistency of that
   ordering across ALL THREE landscapes.

**Analysis populations and denominators (audit m5, pre-declared):**
the denominator for every success statistic is n = 100 attempted runs
per (landscape, arm) — no exclusions from the denominator. A failed
run (per §4's semantics: a provider SCC non-convergence mid-run, or a
fence-rejection-cap saturation) is never a success — excluded from the
NUMERATOR only — and is reported separately: failed-run count +
failure-mode breakdown (SCC vs fence-saturation vs other) + a pointer
to each failed run's `status: failed` JSON. Budget-not-reached runs
(§6) are COMPLETED runs that spent fewer queries than a grid entry:
they remain in the denominator at every budget they spent past, and
each reports budget-not-reached for the entries it did not reach.

**The cross-arm (A vs B) analysis (audit m6, pre-declared):** PAIRED —
identical seeds across the two arms. Per-seed deltas in
generations-to-success (T_g) and in queries-to-success (T_s) at each
ε; the paired Wilcoxon signed-rank test is the inferential anchor, the
sign test the small-sample fallback (pre-declared condition: when
fewer than 20 paired seeds both succeed at the (ε, budget) point —
e.g. the strict ε = 0.25 cells); per-seed delta distributions are
reported descriptively regardless of the test. The per-arm summaries
(outputs 1–6) stand independently per landscape; the ablation's report
role is the ch. 6 descent discussion, and a no-difference result is a
finding there too.

**Cross-landscape aggregation (the methods test's headline; the only
aggregation in the thesis):** per landscape and arm, the success-vs-G
curves and the ε-band statements are compared; the consistency claim
("the GA reproduces the BF answer consistently, and faster,
consistently") is stated as: (i) at a common (G, ε), the success
fractions agree within the binomial overlap across all three
landscapes; (ii) the faster claim holds per landscape (ERT below the
BF cost) for the same (G, ε). **The binomial-overlap threshold
(audit m4, pre-declared): UNADJUSTED 95 % Wilson CIs** — pairwise CI
overlap per cell, no multiplicity correction; the three-landscape
aggregation is exploratory in character, and the inferential anchor is
the item-5 guardrail (consistent across BOTH replicates = finding;
single-replicate deviation = observation). **The headline anchor
(audit m3, pre-specified):** the methods-test claim's headline (G, ε)
cell is **(G = G_battery, ε = 1.0 kcal/mol)** — the budget-cap end,
the consistency-favourable band — and the headline per-cell success
fraction is reported at **ε = 0.25 kcal/mol** (the strictest band);
both anchors are fixed now, and every per-(G, ε) table still reports
ALL THREE ε regardless. Consistency ACROSS BOTH replicates is
the finding pattern (the item-5 guardrail: a difference consistent in
both replicates is a finding; a single-replicate deviation is an
observation).

**Artifacts (the provenance pattern, laptop-side):**

- `battery/runs/<landscape>/<arm>/` — per-run `ga_run_v2` JSONs
  (+ sha256 sidecars) via `--out-root` (the runner never writes into
  `data/derived/`); the vendored engine runs HERE, not in
  `GOCAT_Program/`.
- `battery/runs/seed_ledger_<landscape>_<arm>.json` — (seed → run
  file, sha, queries, submitted, memo_hits, unique, final best_f,
  success flags per ε, generations-to-success per ε).
- `battery/runs/battery_summary_<landscape>_<arm>.json` — the §8
  statistics computed FROM the ledgers (success curves, ERT tables,
  quantile statements) + every input artifact's sha256; registered in
  `04` at landing (§12's battery section).
- The aggregation headline file after both arms + all three landscapes
  land.

## 9. Wording discipline (carried verbatim)

The GA spends **fitness queries** against the oracle; n_evals counts
every call (duplicates, memo hits, local search included). "Global
optimum" statements mean the DFT-arbitrated certified optimum and say
so. The battery's per-landscape targets are the xTB-tier oracle
argmins — never called "the global optimum" unqualified. No conclusions
from synthetic runs; failure fails hard; a negative GA-vs-BF result is
a FINDING. Tuning lives only in the declared scoping phase — never on
reported numbers, ever.

## 10. Audit trail

The independent pre-freeze audit **REVIEW-GOCAT-BATTERY-2026-10-19-01**
(AI-assisted, author-commissioned, local; intake + report + artifacts
+ FIXLIST in `thesis/audits/2026-10-19_battery-audit_*`; verdict
**continue-with-changes**, 0 Critical / 3 Major / 7 Minor) was
incorporated into this draft BEFORE the freeze as the v1 → v2
revision. Resolution map: **M1** — §4 rewritten forward-tense (the
snapshot is declared as a requirement, not a landed fact; its
manifest + the locked Python environment spec moved into the §0
stage-2 content list) + the `battery/README.md` tense line fixed;
**M2** — §6's verbatim-grid claim replaced by the explicit two-grid
resolution paragraph (the success-vs-queries curve reads off the
ledger's samples; the success-vs-G curve is the primary curve);
**M3** — §8 output 4's n_success = 0 convention ("≥ B", count
carried, no substitute estimate); **m1** — §0's explicit ordering
statement; **m2** — §7 Phase 7a's corner-case contingency (surface
and stop, never tune away); **m3** + **m4** — §8's pre-specified
headline anchor (G_battery, ε = 1.0) + headline success fraction
(ε = 0.25) + the unadjusted 95 % Wilson-overlap criterion with the
item-5 guardrail as the inferential anchor; **m5** — §8's analysis
populations (n = 100 denominators, failed-run numerator exclusion +
failure-mode breakdown, budget-not-reached stays in the denominator);
**m6** — §8's pre-declared PAIRED cross-arm analysis (Wilcoxon
signed-rank, sign-test fallback condition); **m7** — §6's
citation-resolution rule (placeholder now, binds at the stage-2
addendum). All edits are battery-side prereg text; no previously
completed work was changed (the containment rule, respected). Optional
close-out: a delta re-audit by the same independent auditor at or
before the stage-1 freeze — Jonny's call. The amended draft's sha256
is computed and recorded in PROJECT_STATE on this revision's landing;
the freeze re-hashes it into `04`.