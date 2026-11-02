# Glossary — the project's internal vocabulary, translated

The code was written inside a working project that kept its own plan, registry
and decision log. Those documents are not published, and the code still refers
to them. Nothing here changes a claim; it only tells you what a term points at.

| Term in the code | What it means |
|---|---|
| **"item N", "work item 4b"** | a number in the project's work order — a dependency-ordered list of build stages. `item 4` = the search, `item 6` = the battery, `item 7` = the reference tier, `item 9` = the analyses. |
| **"the briefing", "spec N"** | the written specification handed to the coding agent for one stage. Not published; its requirements survive as the module docstrings and the tests. |
| **"`04`", "the registry", "`§12`"** | the project's numbers registry — the file that recorded every measured number with its provenance. Not published; every number it held is re-derivable from `data/`. |
| **"ruling", "the author's ruling, 2026-10-28"** | a decision the author recorded on that date, after the options were laid out. The dated comments are the decision log. |
| **"Jonny"** | the author. Appears in decision records and in a few docstrings; runtime messages no longer name him. |
| **"Pi"** | the coding agent that implemented the stages. One test docstring mentions it. |
| **"the pre-registration"** | `docs/preregistration.md` — the design frozen before any reported run. |
| **"the addendum", "stage-2"** | `docs/scoping-addendum.md` — the second freeze, after the unreported scoping sweep, fixing the operator set and the generation budget. Its two sha256 values are explained in the README's provenance paragraph. |
| **"the oracle"** | the exhaustively enumerated landscape: every configuration's fitness, computed once. The search reads it by lookup. |
| **"cloud v2", "v1"** | two revisions of the candidate-site cloud. v2 (the published one) sits on the path-union surface; v1 (826 points, TS surface only) is superseded and not shipped. |
| **"a roll", "roll k"** | a position in the CVT sampler's internal random stream. The canonical cloud is roll 0 and replicate *k* is roll *k*; see `docs/cvt_seed_probe.py`. |
| **"the fence"** | the configurations excluded from the search space — pairs closer than 1.0 Å, and (per landscape) the one configuration whose self-consistent field would not converge. Recorded, never penalised. |
| **"T2"** | the frozen convergence protocol for the xTB tier (`maxiterations=2000`, `broydamp=0.1`), adopted once for every configuration. |
| **"PENDING_DECISION", "BLOCKED"** | leftover markers from the build workflow. Where they survived into shipped text they have been removed; where a docstring still carries one, it records a state that has since passed. |

## Renamed files

| Working-project name | Here |
|---|---|
| `harness_battery.py` | `experiments/battery.py` |
| `harness_scoping.py` | `experiments/cells.py` |
| `s60_ga_param_study.py` | `experiments/screen.py` |
| `2026-10-19_preregistration.md` | `docs/preregistration.md` |
| `2026-10-19_scoping-addendum.md` | `docs/scoping-addendum.md` |

## Where to start

If you are marking the method rather than reading the codebase, read four
files: `ga/ga.py` (the search), `ga/space.py` (the configuration arithmetic),
`run_ga.py` (one run, end to end), and `experiments/battery.py` (the reported
600-run experiment). Their shared interface is `fitness/contract.py`. The rest
is per-stage machinery, each with its assumptions at the top of the file.
