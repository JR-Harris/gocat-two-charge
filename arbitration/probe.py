"""Step C — the fidelity probe (sampler + statistics).

~192 configs sampled from the WHOLE landscape (not the top), stratified
per the mini-cloud pattern (the strata_rules of
data/derived/mini_cloud_summary.json): core = nearest-TS-atom
1-indexed in {1,7,9,11,14,16}, carboxylate = {13,3,4,15,5,6}, spread =
the remainder.  RULING 1 (Jonny, 2026-10-28): the config->stratum rule
is 'both' — core = BOTH sites core; carboxylate = BOTH sites in the
carboxylate classes and neither in core; spread = the remainder — and
the rev-2 targets are DOUBLED: 60/48/84 = 192 configs.  Seeded rng
20260925, draw order core -> carboxylate -> spread, per-stratum shuffle
of canon-(row-)index-order pools.  ADDITIONALLY each pool is stratified
by f-quintile (landscape-wide edges over the finite rows) so the probe
spans the fitness range: ~ceil(k/5) per (stratum, quintile) cell,
filled in the recorded draw order; irregular cells are recorded in the
summary, never silently tuned.  The domain-excluded row (NaN f) is
never sampleable.

POOL-COUNT GATE (the ruling's prescribed gate): the artifact-derived
pool counts are core 13,224 / carboxylate-both 108,788 / spread 1,577,119
(sum = the 1,699,131 finite rows, exactly).  The ruling's count
triplet's spread number 1,288,599 is NOT the remainder — its triplet
sums to 1,410,611 = the total minus exactly the 288,520 MIXED-CORE
rows (one site core, one not), which the triplet orphans; the rule
TEXT's 'remainder' gives 1,577,119.  The gate asserts the
artifact-derived triplet; the discrepancy is in every audit and was
reported to Jonny (his ruling on the spread definition governs; a flip
is a one-constant change).

RULING 2 (statistics): 'the top stratum' := the SPREAD stratum — the
landscape's top region is 100 % spread-stratum (the certified
optimum's sites point at the H periphery, 0-based nearest_ts_atom
class 24); the rev-2 briefing's equation with the core stratum is
retired as a briefing defect.  The step-C statistics report the
overall rho, the spread-stratum (top) grouping, the per-stratum table
(n, rho, RMS, quintile spread) for all three strata; the step-A
top-100 rank correlation (xTB rank vs DFT rank) is computed in step
D's summary.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as sps

from fitness.orca import KCAL_PER_EH  # the frozen conversion (recorded)

REPO_ROOT = Path(__file__).resolve().parent.parent
MINI_CLOUD_SUMMARY = (REPO_ROOT / "data" / "derived"
                      / "mini_cloud_summary.json")

# The strata rules, 1-INDEXED (belt-gated against the mini-cloud summary).
CORE_1IDX = (1, 7, 9, 11, 14, 16)
CARBOXYLATE_1IDX = (13, 3, 4, 15, 5, 6)

SEED = 20260925                       # the mini-cloud's pinned seed

# The probe targets (Ruling 1, Jonny 2026-10-28): the rev-2 targets
# DOUBLED -> 30/24/42 = 96 becomes 60/48/84 = 192 configs (384 SPs).
TARGETS = {"core": 60, "carboxylate": 48, "spread": 84}
DRAW_ORDER = ("core", "carboxylate", "spread")

# RULING 1 recorded (Jonny, 2026-10-28): the config->stratum rule is
# 'both' (core = both sites in the core classes; carboxylate = both
# sites in the carboxylate classes and neither in core; spread = the
# remainder) — PENDING_DECISION is cleared.
#
# The pool-count gate the ruling prescribed: the independently derived
# counts (finite rows, artifact-derived) are core 13,224 and
# carboxylate-both 108,788 (both CONFIRM the ruling's numbers) but the
# ruling's spread number 1,288,599 is NOT the remainder: the triplet
# sums to 1,410,611 = 1,699,131 minus exactly the 288,520 MIXED-CORE
# rows (one site core, one not), which the ruling's count triplet
# orphans. The rule TEXT ('spread = the remainder') gives 1,577,119 and
# partitions the finite rows exactly. Implemented per the rule TEXT,
# with the discrepancy recorded in every audit + reported to Jonny.
PINNED_POOLS = {"core": 13224, "carboxylate": 108788, "spread": 1577119}
RULING_POOL_NUMBERS = {"core": 13224, "carboxylate": 108788,
                       "spread": 1288599}
MIXED_CORE_ROWS = 288520

# The recorded ruling; kept as a named constant (switchable, audited).
CONFIG_STRATUM_RULE = "both"

STRATA = ("core", "carboxylate", "spread")


def _belt_gate():
    """Fail hard if the mini-cloud summary's strata rules have drifted
    from the pinned 1-indexed lists."""
    ms = json.loads(MINI_CLOUD_SUMMARY.read_text())
    rules = ms["strata_rules"]
    got_core = sorted(rules["core"]["nearest_ts_atom_1idx"])
    carbox = (rules["carboxylate"].get("ring_cooH_1idx", [])
              + rules["carboxylate"].get("pyruvate_cooH_1idx", []))
    if got_core != sorted(CORE_1IDX):
        raise ValueError(
            f"mini-cloud strata drift: core rule {got_core} != pinned "
            f"{sorted(CORE_1IDX)}")
    if sorted(carbox) != sorted(CARBOXYLATE_1IDX):
        raise ValueError(
            f"mini-cloud strata drift: carboxylate rule {sorted(carbox)} "
            f"!= pinned {sorted(CARBOXYLATE_1IDX)}")
    if int(ms.get("seed", -1)) != SEED:
        raise ValueError(
            f"mini-cloud drift: seed {ms.get('seed')!r} != pinned {SEED}")


def site_strata(nta):
    """Site-level strata masks (0-based nearest_ts_atom classes):
    {core, carboxylate} -- spread is neither."""
    core = np.isin(nta, tuple(c - 1 for c in CORE_1IDX))
    carbox = np.isin(nta, tuple(c - 1 for c in CARBOXYLATE_1IDX))
    return core, carbox


def row_strata(i_sites, j_sites, core, carbox, rule=CONFIG_STRATUM_RULE):
    """Config-level stratum per row (vectorized; rule per the constant)."""
    ci, cj = core[i_sites], core[j_sites]
    bi, bj = carbox[i_sites], carbox[j_sites]
    if rule == "both":
        is_core = ci & cj
        is_carbox = (~is_core) & bi & bj
    elif rule == "any":
        is_core = ci | cj
        is_carbox = (~is_core) & (bi | bj)
    else:
        raise ValueError(f"unknown CONFIG_STRATUM_RULE {rule!r} "
                         "(implemented: 'both', 'any')")
    stratum = np.full(i_sites.shape, "spread", dtype="<U11")
    stratum[is_carbox] = "carboxylate"
    stratum[is_core] = "core"
    return stratum


def quintile_edges(f):
    """Landscape-wide f-quintile edges (4 edges, finite rows only)."""
    finite = f[np.isfinite(f)]
    return np.quantile(finite.astype(np.float64), [0.2, 0.4, 0.6, 0.8])


def quintile_of(values, edges):
    """Quintile bin 0..4 per value (np.searchsorted side='right')."""
    return np.searchsorted(edges, values, side="right")


def sample(engine, rule=CONFIG_STRATUM_RULE):
    """The seeded probe sample -> (sorted rows, audit dict)."""
    _belt_gate()
    landscape = engine.landscape
    f = landscape.f
    nta = landscape.provenance()["nearest_ts_atom"]
    core, carbox = site_strata(nta)
    i_sites = landscape.pair_indices[:, 0]
    j_sites = landscape.pair_indices[:, 1]
    stratum = row_strata(i_sites, j_sites, core, carbox, rule)
    edges = quintile_edges(f)
    bins = quintile_of(f.astype(np.float64), edges)

    rng = np.random.default_rng(SEED)
    all_rows = []
    audit = {"config_stratum_rule": rule, "seed": SEED,
             "targets": dict(TARGETS), "draw_order": list(DRAW_ORDER),
             "quintile_edges_kcal": [float(e) for e in edges],
             "site_counts": {"core": int(core.sum()),
                             "carboxylate": int(carbox.sum()),
                             "spread": int(nta.size - core.sum()
                                           - carbox.sum())},
             "strata": {}}
    for s in DRAW_ORDER:
        # canon (row) order pool; finite rows only (the domain-excluded
        # NaN row is never sampleable)
        pool = np.flatnonzero((stratum == s) & np.isfinite(f)).copy()
        rng.shuffle(pool)                # in place, seeded
        quota = -(-TARGETS[s] // 5)      # ceil(k/5)
        cell_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        taken = []
        for row, cell in zip(pool, bins[pool]):
            if len(taken) >= TARGETS[s]:
                break
            if cell_counts[int(cell)] < quota:
                cell_counts[int(cell)] += 1
                taken.append(int(row))
        taken.sort()
        under = [q for q in range(5) if cell_counts[q] < quota]
        entry = {
            "pool_size": int(pool.size), "target": TARGETS[s],
            "quota_per_cell": quota,
            "cell_counts": {f"q{q + 1}": cell_counts[q] for q in range(5)},
            "taken": len(taken), "taken_rows": taken,
            "undersupplied_cells": (["q%d" % (q + 1) for q in under]
                                    if len(taken) < TARGETS[s] else []),
        }
        if len(taken) < TARGETS[s]:
            entry["irregular"] = (
                "stratum undershot its target: the cells that could not "
                "supply their quota are recorded above (never silently "
                "tuned)")
        audit["strata"][s] = entry
        all_rows.extend(taken)
    audit["irregular_cells_present"] = any(
        st.get("undersupplied_cells") for st in audit["strata"].values())
    # the pool-count gate (Ruling 1's prescribed gate, pinned to the
    # artifact-derived counts — see the module docstring's reconciliation)
    audit["pool_count_gate"] = {
        "pinned": dict(PINNED_POOLS),
        "measured": {s: audit["strata"][s]["pool_size"] for s in DRAW_ORDER},
        "ruling_numbers": dict(RULING_POOL_NUMBERS),
        "discrepancy": (
            "the ruling's spread count 1,288,599 orphans the 288,520 "
            "mixed-core rows (its triplet sums to 1,410,611, not the "
            "1,699,131 finite rows); implemented per the rule TEXT: "
            "spread = the remainder = 1,577,119; flagged to the author"),
    }
    for s in DRAW_ORDER:
        if PINNED_POOLS[s] != audit["strata"][s]["pool_size"]:
            raise AssertionError(
                f"pool-count gate ({s}): "
                f"{audit['strata'][s]['pool_size']} != {PINNED_POOLS[s]}")
    return sorted(all_rows), audit


def stratum_by_row(audit):
    """{row: stratum} from the sampler's audit."""
    out = {}
    for s in DRAW_ORDER:
        for r in audit["strata"][s]["taken_rows"]:
            out[r] = s
    return out


def statistics(records, stratum_by_row, quintile_edges=None):
    """The probe's deliverables from the C records (the float64 DFT f is
    the authority; f_xTB is the float32 landscape value carried in the
    record).

    RULING 2 (Jonny, 2026-10-28): 'the top stratum' := the SPREAD
    stratum (the top region of the landscape is 100 % spread — the
    rev-2 line equating it with the core stratum is retired as a
    briefing defect).  Reported: the overall Pearson/Spearman; the
    SPREAD-stratum grouping (the top stratum); the per-stratum table
    (n, rho, RMS, quintile spread) for all three strata.
    """
    rows = sorted(records)
    f_xtb = np.array([records[r]["f_xtb_kcal"] for r in rows],
                     dtype=np.float64)
    f_dft = np.array([records[r]["f_kcal"] for r in rows], dtype=np.float64)
    resid = f_xtb - f_dft
    out = {
        "n": len(rows),
        "pearson_overall": float(sps.pearsonr(f_xtb, f_dft)[0]),
        "spearman_overall": float(sps.spearmanr(f_xtb, f_dft)[0]),
        "rms_overall_kcal": float(np.sqrt(np.mean(resid ** 2))),
    }

    def _grp(rs):
        mask = np.isin(np.array(rows), rs)
        x, d, r = f_xtb[mask], f_dft[mask], resid[mask]
        return {"n": len(rs),
                "pearson": float(sps.pearsonr(x, d)[0]),
                "spearman": float(sps.spearmanr(x, d)[0]),
                "rms_kcal": float(np.sqrt(np.mean(r ** 2)))}

    # the per-stratum table (all three; the SPREAD entry is the ruling's
    # 'top stratum')
    table = {}
    for s in DRAW_ORDER:
        srows = [r for r in rows if stratum_by_row[r] == s]
        entry = _grp(srows) if srows else {"n": 0}
        if srows and quintile_edges is not None:
            bins = quintile_of(np.array([records[r]["f_xtb_kcal"]
                                         for r in srows], dtype=np.float64),
                               quintile_edges)
            vals, cnts = np.unique(bins, return_counts=True)
            counts = {f"q{q + 1}": int(cnts[vals.tolist().index(q)])
                      if q in vals.tolist() else 0 for q in range(5)}
            entry["quintile_spread"] = counts
        table[s] = entry
    out["per_stratum"] = table
    out["top_stratum"] = "spread"
    out["ruling2_note"] = (
        "'the top stratum' := the spread stratum (Ruling 2, 2026-10-28): "
        "the top region of the landscape is 100 % spread-stratum; the "
        "briefing-rev-2 equation with the core stratum is retired")
    out["residual_scatter"] = [
        {"row": int(r), "stratum": stratum_by_row[r],
         "f_xtb_kcal": float(x), "f_dft_kcal": float(d),
         "residual_kcal": float(x - d)}
        for r, x, d in zip(rows, f_xtb, f_dft)]
    return out


def run_step_C(engine):
    """Step C: sample -> execute the probe configs -> records + the
    statistics + the summary.  (The stratum-rule decision is RECORDED
    (Ruling 1, 2026-10-28: 'both', 192 targets); the pool-count gate
    runs inside sample().)"""
    rows, audit = sample(engine)
    _, modes = engine._dispatch_diff(engine.rows_jobs(rows), "C")
    records = engine.read_records("C")
    stats_out = statistics(records, stratum_by_row(audit),
                           quintile_edges=np.array(
                               audit["quintile_edges_kcal"]))
    summary = {
        "purpose": ("step C: the landscape-wide strata-fidelity probe "
                    "(Ruling-1 doubled targets 60/48/84 = 192 configs, "
                    "f-quintile cells)"),
        "config_stratum_rule": audit["config_stratum_rule"],
        "sampler": audit,
        "n_expected": sum(TARGETS.values()),
        "executed": sum(1 for m in modes.values() if m == "executed"),
        "from_cache": sum(1 for m in modes.values() if m == "cache"),
        "resumed": sum(1 for m in modes.values() if m == "resumed"),
        "statistics": stats_out,
        "provenance": engine.provenance_block(),
    }
    return summary, engine.write_summary("C", summary)