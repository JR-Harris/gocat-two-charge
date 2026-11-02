#!/usr/bin/env python3
"""The cross-cloud matched-configuration correlation (the reported ρ ≈ 0.98).

The three candidate clouds are independent draws on the same surface, so a
site index means nothing across clouds: site 17 on the canonical cloud is not
site 17 on a replicate. The comparison therefore matches geometrically -- each
canonical site is paired with its NEAREST site in the replicate cloud -- and
compares the same pair of mapped sites carrying the same signs.

Pairs are dropped when the mapping degenerates: when both canonical sites map
onto one replicate site (a collision), or when the mapped pair falls inside
the replicate's 1.0 Å separation floor (so it is not a configuration of that
landscape at all). Everything else is scored.

Reports Spearman's rho over the surviving configurations -- the number quoted
in the results (0.9851 canonical vs replicate-1, 0.9854 vs replicate-2) --
together with the survival fraction and the drop counts, so the rule's effect
is visible rather than asserted.

    python experiments/matched_correlation.py
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
SIGN_CANON = ((1, 1), (1, -1), (-1, 1), (-1, -1))
D_MIN = 1.0

LANDSCAPES = {
    "canonical": (BASE / "data" / "derived" / "enumeration_v2",
                  BASE / "data" / "derived" / "candidates_v2.npy"),
    "replicate-1": (BASE / "data" / "replicates" / "replicate-1" / "derived"
                    / "enumeration_replicate-1",
                    BASE / "data" / "replicates" / "replicate-1"
                    / "candidates_replicate-1.npy"),
    "replicate-2": (BASE / "data" / "replicates" / "replicate-2" / "derived"
                    / "enumeration_replicate-2",
                    BASE / "data" / "replicates" / "replicate-2"
                    / "candidates_replicate-2.npy"),
}


def load(name):
    ldir, cloud = LANDSCAPES[name]
    return (np.load(cloud), np.load(ldir / "f.npy"),
            np.load(ldir / "pair_indices.npy"),
            np.load(ldir / "sign_index.npy"))


def nearest_map(src_cloud, dst_cloud):
    """Index of the nearest dst site for every src site (brute force: the
    clouds are ~900 points, so this is a 900x900 distance matrix)."""
    d2 = ((src_cloud[:, None, :] - dst_cloud[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1)


def spearman(x, y):
    """Spearman's rho without scipy: average ranks, then Pearson."""
    def ranks(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1, dtype=np.float64)
        # average ties
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = r[order[i:j + 1]].mean()
            i = j + 1
        return r
    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def compare(base, other):
    cb, fb, pb, sb = load(base)
    co, fo, po, so = load(other)
    m = nearest_map(cb, co)

    # replicate lookup: (lo, hi, sign_at_lo, sign_at_hi) -> row, where the
    # signs are attached to the site they belong to (the pair order may be
    # swapped between landscapes, and a swap swaps the signs with it)
    lo = np.minimum(po[:, 0], po[:, 1])
    hi = np.maximum(po[:, 0], po[:, 1])
    swap = po[:, 0] > po[:, 1]
    dec = np.array([SIGN_CANON[int(k)] for k in so])          # (n, 2)
    s_lo = np.where(swap, dec[:, 1], dec[:, 0])
    s_hi = np.where(swap, dec[:, 0], dec[:, 1])
    table = {}
    for k in range(len(lo)):
        table[(int(lo[k]), int(hi[k]), int(s_lo[k]), int(s_hi[k]))] = k

    xb, xo, dropped = [], [], {"collision": 0, "floor": 0, "nan": 0}
    for r in range(len(fb)):
        i, j = int(pb[r, 0]), int(pb[r, 1])
        mi, mj = int(m[i]), int(m[j])
        if mi == mj:
            dropped["collision"] += 1
            continue
        si, sj = SIGN_CANON[int(sb[r])]       # pair order is (i, j) here
        if mi < mj:
            key = (mi, mj, si, sj)
        else:
            key = (mj, mi, sj, si)
        q = table.get(key)
        if q is None:
            # not a configuration of the other landscape (inside its floor)
            dropped["floor"] += 1
            continue
        if not np.isfinite(fo[q]) or not np.isfinite(fb[r]):
            dropped["nan"] += 1
            continue
        xb.append(float(fb[r])); xo.append(float(fo[q]))

    n_total = len(fb)
    n_used = len(xb)
    return {"base": base, "other": other, "n_rows": n_total,
            "n_compared": n_used,
            "survival_fraction": round(n_used / n_total, 4),
            "dropped": dropped,
            "spearman": round(spearman(xb, xo), 4)}


def main():
    out = {}
    for other in ("replicate-1", "replicate-2"):
        res = compare("canonical", other)
        out[other] = res
        print(f"canonical vs {other}: rho = {res['spearman']:.4f} over "
              f"{res['n_compared']:,} matched configurations "
              f"({100 * res['survival_fraction']:.1f}% survive; "
              f"dropped {res['dropped']})")
    dest = BASE / "data" / "results" / "matched_correlation.json"
    dest.write_text(json.dumps({"schema": "matched_correlation_v1",
                                "note": ("nearest-site matching, same signs; "
                                         "collisions and sub-floor images "
                                         "dropped"),
                                "results": out}, indent=1) + "\n")
    print(f"written: {dest.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
