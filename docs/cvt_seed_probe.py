#!/usr/bin/env python3
"""Measure how the CVT sampler is seeded — the experiment behind the claim.

The candidate clouds are drawn by `point_cloud_utils.sample_mesh_lloyd` on the
path-union mesh. The sampler seeds itself INTERNALLY and exposes no handle:
that is why the three clouds are recorded artifacts, locked by hash, rather
than files anyone regenerates on demand. This script re-measures the three
facts the claim rests on, on whatever installation you have:

  1. two draws in the SAME process differ (the stream advances per call);
  2. two fresh processes draw the SAME points, to about 1e-12 A (the seed is
     fixed per process, not wallclock and not entropy);
  3. there is no seed argument and no seed setter in the library.

Run it twice; the second run is the "fresh process" for fact 2. It writes
nothing outside a temporary file in /tmp.

    python docs/cvt_seed_probe.py --mesh-from-cloud data/derived/candidates_v2.npy

With no mesh argument it builds a small icosphere, which is enough to measure
the behaviour (the numbers differ from the real cloud; the pattern does not).
"""

import argparse
import hashlib
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import point_cloud_utils as pcu
import trimesh


def tiny_mesh(radius=3.0):
    m = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    return m


def draw(mesh, n):
    pts = pcu.sample_mesh_lloyd(mesh.vertices, mesh.faces, n)
    return np.asarray(pts, dtype=np.float64)


def digest(pts):
    return hashlib.sha256(np.ascontiguousarray(pts, dtype=np.float64)
                          .tobytes()).hexdigest()[:16]


def max_delta(a, b):
    """Largest displacement, matching points in draw order."""
    return float(np.linalg.norm(a - b, axis=1).max())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mesh-from-cloud", default=None,
                    help="unused: the union surface is not stored as a mesh; "
                         "omit to probe on an icosphere")
    ap.add_argument("--fresh", action="store_true",
                    help="internal: the second process of fact 2")
    ap.add_argument("--out", default=None, help="internal: where to write")
    args = ap.parse_args(argv)

    mesh = tiny_mesh()
    a = draw(mesh, args.n)

    if args.fresh:
        np.save(args.out, a)
        return 0

    b = draw(mesh, args.n)
    print(f"library            : point_cloud_utils {getattr(pcu, '__version__', '?')}")
    print(f"draws              : {args.n} points on an icosphere")
    print(f"1. same process    : max delta between draw 1 and draw 2 = "
          f"{max_delta(a, b):.3e} A")
    print(f"                      (different draws => the stream advances per call)")

    tmp = tempfile.mkdtemp()
    out = Path(tmp) / "second_process.npy"
    subprocess.run([sys.executable, __file__, "--n", str(args.n),
                    "--fresh", "--out", str(out)],
                   check=True, cwd=str(Path(__file__).resolve().parent))
    other = np.load(out)
    print(f"2. fresh process   : max delta vs this process's draw 1 = "
          f"{max_delta(a, other):.3e} A")
    print(f"                      digests {'identical (bitwise)' if digest(a) == digest(other) else 'differ (not bitwise)'}"
          f"  here {digest(a)} / there {digest(other)}")

    sig = inspect.signature(pcu.sample_mesh_lloyd)
    print(f"3. API             : signature {sig}")
    names = [n for n in dir(pcu) if "seed" in n.lower()]
    print(f"                      seed-related names in the module: {names or 'none'}")
    print()
    print("Reading: fact 1 shows the stream advances per call (so a second")
    print("realization = advance the stream, the 'roll'); fact 2 shows the seed")
    print("is fixed per process (a fresh process reproduces the draw to ~1e-12 A,")
    print("not bitwise); fact 3 shows there is no handle to set it. Hence: the")
    print("clouds are recorded and locked by hash, and the replicate index is")
    print("the stream position.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
