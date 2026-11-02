#!/usr/bin/env python3
"""The GA runner (the discrete memetic GA): one invocation = one run.

  python run_ga.py --mode {oracle,stub,xtb} --generations G --seed S
      [--param-set smoke] [--landscape-dir PATH]
      [--cloud PATH --cloud-sha256 HEX] [--out-root PATH] [--force]

    Runs the GA with exactly G generations (the primary stop rule; G is
    an explicit argument of every invocation, never a default -- there
    is no evaluation-budget stop and no batch truncation anywhere;
    every generation completes).

Modes + identity:

  * oracle (default): the fitness is a LOOKUP into the certified
    landscape (--landscape-dir, default data/derived/enumeration_v2/).
    The cloud, its sha256 and the domain exclusions resolve FROM the
    landscape artifact itself -- one source of truth; a --cloud
    override must match the summary's recorded sha256 or the run is
    refused. Oracle runs are table-priced (seconds at the smoke
    sizings); the later GA-vs-bf budgets read off the trajectory.
  * stub: the synthetic fitness.stub provider (plumbing verification
    ONLY -- no scientific meaning).
  * xtb (live): the real GFN2-xTB provider, a FRESH instance per run
    (the n_evals ledger is per-run state), OMP=1 as shipped; ~0.15 s
    per evaluation (the T2 sizing) -- G = 20 at pop 50 spends roughly
    1.2e3-2.5e3 evaluations ~ 3-6 min/run (MEASURE the first live run,
    then size further loops).

  Cloud identity in stub/xtb modes follows the enumeration driver's
  rule: the canonical cloud WITH its frozen sha256 by default; any
  other --cloud REQUIRES --cloud-sha256 (no implicit acceptance). The
  canonical landscape's domain exclusions fence the search space
  whenever the cloud is canonical (else the fence is empty;
  --landscape-dir names the exclusions' source in stub/xtb mode,
  default the canonical enumeration_v2).

  --param-set names the frozen operator record (ga/params.py); the
  registered sets print on an unknown name. The registered set is the
  stage-2 frozen set: the pre-declared scoping sweep selected exactly these
  values, and every reported battery run used them.

Result record: runs/ga/g{G:06d}_s{seed:05d}_{fitness_id}_{cloud12}.json
(+ a sha256 sidecar) -- schema "ga_run_v2", atomic write (tmp + fsync +
rename); --force is required to overwrite an existing identity, else
the run is REFUSED (exit 2, nothing touched). A failed or interrupted
invocation leaves NO artifact -- EXCEPT a fail-hard provider failure
(e.g. a live SCC non-convergence on a configuration the fence did not
know), which writes the run with status: failed + the failing config's
identity, then exits nonzero (the pre-registered domain rule: such a
failure is terminal -- never re-run, never penalised).

Exit codes: 0 = the run wrote its record; 1 = a failed run (a provider
failure writes its failed record first; an init/gate failure writes
nothing); 2 = the CLI refused the run (identity exists without
--force, incoherent cloud identity, unknown param set, out-root inside
the canonical tree).

The runner NEVER writes under data/derived/. The battery runs as a
shell loop over (G, seed) pairs on this CLI; nothing here is a
reported benchmark (the PLUMBING item): the operator rates + the G
sweep arrive at item 6's pre-registration.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(
    os.path.realpath(__file__))))

from ga import ga as ga_mod
from ga import space as ga_space
from ga.oracle_provider import landscape_oracle
from ga.params import get_param_set
from fitness import stub as stub_module
from fitness import xtb as xtb_module

REPO_ROOT = Path(os.path.dirname(os.path.abspath(
    os.path.realpath(__file__))))

# The canonical cloud identity (the run_exhaustive_search.py constants;
# the brief cross-checks this equivalence -- pinned in ga/test_ga.py).
CANONICAL_CLOUD = "data/derived/candidates_v2.npy"
CANONICAL_CLOUD_SHA256 = (
    "084e48a6b5267900a995ba2d4f0412f77f5bb13446ba74eafaf474c644349e71")
CANONICAL_LANDSCAPE_DIR = "data/derived/enumeration_v2"

DEFAULT_OUT_ROOT = "runs/ga"
MODE_LABEL = {"oracle": "oracle", "stub": "stub", "xtb": "live"}


class Refused(Exception):
    """The CLI refuses the run (exit 2; nothing touched)."""


class Failed(Exception):
    """The run fails before/without a record (exit 1)."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip() or ""
    except Exception:
        return ""


def read_domain_exclusions(landscape_dir, summary_sha=None):
    """-> (canonical_tuples, source_path) from a landscape's
    domain_exclusions.json; the file's sha256 is verified against the
    summary's certification whenever one is given."""
    path = Path(landscape_dir) / "domain_exclusions.json"
    if not path.is_file():
        raise Failed(f"{path}: no domain_exclusions.json in the landscape")
    if summary_sha is not None:
        got = sha256_file(path)
        if got != summary_sha:
            raise Failed(
                f"{path}: domain-exclusions sha256 mismatch against the "
                f"summary's certification (got {got}, expected "
                f"{summary_sha})")
    data = json.loads(path.read_text())
    rows = [int(r) for r in (data.get("rows") or [])]
    entries = data.get("entries") or []
    if int(data.get("count", -1)) != len(rows) or len(entries) != len(rows):
        raise Failed(f"{path}: the exclusion count/list/entries disagree")
    tuples = []
    for e, row in zip(entries, rows):
        cfg = e.get("config") or []
        if int(e["row"]) != row or len(cfg) != 4:
            raise Failed(f"{path}: malformed exclusion entry {e!r}")
        tuples.append((int(cfg[0]), int(cfg[1]),
                       1 if int(cfg[2]) > 0 else -1,
                       1 if int(cfg[3]) > 0 else -1))
    return tuples, str(path)


def resolve_cloud(canonical_default, args):
    """(stub/xtb modes) -> (cloud_path, sha256, is_canonical). Any
    non-default cloud REQUIRES its own --cloud-sha256 (no implicit
    acceptance); --cloud-sha256 on the default cloud must still match
    the frozen constant."""
    cloud = Path(args.cloud) if args.cloud else REPO_ROOT / canonical_default
    if args.cloud and args.cloud_sha256 is None:
        raise Refused(
            f"the cloud {args.cloud} is not the default and no "
            "--cloud-sha256 was given (no implicit acceptance of a "
            "wrong cloud: the record takes the cloud's sha256 from "
            "the arguments)")
    sha = args.cloud_sha256 if args.cloud else CANONICAL_CLOUD_SHA256
    if not cloud.is_file():
        raise Refused(f"{cloud}: the cloud file not found")
    got = sha256_file(cloud)
    if got != sha:
        raise Refused(
            f"{cloud}: cloud sha256 mismatch (got {got}, expected "
            f"{sha}) -- tampered or wrong input")
    return cloud, got, got == CANONICAL_CLOUD_SHA256


def canonical_exclusions(args):
    """-> (tuples, source) for a CANONICAL cloud: the certified
    landscape's domain exclusions, sha-verified against its summary."""
    ldir = Path(args.landscape_dir or (REPO_ROOT / CANONICAL_LANDSCAPE_DIR))
    summary_path = ldir / "enumeration_summary.json"
    if not summary_path.is_file():
        raise Failed(
            f"{summary_path}: the canonical cloud's exclusions source "
            "is missing its landscape summary")
    summary = json.loads(summary_path.read_text())
    excl_sha = ((summary.get("artifacts") or {})
                .get("domain_exclusions") or {}).get("sha256")
    return read_domain_exclusions(ldir, excl_sha)


# --------------------------------------------------------------------------
# the run record
# --------------------------------------------------------------------------

def compose_record(result, *, mode, fitness_id, provider_meta, cloud_path,
                   cloud_sha256, d_min, exclusions, exclusion_source,
                   landscape_dir=None, artifact_sha256s=None, param_set=None,
                   created_utc=None, git_head=None, status="ok",
                   failure=None):
    """The full "ga_run_v2" record dict: the engine result verbatim plus
    the run's provenance."""
    d = result.as_dict()
    record = {
        "schema": d["schema"],
        "created_utc": created_utc if created_utc is not None else create_utc(),
        "git_head": git_head if git_head is not None else git_head_sha(),
        "mode": mode,
        "status": status,
        "fitness_id": fitness_id,
        "provider_meta": provider_meta,
        "param_set": param_set,
        "generations": d["generations"],
        "seed": d["seed"],
        "params": d["params"],
        "cloud": {"path": str(cloud_path), "sha256": cloud_sha256},
        "d_min": d_min,
        "exclusions": {"list": [list(t) for t in exclusions],
                       "source": exclusion_source},
        "counts": d["counts"],
        "n_generations_executed": d["generations_executed"],
        "ledger": d["ledger"],
        "trajectory": d["trajectory"],
        "best": d["best"],
        "wall_seconds": d["wall_seconds"],
    }
    if landscape_dir is not None or artifact_sha256s is not None:
        record["oracle"] = {
            "landscape_dir": (str(landscape_dir)
                              if landscape_dir is not None else None),
            "cloud_sha256": cloud_sha256,
            "artifact_sha256s": (dict(artifact_sha256s) if artifact_sha256s
                                 else None),
        }
    if failure is not None:
        record["failure"] = failure
    return record


def failing_config(exc):
    """-> the failing configuration [i, j, s1, s2] from a
    ProviderFailure whose provider error carries .config (the
    domain-exclusion class); None otherwise."""
    pe = getattr(exc, "provider_error", None)
    cfg = getattr(pe, "config", None)
    if cfg is None:
        return None
    try:
        return [int(cfg[0]), int(cfg[1]),
                1 if int(cfg[2]) > 0 else -1,
                1 if int(cfg[3]) > 0 else -1]
    except Exception:
        return None


def record_path(out_root, generations, seed, fitness_id, cloud_sha256):
    name = (f"g{generations:06d}_s{seed:05d}_{fitness_id}_"
            f"{cloud_sha256[:12]}.json")
    return Path(out_root) / name


def write_record(record, path):
    """Atomic write (tmp + fsync + rename) + the sha256 sidecar
    (sha256sum-compatible: '<sha>  <name>'). A failure mid-write
    cleans its temp files: a failed or interrupted invocation leaves
    no artifact."""
    payload = (json.dumps(record, indent=1) + "\n").encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_name(path.name + ".sha256")
    tmp_json = path.with_name("." + path.name + ".tmp")
    tmp_side = sidecar.with_name("." + sidecar.name + ".tmp")
    try:
        with open(tmp_json, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        with open(tmp_side, "wb") as fh:
            fh.write(f"{sha}  {path.name}\n".encode("ascii"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_json, path)
        os.replace(tmp_side, sidecar)
    finally:
        for t in (tmp_json, tmp_side):
            if t.exists():
                t.unlink()
    return sha


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def resolve_mode(args):
    """-> dict(provider, fitness, fitness_id, provider_meta, cloud,
    cloud_sha256, d_min, exclusions, exclusion_source, mode_label,
    landscape) -- a FRESH provider instance per run."""
    if args.mode == "oracle":
        landscape_dir = Path(args.landscape_dir or
                             (REPO_ROOT / CANONICAL_LANDSCAPE_DIR))
        summary_path = landscape_dir / "enumeration_summary.json"
        if not summary_path.is_file():
            raise Failed(f"{summary_path}: no landscape summary")
        summary = json.loads(summary_path.read_text())
        if args.cloud:
            cloud = Path(args.cloud)
            if not cloud.is_file():
                raise Refused(f"{cloud}: the cloud file not found")
            got = sha256_file(cloud)
            want = str(summary.get("cloud_sha256"))
            if got != want:
                raise Refused(
                    f"{cloud}: the --cloud override does not match the "
                    f"landscape's recorded cloud sha256 (got {got}, "
                    f"expected {want}) -- one source of truth")
        else:
            rec = (summary.get("provider_meta") or {}).get("cloud") or {}
            rel = rec.get("path")
            if not rel:
                raise Failed(
                    f"{summary_path}: the summary records no cloud path; "
                    "pass --cloud explicitly")
            cloud = (Path(rel) if Path(rel).is_absolute()
                     else REPO_ROOT / rel)
            if not cloud.is_file():
                raise Failed(
                    f"{cloud}: the landscape's recorded cloud file is "
                    "missing; pass --cloud explicitly")
        excl_sha = ((summary.get("artifacts") or {})
                    .get("domain_exclusions") or {}).get("sha256")
        tuples, source = read_domain_exclusions(landscape_dir, excl_sha)
        provider = landscape_oracle(landscape_dir, cloud)
        return {
            "provider": provider, "fitness": provider.eval_batch,
            "fitness_id": provider.FITNESS_ID,
            "provider_meta": provider.PROVIDER_META,
            "cloud": cloud, "cloud_sha256": provider.cloud_sha256,
            "d_min": float((summary.get("universe") or {}).get(
                "d_min", ga_space.D_MIN_A)),
            "exclusions": tuples, "exclusion_source": source,
            "mode_label": MODE_LABEL["oracle"],
            "landscape": {"landscape_dir": str(landscape_dir),
                          "cloud_sha256": provider.cloud_sha256,
                          "artifact_sha256s":
                              dict(provider.artifact_sha256s)},
        }

    # --- stub / xtb: cloud identity from the arguments ---------------------
    cloud, cloud_sha, canonical = resolve_cloud(CANONICAL_CLOUD, args)
    tuples, source = canonical_exclusions(args) if canonical else ([], None)
    if args.mode == "stub":
        provider = stub_module.StubProvider()        # FRESH per run
    else:
        provider = xtb_module.XtbProvider(           # FRESH per run
            cloud_path=cloud,
            cloud_sha256=(cloud_sha if args.cloud else None))
    return {
        "provider": provider, "fitness": provider.eval_batch,
        "fitness_id": provider.FITNESS_ID,
        "provider_meta": provider.PROVIDER_META,
        "cloud": cloud, "cloud_sha256": cloud_sha,
        "d_min": ga_space.D_MIN_A,
        "exclusions": tuples, "exclusion_source": source,
        "mode_label": MODE_LABEL[args.mode],
        "landscape": None,
    }


def check_out_root(out_root):
    out = Path(out_root).resolve()
    canonical = (REPO_ROOT / "data" / "derived").resolve()
    if out == canonical or str(out).startswith(str(canonical) + os.sep):
        raise Refused(
            f"{out_root}: the runner NEVER writes under data/derived/ "
            "(the run tree is runs/)")


def run_one(args) -> int:
    try:
        params = get_param_set(args.param_set)
    except ValueError as e:
        raise Refused(str(e))
    if args.generations < 1:
        raise Refused(
            f"--generations must be an int >= 1 (the stop rule's "
            f"explicit argument), got {args.generations}")
    if args.seed < 0:
        raise Refused(f"--seed must be an int >= 0, got {args.seed}")
    out_root = Path(args.out_root or DEFAULT_OUT_ROOT)
    check_out_root(out_root)

    mode = resolve_mode(args)
    path = record_path(out_root, args.generations, args.seed,
                       mode["fitness_id"], mode["cloud_sha256"])
    if path.exists() and not args.force:
        raise Refused(
            f"{path}: the identity already exists; --force overwrites "
            "(nothing touched)")

    candidates = np.load(mode["cloud"])
    space = ga_space.Space(candidates, d_min=mode["d_min"],
                           exclusions=mode["exclusions"],
                           nn_k=params.nn_k)
    if space.n_exclusions != len(mode["exclusions"]):
        raise Failed("the fence lost exclusions at construction")

    try:
        result = ga_mod.run_ga(mode["fitness"], space, params, args.seed,
                               G=args.generations)
    except ga_mod.ProviderFailure as exc:
        cfg = failing_config(exc)
        labels = exc.batch_labels
        failure = {
            "error": str(exc),
            "provider_error": str(exc.provider_error),
            "failing_config": cfg,
            "first_batch_label": labels[0] if labels else None,
            "n_batch_rows": len(exc.batch_rows),
        }
        record = compose_record(
            exc.partial, mode=mode["mode_label"],
            fitness_id=mode["fitness_id"],
            provider_meta=mode["provider_meta"],
            cloud_path=mode["cloud"], cloud_sha256=mode["cloud_sha256"],
            d_min=mode["d_min"], exclusions=mode["exclusions"],
            exclusion_source=mode["exclusion_source"],
            landscape_dir=(mode["landscape"] or {}).get("landscape_dir"),
            artifact_sha256s=(mode["landscape"] or {}).get(
                "artifact_sha256s"),
            param_set=args.param_set, status="failed", failure=failure)
        sha = write_record(record, path)
        print(f"run_ga: FAILED (record written, status=failed) "
              f"failing_config={cfg or (labels[0] if labels else '?')} "
              f"-> {path} ({sha[:12]})")
        return 1
    record = compose_record(
        result, mode=mode["mode_label"], fitness_id=mode["fitness_id"],
        provider_meta=mode["provider_meta"], cloud_path=mode["cloud"],
        cloud_sha256=mode["cloud_sha256"], d_min=mode["d_min"],
        exclusions=mode["exclusions"],
        exclusion_source=mode["exclusion_source"],
        landscape_dir=(mode["landscape"] or {}).get("landscape_dir"),
        artifact_sha256s=(mode["landscape"] or {}).get("artifact_sha256s"),
        param_set=args.param_set)
    sha = write_record(record, path)
    b = record["best"]
    print(f"run_ga: OK g={args.generations} seed={args.seed:05d} "
          f"mode={mode['mode_label']} best_f={b['f']:.6f} "
          f"config=({b['config'][0]}, {b['config'][1]}, "
          f"{b['config'][2]:+d}, {b['config'][3]:+d}) row={b['row']} "
          f"queries={record['counts']['queries']} "
          f"submitted={record['counts']['submitted']} "
          f"memo_hits={record['counts']['memo_hits']} "
          f"unique={record['counts']['unique']} "
          f"wall={record['wall_seconds']:.2f}s -> {path} ({sha[:12]})")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="run_ga.py -- the GA runner: one invocation = one "
                    "run (exactly G generations)")
    parser.add_argument("--mode", choices=("oracle", "stub", "xtb"),
                        default="oracle",
                        help="oracle (default): lookup into the certified "
                             "landscape; stub: the synthetic provider; "
                             "xtb: the live GFN2-xTB provider")
    parser.add_argument("--generations", type=int, required=True,
                        help="G -- the stop rule's explicit argument (no "
                             "default exists)")
    parser.add_argument("--seed", type=int, required=True,
                        help="the run's PCG64 seed")
    parser.add_argument("--param-set", default="smoke",
                        help="the frozen operator record (ga/params.py)")
    parser.add_argument("--landscape-dir", default=None,
                        help="oracle mode: the certified landscape (default "
                             "data/derived/enumeration_v2/); stub/xtb: the "
                             "exclusions' source landscape")
    parser.add_argument("--cloud", default=None,
                        help="the cloud path (oracle: must match the "
                             "landscape's recorded sha256; stub/xtb: any "
                             "non-default cloud requires --cloud-sha256)")
    parser.add_argument("--cloud-sha256", default=None,
                        help="the cloud's sha256 (required with a "
                             "non-default --cloud in stub/xtb mode)")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help="the write root (default runs/ga; NEVER "
                             "data/derived)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing run identity")
    args = parser.parse_args(argv)
    try:
        return run_one(args)
    except Refused as e:
        print(f"run_ga: REFUSED: {e}")
        return 2
    except Failed as e:
        print(f"run_ga: FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())