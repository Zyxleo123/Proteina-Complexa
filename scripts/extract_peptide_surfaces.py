"""Build peptide-interface surface caches for one PDB or a whole CPSea dataset.

One cache per complex at ``<output-dir>/<hh>/<hh>/<example_id>.surface.npz``
(sharded; legacy flat paths are still found). See
`proteinfoundation.surface.peptide_surface` for the array layout and PepBridge
attribution.

Whole dataset (chains come from the dataset's metadata parquet, `binder_chain_id`):

    python scripts/extract_peptide_surfaces.py \
        --dataset-config configs/dataset/unified/cpsea_peptide.yaml \
        --output-dir surfaces/cpsea \
        --cutoff 4.0 --num-points 96 --seed 0

Single PDB (chains given explicitly; receptor defaults to "every chain that is not the
peptide"):

    python scripts/extract_peptide_surfaces.py \
        --pdb CPSea_data/.../complex.pdb --peptide-chains B \
        --output-dir surfaces/one

A dataset run never aborts on a bad complex: each failure is logged with its reason and
counted, and the run ends with a summary (successes / failures by reason / empty
interfaces / point-count quantiles) plus an optional ``--report`` JSON. Existing valid
caches are skipped by default -- "valid" means readable *and* built with the same cutoff,
sample count, seed and extractor version, so changing any of those rebuilds rather than
silently mixing two settings into one directory. ``--overwrite`` rebuilds everything.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Pin BLAS/OpenMP to 1 thread *before* numpy/scipy import in workers. With
# ProcessPool + cKDTree(workers=-1)/OpenBLAS, 32 children each spawn 32 threads
# and extract collapses from ~50ms to multi-seconds.
for _env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env, "1")

import numpy as np

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from proteinfoundation.surface.peptide_surface import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_CUTOFF,
    DEFAULT_NUM_POINTS,
    DEFAULT_SAS_POINTS_PER_ATOM,
    DEFAULT_SEED,
    EXTRACTOR_VERSION,
    SurfaceExtractionError,
    cache_path_for,
    extract_peptide_surface,
    is_cache_valid,
    load_surface_cache,
    resolve_cache_path,
    resolve_chain_assignment,
    save_surface_cache,
)

logger = logging.getLogger("extract_peptide_surfaces")


# ---------------------------------------------------------------------------
# Work-item discovery
# ---------------------------------------------------------------------------


def _resolve_env_placeholders(text: str) -> str:
    """Expand OmegaConf-style ``${oc.env:NAME}`` / ``${oc.env:NAME,default}`` in a path.

    The dataset YAMLs are Hydra configs, but importing Hydra just to read two filenames
    would pull the whole training stack into a preprocessing script. Only the one
    interpolation form these files actually use is supported; anything else is reported
    as an unresolved path rather than guessed at.
    """
    import re

    def sub(match):
        body = match.group(1)
        name, _, default = body.partition(",")
        return os.environ.get(name.strip(), default.strip())

    return re.sub(r"\$\{oc\.env:([^}]*)\}", sub, text)


def _metadata_files_from_config(config_path: Path, splits: list[str]) -> list[Path]:
    import yaml

    cfg = yaml.safe_load(config_path.read_text())
    dm = cfg.get("datamodule", {})
    keys = {
        "train": "metadata_file",
        "val": "val_metadata_file",
        "test": "test_metadata_file",
    }
    files: list[Path] = []
    for split in splits:
        key = keys.get(split)
        raw = dm.get(key) if key else None
        if raw is None:
            # `test` is genuinely absent from most of our dataset configs; derive it from
            # the train path's naming convention rather than failing the whole run.
            train = dm.get("metadata_file")
            if split == "test" and train:
                raw = str(train).replace("_train.parquet", f"_{split}.parquet")
            else:
                logger.warning("split %r not declared in %s; skipping", split, config_path)
                continue
        path = Path(_resolve_env_placeholders(str(raw)))
        if "${" in str(path):
            logger.warning("unresolved interpolation in %s; skipping", path)
            continue
        if not path.exists():
            logger.warning("metadata file %s (split %s) does not exist; skipping", path, split)
            continue
        files.append(path)
    return files


def _items_from_dataset(
    config_path: Path, splits: list[str], limit: int | None, sample: int | None, seed: int
) -> list[dict]:
    import pandas as pd

    frames = []
    for meta_path in _metadata_files_from_config(config_path, splits):
        df = pd.read_parquet(meta_path)
        logger.info("loaded %d rows from %s", len(df), meta_path)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no usable metadata files resolved from {config_path}")

    df = pd.concat(frames, ignore_index=True)
    for column in ("example_id", "path"):
        if column not in df.columns:
            raise SystemExit(f"metadata is missing required column {column!r}")

    items = []
    for row in df.itertuples(index=False):
        record = row._asdict()
        items.append(
            {
                "example_id": str(record["example_id"]),
                "pdb": str(record["path"]),
                "peptide_chains": str(record.get("binder_chain_id") or "B"),
                "receptor_chains": None,
                "peptide_length": record.get("peptide_length"),
                "cyclization_type": record.get("cyclization_type"),
            }
        )

    if sample is not None and sample < len(items):
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(items), size=sample, replace=False)
        items = [items[i] for i in sorted(picked.tolist())]
    if limit is not None:
        items = items[:limit]
    return items


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _quantiles(values: list[int]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    qs = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100])
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "min": int(qs[0]),
        "p05": round(float(qs[1]), 1),
        "p25": round(float(qs[2]), 1),
        "median": round(float(qs[3]), 1),
        "p75": round(float(qs[4]), 1),
        "p95": round(float(qs[5]), 1),
        "max": int(qs[6]),
    }


def _log_summary(stats: dict) -> None:
    logger.info("=" * 72)
    logger.info(
        "processed %d  |  ok %d (new %d, skipped %d)  |  failed %d",
        stats["total"],
        stats["ok"],
        stats["written"],
        stats["skipped"],
        stats["failed"],
    )
    if stats["failures_by_reason"]:
        logger.info("failures by reason:")
        for reason, count in sorted(stats["failures_by_reason"].items(), key=lambda kv: -kv[1]):
            logger.info("  %-20s %d", reason, count)
    logger.info(
        "empty interfaces: %d (%.2f%% of attempted)",
        stats["empty_interface"],
        100.0 * stats["empty_interface"] / max(stats["attempted"], 1),
    )
    for name, key in (
        ("retained interface vertices", "interface_vertex_counts"),
        ("valid sampled points", "sampled_valid_counts"),
        ("full peptide vertices", "full_vertex_counts"),
    ):
        q = stats["point_counts"].get(key)
        if q:
            logger.info(
                "%-28s n=%d mean=%.1f  min=%d p25=%.1f med=%.1f p75=%.1f max=%d",
                name,
                q["count"],
                q["mean"],
                q["min"],
                q["p25"],
                q["median"],
                q["p75"],
                q["max"],
            )
    under = stats.get("under_filled")
    if under:
        logger.warning(
            "%d/%d caches are padded (fewer than %d interface points)",
            under,
            stats["ok"],
            stats["num_points"],
        )
    logger.info("elapsed %.1fs", stats["elapsed_s"])
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Per-complex work (safe under multiprocessing.spawn — PyMOL is per-process)
# ---------------------------------------------------------------------------


def _process_one(payload: dict) -> dict:
    """Extract one complex. Returns a picklable result dict for the parent process."""
    item = payload["item"]
    example_id = item["example_id"]
    cache = cache_path_for(payload["output_dir"], example_id)  # sharded write path
    out: dict = {"example_id": example_id, "status": "failed", "reason": "", "message": ""}

    if payload["skip_existing"] and not payload["overwrite"]:
        existing = resolve_cache_path(payload["output_dir"], example_id)
        if is_cache_valid(
            existing,
            payload["cutoff"],
            payload["num_points"],
            payload["seed"],
            EXTRACTOR_VERSION,
        ):
            surface = load_surface_cache(existing)
            out.update(
                {
                    "status": "skipped",
                    "num_full": surface.num_full,
                    "num_interface": surface.num_interface,
                    "num_sampled": surface.num_sampled,
                }
            )
            return out

    try:
        backend = payload.get("backend", DEFAULT_BACKEND)
        receptor = item.get("receptor_chains")
        peptide = item["peptide_chains"]
        # SAS resolves "receptor = every non-peptide chain" inside one PDB scan.
        # PyMOL still needs an explicit receptor list before split_chains.
        if backend == "pymol" or receptor is not None:
            receptor, peptide = resolve_chain_assignment(
                item["pdb"], peptide, receptor
            )
        surface = extract_peptide_surface(
            item["pdb"],
            receptor_chains=receptor,
            peptide_chains=peptide,
            cutoff=payload["cutoff"],
            num_points=payload["num_points"],
            seed=payload["seed"],
            solvent_radius=payload["solvent_radius"],
            surface_quality=payload["surface_quality"],
            backend=backend,
            sas_points_per_atom=int(
                payload.get("sas_points_per_atom", DEFAULT_SAS_POINTS_PER_ATOM)
            ),
        )
        for key in ("peptide_length", "cyclization_type"):
            if item.get(key) is not None:
                surface.metadata[key] = (
                    int(item[key]) if key == "peptide_length" else str(item[key])
                )
        surface.metadata["example_id"] = example_id
        save_surface_cache(cache, surface)
    except SurfaceExtractionError as exc:
        out["reason"] = exc.reason
        out["message"] = str(exc)
        return out
    except Exception as exc:  # noqa: BLE001 — one bad complex must not kill the pool
        out["reason"] = "unexpected"
        out["message"] = f"{type(exc).__name__}: {exc}"
        return out

    out.update(
        {
            "status": "written",
            "num_full": surface.num_full,
            "num_interface": surface.num_interface,
            "num_sampled": surface.num_sampled,
        }
    )
    return out


def _run_serial(items: list[dict], payload_base: dict, num_points: int) -> tuple:
    ok = written = skipped = 0
    failures: list[tuple[str, str, str]] = []
    reasons: Counter = Counter()
    full_counts: list[int] = []
    interface_counts: list[int] = []
    valid_counts: list[int] = []
    under_filled = 0
    attempted = 0
    total = len(items)

    for n, item in enumerate(items, start=1):
        result = _process_one({**payload_base, "item": item})
        status = result["status"]
        if status == "skipped":
            full_counts.append(result["num_full"])
            interface_counts.append(result["num_interface"])
            valid_counts.append(result["num_sampled"])
            under_filled += int(result["num_sampled"] < num_points)
            ok += 1
            skipped += 1
            logger.debug("[%d/%d] %s: valid cache, skipped", n, total, result["example_id"])
            continue
        if status == "failed":
            attempted += 1
            reasons[result["reason"]] += 1
            failures.append((result["example_id"], result["reason"], result["message"]))
            logger.error(
                "[%d/%d] %s FAILED (%s): %s",
                n,
                total,
                result["example_id"],
                result["reason"],
                result["message"],
            )
            continue
        attempted += 1
        full_counts.append(result["num_full"])
        interface_counts.append(result["num_interface"])
        valid_counts.append(result["num_sampled"])
        under_filled += int(result["num_sampled"] < num_points)
        ok += 1
        written += 1
        logger.info(
            "[%d/%d] %s: full=%d interface=%d sampled=%d/%d",
            n,
            total,
            result["example_id"],
            result["num_full"],
            result["num_interface"],
            result["num_sampled"],
            num_points,
        )
    return ok, written, skipped, failures, reasons, full_counts, interface_counts, valid_counts, under_filled, attempted


def _run_pool(items: list[dict], payload_base: dict, num_points: int, workers: int) -> tuple:
    """One-node CPU parallelism via ProcessPool.

    ``sas`` uses ``forkserver`` (cheap workers, no PyMOL). ``pymol`` uses ``spawn`` so
    each child owns an isolated interpreter/PyMOL.

    Critical for large corpora:
      * Pre-filter existing caches in the parent (do not ship 2.7M skip jobs to the pool).
      * Bound in-flight futures to ``workers * 4`` — submitting every future upfront OOMs.
      * Pin OpenMP/BLAS to 1 thread (module top) — otherwise 32 workers thrash.
    """
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    ok = written = skipped = 0
    failures: list[tuple[str, str, str]] = []
    reasons: Counter = Counter()
    full_counts: list[int] = []
    interface_counts: list[int] = []
    valid_counts: list[int] = []
    under_filled = 0
    attempted = 0
    total = len(items)

    # Parent-side skip: honour extractor_version / cutoff / seed. Glob once, then only
    # open npz for ids that exist (avoids 2.7M stat calls; also drops stale v1 caches).
    todo = items
    if payload_base["skip_existing"] and not payload_base["overwrite"]:
        from concurrent.futures import ThreadPoolExecutor

        out_dir = Path(payload_base["output_dir"])
        # Flat + sharded layouts (rglob). Keep full paths so we don't re-stat via
        # resolve_cache_path (that doubled ZFS metadata traffic per cache).
        on_disk: dict[str, Path] = {}
        for p in out_dir.rglob("*.surface.npz"):
            if p.is_file() and not p.name.endswith(".npz.tmp"):
                on_disk[p.name[: -len(".surface.npz")]] = p
        logger.info("prefilter: found %d caches on disk; validating...", len(on_disk))

        cutoff = payload_base["cutoff"]
        num_points = payload_base["num_points"]
        seed = payload_base["seed"]

        def _valid(eid: str) -> bool:
            return is_cache_valid(on_disk[eid], cutoff, num_points, seed, EXTRACTOR_VERSION)

        # Metadata zip reads are ZFS-latency bound — threads help a lot vs serial 26ms/cache.
        valid_ids: set[str] = set()
        eids = list(on_disk)
        with ThreadPoolExecutor(max_workers=32) as pool:
            for eid, good in zip(eids, pool.map(_valid, eids, chunksize=64)):
                if good:
                    valid_ids.add(eid)

        todo = []
        for item in items:
            if item["example_id"] in valid_ids:
                ok += 1
                skipped += 1
            else:
                todo.append(item)
        logger.info(
            "prefilter: %d valid caches (of %d on disk), %d remaining (of %d)",
            skipped,
            len(on_disk),
            len(todo),
            total,
        )

    if not todo:
        return (
            ok,
            written,
            skipped,
            failures,
            reasons,
            full_counts,
            interface_counts,
            valid_counts,
            under_filled,
            attempted,
        )

    backend = str(payload_base.get("backend", DEFAULT_BACKEND)).lower()
    # forkserver: one-time import in a server process, then fork workers (fast for SAS).
    # spawn: required for PyMOL (native libs + singleton are not fork-safe).
    start_method = "spawn" if backend == "pymol" else "forkserver"
    ctx = mp.get_context(start_method)
    # Force workers to extract (parent already skipped existing).
    worker_base = {**payload_base, "skip_existing": False, "overwrite": False}
    max_inflight = max(workers * 4, workers)
    logger.info(
        "ProcessPool workers=%d (%s) remaining=%d max_inflight=%d",
        workers,
        start_method,
        len(todo),
        max_inflight,
    )

    item_iter = iter(todo)
    pending: dict = {}
    done = skipped  # count prefiltered skips toward progress denominator

    def _submit(pool: ProcessPoolExecutor) -> None:
        while len(pending) < max_inflight:
            try:
                item = next(item_iter)
            except StopIteration:
                return
            fut = pool.submit(_process_one, {**worker_base, "item": item})
            pending[fut] = item["example_id"]

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        _submit(pool)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                eid = pending.pop(fut)
                done += 1
                try:
                    result = fut.result()
                except Exception as exc:  # noqa: BLE001
                    reasons["pool_crash"] += 1
                    failures.append((eid, "pool_crash", f"{type(exc).__name__}: {exc}"))
                    attempted += 1
                    logger.error("[%d/%d] %s FAILED (pool): %s", done, total, eid, exc)
                    continue

                status = result["status"]
                if status == "skipped":
                    # Race: another writer finished this id between prefilter and worker.
                    ok += 1
                    skipped += 1
                    continue
                if status == "failed":
                    attempted += 1
                    reasons[result["reason"]] += 1
                    failures.append(
                        (result["example_id"], result["reason"], result["message"])
                    )
                    logger.error(
                        "[%d/%d] %s FAILED (%s): %s",
                        done,
                        total,
                        result["example_id"],
                        result["reason"],
                        result["message"],
                    )
                    continue
                attempted += 1
                full_counts.append(result["num_full"])
                interface_counts.append(result["num_interface"])
                valid_counts.append(result["num_sampled"])
                under_filled += int(result["num_sampled"] < num_points)
                ok += 1
                written += 1
                if written <= 20 or written % 50 == 0 or done % 200 == 0:
                    logger.info(
                        "[%d/%d] %s: full=%d interface=%d sampled=%d/%d  (written=%d)",
                        done,
                        total,
                        result["example_id"],
                        result["num_full"],
                        result["num_interface"],
                        result["num_sampled"],
                        num_points,
                        written,
                    )
            _submit(pool)

    return ok, written, skipped, failures, reasons, full_counts, interface_counts, valid_counts, under_filled, attempted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-config", type=Path, help="dataset YAML (whole dataset mode)")
    source.add_argument("--pdb", type=Path, help="a single complex PDB (single mode)")

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF)
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--peptide-chains", default=None, help="comma-separated, single-PDB mode")
    parser.add_argument(
        "--receptor-chains", default=None, help="comma-separated; default = all non-peptide chains"
    )
    parser.add_argument("--example-id", default=None, help="cache basename, single-PDB mode")

    parser.add_argument("--splits", default="train,val,test", help="comma-separated splits")
    parser.add_argument("--limit", type=int, default=None, help="process at most N complexes")
    parser.add_argument(
        "--sample", type=int, default=None, help="randomly pick N complexes (uses --seed)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="optional multi-job partition (items[i::N]); prefer --workers for one node",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="this job's shard in [0, num-shards)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="in-process CPU parallelism on one node (spawn ProcessPool; each owns a PyMOL). "
        "Keep workers × ~2GB under --mem; submitting millions of futures is avoided internally.",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild caches even if a valid one exists"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="do not skip valid existing caches (equivalent to --overwrite for valid caches)",
    )
    parser.set_defaults(skip_existing=True)

    parser.add_argument("--solvent-radius", type=float, default=None)
    parser.add_argument("--surface-quality", type=int, default=None)
    parser.add_argument(
        "--backend",
        choices=("sas", "pymol"),
        default=DEFAULT_BACKEND,
        help="sas = fast Shrake–Rupley + atom receptor (~50×); pymol = PepBridge SES (slow)",
    )
    parser.add_argument(
        "--sas-points-per-atom",
        type=int,
        default=DEFAULT_SAS_POINTS_PER_ATOM,
        help="Fibonacci sphere density for --backend sas",
    )
    parser.add_argument("--report", type=Path, default=None, help="write the summary as JSON")
    parser.add_argument("--failures-log", type=Path, default=None, help="write per-failure TSV")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.pdb is not None and not args.peptide_chains:
        parser.error("--peptide-chains is required with --pdb")

    if args.pdb is not None:
        items = [
            {
                "example_id": args.example_id or args.pdb.name.split(".pdb")[0],
                "pdb": str(args.pdb),
                "peptide_chains": args.peptide_chains,
                "receptor_chains": args.receptor_chains,
            }
        ]
    else:
        items = _items_from_dataset(
            args.dataset_config,
            [s for s in args.splits.split(",") if s],
            args.limit,
            args.sample,
            args.seed,
        )

    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard-index must be in [0, {args.num_shards})")
    if args.num_shards > 1:
        # Stable partition so relaunches / mixed shard counts don't reshuffle ownership
        # of unfinished IDs more than necessary. skip-existing makes overlap safe.
        items = sorted(items, key=lambda x: x["example_id"])
        before = len(items)
        items = items[args.shard_index :: args.num_shards]
        logger.info(
            "shard %d/%d: %d / %d complexes",
            args.shard_index,
            args.num_shards,
            len(items),
            before,
        )

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    logger.info(
        "%d complexes to process -> %s  (workers=%d)",
        len(items),
        args.output_dir,
        args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload_base = {
        "output_dir": str(args.output_dir),
        "cutoff": args.cutoff,
        "num_points": args.num_points,
        "seed": args.seed,
        "skip_existing": args.skip_existing,
        "overwrite": args.overwrite,
        "solvent_radius": args.solvent_radius,
        "surface_quality": args.surface_quality,
        "backend": args.backend,
        "sas_points_per_atom": args.sas_points_per_atom,
    }
    logger.info("backend=%s  extractor_version=%s", args.backend, EXTRACTOR_VERSION)

    started = time.time()
    if args.workers == 1:
        (
            ok,
            written,
            skipped,
            failures,
            reasons,
            full_counts,
            interface_counts,
            valid_counts,
            under_filled,
            attempted,
        ) = _run_serial(items, payload_base, args.num_points)
    else:
        (
            ok,
            written,
            skipped,
            failures,
            reasons,
            full_counts,
            interface_counts,
            valid_counts,
            under_filled,
            attempted,
        ) = _run_pool(items, payload_base, args.num_points, args.workers)

    stats = {
        "extractor_version": EXTRACTOR_VERSION,
        "cutoff": args.cutoff,
        "num_points": args.num_points,
        "seed": args.seed,
        "output_dir": str(args.output_dir),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "workers": args.workers,
        "total": len(items),
        "attempted": attempted,
        "ok": ok,
        "written": written,
        "skipped": skipped,
        "failed": len(failures),
        "empty_interface": reasons.get("empty_interface", 0),
        "failures_by_reason": dict(reasons),
        "under_filled": under_filled,
        "point_counts": {
            "full_vertex_counts": _quantiles(full_counts),
            "interface_vertex_counts": _quantiles(interface_counts),
            "sampled_valid_counts": _quantiles(valid_counts),
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    _log_summary(stats)

    if args.failures_log and failures:
        args.failures_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.failures_log, "w") as fh:
            fh.write("example_id\treason\tmessage\n")
            for example_id, reason, message in failures:
                fh.write(f"{example_id}\t{reason}\t{message.replace(chr(9), ' ')}\n")
        logger.info("wrote %d failures to %s", len(failures), args.failures_log)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = dict(stats)
        report["failures"] = [
            {"example_id": e, "reason": r, "message": m} for e, r, m in failures
        ]
        args.report.write_text(json.dumps(report, indent=2))
        logger.info("wrote report to %s", args.report)

    # Non-zero only when nothing succeeded: partial failure on a big dataset is expected
    # and is reported in the summary, not by killing the caller's pipeline.
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
