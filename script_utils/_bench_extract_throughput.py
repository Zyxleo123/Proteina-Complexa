"""Microbench: concurrent SAS extract+save to local vs ZFS."""
from __future__ import annotations

import os

for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from proteinfoundation.surface.peptide_surface import (  # noqa: E402
    cache_path_for,
    extract_peptide_surface,
    load_surface_cache,
    save_surface_cache,
)


def work(args):
    pdb, pep, eid, out_dir = args
    t0 = time.perf_counter()
    surf = extract_peptide_surface(pdb, None, pep, backend="sas")
    t1 = time.perf_counter()
    save_surface_cache(cache_path_for(out_dir, eid), surf)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def run_to(items, out_dir: Path, workers: int, start_method: str = "forkserver") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.rglob("*.surface.npz"):
        p.unlink()
    payload = [(*it, str(out_dir)) for it in items]
    ctx = mp.get_context(start_method)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        times = list(pool.map(work, payload, chunksize=8))
    wall = time.perf_counter() - t0
    ext = sum(a for a, _ in times) / len(times)
    sav = sum(b for _, b in times) / len(times)
    print(
        f"{out_dir}: wall={wall:.2f}s rate={len(items) / wall:.1f}/s  "
        f"mean_extract={ext:.3f}s mean_save={sav:.3f}s workers={workers} ({start_method})"
    )


def main() -> None:
    caches = list(
        Path("/zfsauton/scratch/yixiz/Proteina-Complexa/surfaces/cpsea").glob(
            "*.surface.npz"
        )
    )[:512]
    items = []
    for c in caches:
        s = load_surface_cache(c)
        m = s.metadata
        items.append(
            (
                m["source_pdb"],
                m["peptide_chains"],
                c.name[: -len(".surface.npz")],
            )
        )
    print(f"items={len(items)}")
    run_to(items, Path("/tmp/surf_bench_local"), 32)
    run_to(
        items,
        Path("/zfsauton/scratch/yixiz/Proteina-Complexa/surfaces/_bench_shard"),
        32,
    )
    run_to(items, Path("/tmp/surf_bench_w8"), 8)


if __name__ == "__main__":
    main()
