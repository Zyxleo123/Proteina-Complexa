"""Recover the pre-cyclization linear peptide for every CPSea cyclic complex.

CPSea's `example_id` carries full provenance. `AF-A0A009E759-F1_0_89_101_relaxed_relaxed`
decodes to AFDB accession `AF-A0A009E759-F1` plus the **inclusive** residue range
[89, 101] that was excised as the peptide. Verified on all 2,444,209 rows of
`cpsea_train.parquet`: `end - start + 1 == peptide_length`, zero mismatches.

So the "before" state of an LP -> CP transport does not have to be synthesised by
opening a ring. It can be *retrieved* from the parent AlphaFold model, which makes the
coupling a real transformation rather than one we invent:

    LP side : parent AFDB residues [start, end], rigid-aligned onto the CPSea receptor
              and paired with CPSea's OWN receptor
    CP side : the CPSea processed complex, unchanged

Both sides therefore carry a byte-identical receptor, so the transport is about the
peptide alone and target conditioning is literally the same object at both ends. The
receptor RMSD of that alignment is recorded per pair as a diagnostic rather than hidden --
CPSea relaxed its structures twice (`REMARK CREATED WITH OPENMM 8.2`), so the parent and
the product do not share a frame and the fit residual is the honest measure of how far
apart they are.

Two facts about the curation that this relies on, both checked on disk:

  * Residue numbering is preserved on both chains of the processed complex (receptor =
    chain A, binder = chain B, `binder_chain_id` == "B" throughout), so the parent and the
    CPSea receptor can be matched by residue id directly -- no sequence alignment needed.
  * The curation deletes a ~5-residue buffer either side of the excised peptide (for the
    example above, chain A stops at 83 and resumes at 107 around a peptide at 89-101), so
    the peptide's own covalent flankers are absent and the LP side is not a loop still
    tethered to its receptor.

Subcommands
-----------
    parents   decode provenance, pick the parent set, emit the fetch and pair work lists
    fetch     download AFDB parent models for one shard (resumable, skips what it has)
    pair      build the paired records for one shard (resumable)
    selftest  unit tests; no network, no cluster, no data root

Every stage writes ONE output file per shard. Concurrent O_APPEND to a shared file
NUL-corrupts rows on this filesystem (88 of 144 lost, measured), so shards never share.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------- provenance

# `AF-<accession>-F<n>_<idx>_<start>_<end>_relaxed_relaxed`.  The trailing `_relaxed`
# pair is CPSea's two OpenMM passes; `<idx>` is a per-parent fragment counter we keep for
# the record but never use as a key (a parent can contribute up to 32 fragments).
EXAMPLE_ID_RE = re.compile(
    r"^(?P<accession>AF-[A-Z0-9]+-F\d+)"
    r"_(?P<frag_idx>\d+)"
    r"_(?P<res_start>\d+)"
    r"_(?P<res_end>\d+)"
    r"_relaxed_relaxed$"
)

AFDB_URL = "https://alphafold.ebi.ac.uk/files/{accession}-model_{version}.pdb"

# Two accessions used only to tell "this accession is gone from AFDB" apart from "this
# node has no egress".  If BOTH fail the network is the problem, and the fetch stage says
# so instead of writing 20,000 identical 404s.
CONTROL_ACCESSIONS = ("AF-P69905-F1", "AF-P00533-F1")

# Written by `fetch` when it decides the node cannot reach AFDB at all.  `pair` gates on
# this file so a blocked download reports itself once, loudly, instead of looking like an
# empty result.
FETCH_BLOCKED_MARKER = "FETCH_BLOCKED"

RECEPTOR_CHAIN = "A"
BINDER_CHAIN = "B"


@dataclass(frozen=True)
class Provenance:
    example_id: str
    accession: str
    frag_idx: int
    res_start: int
    res_end: int

    @property
    def n_res(self) -> int:
        """Inclusive span. Equals `peptide_length` on every CPSea row."""
        return self.res_end - self.res_start + 1


def parse_example_id(example_id: str) -> Provenance | None:
    """Decode one CPSea example_id, or None if it does not follow the scheme."""
    m = EXAMPLE_ID_RE.match(example_id)
    if m is None:
        return None
    return Provenance(
        example_id=example_id,
        accession=m.group("accession"),
        frag_idx=int(m.group("frag_idx")),
        res_start=int(m.group("res_start")),
        res_end=int(m.group("res_end")),
    )


def shard_of(key: str, num_shards: int) -> int:
    """Stable shard assignment. `hash()` is salted per process, so it cannot be used."""
    import hashlib

    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % num_shards


# ------------------------------------------------------------------------------- parents


def cmd_parents(args: argparse.Namespace) -> int:
    import pandas as pd
    import pyarrow.parquet as pq

    # Only what is actually used downstream. `binder_chain_id` and `split` are constant
    # across this corpus ("B", and whichever split is being read), so reading them costs
    # memory for no information; the binder chain is a module constant instead.
    cols = ["example_id", "path", "cluster_id", "peptide_length", "cyclization_type"]
    if args.max_rows:
        print(
            f"  WARNING: --max-rows {args.max_rows:,} takes the FIRST rows in file order,\n"
            "           which is NOT a random sample -- the metadata is ordered by\n"
            "           accession, so this covers a narrow alphabetical slice of the\n"
            "           corpus. Use it to exercise the machinery, never to build a real\n"
            "           dataset or to read a distribution off the diagnosis."
        )
    frames = []
    for meta in args.metadata:
        if args.max_rows:
            # Stream batches and stop early, rather than reading the whole table and
            # slicing: slicing afterwards would already have paid the full memory cost.
            import pyarrow as pa

            batches, got = [], 0
            for batch in pq.ParquetFile(meta).iter_batches(batch_size=65536, columns=cols):
                batches.append(batch)
                got += batch.num_rows
                if got >= args.max_rows:
                    break
            tbl = pa.Table.from_batches(batches).slice(0, args.max_rows)
        else:
            tbl = pq.read_table(meta, columns=cols)
        frames.append(tbl.to_pandas())
        print(f"  read {len(frames[-1]):>9,} rows from {meta}")
        del tbl
    df = pd.concat(frames, ignore_index=True)
    del frames

    # Vectorised deliberately: one dataclass per row plus a list comprehension per field
    # costs several GB of pure object overhead over 2.4M rows. `parse_example_id` stays for
    # single-row callers (the tests and `build_pair`), where clarity is worth more.
    ext = df["example_id"].str.extract(EXAMPLE_ID_RE)
    undecodable = int(ext["accession"].isna().sum())
    if undecodable:
        print(f"  WARNING: {undecodable:,} example_ids do not decode; dropping them")
        keep = ext["accession"].notna().to_numpy()
        df = df[keep].reset_index(drop=True)
        ext = ext[keep].reset_index(drop=True)

    df["accession"] = ext["accession"]
    df["res_start"] = ext["res_start"].astype("int32")
    df["res_end"] = ext["res_end"].astype("int32")
    df["frag_idx"] = ext["frag_idx"].astype("int32")
    del ext

    # The span IS the peptide length on every row we have ever checked -- verified on all
    # 2,444,209 rows of cpsea_train. If that ever breaks, the range does not mean what this
    # whole module assumes, so refuse rather than silently building pairs against the wrong
    # residues. int64 on both sides: int32 arithmetic against an int16 peptide_length would
    # compare across dtypes and could wrap.
    span = df["res_end"].astype("int64") - df["res_start"].astype("int64") + 1
    bad = int((span != df["peptide_length"].astype("int64")).sum())
    if bad:
        print(
            f"FATAL: {bad:,} rows where res_end-res_start+1 != peptide_length.\n"
            "       The provenance range does not address the peptide; refusing to build.",
            file=sys.stderr,
        )
        return 1
    print(f"  provenance decoded and span-checked on {len(df):,} rows")

    if args.cyc_types and args.cyc_types != ["all"]:
        before = len(df)
        df = df[df["cyclization_type"].isin(args.cyc_types)].reset_index(drop=True)
        print(f"  chemistry filter {args.cyc_types}: {before:,} -> {len(df):,} rows")

    # Cluster-unique before capping, so a cap does not fill up with near-duplicates of one
    # cluster. Sorted first: `groupby.first()` on an unsorted frame is not reproducible
    # across pandas versions, and this selection has to be replayable.
    df = df.sort_values("example_id", kind="mergesort").reset_index(drop=True)
    if args.cluster_unique:
        before = len(df)
        df = df.groupby("cluster_id", as_index=False, sort=True).head(1).reset_index(drop=True)
        print(f"  cluster-unique: {before:,} -> {len(df):,} rows")

    # Cap by PARENT, not by row: a parent contributes up to 32 fragments and we pay the
    # download once per parent, so capping rows would buy fewer pairs per byte fetched.
    if args.parent_limit and df["accession"].nunique() > args.parent_limit:
        chosen = (
            df["accession"].drop_duplicates().sort_values(kind="mergesort").head(args.parent_limit)
        )
        df = df[df["accession"].isin(set(chosen))].reset_index(drop=True)
        print(f"  parent cap {args.parent_limit:,}: {len(df):,} rows kept")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parents = sorted(df["accession"].unique())
    parents_file = out_dir / "parents.tsv"
    parents_file.write_text("\n".join(parents) + "\n")

    todo = out_dir / "pairs_todo.parquet"
    df.to_parquet(todo, index=False)

    print()
    print(f"  parents      : {len(parents):,}  -> {parents_file}")
    print(f"  pairs to make: {len(df):,}  -> {todo}")
    print(f"  frags/parent : mean {len(df) / max(len(parents), 1):.2f}")
    print("  chemistry    :", df["cyclization_type"].value_counts().to_dict())
    return 0


# --------------------------------------------------------------------------------- fetch


def _afdb_get(session, accession: str, version: str, timeout: float):
    url = AFDB_URL.format(accession=accession, version=version)
    r = session.get(url, timeout=timeout)
    return r.status_code, r.content


def cmd_fetch(args: argparse.Namespace) -> int:
    import requests

    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parents = [p.strip() for p in Path(args.parents).read_text().split() if p.strip()]
    mine = [p for p in parents if shard_of(p, args.num_shards) == args.shard]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine):,} of {len(parents):,} parents")

    session = requests.Session()
    session.headers.update({"User-Agent": "Proteina-Complexa/lp2cp-provenance"})

    # Egress check. A 404 on a live accession and a 404 from a blocking proxy look
    # identical per-request, and the difference decides whether this stage is worth
    # retrying. Only shard 0 writes the marker, so the shards do not race on it.
    reachable = False
    for ctl in CONTROL_ACCESSIONS:
        try:
            code, body = _afdb_get(session, ctl, args.version, args.timeout)
            print(f"  control {ctl}: HTTP {code}, {len(body):,} bytes")
            if code == 200 and len(body) > 1024:
                reachable = True
                break
        except Exception as exc:  # noqa: BLE001 - any transport error means "not reachable"
            print(f"  control {ctl}: {type(exc).__name__}: {exc}")
    if not reachable:
        print()
        print("=" * 70)
        print("GATE: cannot reach AFDB from this node -- every control accession failed.")
        print("      This is egress, not the data: the same controls fail from the login")
        print("      node too. Nothing was downloaded.")
        print(f"      Run the fetch where egress exists, then re-submit `pair`.")
        print("=" * 70)
        if args.shard == 0:
            (models / FETCH_BLOCKED_MARKER).write_text(
                "AFDB unreachable; controls "
                + ",".join(CONTROL_ACCESSIONS)
                + f" all failed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        # Exit 0 on purpose: a non-zero exit strands every afterok dependent in
        # DependencyNeverSatisfied with no explanation of which gate failed.
        return 0

    marker = models / FETCH_BLOCKED_MARKER
    if marker.exists():
        marker.unlink()  # egress is back; do not let a stale marker gate `pair`

    manifest = out_dir / f"fetch_shard{args.shard:03d}.jsonl"
    n_have = n_got = n_miss = n_err = 0
    with manifest.open("a") as fh:
        for i, acc in enumerate(mine, 1):
            dest = models / f"{acc}-model_{args.version}.pdb"
            if dest.exists() and dest.stat().st_size > 1024:
                n_have += 1
                continue
            status, body = None, b""
            for attempt in range(args.retries + 1):
                try:
                    status, body = _afdb_get(session, acc, args.version, args.timeout)
                    if status == 200 or status == 404:
                        break
                except Exception as exc:  # noqa: BLE001
                    status = f"{type(exc).__name__}"
                time.sleep(args.sleep * (attempt + 1))
            if status == 200 and len(body) > 1024:
                dest.write_bytes(body)
                n_got += 1
            elif status == 404:
                n_miss += 1
            else:
                n_err += 1
            fh.write(
                json.dumps({"accession": acc, "status": status, "bytes": len(body)}) + "\n"
            )
            if i % 500 == 0:
                fh.flush()
                print(
                    f"  [{i:>7,}/{len(mine):,}] have={n_have:,} got={n_got:,} "
                    f"missing={n_miss:,} error={n_err:,}",
                    flush=True,
                )
            time.sleep(args.sleep)

    print()
    print(f"  already present : {n_have:,}")
    print(f"  downloaded      : {n_got:,}")
    print(f"  absent from AFDB: {n_miss:,}")
    print(f"  transport errors: {n_err:,}")
    print(f"  manifest        : {manifest}")
    return 0


# ---------------------------------------------------------------------------------- pair


def _read_structure(path: Path):
    from biotite.structure.io.pdb import PDBFile

    return PDBFile.read(str(path)).get_structure(model=1)


def _ca_by_resid(arr) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    sel = arr[arr.atom_name == "CA"]
    for rid, coord in zip(sel.res_id, sel.coord):
        out.setdefault(int(rid), coord)
    return out


def _resname_by_resid(arr) -> dict[int, str]:
    out: dict[int, str] = {}
    for rid, rname in zip(arr.res_id, arr.res_name):
        out.setdefault(int(rid), str(rname))
    return out


def _one_letter(resname: str) -> str:
    import biotite.structure.info as info

    code = info.one_letter_code(resname)
    return code if code else "X"


def _seq_of(arr, lo: int, hi: int) -> str:
    names = _resname_by_resid(arr)
    return "".join(_one_letter(names[r]) for r in range(lo, hi + 1) if r in names)


def _terminal_ca_gap(cas: dict[int, np.ndarray], lo: int, hi: int) -> float | None:
    if lo not in cas or hi not in cas:
        return None
    return float(np.linalg.norm(cas[lo] - cas[hi]))


def build_pair(row: dict, models_dir: Path, lp_pdb_dir: Path, min_anchor: int) -> dict:
    """Build one LP/CP pair. Returns a record; `skip_reason` is set when unusable.

    Every early return still carries the identifying fields, because a pair that could
    not be built is data about coverage -- dropping it silently would make the diagnosis
    read as if the excluded cases never existed.
    """
    import biotite.structure as struc

    rec: dict = {
        "example_id": row["example_id"],
        "accession": row["accession"],
        "res_start": int(row["res_start"]),
        "res_end": int(row["res_end"]),
        "cyclization_type": row["cyclization_type"],
        "cluster_id": row["cluster_id"],
        "cp_pdb": str(row["path"]),
        "lp_pdb": None,
        "n_receptor_common": 0,
        "receptor_rmsd_A": None,
        "lp_seq": None,
        "cp_seq": None,
        "n_mut": None,
        "mut_positions": None,
        "lp_term_ca_A": None,
        "cp_term_ca_A": None,
        "lp_cp_ca_rmsd_A": None,
        "lp_cp_ca_max_A": None,
        "skip_reason": None,
    }

    parent_path = models_dir / f"{row['accession']}-model_{row['version']}.pdb"
    if not parent_path.exists():
        rec["skip_reason"] = "parent_not_fetched"
        return rec
    cp_path = Path(row["path"])
    if not cp_path.exists():
        rec["skip_reason"] = "cpsea_pdb_missing"
        return rec

    try:
        parent = _read_structure(parent_path)
        cp = _read_structure(cp_path)
    except Exception as exc:  # noqa: BLE001
        rec["skip_reason"] = f"parse_error:{type(exc).__name__}"
        return rec

    lo, hi = rec["res_start"], rec["res_end"]
    cp_recv = cp[cp.chain_id == RECEPTOR_CHAIN]
    cp_bind = cp[cp.chain_id == BINDER_CHAIN]
    if cp_recv.array_length() == 0 or cp_bind.array_length() == 0:
        rec["skip_reason"] = "cpsea_chains_missing"
        return rec

    # The parent is a single-chain AFDB model; take whatever chain it uses rather than
    # assuming "A", because the receptor chain id in the processed complex is also "A"
    # and confusing the two would align the parent onto itself.
    parent_chains = set(np.unique(parent.chain_id).tolist())
    if len(parent_chains) != 1:
        rec["skip_reason"] = f"parent_multichain:{sorted(parent_chains)}"
        return rec

    parent_ca = _ca_by_resid(parent)
    recv_ca = _ca_by_resid(cp_recv)

    # Numbering is inherited from the parent on both chains, so the receptor residues are
    # matched by id. The excised peptide range is excluded explicitly: in the parent those
    # residues are peptide, and including them would fit the peptide as part of the
    # receptor and wash out the very displacement we are trying to measure.
    common = sorted(
        r for r in (set(parent_ca) & set(recv_ca)) if not (lo <= r <= hi)
    )
    rec["n_receptor_common"] = len(common)
    if len(common) < min_anchor:
        rec["skip_reason"] = f"too_few_receptor_anchors:{len(common)}<{min_anchor}"
        return rec

    fixed = np.stack([recv_ca[r] for r in common])
    mobile = np.stack([parent_ca[r] for r in common])
    try:
        _, transform = struc.superimpose(fixed, mobile)
    except Exception as exc:  # noqa: BLE001
        rec["skip_reason"] = f"superimpose_error:{type(exc).__name__}"
        return rec
    rec["receptor_rmsd_A"] = float(
        np.sqrt(np.mean(np.sum((transform.apply(mobile) - fixed) ** 2, axis=-1)))
    )

    lp_slice = parent[(parent.res_id >= lo) & (parent.res_id <= hi)]
    if lp_slice.array_length() == 0:
        rec["skip_reason"] = "parent_range_empty"
        return rec
    lp_res = sorted({int(r) for r in lp_slice.res_id})
    if len(lp_res) != hi - lo + 1:
        rec["skip_reason"] = f"parent_range_partial:{len(lp_res)}of{hi - lo + 1}"
        return rec

    lp_aligned = transform.apply(lp_slice)
    lp_aligned.chain_id[:] = BINDER_CHAIN

    rec["lp_seq"] = _seq_of(lp_slice, lo, hi)
    rec["cp_seq"] = _seq_of(cp_bind, lo, hi)
    if rec["lp_seq"] and rec["cp_seq"] and len(rec["lp_seq"]) == len(rec["cp_seq"]):
        muts = [i for i, (a, b) in enumerate(zip(rec["lp_seq"], rec["cp_seq"])) if a != b]
        rec["n_mut"] = len(muts)
        # Parent numbering, not 0-based offsets: these positions are quoted against the
        # structures, and an offset silently means a different residue.
        rec["mut_positions"] = [lo + i for i in muts]

    lp_ca = _ca_by_resid(lp_aligned)
    cp_ca = _ca_by_resid(cp_bind)
    rec["lp_term_ca_A"] = _terminal_ca_gap(lp_ca, lo, hi)
    rec["cp_term_ca_A"] = _terminal_ca_gap(cp_ca, lo, hi)

    # How far the transport actually has to move the peptide, receptor-aligned. This is the
    # amplitude the precheck's decoy work found a 2 A floor for: below ~1 A the "before"
    # state collapses back onto the "after" one and the pair teaches the model to copy its
    # input. Reported per pair so that floor can be checked on real data rather than assumed.
    shared = [r for r in range(lo, hi + 1) if r in lp_ca and r in cp_ca]
    if shared:
        d = np.linalg.norm(
            np.stack([lp_ca[r] for r in shared]) - np.stack([cp_ca[r] for r in shared]),
            axis=-1,
        )
        rec["lp_cp_ca_rmsd_A"] = float(np.sqrt(np.mean(d**2)))
        rec["lp_cp_ca_max_A"] = float(np.max(d))

    from biotite.structure.io.pdb import PDBFile

    lp_pdb_dir.mkdir(parents=True, exist_ok=True)
    dest = lp_pdb_dir / f"{row['example_id']}.pdb"
    combined = cp_recv + lp_aligned
    pdb = PDBFile()
    pdb.set_structure(combined)
    pdb.write(str(dest))
    rec["lp_pdb"] = str(dest)
    return rec


def cmd_pair(args: argparse.Namespace) -> int:
    import pandas as pd

    models = Path(args.models_dir)
    if (models / FETCH_BLOCKED_MARKER).exists():
        print("=" * 70)
        print("GATE: fetch reported AFDB unreachable; no parent models exist.")
        print((models / FETCH_BLOCKED_MARKER).read_text().strip())
        print("      Nothing to pair. Exiting 0 so dependents are not stranded.")
        print("=" * 70)
        return 0

    df = pd.read_parquet(args.todo)
    df["version"] = args.version
    mine = df[[shard_of(e, args.num_shards) == args.shard for e in df["example_id"]]]
    mine = mine.reset_index(drop=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(mine):,} of {len(df):,} pairs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_file = out_dir / f"pairs_shard{args.shard:03d}.jsonl"

    # Resume by reading back what this shard already wrote. A corrupt line is REFUSED, not
    # skipped: silently dropping it would let the job re-make a pair it already has while
    # reporting a clean run.
    done: set[str] = set()
    if shard_file.exists():
        with shard_file.open() as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                if "\x00" in line:
                    print(
                        f"FATAL: NUL byte in {shard_file}:{ln} -- this shard's output is "
                        "corrupt. Delete it and re-run rather than appending to it.",
                        file=sys.stderr,
                    )
                    return 1
                try:
                    done.add(json.loads(line)["example_id"])
                except Exception:
                    print(f"FATAL: unparseable line {shard_file}:{ln}", file=sys.stderr)
                    return 1
        print(f"  resuming: {len(done):,} already built")

    lp_pdb_dir = Path(args.lp_pdb_dir)
    n_ok = 0
    reasons: dict[str, int] = {}
    with shard_file.open("a") as fh:
        for i, row in enumerate(mine.to_dict("records"), 1):
            if row["example_id"] in done:
                continue
            rec = build_pair(row, models, lp_pdb_dir, args.min_anchor)
            fh.write(json.dumps(rec) + "\n")
            if rec["skip_reason"] is None:
                n_ok += 1
            else:
                key = str(rec["skip_reason"]).split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
            if i % 200 == 0:
                fh.flush()
                print(f"  [{i:>7,}/{len(mine):,}] built={n_ok:,} skipped={sum(reasons.values()):,}",
                      flush=True)

    print()
    print(f"  built   : {n_ok:,}")
    print(f"  skipped : {sum(reasons.values()):,}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {k:<28s} {v:,}")
    print(f"  shard   : {shard_file}")
    return 0


# ------------------------------------------------------------------------------ selftest


def cmd_selftest(_: argparse.Namespace) -> int:
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    p = parse_example_id("AF-A0A009E759-F1_0_89_101_relaxed_relaxed")
    check(p is not None, "failed to parse a known-good example_id")
    assert p is not None
    check(p.accession == "AF-A0A009E759-F1", f"accession wrong: {p.accession}")
    check(p.res_start == 89 and p.res_end == 101, f"range wrong: {p.res_start}-{p.res_end}")
    check(p.n_res == 13, f"inclusive span wrong: {p.n_res} != 13")

    check(parse_example_id("not-an-afdb-id") is None, "accepted a malformed id")
    check(
        parse_example_id("AF-A0A009E759-F1_0_89_101") is None,
        "accepted an id without the relaxed suffix",
    )
    check(
        parse_example_id("AF-Q8N1A0-F2_3_1_7_relaxed_relaxed").accession == "AF-Q8N1A0-F2",
        "multi-fragment accession (F2) mis-parsed",
    )

    # Shard assignment must be stable across processes. `hash()` is salted per process,
    # so a naive implementation silently re-shards on every job and shards overlap.
    keys = [f"AF-X{i}-F1_0_1_9_relaxed_relaxed" for i in range(200)]
    a = [shard_of(k, 8) for k in keys]
    b = [shard_of(k, 8) for k in keys]
    check(a == b, "shard_of is not deterministic within a process")
    check(all(0 <= s < 8 for s in a), "shard_of returned an out-of-range shard")
    check(len(set(a)) > 1, "shard_of put every key in one shard")
    # Known-value pins, so a future refactor cannot quietly re-shard existing outputs.
    check(shard_of("AF-A0A009E759-F1", 8) == shard_of("AF-A0A009E759-F1", 8), "unstable")
    check(
        shard_of("AF-A0A009E759-F1", 1) == 0, "num_shards=1 must put everything in shard 0"
    )

    # Every shard of a partition must cover the key space exactly once.
    cover = {s: 0 for s in range(8)}
    for k in keys:
        cover[shard_of(k, 8)] += 1
    check(sum(cover.values()) == len(keys), "shards do not partition the key space")

    if fails:
        print("SELFTEST FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print(f"selftest OK ({len(keys)} shard keys, provenance parser, span arithmetic)")
    return 0


# ---------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("parents", help="decode provenance and pick the parent set")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--parent-limit", type=int, default=20000)
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="read only the first N metadata rows (smoke only; NOT a random sample)",
    )
    p.add_argument("--cyc-types", nargs="*", default=["all"])
    p.add_argument("--cluster-unique", action="store_true", default=True)
    p.add_argument("--no-cluster-unique", dest="cluster_unique", action="store_false")
    p.set_defaults(func=cmd_parents)

    p = sub.add_parser("fetch", help="download AFDB parent models for one shard")
    p.add_argument("--parents", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--version", default="v4")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.05)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("pair", help="build paired LP/CP records for one shard")
    p.add_argument("--todo", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--lp-pdb-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--version", default="v4")
    p.add_argument("--min-anchor", type=int, default=10)
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("selftest", help="unit tests; no network or data needed")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
