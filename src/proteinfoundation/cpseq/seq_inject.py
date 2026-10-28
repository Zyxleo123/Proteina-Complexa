"""Hand the sequence-conditioned generator a sequence that is not the dataset's.

`eval_val_gen_testset.py --seq-cond` reveals the binder sequence that the *batch* carries,
so the only peptide the model can be asked about is the native one. Asking whether a
**searched** sequence is good for the model -- rather than good for the predictor that
scored it -- needs the requested sequence substituted into `batch["residue_type"]` before
`validation_step_generate` resolves the conditioning mask.

Three things this module refuses to guess.

**Attribution is by the batch's own native sequence, never by batch order.** The val
dataloader's ordering is not part of any contract, and a wrong pairing would attribute a
designed peptide to the wrong receptor -- which looks like a result rather than a bug. Each
batch row is identified by decoding the native sequence it already carries and looking that
up, so the pairing is checkable from the batch alone.

**The residue-type alphabet is verified, not assumed.** `residue_type` is written from
atomworks' `restype_idx` (`atomworks_transforms.py:104`), while the PDB writers decode
through OpenFold's `residue_constants.restypes` (`transforms.py:2302`). Those two agreeing
is an assumption, and if it were wrong every injected sequence would be silently permuted.
Decoding the native sequence and requiring it to match a known reference makes the
assumption self-checking: a wrong alphabet matches nothing and the run refuses to start.

**A request whose length differs from the native peptide is an error, not a resize.** The
searches here substitute residues and never change length, so a length mismatch means the
request was paired with the wrong parent. Padding or truncating it would hide that.

The emitted provenance is written beside the structure dump as `requested_sequences.json`,
which is how `cpseq_score_seqcond.py` learns to measure compliance against the *requested*
peptide instead of the dataset's reference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from openfold.np import residue_constants as rc

# The decode table. Imported, never written out here: a literal would be a second copy of
# the ordering this module exists to avoid assuming.
_AA_BY_INDEX: tuple[str, ...] = tuple(rc.restypes)
_INDEX_BY_AA: dict[str, int] = {a: i for i, a in enumerate(_AA_BY_INDEX)}
UNKNOWN_INDEX = rc.restype_num  # 20; anything >= this is not a standard residue


class SequenceInjectionError(RuntimeError):
    """Raised for any condition under which the run would produce a plausible wrong result."""


@dataclass
class Request:
    """One requested peptide, and enough provenance to read the result afterwards."""

    parent_id: str
    native_sequence: str
    requested_sequence: str
    label: str = "requested"
    extra: dict = field(default_factory=dict)

    @property
    def n_substitutions(self) -> int:
        return sum(a != b for a, b in zip(self.native_sequence, self.requested_sequence))

    @property
    def is_native(self) -> bool:
        return self.requested_sequence == self.native_sequence

    def to_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "native_sequence": self.native_sequence,
            "requested_sequence": self.requested_sequence,
            "label": self.label,
            "n_substitutions": self.n_substitutions,
            "is_native": self.is_native,
            **({"extra": self.extra} if self.extra else {}),
        }


def decode_sequence(residue_type_row: torch.Tensor, mask_row: torch.Tensor) -> str:
    """One batch row's peptide sequence, over the masked (real) positions only.

    Non-standard or padding indices decode to 'X' so a partial decode is visible rather
    than quietly shortened.
    """
    ids = residue_type_row[mask_row.bool()].tolist()
    return "".join(_AA_BY_INDEX[i] if 0 <= i < len(_AA_BY_INDEX) else "X" for i in ids)


def encode_sequence(seq: str) -> list[int]:
    """Residue-type indices for a one-letter sequence. Refuses anything non-standard."""
    bad = sorted({c for c in seq if c not in _INDEX_BY_AA})
    if bad:
        raise SequenceInjectionError(
            f"sequence {seq!r} contains residues outside the 20 standard types: {bad}. "
            "The model's residue-type feature has no channel for them."
        )
    return [_INDEX_BY_AA[c] for c in seq]


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def native_sequences(dataset_dir: Path) -> dict[str, str]:
    """{parent_id -> native peptide sequence} from a prepared dataset's manifest.

    Uniqueness of the sequences is NOT required here. It is required only by consumers that
    identify a parent BY its native sequence -- `load_requests`, which keys its result that
    way so a batch row can be matched without trusting the dataloader's order. A consumer
    that iterates parents explicitly, like a structure predictor folding a list, is
    unaffected by two parents sharing a peptide, and `pepbench_valid` has two such pairs.
    """
    import pandas as pd

    targets = pd.read_csv(Path(dataset_dir) / "manifest" / "targets.csv")
    return dict(zip(targets.parent_id.astype(str), targets.reference_sequence.astype(str)))


def _refuse_ambiguous_native(natives: dict[str, str], parent: str, native: str, where: str) -> None:
    """Refuse a parent whose native peptide is shared with another parent in the manifest.

    Scoped to the parent being requested, and checked against the WHOLE manifest rather
    than against the request list: a batch row belonging to the OTHER parent would decode
    to the same sequence and collect this parent's request, even though that parent was
    never asked for. Requesting one of a colliding pair is therefore already ambiguous.
    """
    sharers = sorted(p for p, s in natives.items() if s == native)
    if len(sharers) > 1:
        raise SequenceInjectionError(
            f"{where}: parent {parent}'s native peptide {native!r} is shared with "
            f"{[p for p in sharers if p != parent]}. A batch row cannot be attributed to one of "
            "them by its decoded sequence, so an injected request could land on the wrong "
            "receptor. Fold these with a consumer that addresses parents explicitly, or give "
            "the dataset a per-row id."
        )


def _rows_from_path(path: Path) -> list[dict]:
    """Rows from .json (list, or {'rows': [...]}), .jsonl, or .csv -- whichever it is."""
    text = Path(path).read_text()
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("rows", "requests", "candidates"):
                if isinstance(obj.get(key), list):
                    return obj[key]
            raise SequenceInjectionError(
                f"{path} is a JSON object with no 'rows'/'requests'/'candidates' list."
            )
        if not isinstance(obj, list):
            raise SequenceInjectionError(f"{path} must hold a list of rows.")
        return obj
    if suffix == ".csv":
        return list(csv.DictReader(text.splitlines()))
    raise SequenceInjectionError(f"unsupported sequence source {path.suffix!r}; use .json, .jsonl or .csv")


def load_requests(path: Path, dataset_dir: Path, label: str | None = None) -> dict[str, Request]:
    """{native sequence -> Request}, the form `apply_to_batch` looks up.

    A row needs a `sequence` (the peptide to request) and a `parent_id`. The parent's
    native sequence comes from the dataset manifest, so a row cannot name a parent that
    was never prepared, and a row's own `native_sequence`, if it carries one, is checked
    against the manifest rather than trusted.

    `label` selects one side of a paired file. One generation run asks one question, so a
    file holding both the searched peptide and its native would otherwise be refused for
    naming a parent twice; filtering by label lets the two runs of a paired comparison come
    from one file, which is what makes them the same pair.
    """
    natives = native_sequences(dataset_dir)
    rows = _rows_from_path(Path(path))
    if not rows:
        raise SequenceInjectionError(f"{path} holds no rows.")
    if label is not None:
        present = sorted({str(r.get("label") or "requested") for r in rows})
        rows = [r for r in rows if str(r.get("label") or "requested") == label]
        if not rows:
            raise SequenceInjectionError(
                f"{path} has no rows labelled {label!r}; it carries {present}."
            )

    by_native: dict[str, Request] = {}
    for i, row in enumerate(rows):
        seq = str(row.get("sequence") or row.get("requested_sequence") or "").strip().upper()
        parent = str(row.get("parent_id") or "").strip()
        if not seq:
            raise SequenceInjectionError(f"{path} row {i} has no 'sequence'.")
        if not parent:
            raise SequenceInjectionError(f"{path} row {i} has no 'parent_id'.")
        if parent not in natives:
            raise SequenceInjectionError(
                f"{path} row {i} names parent {parent!r}, which is not in "
                f"{dataset_dir}/manifest/targets.csv."
            )
        native = natives[parent]
        stated = str(row.get("native_sequence") or "").strip().upper()
        if stated and stated != native:
            raise SequenceInjectionError(
                f"{path} row {i} says parent {parent}'s native sequence is {stated!r}, but the "
                f"manifest says {native!r}. One of the two is paired wrong."
            )
        if len(seq) != len(native):
            raise SequenceInjectionError(
                f"{path} row {i}: requested {seq!r} is {len(seq)} residues but parent {parent}'s "
                f"peptide is {len(native)} ({native!r}). These searches substitute residues and "
                "never change length, so this is a mis-pairing, not a resize."
            )
        # This form is keyed by native sequence, so a parent sharing its peptide with
        # another cannot be matched unambiguously.
        _refuse_ambiguous_native(natives, parent, native, f"{path} row {i}")
        encode_sequence(seq)  # fail now, on the file, not mid-run on a GPU
        if native in by_native:
            raise SequenceInjectionError(
                f"{path} requests two sequences for parent {parent} ({by_native[native].requested_sequence!r} "
                f"and {seq!r}). One run asks one question per parent; use two runs."
            )
        by_native[native] = Request(
            parent_id=parent,
            native_sequence=native,
            requested_sequence=seq,
            label=str(row.get("label") or "requested"),
            extra={
                k: row[k]
                for k in ("candidate_id", "arm", "rung", "selected_on", "source_run")
                if k in row and row[k] not in (None, "")
            },
        )
    return by_native


def load_requests_multi(path: Path, dataset_dir: Path) -> dict[str, list[Request]]:
    """{parent_id -> [Request, ...]} for consumers that fold SEVERAL sequences per parent.

    The generator takes one sequence per parent per run, because a conditioning pass that
    mixed a searched peptide and its native would answer two questions under one label --
    `load_requests` enforces that. A structure predictor has no such constraint: folding
    the searched peptide and the native one in the same run is how the pair gets the same
    predictor, the same alignment and the same settings, which is what makes the paired
    comparison a comparison.

    Every per-row rule from `load_requests` still applies; only the one-per-parent rule is
    lifted, and two rows carrying the same sequence AND label for one parent are still a
    duplicate rather than two folds.
    """
    natives = native_sequences(dataset_dir)
    rows = _rows_from_path(Path(path))
    if not rows:
        raise SequenceInjectionError(f"{path} holds no rows.")

    out: dict[str, list[Request]] = {}
    for i, row in enumerate(rows):
        seq = str(row.get("sequence") or row.get("requested_sequence") or "").strip().upper()
        parent = str(row.get("parent_id") or "").strip()
        label = str(row.get("label") or "requested")
        if not seq:
            raise SequenceInjectionError(f"{path} row {i} has no 'sequence'.")
        if not parent:
            raise SequenceInjectionError(f"{path} row {i} has no 'parent_id'.")
        if parent not in natives:
            raise SequenceInjectionError(
                f"{path} row {i} names parent {parent!r}, which is not in "
                f"{dataset_dir}/manifest/targets.csv."
            )
        native = natives[parent]
        stated = str(row.get("native_sequence") or "").strip().upper()
        if stated and stated != native:
            raise SequenceInjectionError(
                f"{path} row {i} says parent {parent}'s native sequence is {stated!r}, but the "
                f"manifest says {native!r}. One of the two is paired wrong."
            )
        if len(seq) != len(native):
            raise SequenceInjectionError(
                f"{path} row {i}: requested {seq!r} is {len(seq)} residues but parent {parent}'s "
                f"peptide is {len(native)} ({native!r})."
            )
        encode_sequence(seq)
        req = Request(
            parent_id=parent,
            native_sequence=native,
            requested_sequence=seq,
            label=label,
            extra={
                k: row[k]
                for k in ("candidate_id", "arm", "rung", "selected_on", "source_run")
                if k in row and row[k] not in (None, "")
            },
        )
        bucket = out.setdefault(parent, [])
        if any(r.requested_sequence == seq and r.label == label for r in bucket):
            raise SequenceInjectionError(
                f"{path} row {i} repeats sequence {seq!r} under label {label!r} for parent "
                f"{parent}. Folding the same input twice under one label is a duplicate, not a "
                "replicate -- replicates come from the predictor's seeds."
            )
        bucket.append(req)
    return out


def apply_to_batch(
    batch: dict,
    table: dict[str, Request],
    *,
    on_missing: str = "error",
) -> list[dict]:
    """Substitute the requested peptides into `batch["residue_type"]`, in place.

    Returns one provenance record per batch row. `on_missing` governs a row whose native
    sequence has no request: "error" (the default -- a half-injected batch makes the run
    ambiguous) or "native" (leave it, and say so in the record).
    """
    if on_missing not in ("error", "native"):
        raise SequenceInjectionError(f"on_missing must be 'error' or 'native', not {on_missing!r}")
    if "residue_type" not in batch:
        raise SequenceInjectionError(
            "batch has no `residue_type`, so there is no sequence to replace. This checkpoint's "
            "dataloader is not emitting the binder sequence feature, and --seq-cond would be "
            "skipped for the same reason."
        )

    residue_type = batch["residue_type"]
    mask = batch["mask"].bool() if "mask" in batch else batch["coord_mask"][..., 1].bool()
    if residue_type.shape[:2] != mask.shape[:2]:
        raise SequenceInjectionError(
            f"residue_type {tuple(residue_type.shape)} and mask {tuple(mask.shape)} disagree on "
            "batch/length; refusing to index one with the other."
        )

    records: list[dict] = []
    for b in range(residue_type.shape[0]):
        native = decode_sequence(residue_type[b], mask[b])
        req = table.get(native)
        if req is None:
            if on_missing == "error":
                raise SequenceInjectionError(
                    f"batch row {b} decodes to native peptide {native!r}, which has no request in "
                    f"the sequence source. Known natives: {sorted(table)}. If this decode looks "
                    "like nonsense rather than a peptide, the residue-type alphabet assumption is "
                    "wrong and nothing should be injected."
                )
            records.append({"row": b, "native_sequence": native, "injected": False, "reason": "no request"})
            continue

        ids = encode_sequence(req.requested_sequence)
        positions = mask[b].nonzero(as_tuple=True)[0]
        if len(positions) != len(ids):
            raise SequenceInjectionError(
                f"batch row {b} has {len(positions)} unmasked residues but the request is "
                f"{len(ids)} long. The decode said {native!r}."
            )
        residue_type[b, positions] = torch.tensor(
            ids, dtype=residue_type.dtype, device=residue_type.device
        )
        # Read it back: the write is the whole point of this module, and a dtype or device
        # surprise here would otherwise surface as a sequence-recovery number.
        written = decode_sequence(residue_type[b], mask[b])
        if written != req.requested_sequence:
            raise SequenceInjectionError(
                f"batch row {b}: wrote {req.requested_sequence!r} but it reads back as {written!r}."
            )
        rec = {"row": b, "injected": True, **req.to_dict()}
        records.append(rec)

    if not any(r["injected"] for r in records):
        raise SequenceInjectionError(
            "no row in this batch received a requested sequence; the run would measure the "
            "native peptides under a misleading label."
        )
    return records


def write_sidecar(
    dump_dir: Path,
    table: dict[str, Request],
    *,
    source_path: Path,
    dataset_dir: Path,
    extra: dict | None = None,
) -> Path:
    """Record what was requested, beside the structures, keyed by native sequence.

    `cpseq_score_seqcond.py` resolves a dumped complex to its parent from the structure and
    then needs to know which peptide was *asked for*, or it will compare every decode
    against the dataset's reference and flag the whole run as non-compliant.
    """
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    path = dump_dir / "requested_sequences.json"
    payload = {
        "schema": "cpseq-requested-sequences-1",
        "source_path": str(source_path),
        "source_sha256_16": file_fingerprint(source_path),
        "dataset_dir": str(dataset_dir),
        "n_requests": len(table),
        "by_native_sequence": {native: req.to_dict() for native, req in sorted(table.items())},
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path
