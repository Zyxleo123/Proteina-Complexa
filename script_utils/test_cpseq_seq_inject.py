#!/usr/bin/env python3
"""Checks for `cpseq.seq_inject`. No GPU, no dataset, no model.

The failure this guards against is not a crash. It is a run that completes, writes
plausible energies, and has attributed designed peptides to the wrong receptors -- or
permuted every one of them because the residue-type alphabet differed from the decoder's.
So most of these assert on *refusal*.

    python script_utils/test_cpseq_seq_inject.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proteinfoundation.cpseq import seq_inject as si  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def expect_refusal(name: str, fn) -> None:
    try:
        fn()
    except si.SequenceInjectionError:
        check(name, True)
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"raised {type(exc).__name__}: {exc}")
    else:
        check(name, False, "did not refuse")


def make_batch(sequences: list[str], n_pad: int = 3) -> dict:
    """A batch shaped like the real one: [b, n] residue types plus a padding mask."""
    width = max(len(s) for s in sequences) + n_pad
    rt = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for i, s in enumerate(sequences):
        ids = si.encode_sequence(s)
        rt[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = True
    return {"residue_type": rt, "mask": mask}


def make_dataset(tmp: Path, parents: dict[str, str]) -> Path:
    d = tmp / "ds" / "manifest"
    d.mkdir(parents=True, exist_ok=True)
    rows = ["parent_id,reference_sequence"] + [f"{k},{v}" for k, v in parents.items()]
    (d / "targets.csv").write_text("\n".join(rows) + "\n")
    return tmp / "ds"


def source(tmp: Path, rows: list[dict], name: str = "req.jsonl") -> Path:
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _mask_batch(seqs: list[str], n: int) -> dict:
    """A minimal batch: `residue_type` plus a residue mask, padded to width `n`."""
    from openfold.np import residue_constants as rc

    b = len(seqs)
    rt = torch.full((b, n), rc.restype_num, dtype=torch.long)
    m = torch.zeros(b, n, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        ids = si.encode_sequence(seq)
        rt[i, : len(ids)] = torch.tensor(ids)
        m[i, : len(ids)] = True
    return {"residue_type": rt, "mask": m}


def test_reveal_mask_for_batch() -> None:
    """The mask arm's attribution: which positions get hidden, on which row."""
    print("\nreveal_mask_for_batch")
    seqs = ["MEEVD", "RIIYDRKFLMECRN"]
    masks = {"MEEVD": [1, 3], "RIIYDRKFLMECRN": [2, 5, 11]}

    batch = _mask_batch(seqs, 16)
    reveal, recs = si.reveal_mask_for_batch(batch, masks)
    check("every row is attributed", all(r["masked"] for r in recs), f"{recs}")
    check("the named positions are hidden and nothing else",
          reveal[0, :5].tolist() == [True, False, True, False, True],
          f"{reveal[0, :5].tolist()}")
    check("hidden counts are recorded per row",
          [r["n_hidden"] for r in recs] == [2, 3], f"{recs}")
    check("padding is never revealed", not bool(reveal[0, 5:].any()))

    # The property the whole module exists for. The dataloader's ordering is not a
    # contract, and a mask paired with the wrong receptor opens positions on one peptide
    # that were chosen by looking at another -- a plausible result, not an error.
    flipped = _mask_batch(list(reversed(seqs)), 16)
    reveal2, _ = si.reveal_mask_for_batch(flipped, masks)
    check("attribution does not depend on batch order",
          reveal2[1, :5].tolist() == reveal[0, :5].tolist(),
          f"{reveal2[1, :5].tolist()} vs {reveal[0, :5].tolist()}")

    for bad, label, needle in (
        ({"MEEVD": [0, 1, 2, 3, 4], "RIIYDRKFLMECRN": [2]},
         "a fully hidden row", "unconditional generation"),
        ({"MEEVD": [9], "RIIYDRKFLMECRN": [2]},
         "a position past the end", "outside the peptide"),
        ({"NOPE": [1]}, "a native with no mask", "has no mask"),
    ):
        try:
            si.reveal_mask_for_batch(batch, bad)
            check(f"{label} is refused", False, "no error raised")
        except si.SequenceInjectionError as e:
            check(f"{label} is refused", needle in str(e), str(e)[:100])

    reveal3, recs3 = si.reveal_mask_for_batch(
        batch, {"MEEVD": [1]}, on_missing="reveal_all"
    )
    check("on_missing='reveal_all' leaves the unrequested row fully revealed",
          bool(reveal3[1, :14].all()) and not recs3[1]["masked"], f"{recs3}")


def main() -> int:
    print("=== cpseq.seq_inject ===")

    # --- alphabet round-trip ---------------------------------------------------------
    # If this fails, nothing else in the module is trustworthy: the decode is what makes
    # attribution and the alphabet check work.
    for seq in ("MEEVD", "SEKFKLLFQSY", "VPLRARNLPPSFFTEP", "ACDEFGHIKLMNPQRSTVWY"):
        b = make_batch([seq])
        got = si.decode_sequence(b["residue_type"][0], b["mask"][0])
        check(f"round-trip {seq[:8]}", got == seq, f"got {got!r}")

    check("padding excluded from decode",
          si.decode_sequence(*[make_batch(["MEEVD"], n_pad=9)[k][0] for k in ("residue_type", "mask")]) == "MEEVD")
    expect_refusal("non-standard residue refused", lambda: si.encode_sequence("MEEVX"))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parents = {"lnr_1elr": "MEEVD", "lnr_6efk": "IEEVD", "lnr_4j86": "TFKKTN"}
        ds = make_dataset(tmp, parents)

        # --- the manifest contract ---------------------------------------------------
        nat = si.native_sequences(ds)
        check("natives loaded", nat == parents, f"got {nat}")

        # Duplicate natives are refused where they actually break something -- the
        # by-native-sequence form -- and tolerated where they do not. pepbench_valid has
        # two such pairs, and refusing it outright blocked folding it with a predictor
        # that addresses parents explicitly.
        dupdir = make_dataset(tmp / "dupds", {"a": "MEEVD", "b": "MEEVD", "c": "TFKKTN"})
        check("native_sequences tolerates duplicates",
              si.native_sequences(dupdir) == {"a": "MEEVD", "b": "MEEVD", "c": "TFKKTN"})
        dsrc = source(tmp, [{"parent_id": "a", "sequence": "MEWVD"}], "dup_req.jsonl")
        expect_refusal("by-native form refuses an ambiguous parent",
                       lambda: si.load_requests(dsrc, dupdir))
        check("multi form folds an ambiguous parent anyway",
              si.load_requests_multi(dsrc, dupdir)["a"][0].requested_sequence == "MEWVD")
        csrc = source(tmp, [{"parent_id": "c", "sequence": "TFKKTK"}], "unamb_req.jsonl")
        check("by-native form accepts an unambiguous parent in the same dataset",
              si.load_requests(csrc, dupdir)["TFKKTN"].requested_sequence == "TFKKTK")

        # --- the source file ---------------------------------------------------------
        good = source(tmp, [
            {"parent_id": "lnr_1elr", "sequence": "MEWVD", "label": "searched"},
            {"parent_id": "lnr_6efk", "sequence": "IEEVD", "label": "native"},
        ])
        table = si.load_requests(good, ds)
        check("two requests loaded", len(table) == 2)
        check("keyed by native sequence", set(table) == {"MEEVD", "IEEVD"}, f"got {set(table)}")
        check("substitutions counted", table["MEEVD"].n_substitutions == 1)
        check("native flagged as native", table["IEEVD"].is_native)
        check("designed not flagged native", not table["MEEVD"].is_native)

        expect_refusal("length mismatch refused", lambda: si.load_requests(
            source(tmp, [{"parent_id": "lnr_1elr", "sequence": "MEEVDK"}], "len.jsonl"), ds))
        expect_refusal("unknown parent refused", lambda: si.load_requests(
            source(tmp, [{"parent_id": "lnr_9xxx", "sequence": "MEEVD"}], "unk.jsonl"), ds))
        expect_refusal("wrong stated native refused", lambda: si.load_requests(
            source(tmp, [{"parent_id": "lnr_1elr", "sequence": "MEWVD",
                          "native_sequence": "TFKKTN"}], "mis.jsonl"), ds))
        expect_refusal("two requests per parent refused", lambda: si.load_requests(
            source(tmp, [{"parent_id": "lnr_1elr", "sequence": "MEWVD"},
                         {"parent_id": "lnr_1elr", "sequence": "MEYVD"}], "two.jsonl"), ds))
        expect_refusal("missing sequence field refused", lambda: si.load_requests(
            source(tmp, [{"parent_id": "lnr_1elr"}], "nos.jsonl"), ds))
        expect_refusal("empty file refused", lambda: si.load_requests(
            source(tmp, [], "empty.jsonl") if False else _write(tmp / "e.jsonl", "\n"), ds))

        # csv and json forms carry the same meaning
        csv_p = tmp / "req.csv"
        csv_p.write_text("parent_id,sequence,label\nlnr_1elr,MEWVD,searched\n")
        check("csv source", si.load_requests(csv_p, ds)["MEEVD"].requested_sequence == "MEWVD")
        json_p = tmp / "req.json"
        json_p.write_text(json.dumps({"rows": [{"parent_id": "lnr_4j86", "sequence": "TFKKTK"}]}))
        check("json source", si.load_requests(json_p, ds)["TFKKTN"].requested_sequence == "TFKKTK")

        # --- label filtering: one paired file, two runs -------------------------------
        paired = source(tmp, [
            {"parent_id": "lnr_1elr", "sequence": "MEWVD", "label": "searched"},
            {"parent_id": "lnr_1elr", "sequence": "MEEVD", "label": "native"},
            {"parent_id": "lnr_4j86", "sequence": "TFKKTK", "label": "searched"},
            {"parent_id": "lnr_4j86", "sequence": "TFKKTN", "label": "native"},
        ], "paired.jsonl")
        expect_refusal("paired file refused without a label",
                       lambda: si.load_requests(paired, ds))
        srch = si.load_requests(paired, ds, label="searched")
        check("searched side loads", len(srch) == 2)
        check("searched side is the designed one", srch["MEEVD"].requested_sequence == "MEWVD")
        natv = si.load_requests(paired, ds, label="native")
        check("native side loads", len(natv) == 2)
        check("native side is native", all(r.is_native for r in natv.values()))
        check("both sides cover the same parents",
              {r.parent_id for r in srch.values()} == {r.parent_id for r in natv.values()})
        expect_refusal("absent label refused",
                       lambda: si.load_requests(paired, ds, label="nosuch"))

        # the multi form is what a predictor folds: both sides, one call
        multi = si.load_requests_multi(paired, ds)
        check("multi groups by parent", set(multi) == {"lnr_1elr", "lnr_4j86"})
        check("multi keeps both sides", all(len(v) == 2 for v in multi.values()))
        check("multi labels preserved",
              sorted(r.label for r in multi["lnr_1elr"]) == ["native", "searched"])
        expect_refusal("multi still refuses a length mismatch", lambda: si.load_requests_multi(
            source(tmp, [{"parent_id": "lnr_1elr", "sequence": "MEEVDKK"}], "ml.jsonl"), ds))
        expect_refusal("multi refuses an exact duplicate", lambda: si.load_requests_multi(
            source(tmp, [{"parent_id": "lnr_1elr", "sequence": "MEWVD", "label": "searched"},
                         {"parent_id": "lnr_1elr", "sequence": "MEWVD", "label": "searched"}],
                   "md.jsonl"), ds))

        # --- injection ---------------------------------------------------------------
        batch = make_batch(["MEEVD", "IEEVD"])
        recs = si.apply_to_batch(batch, table)
        check("both rows injected", all(r["injected"] for r in recs))
        check("row 0 now reads the request",
              si.decode_sequence(batch["residue_type"][0], batch["mask"][0]) == "MEWVD")
        check("row 1 unchanged (native was requested)",
              si.decode_sequence(batch["residue_type"][1], batch["mask"][1]) == "IEEVD")
        check("attribution is by sequence, not order",
              [r["parent_id"] for r in recs] == ["lnr_1elr", "lnr_6efk"],
              f"got {[r['parent_id'] for r in recs]}")

        # the same table against a batch in the OPPOSITE order must attribute the same way
        rev = make_batch(["IEEVD", "MEEVD"])
        rrecs = si.apply_to_batch(rev, table)
        check("reversed batch attributed correctly",
              [r["parent_id"] for r in rrecs] == ["lnr_6efk", "lnr_1elr"],
              f"got {[r['parent_id'] for r in rrecs]}")
        check("reversed batch row 1 got lnr_1elr's request",
              si.decode_sequence(rev["residue_type"][1], rev["mask"][1]) == "MEWVD")

        # a row nobody asked about
        expect_refusal("unrequested row refused by default",
                       lambda: si.apply_to_batch(make_batch(["MEEVD", "TFKKTN"]), table))
        lenient = si.apply_to_batch(make_batch(["MEEVD", "TFKKTN"]), table, on_missing="native")
        check("on_missing=native leaves the row", lenient[1]["injected"] is False)
        check("on_missing=native records why", lenient[1]["reason"] == "no request")
        check("on_missing=native still injects the known row", lenient[0]["injected"] is True)

        # a batch where NO row is known is not a silent native run
        expect_refusal("wholly unmatched batch refused",
                       lambda: si.apply_to_batch(make_batch(["TFKKTN"]), table, on_missing="native"))
        expect_refusal("bad on_missing refused",
                       lambda: si.apply_to_batch(make_batch(["MEEVD"]), table, on_missing="whatever"))
        expect_refusal("batch without residue_type refused",
                       lambda: si.apply_to_batch({"mask": torch.ones(1, 5, dtype=torch.bool)}, table))

        # --- the sidecar -------------------------------------------------------------
        dump = tmp / "rosetta_dump"
        path = si.write_sidecar(dump, table, source_path=good, dataset_dir=ds,
                                extra={"ckpt_name": "last-v3-EMA.ckpt"})
        payload = json.loads(path.read_text())
        check("sidecar schema", payload["schema"] == "cpseq-requested-sequences-1")
        check("sidecar keyed by native", set(payload["by_native_sequence"]) == {"MEEVD", "IEEVD"})
        check("sidecar carries the request",
              payload["by_native_sequence"]["MEEVD"]["requested_sequence"] == "MEWVD")
        check("sidecar records substitution count",
              payload["by_native_sequence"]["MEEVD"]["n_substitutions"] == 1)
        check("sidecar fingerprints its source", len(payload["source_sha256_16"]) == 16)
        check("sidecar carries extras", payload["ckpt_name"] == "last-v3-EMA.ckpt")

    test_reveal_mask_for_batch()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


if __name__ == "__main__":
    raise SystemExit(main())
