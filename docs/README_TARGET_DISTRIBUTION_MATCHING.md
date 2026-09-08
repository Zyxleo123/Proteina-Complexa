# Matching a target set to the training distribution

How to configure a staged target set so the denoiser sees a receptor like the ones it was
trained on, and how to verify it before spending a GPU arm.

This exists because the largest effect ever measured on LNR ring closure — **+22 to +34
points** — came from none of the knobs anyone was tuning. It came from how the receptor was
cut up. Nothing about it raises an error, appears in the metadata, or shows up in a config
diff, so it has to be measured explicitly.

## What the model actually reads

`SegmentAwareResidueFeaturesTransform` ([src/proteinfoundation/datasets/transforms.py:138])
labels every residue with `segment_id`, `pos_in_segment`, and `effective_chain_id`. The
denoiser consumes the last two through `ChainIdxPairFeat` and `rel_seq_sep` (at
`aggressive`). A new segment starts when, relative to the previous residue
([segment_utils.py:121]):

1. the chain changes, **or**
2. the PDB residue number does not increase by exactly 1, **or**
3. the C(i−1) → N(i) distance exceeds `cn_break_cutoff` (2.0 Å).

CPSea's preprocessed PDBs store the receptor **already spatially cropped to the pocket with
original numbering kept**, so every training example presents ~13 disjoint fragments. A
target set that ships complete receptor chains presents ~1, with `pos_in_segment` running
past 200 — values the model has never seen.

**Rule 3 dominates.** Measured on both `cpsea_val` and every LNR staging: segments computed
ignoring numbering are *identical* to segments computed with it, because a spatial crop
leaves real backbone gaps. Two consequences:

- You cannot de-fragment a cropped set by renumbering it contiguously. That manipulation is
  inert. The only way back is to restore the deleted residues from `source_path`.
- You *can* add fragmentation by numbering alone, to a physically intact chain — but see the
  D-num caveat below, because it does not reproduce the shape of the distribution.

## The four statistics that matter

Reported by `scripts/audit_target_distribution.py`. The reference row is the training/val
distribution; everything else is a candidate target set.

| set | segments | max_run | seg_med | seg_p90 | frag<5 | tgt_len | pep_len |
|---|---|---|---|---|---|---|---|
| **CPSea_ctl (reference)** | 13.0 | 34.0 | 2.0 | 23.0 | 59.3% | 106.0 | 13.5 |
| LNR original (full chains) | 1.0 | 201.0 | 138.5 | 341.5 | 0.0% | 225.5 | 10.0 |
| LNR 18 Å crop | 9.5 | 54.5 | 3.0 | 32.3 | 51.6% | 126.5 | 10.0 |
| **LNR 14 Å crop** | 11.0 | **32.0** | **2.0** | **24.0** | **59.4%** | 90.5 | 10.0 |
| LNR renumber-to-31 | 11.0 | 31.0 | 6.0 | 31.0 | 46.8% | 126.5 | 10.0 |

`segments` alone is a trap: it plateaus at 9–14 for every crop radius from 6 to 18 Å, which
is why an early sweep concluded the radius knob was exhausted when it was not. **`max_run`
and `frag<5` are the live statistics.**

## Configuring a run

### Recipe 1 — physical crop (use this by default)

Cuts the receptor to a heavy-atom shell around the peptide. Real deletions produce real
backbone breaks, so numbering, geometry and features all agree.

```bash
python scripts/restage_lnr_pocket.py \
    --in-metadata CPSea_data/lnr_staged/metadata/lnr_test.parquet \
    --out-dir     CPSea_data/lnr_pocket14 \
    --radius      14.0
```

Radius is the knob. Measured on 20 LNR targets:

| radius | segments | max_run | tgt_len |
|---|---|---|---|
| 18 | 11.0 | 51.0 | 134 |
| **14** | 13.0 | **33.5** | 101.5 |
| 12 | 11.5 | 28.5 | 83.5 |
| 10 | 9.0 | 20.0 | 65.5 |
| 8 | 10.0 | 14.0 | 50.0 |

Pick the radius whose `max_run` matches your reference, then confirm with the audit. **14 Å
reproduces CPSea on `max_run`, `seg_med`, `seg_p90` and `frag<5` simultaneously.**

Below ~8 Å the crop starts dropping targets under `--min-receptor-residues` (default 30,
matching the training crop's `target_min_length`); dropped examples are recorded in
`dropped.csv`, never silently padded.

Cropping does **not** cost you real contacts at sensible radii: going 18 Å → 14 Å removes
**zero** receptor residues within 8 Å of the peptide across all 60 LNR targets, keeping 75%
of the residues (min 58%). It strips outer shell only. Verify this for any new radius before
trusting it.

### Recipe 2 — numbering-only fragmentation (a probe, not a default)

Same atoms, same coordinates; only `resSeq`/`iCode` change, so runs longer than
`--max-seg-len` are split by the numbering rule.

```bash
python scripts/inject_numbering_gaps.py \
    --in-metadata CPSea_data/lnr_pocket/metadata/lnr_test_pocket.parquet \
    --out-dir     CPSea_data/lnr_gapped31 \
    --max-seg-len 31
```

Use it **only** to separate "the model wants short segments as a feature" from "the model
wants a smaller receptor" — a physical crop moves both at once. Three caveats:

- It writes numbering that **disagrees with the geometry**: residues 1.3 Å apart are told
  they are in different fragments. Never stage designs from the output.
- It matches `max_run` but **not the shape**: chopping long runs into ≤31 chunks yields
  `seg_med` 6.0 and `frag<5` 46.8%, against the reference's 2.0 and 59.3%. A physical crop
  produces many tiny fragments; this does not. Read it as a partial manipulation.
- `tgt_len` is unchanged by construction (126.5), which is exactly why it is the control.

### Recipe 3 — launching the arms

Both submitters take the staged parquet positionally through `--metadata`; everything else
(checkpoint, AE, sampler, seeds) is held fixed, so arms differ in exactly one respect.

```bash
bash scripts/submit_uncond_gen.sh \
    --metadata CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet \
    --tag pocket14 --seeds "0 1 2 3 4 5 6 7"
```

Seed count is a power decision, not a detail. After the pocket crop the remaining headroom
is ~0.10 per attempt and `pass@4` is saturated at 92–97%, so read **per-attempt** closure and
budget 8 seeds; 4 seeds gives roughly 2.5σ on a +0.10 effect over 60 targets.

## Verifying before you launch

```bash
python scripts/audit_target_distribution.py \
    --set "CPSea_ctl (ref)=CPSea_data/control/cpsea_val_control.parquet" \
    --set "my set=CPSea_data/<my_set>/metadata/<my_set>.parquet"
```

The first `--set` is the reference; deltas are printed against it. The staging job
`scripts/restage_gapmore.sbatch` runs this automatically after building each set, so a
mis-staged run announces itself in the `.out` file rather than in a null result three
GPU-hours later.

Check the integrity block too. It should be identical to the reference on every row:

- `fullBB%` / `heavy-at/res` — completeness. A drop means residues are missing atoms.
- `chains` — receptor chain count.
- `DUPLICATE ATOMS` — **must be 0.** Non-zero means altloc copies are reaching the
  featurizer, which corrupts the per-residue features. Crystal structures need altloc≠A
  dropped at staging (`scripts/build_lnr_metadata.py:79` does this; the `altloc-files` and
  `occ<1-files` counts are cosmetic column differences and are expected to be non-zero for
  crystal-derived sets).

## Known residual gaps (LNR, as of 2026-09-06)

- **Peptide length.** LNR median 10 vs CPSea 14; 21 of 60 targets fall outside CPSea's 5–95th
  percentile (8–16), mostly short. The largest surviving difference. Not fixable by
  restaging — only by subsetting, which costs power.
- **Truncated pockets, 4 of 60.** In `4w50`, `4xob`, `6efk`, `6mlc` the peptide contacts
  receptor chains within 5 Å that staging discarded (staging keeps one receptor chain). Those
  four score *above* average (0.844 vs 0.788 pooled), so this is a correctness issue for
  design staging, not a measurement bias.
- **Provenance is NOT a gap.** CPSea is 100% AlphaFold + relaxed, LNR is 100% raw crystal.
  This looked like the obvious axis and was falsified by a paired relaxation arm: all three
  chemistries within ±0.05. Do not resurrect it.

## Traps

- Reading `segments` and concluding the radius knob is exhausted. Read `max_run`.
- De-gapping a cropped set by contiguous renumbering. Inert — rule 3 dominates.
- Judging an arm on `pass@k` after the pocket fix. It saturates at 92–98%; the signal moved
  to per-attempt closure.
- Filtering closure rows naively. Abstained rows carry **NaN**, which is truthy, so a naive
  filter scores a 100%-abstention cell as 100% closure. Gate on `requested_type_satisfied`
  first, then drop NaN.
- Reading a partial arm. Per-target closure is the dominant variance component; an n=34 read
  once showed +0.102 that went to +0.007 at full n. Bootstrap over targets, not attempts.

[src/proteinfoundation/datasets/transforms.py:138]: ../src/proteinfoundation/datasets/transforms.py#L138
[segment_utils.py:121]: ../src/proteinfoundation/datasets/segment_utils.py#L121
