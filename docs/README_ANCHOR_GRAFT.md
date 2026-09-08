# Changing a residue's identity before the autoencoder sees it

How to graft the anchor residues a cyclization chemistry needs onto a peptide's termini
without the model silently undoing it, and the four ways this failed while looking correct.

This exists because the first version of the graft ran end to end, produced no error, and was
**completely erased** — every grafted endpoint came back decoded as alanine. Nothing about
that raises an exception, appears in a metric, or shows up in a config diff. Each failure
below has the same shape: perfect local geometry, plausible output, wrong answer.

## Why graft at all

The cyclization head abstains — emits a null edge — when the decoded sequence admits no
candidate anchor pair: two CYS for a disulfide, a LYS plus an ASP/GLU/ASN/GLN for an
isopeptide. On the LNR inputs that is the dominant failure, and it is governed entirely by the
sequence budget `t_lat`:

    t_lat            0.2    0.4    0.6    0.8    1.0
    mean subs        8.5    7.3    3.1    0.1    0.0
    disulfide abst  0.033  0.113  0.950  1.000  1.000
    isopeptide abst 0.046  0.113  0.796  0.983  0.983

The inputs are linear binders of median length 10 and 87% contain no cysteine at all, so a
disulfide needs two specific mutations at two specific positions. At `t_lat >= 0.8` the
sampler is allowed ~0 substitutions, so it can never make them and the sequence-preserving
half of the grid is unmeasurable.

`scripts/anchor_graft.py` writes the anchors into the **input** instead: disulfide CYS/CYS,
isopeptide LYS at the first endpoint and ASP at the last, mainchain a declared no-op.
Abstention then goes to ~0 by construction rather than by persuasion. It also makes the
bond-distance guidance term meaningful — `atoms_valid` is what zeroes that term on an
abstaining sample.

The isopeptide orientation is measured, not conventional: of 499 isopeptide rings the model
placed for itself in `sdedit_pocket14_20260906_213400`, it put LYS at the lower-index endpoint
in **100%** and paired it with ASP in **98%**.

---

## Trap 1 — truncating the side chain is a silent relabel to alanine

The first implementation kept the backbone plus CB and dropped every other side-chain atom,
on the reasoning that inventing a rotamer would be dishonest. Result:

    graft  disulfide  A.VQGGAAGH.S    (wanted C ... C)
    graft  disulfide  A.GLTIYAQKQ.A   (wanted C ... C)

**ALA's complete atom37 set is exactly `[N, CA, C, CB, O]`** — precisely what that truncation
leaves behind. The encoder reads three things about a residue, and two of them then said
alanine. Measured through the encoder's own feature functions
([seq_feats.py `OpenfoldSideChainAnglesSeqFeat`]):

    truncate  n_atoms=5  chi1=ABSENT       <- bit-identical to a real ALA
    build     n_atoms=6  chi1=ok, -65 deg  <- a cysteine
    real ALA  n_atoms=5  chi1=ABSENT

Only the 20-d `x1_aatype` one-hot still said CYS, against a 37-slot occupancy mask and a
chi-angle feature that both said ALA — a combination the AE has never seen contradicted in
training. The decoder resolved it in favour of the geometry.

**Truncating is not the conservative choice, it is a relabel.** The rule that generalises: when
you edit residue identity in a batch the AE will encode, make the occupancy mask and the chi
angles agree with the new aatype, and check the result is not accidentally some *other*
residue's exact atom set. The proof has to be at the feature level, not the tensor level.

Kept reachable as `--graft-sidechain truncate`, the negative control it turned out to be.
`test_anchor_graft.py::test_the_grafted_residue_is_not_an_alanine` pins both directions.

## Trap 2 — the group-0 frame needs openfold's fixup, or every side chain is misplaced

`scripts/rotamer.py` builds the replacement side chain from openfold's own
`torsion_angles_to_frames` + `frames_and_literature_positions_to_atom14_pos`, fed the
**atom37** constants (both are generic in the atom axis, so atom37 in gives atom37 out).

`Rigid.from_3_points(C, CA, N)` is **not** the rigid-group-0 frame. openfold then rotates
group 0 by `diag(-1, 1, -1)` — the `rots[..., 0, 0, 0] = -1` block at the end of
`data_transforms.atom37_to_frames` — because Algorithm 21's x-axis runs CA→C while
`restype_atom37_rigid_group_positions` is expressed in the opposite-handed convention.

Measured over 1945 real crystal residues, CB's distance from where it actually belongs:

| | median | mean | max |
|---|---|---|---|
| with the fixup | **0.054 Å** | 0.064 | 0.541 |
| without it | **2.631 Å** | 2.629 | 2.729 |

Every bond length and bond angle is ideal in both cases. Nothing checkable against the
constants catches it, because the constants are being used consistently — which is exactly why
the test validates against **real crystal side chains** instead.

## Trap 3 — `restype_atom37_mask` gives OXT to no residue

Intersecting the new residue's allowed atom set with the source mask strips the terminal
carboxylate off every grafted C-terminal residue — i.e. every `j` endpoint the graft touches.
OXT's presence depends on being terminal, not on residue identity, so it is copied through
from the source on the same footing as N/CA/C/O.

## Trap 4 — `coords` and `coords_nm` are different rigid frames

They are **not** one tensor scaled by 10. Measured on the LNR inputs through
`build_dataset` + `structure_collate_fn`:

    max |coords - coords_nm * 10|      12.8 - 19.0 A   (three examples)
    pairwise-distance difference       0.011 A
    rigid fit                          1e-6 A RMSD, det(R) = +1

Same structure, different frames: `coords_nm` is centred and rotated by the dataset transform
while `coords` keeps the crystal frame. (`atomworks_dataloader_utils.py` really does set
`coords = coords_nm * 10`, but that is the atomworks path, not the `structure_data.py` path
CPSea uses — do not generalise from it.)

**Both are read by the encoder, by different features:**

| tensor | feature | what it feeds |
|---|---|---|
| `coords_nm` | `Atom37NanometersCoorsSeqFeat` | `x1_a37coors_nm` — coordinates **and** the 37-slot occupancy |
| `coords` | `OpenfoldSideChainAnglesSeqFeat`, `BackboneTorsionAnglesSeqFeat` | chi and backbone torsions |

The angle features are dihedral-based and so frame-invariant, which is why the split is
survivable at all. But anything **writing** coordinates must write each tensor in that
tensor's own frame. Computing a position from one and storing it in the other places it
somewhere arbitrary while every bond length still measures perfect. `graft_anchors` therefore
runs the side-chain build **once per tensor** rather than building once and scaling.

To check two coordinate tensors hold the same structure when their frames differ, compare
**pairwise distances** — frame-invariant. Do **not** compare the internal geometry of
something you just built from ideal constants: that agrees by construction in any frame and is
blind to a bad backbone. (My first desync check made exactly that mistake and passed a
deliberately corrupted CA.)

---

## Validating a geometry builder

The load-bearing test rebuilds **real crystal side chains from their own measured chi** and
compares atom for atom — an external reference, so it fails on a wrong frame, wrong
handedness, wrong group composition, or wrong literature positions alike.

Worst-atom deviation, ideal geometry on a real backbone (n=1945):

    CYS 0.140   ASN 0.172   ASP 0.174   GLN 0.234   GLU 0.240   LYS 0.324   (median, A)

The residual is ideal-vs-real internal geometry compounding along the chain — CYS is one bond
past CB, LYS is four. Chirality is checked against the same structures rather than a
remembered constant: built improper N-C-CA-CB is **-57.47°** against a measured
**-57.61 ± 2.84°**.

**chi2 and beyond round-trip exactly; chi1 does not, and should not.** chi1 is measured as
N-CA-CB-SG and N belongs to the real backbone, so when a residue's real N-CA-C angle differs
from the ideal 111° the offset shows through. Correlation between (N-CA-C − 111°) and the chi1
error is **r = 0.998** over 468 residues, with the backbone angle spanning 98–120°. That
correlation is itself a test — a genuinely misplaced side chain would be off by an amount
unrelated to the residue's own backbone.

---

## Checkpoint compatibility: growing the conditioning-type table

`NUM_CYCLIZATION_COND_TYPES` ([cyclization/constants.py]) sizes
`CyclizationTypeSeqFeat.embedding` ([seq_cond_feats.py]). Bumping it 4 → 5 to add `LINEAR`
made **every checkpoint trained before that edit unloadable**, in every consumer — training,
design, eval, SDEdit:

    size mismatch for nn.cond_factory.feat_creators.2.embedding.weight:
      copying a param with shape torch.Size([4, 256]), shape in current model is [5, 256]

`strict=False` does not help — a size mismatch is fatal regardless. This killed all 12 shards
of one sweep at load time, 0 rows, while a job launched 90 minutes earlier from the same tree
ran fine.

`CyclizationTypeSeqFeat._load_from_state_dict` now pads a short embedding with **zero** rows so
old checkpoints load, and `forward` **raises** if a padded (untrained) type is ever requested.
A zero row would otherwise return a vector indistinguishable from a real conditioning signal,
so the run would look like it honoured the request and silently ignore it. Padding makes an old
checkpoint *loadable*, not *able to honour a type it never saw*.

Verified not to change the model: a smoke re-run after the pad is **bit-identical** to the
pre-pad run on all 10 rows of one peptide, both chemistries. The type embedding is a single
global lookup shared by every sample requesting that type, so if row 1 or 2 had shifted, every
row would have moved.

**This is sound only for APPENDED types.** Rows 0..n-1 keep their meaning only while
`MAINCHAIN=0, DISULFIDE=1, ISOPEPTIDE=2, UNSPECIFIED=3` hold. **Reordering** — inserting
`LINEAR` at 3 and pushing `UNSPECIFIED` to 4 — would silently read a row trained as one type as
another, and nothing can detect it: checkpoints store indices, not names. Reorder ⇒ remap or
retrain, never pad.

General rule: bumping a constant that sizes a layer is a checkpoint-breaking change. Bump it in
the same commit as a load-time shim, or every prior checkpoint dies.

---

## Operational findings

**Shards run in parallel; arms run sequentially inside each shard.** `sdedit_sweep.sbatch`
loops `for ARM in ${GUIDANCE_ARMS}` with one `srun` per arm, so an 8-arm list is a *sum*, not a
division. Measured on the pocket14 grid: 225 edits per shard per arm, **11.9 s/edit** unguided
and **24.3 s/edit** with an identity-mode hook, making an 8-arm run ~12.9 h/shard against a 12 h
wall — it would have died in the last arm, which was a grafted one. Put the arms you care about
**first**: a timeout then costs the controls, not the experiment.

**The sampler is not bit-reproducible for every example.** Same seed, same GPU model,
byte-identical inputs (`n_contacts_input`, `frame_residual_A`, `input_nc_gap_angstrom` all
equal), and one peptide reproduced exactly on 10/10 rows while another diverged on 8/10 by up
to 2.2 Å. Read small-n differences accordingly.

**Uncommitted `src/` edits while launching jobs is the underlying hazard.** Two jobs launched
90 minutes apart ran different code from the same working tree, and neither run's results were
attributable to a revision. Commit before launching, or launch from a pinned git worktree.

---

## Reading the closure results

At `t_ca=0.4`, `nsteps=50` (smoke sizing — production is 400 steps):

| arm | satisfied | closed | anchor bond dist | ca_rmsd | retention |
|---|---|---|---|---|---|
| ungrafted, w=0 | 0/4 | — | — | 7.88 Å | 0.42 |
| ungrafted + bond_fb | 0/4 | — | — | 7.79 | 0.42 |
| **GRAFT**, w=0 | 4/4 | 0.50–0.75 | 1.51–1.69 Å | 8.32 | 0.38 |
| **GRAFT** + bond_fb | 4/4 | 1.00 | 1.63–1.65 Å | 8.32 | 0.37 |

**The graft row is a real effect** — abstention 100% → 0%, mechanically guaranteed since the
anchors are given (3/12 → 12/12 on the fuller smoke).

**The guidance row is not yet a result**, for three reasons worth keeping in mind whenever this
metric is quoted:

1. The unguided control gave 0.75 in one run and 0.50 in the next — same config, same seed. Its
   own run-to-run variance is the size of the effect.
2. The mean anchor bond distance does not move consistently (1.51→1.65 one way, 1.69→1.63 the
   other). `closed` reaching 1.00 is a couple of rows crossing a threshold, not a geometry shift.
3. **`bond_fb` is a flat-bottom loss on the anchor distance, and `cyc/*_bond_success` thresholds
   that same distance.** Improving it is close to tautological — the same trap as
   `cyc_cb_window_success`, which saturates and proves nothing.

And note the cost column: **8 Å CA-RMSD on a 10-residue peptide, retention 0.37–0.42.**
Whatever is closing, these edits are not preserving the binder. Closure alone is not the
deliverable; Rosetta dG is what says whether a closed ring is worth anything.

---

## Files

| path | what |
|---|---|
| `scripts/rotamer.py` | ideal side-chain builder; the frame fixup and OXT handling |
| `scripts/anchor_graft.py` | which residue goes where; per-tensor build; run tagging |
| `scripts/test_rotamer.py` | validation against real crystal side chains |
| `scripts/test_anchor_graft.py` | graft semantics, the alanine regression, the two-frame case |
| `scripts/test_cyc_cond_compat.py` | checkpoint padding and the untrained-type refusal |
| `src/proteinfoundation/nn/feature_factory/seq_cond_feats.py` | `_load_from_state_dict` pad |

All three test suites are CPU-only and run in seconds:

    .venv/bin/python scripts/test_rotamer.py
    .venv/bin/python scripts/test_anchor_graft.py
    .venv/bin/python scripts/test_cyc_cond_compat.py

**Resume trap:** `graft_tag` is in the run_key and the PDB filename. Build mode tags `graftsc`;
the truncating modes keep the historical `graft` / `graftnocb`. Rows written before the fix are
wrong but carry no marker saying so, and resume keys off this string — reusing `graft` would
make a fixed run skip exactly the rows it exists to redo.
