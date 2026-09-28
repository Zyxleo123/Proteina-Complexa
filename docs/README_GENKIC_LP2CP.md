# Rosetta / GenKIC: cyclic versions of bound linear peptides

Finding macrocyclic versions of **bound linear peptides** that keep both the sequence and the
binding pose. The receptor is held fixed, a small region around the two termini is released,
and closed conformations are searched through the intended terminal bond. Quality of the
surviving candidates is the objective; single-sample closure rate is not.

Branch `cpsea-genkic`, worktree `.claude/worktrees/genkic`.

---

## 1. The quality contract

Fixed for the experiment. Implemented once, in [`script_utils/cp_contract.py`](../script_utils/cp_contract.py),
so every arm is graded by the same code. **REU are Rosetta energy units — not kcal/mol, and not
free energies.**

| Criterion | Bar |
|---|---|
| Chemistry | the requested terminal chemistry, an **explicitly declared** covalent bond, bond length and both angles inside the chemistry's window. Cβ proximity is **not** closure. |
| Sequence | ≥ 80 % identity to the **original LP** *and* ≤ 2 substitutions, anchor mutations included. Supplied protected residues unchanged. |
| Contacts | ≥ 80 % of the original peptide–receptor **residue pairs** retained; a contact is any heavy-atom pair ≤ 4.5 Å. Terminal contacts count like every other. |
| Pose | whole-peptide Cα RMSD ≤ 2 Å **and** interface-residue Cα RMSD ≤ 1 Å, interface residues taken from the original LP's heavy-atom contacts, measured in the receptor frame with no separate peptide alignment. |
| Energy | paired interface degradation ≤ **+5 REU** (CP − LP, identical preparation, scoring, relaxation). The stricter ≤ 0 result is reported alongside, never instead. |
| Integrity | no severe clashes, no stereochemical defects (chirality, backbone geometry). |

Applied to the **final relaxed** candidate; pre-relax values are recorded beside each result
under a `pre_` prefix, so a candidate that only became acceptable through relaxation is
visible as such.

Reported but **not** substituted for the primary bar: mutation budgets 0/1/2/3 and identity
tiers 90/80/70 %. In every tier cell, all non-sequence criteria stay at their primary values —
a weaker tier is never relabelled a primary success. The legacy Cα-based contact metric
(1.0 nm Cα–Cα, from `scripts/sdedit_cyclize.py`) is retained as `contact_retention_ca10` for
historical comparison only; it measured r² 0.004 against interface dG, so the heavy-atom set
is primary.

### Two traps the contract module is built around

1. **Contacts are residue *pairs*, not counts.** Retention is `|kept ∩ original| / |original|`,
   so shedding three native contacts and forming three new ones scores 0.75 on a four-contact
   interface, not 1.0.
2. **"Receptor frame" requires checking the frame.** `align_on_receptor` superimposes on
   **receptor** atoms only and reports the residual. Aligning on the peptide would absorb
   exactly the rigid-body drift the metric exists to catch.

---

## 2. Ring topology: making a macrocycle a macrocycle *to Rosetta*

[`script_utils/cp_topology.py`](../script_utils/cp_topology.py). All three chemistries bond the
peptide's two termini — CPSea is terminal-only (see
[`README_LP2CP_ROUTE_DECISION.md`](README_LP2CP_ROUTE_DECISION.md) §1), and this work keeps that
placement and the peptide length fixed for the primary comparison.

| Chemistry | Bond | Anchors needed | Engine |
|---|---|---|---|
| `mainchain` | C(j) — N(i), head-to-tail amide | none — every residue has a backbone N and C | GenKIC |
| `disulfide` | SG(j) — SG(i) | CYS / CYS | free-terminus anneal |
| `isopeptide` | CG\|CD(j) — NZ(i) | LYS + ASP\|GLU, either way round | free-terminus anneal |

Because the side-chain chemistries need specific anchor identities, they cost mutations, and
**budgets that cannot express them are chemically empty rather than search failures**. A
terminal disulfide costs 2 mutations unless the LP already ends in Cys on both sides, so
budgets 0 and 1 are empty by chemistry on almost every LP. Those cells are emitted as rows with
`status="infeasible_budget"` and stay in the coverage denominator.

### The isopeptide fix (task item 5)

The previously inspected Rosetta path **did not declare its isopeptide bond**: it logged a
warning and returned `False`, so the reported dG described an *open* topology — a linear peptide
sitting in a ring-shaped pose. Any isopeptide dG recorded before this fix describes the wrong
molecule.

Fixed in **both** places, so the repo has one behaviour:
`src/proteinfoundation/evaluation/rosetta_energy.py` and `cp_topology.py`. The mechanism is
Rosetta's own `SidechainConjugation` patch, which opens a real `CONNECT` at LYS:NZ / ASP:CG /
GLU:CD, deletes the extra protons (3 HZ → 1 amide HZ) and the second carboxyl oxygen
(OD1+OD2 → one amide OD), and retypes the atoms `Nbb`/`CObb`. Declaring NZ—CG *without* the
patch leaves NZ carrying three protons, i.e. a five-bonded nitrogen.

Verified on a real complex (`LNR_2qos_C_A`): `is_bonded` True, `LYS:SidechainConjugation`,
NZ retyped `Nbb`, ref2015 finite at −198.5 (vs −243.1 mainchain), receptor displaced 0.0 Å by
the declaration. An isopeptide requested at anchors that cannot form one now **raises** instead
of warning — matching what disulfide already did, and refusing to score an open ring as closed.

A second defect in the same function: endpoints were mapped with
`pdb2pose(binder_chain, i_local + 1)`, which assumes the binder is numbered 1..L. The staged
complexes preserve each crystal's own numbering, and the regression check measured
`pdb2pose(chain, 1) == 0` on `LNR_2qos_C_A` — **the old code could not score any staged LNR
complex at all**. Both are now mapped by chain order.

### Constraints hold the ring closed

ref2015 ships with **zero weight on every constraint term**, so constraints added to a pose are
silently ignored during minimization. Without `constrained_score_function`, `fa_rep` pushes the
declared bond back open and hands back a "cyclic" structure whose ring has quietly reopened.
The ring's length and angles are constrained for every chemistry; the dihedral only for the
amides, because a disulfide's CB–SG–SG–CB dihedral is genuinely bimodal (±85°) and pinning it
would bias the search.

---

## 3. Search

[`script_utils/cp_search.py`](../script_utils/cp_search.py), driven by
[`scripts/genkic_cyclize.py`](../scripts/genkic_cyclize.py).

**The receptor is never moved by either engine**, and the peptide interior is preserved
*kinematically* rather than by restraint — GenKIC through its anchors, the anneal engine through
a fold tree re-rooted at an interior peptide residue so terminal torsions propagate outward.
Both verified by measurement: interior and receptor maximum displacement 0.0000 Å, which is why
`released_local` is recorded per candidate — the set of residues whose coordinates were allowed
to change is exact, not approximate.

**GenKIC (`mainchain`).** Head-to-tail cyclization is a genuine kinematic closure. Loop residues
are supplied in ring order — the C-terminal window, across the new bond, into the N-terminal
window — so GenKIC's auto-assigned anchors are the residues flanking the window.

**Free-terminus anneal (`disulfide`, `isopeptide`).** A side-chain bridge leaves both backbone
termini free, so there is no backbone ring to close; the requirement is that two side-chain
atoms meet at bonding geometry. **GenKIC cannot be used here**: it builds its loop's atom chain
from polymer connections only, so a bond through CB/SG or CE/NZ is not on the chain it searches
— it rejects the closure dihedral as "not found in the chain of atoms to be closed" regardless
of how the atoms are named. Instead the bond is declared, harmonic ring constraints are added,
released terminal torsions are perturbed, and a constrained minimization pulls the bridge in.
The achieved geometry is then **measured** against the chemistry's window, never assumed.

**Window ladder.** `(k, m)` released residues per terminus, ordered smallest-first so the
best-retaining answers are found before more of the peptide is disturbed. Every rung records
exactly which residues it released. GenKIC needs roughly six loop residues to close reliably
(k=m=2 found nothing in 2000 attempts on `LNR_2qos_C_A`; k=m=3 closed in 0.1 s), and a
2-residue loop is refused outright because its 2nd and 3rd pivots coincide.

**Two-phase, by cost.** Closure attempts are cheap and relaxation is not, so the search grades
thousands of candidates on geometry/sequence/pose alone, then a ranked shortlist
(`cp_evaluate.shortlist_key`) pays for packing, relaxation and interface scoring. The ranking
key deliberately excludes energy: it decides who gets *measured*, not who wins.

**The paired baseline is literal.** `refine_lp_baseline` puts the original LP down the identical
pack/relax path — same score function, same coordinate restraints, same packable set, same relax
repeats — differing only in that no ring is declared. Scoring an unrelaxed crystal LP against a
relaxed CP would credit cyclization with the relaxation's own energy gain, which this repo has
measured at hundreds of REU: two orders of magnitude larger than the ±5 REU the contract is
trying to resolve.

---

## 4. Bugs the one real-complex smoke caught

The preflight is a single complex taken end to end, in four stages (offline bookkeeping →
library regression → ring declaration → full search/relax/score). It found three defects that
no offline test could have:

1. **`peptide_internal_energy` aborted on disulfides.** `pdbslice` carries residue *types*
   (including the disulfide variant) but **not** the connections between them, so both CYS
   arrive still flagged as bonded with their partner mapping gone, and ref2015's
   `FullatomDisulfideEnergyContainer` aborts the process. The ring is now re-declared on the
   slice; a slice that cannot be re-declared returns NaN rather than a number, because a
   silently open ring would understate strain by exactly the term being measured.
2. **A 2-residue GenKIC loop aborted the process.** `loop[len//2]` is the last residue, so
   pivots 2 and 3 coincide and Rosetta exits with "The third pivot cannot be before the second
   pivot" — inside `set_pivot_atoms` validation, where no `try/except` around `apply` can catch
   it.
3. **Variant changes obsolete the pose's `PDBInfo`, which killed the entire primary
   chemistry.** Stripping terminus variants for the head-to-tail amide (and GenKIC's pose
   rebuild) leaves the PDBInfo obsolete or absent; `ChainSelector` then raises "received a pose
   without a valid PDBInfo" and **every mainchain relax failed**, while the side-chain
   chemistries — which happen not to trip it — sailed through. An aggregate "did anything
   work?" check passes in that state, so the smoke now asserts **per chemistry** that whatever
   closed also got through relax and scoring.

---

## 5. Comparison: SDEdit under the same contract

[`script_utils/score_sdedit_contract.py`](../script_utils/score_sdedit_contract.py) grades
existing SDEdit outputs with the *same* module, two-phase, acceptance on the relaxed candidate.

This is only possible for SDEdit runs that wrote `complexes/`. Each `edits_shard*.jsonl` row
carries `complex_pdb` (the generated peptide **with** its receptor) and `input_complex_pdb` (the
original LP in the **identical** cropped receptor) — so the paired baseline the contract demands
already exists on disk, with the same crop and the same frame, and the receptor residues
correspond one-to-one. A run that wrote only `pdbs/` holds peptide-only structures and is
**ungradable in principle**: contacts, interface-frame RMSD and paired dG all need the receptor,
and a peptide saved alone cannot be re-united with it afterwards (two loads of one example
differ by ~2.7 Å). The job gates on `complexes/` by name rather than emitting a table of NaNs
that would read as SDEdit failing.

Deliberately **not** done: abstaining rows are *not* filtered out.
`requested_type_satisfied == 0` means the sampled sequence carries no competent anchor pair for
the requested chemistry; those rows are graded and fail on chemistry. Excluding them would
compute the success rate over only the attempts that had already half-succeeded. (They also
carry NaN in the run's own metric columns, and NaN is truthy in Python — a naive filter scores a
100 %-abstention cell as 100 % success.) SDEdit resamples the whole peptide, so its released set
is recorded as every residue and `interior_max_ca_dev_A` is not meaningful for that arm.

---

## 6. Running it

Pure CPU — GenKIC, the anneal, FastRelax and InterfaceAnalyzerMover are all CPU, so everything
goes to the `cpu` partition. Asking for a GPU would only queue this behind real GPU work.

```bash
cd /zfsauton2/home/yixiz/Proteina-Complexa/.claude/worktrees/genkic

./scripts/submit_genkic.sh --smoke     # one real complex, end to end. Read it first.
./scripts/submit_genkic.sh --all       # search array + SDEdit grading (siblings) -> report
```

Individual stages, and overrides:

```bash
./scripts/submit_genkic.sh --search
./scripts/submit_genkic.sh --sdedit SDEDIT_DIR=$PWD/evaluation_results/sdedit_pocket_20260906_180646
./scripts/submit_genkic.sh --report RUN_ID=genkic_20260928_173423

# more search effort, longer shards
./scripts/submit_genkic.sh --search ATTEMPTS_PER_CALL=20000 CALLS=8 TIME_BUDGET_S=25200
# three independent relax/score repeats for the shortlist, per the contract
./scripts/submit_genkic.sh --search RELAX_REPEATS=3 LP_REPEATS=3
# one target, all chemistries
./scripts/submit_genkic.sh --search EXAMPLES="LNR_2qos_C_A"
```

**Resuming.** Resubmit the identical command with the same `RUN_ID`. Each shard reads its own
output file and skips finished `row_key`s, and `TIME_BUDGET_S` stops a shard from *starting* a
target it cannot finish, so hitting the walltime means resubmitting rather than restarting.

```bash
./scripts/submit_genkic.sh --search RUN_ID=<the same run id>
```

Configuration travels as a **positional env-file path**, never `sbatch --export`: any explicit
export list sets `SLURM_GET_USER_ENV=1`, slurmd then fails to rebuild the login environment, and
the job is requeued and **held** with every dependent stranded on `Dependency` (measured here,
jobs 22184 and 22243). One output file per shard, because concurrent `O_APPEND` has NUL-corrupted
rows on this cluster (88 of 144 lost) — and the aggregator **refuses** a corrupt line rather than
skipping it, so data loss stops the run instead of quietly shrinking the table.

---

## 7. Results

`evaluation_results/<RUN_ID>/report/` holds `results.txt`, `per_target.jsonl` and
`summary.json`; representative structures are in `pdbs/`.

Reporting rules the aggregator enforces, each closing off a way a result can look better than it
is:

1. **The denominator is every target in the metadata**, not every target with rows. A dead
   shard, a target that never loaded and a genuine search failure all appear — otherwise
   "accepted / targets-with-results" silently becomes the success rate.
2. **A weaker tier is never a primary success.** Budget and identity tiers sit in their own
   columns beside the primary bar.
3. **Chemically infeasible is not a failure**, and stays visible in coverage.

> **Status (2026-09-28).** Code complete and the smoke's library/topology stages pass on a real
> complex. The substantive search has not yet been launched, so the tables below are not yet
> filled. Do not quote closure or acceptance numbers from this file until this note is replaced
> by a run id.

<!-- FILL AFTER THE RUN:
     - best accepted candidate per LP, and unresolved LPs with the bar that bound them
     - coverage by chemistry x budget, including infeasible cells
     - tier grid (budgets 0/1/2/3, identity 90/80/70)
     - quality vs search effort (by released-residue count)
     - pre- vs post-relax deltas, and shortlist repeat variability
     - GenKIC vs SDEdit on the LPs graded by BOTH arms
-->
