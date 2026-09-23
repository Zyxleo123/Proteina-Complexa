# Milestone 1.5 -- geometric ceiling audit

CPU-only. Builds nothing. Reports, then waits.

Read this before implementing section 3's coverage target or section 5's filters: the
numbers here may change both.

---

## 1. The question

A measured contact retention is uninterpretable on its own. `0.40` is a failure against a
ceiling of `0.95` and near-perfect against a ceiling of `0.45`, and nothing measured so far
distinguishes those two readings. The prior line's 0.37-0.42 was treated as a failure
without ever establishing which one it was.

So, before any pair is constructed: **holding a contiguous window of a peptide at its
native bound position, how much of the native interface can survive ring closure at all?**

The deliverable metric should then be rescaled to `achieved / ceiling`.

## 2. What is measured

For every complex, for each closure chemistry, an exhaustive `O(L^2)` scan over contiguous
held windows `[a, b]`:

| column | meaning |
|---|---|
| `<chem>_ceiling_strict` | **FLOOR.** Retained fraction counting only residues inside the held window; every released residue is written off. Analytic feasibility only |
| `<chem>_ceiling_refined` | **FLOOR.** `_strict` with the window confirmed by a torsion solve. The number to use where the bracket is tight; never alone |
| `<chem>_ceiling_permissive` | **UPPER BOUND.** A released residue keeps a contact whose partner is still inside its reachable ball. Saturates near 1.0, so it bounds without discriminating |
| `<chem>_max_feasible_window_len` | the largest window that can be held while still closing |
| `<chem>_best_a`, `<chem>_best_b` | which window achieves the ceiling |
| `<chem>_native_feasible` | is the peptide already closable with nothing released |
| `mainchain_ceiling_strict_tau_sens` | the same ceiling at a realistic rather than extremal backbone extension |

Contact retention uses **CA-CA within 10 A**, matched byte-for-byte to
`scripts/sdedit_cyclize.py`'s `CONTACT_CUTOFF_NM = 1.0`. A ceiling computed against a
different cutoff is a different quantity and must not be divided into a retention measured
against that one.

### The ceiling is an upper bound, deliberately

No sterics, no Ramachandran, no receptor excluded volume, no side-chain packing. Erring
permissive is the only safe direction: a permissive ceiling can make the model look worse
than it is, never better. Where the idealisation is loose, the report says so rather than
quietly benefiting from it.

### `ceiling_strict` / `ceiling_refined` is a FLOOR, not a ceiling

**This is the single most important thing to know before quoting any number from this
audit**, and it was got wrong twice before the full run settled it.

`retention_ceiling`'s strict accounting counts a contact as kept only when its residue is
**inside** the held window. Every released residue is written off wholesale. But a residue
that moves 2 A does not lose a 10 A CA-CA contact, so the write-off is simply wrong, and
`ceiling_strict` (and `ceiling_refined`, built on it) sits *below* what closure actually
permits.

Measured on the full run (n = 660):

| bound | mainchain median | exceeded by |
|---|---|---|
| `ceiling_refined` | 0.586 | **86.4%** (max 2.74x) |
| `ceiling_permissive` | 1.000 | **0.0%** |

`ceiling_permissive` -- a released residue keeps a contact whose partner is still inside
its reachable ball -- is the valid upper bound. It is also useless on its own, because it
saturates at 1.000.

**So the audit produces a bracket, not a number.** Where the bracket is tight the chemistry
is answered; where it is wide the question is open. On the full run:

| chemistry | bracket | verdict |
|---|---|---|
| isopeptide | [1.000, 1.000] | answered |
| disulfide | [0.981, 1.000] | answered |
| head-to-tail | [~0.59, 1.000] | **open** |

A correction worth recording, because the wrong version was acted on for two days: the
exceedance was first blamed on the brief's **contiguous-window** constraint -- the idea
that a sampler can hold several non-adjacent stretches. That explanation is wrong. The
contiguous window is representable and not the binding constraint; the free-residue
write-off is. The tell is isopeptide, which has **0%** exceedance precisely because its
ceiling holds the *whole* peptide, so nothing is written off. Had contiguity been the
cause, isopeptide would have exceeded too.

Closing the head-to-tail bracket needs conformational sampling under the closure
constraint -- generate closure-satisfying conformers and measure retention directly -- not
a tighter analytic bound. That is Milestone-2-scale work and a scope decision, not a patch.

### Compare closed rings to closed rings

The ceiling is retention **subject to closure**. An open ring keeps whatever contacts it
started with, so including open rings on the achieved side is not a comparison. The report
therefore filters to rows where the requested chemistry was produced *and* the bond
actually formed (`cyc/<type>_bond_success`, never `cyc/cyc_cb_window_success`, which
saturates).

On the smoke that filter is worth a lot: 3,540 ok rows -> 1,972 of the right type -> **742
closed**. Skipping it inflated achieved retention by roughly 2x and pushed the ratio above
1.0 -- above a quantity that is supposed to bound it.

## 3. The two instruments

`script_utils/kinematic_ceiling.py` carries both, on purpose.

**Analytic (primary).** The peptide is a virtual CA chain, 3.80 A bonds, CA-CA-CA
pseudo-angle bounded at `tau_max`. A held window fixes the vector between its two ends;
each free flank reaches an annulus about the window edge; closure is then an
annulus-annulus distance-range test in closed form. `O(1)` per window, so every window of
every complex is enumerated rather than sampled.

The reach bound is exact for the stated pseudo-angle: the extremal conformation is the
planar all-trans zigzag, `n * 3.80 * sin(tau/2)`. At `tau = 150 deg` that is 3.67 A per
residue, so a 13-mer spans 44 A -- which reproduces the "fully extended 13-mer is about
45 A" figure the inventory quoted from a different direction.

**Torsion-space (validator).** A real N/CA/C backbone is rebuilt with NeRF *outward from
the held window*, so the window keeps its native coordinates exactly, and `least_squares`
solves the free phi/psi for closure. Building outward rather than forward from residue 0
is what makes the window genuinely fixed -- a forward build moves the window whenever an
upstream torsion changes, silently converting "hold this window" into "hold nothing".

The validator is sampled **at the decision boundary**: one window just inside the analytic
limit and one just outside. Agreement on obviously-open and obviously-closed cases proves
nothing, and the boundary is where a ceiling is set.

`analytic_UNDER_licensed` is the only fatal column in the validator table. Over-licensing
is expected and measures the bound's looseness; under-licensing would mean the ceiling is
too low and a real deficit could hide behind it.

> A finding from building it: modelling an **in-window terminal atom as free within its
> bond length** added ~3 A of slack and licensed windows the torsion solver could not
> close. Holding a window holds its backbone, so `N(0)` and `C(L-1)` are passed explicitly
> and pinned. The looser "CA trace only" reading is still reachable by omitting them.

### The analytic bound alone is not good enough -- measured, not suspected

Running the validator, the torsion solver closed **0 of 113** boundary windows the
analytic bound called feasible (full run; the smoke saw 0 of 34). Probing one complex
window by window made the cause plain:

```
LNR_1bjr_E_I, L=10, terminal gap 6.05 A
  window   free   analytic   torsion
  [0,8]       1       True     False     <- analytic licenses closure here
  [1,7]       3       True     False
  [2,6]       5       True      True     <- the backbone actually needs 5 free residues
```

The annulus model lets a short flank reach anywhere on a **sphere** about the window edge.
A real backbone's first free residue lies on a **circle** -- a cone about the incoming
chain direction at fixed pseudo-angle, with only the dihedral free. The two agree once
there are enough free residues to wash the directional constraint out, and diverge badly
at the boundary, which is exactly where a ceiling is set.

So the analytic pass is demoted to a **pre-filter** and every reported ceiling is
**confirmed in torsion space** (`refine_with_torsion`). The pre-filter is still worth
keeping: it is a true upper bound, so anything it rejects is genuinely infeasible, and it
supplies the retention-ranked candidate list.

The walk down that list is made cheap by an **exact** prune, not a heuristic: holding more
residues fixed can never make closure easier, so once a window fails, every window
containing it fails too. In practice that costs 2-7 solves per complex, ~23 s.

Effect on the headline, full run:

| set | analytic median | refined median |
|---|---|---|
| LNR mainchain | 0.789 | **0.586** |
| PepBench mainchain | 0.775 | **0.584** |

`analytic_median` is kept in the report only to show what the cone-vs-sphere idealisation
was worth. Never quote it -- and note that `refined` is a floor, not the answer either
(see above).

One caveat the validator does **not** cover: it probes the *analytic* boundary, which is
why it reads 0% agreement. Nothing independently checks `ceiling_refined`, because the
refinement and the validator are the same torsion solver. The refined number is
self-consistent, not independently confirmed.

### Bridged ceilings assume the anchors can carry the chemistry

A bridged ceiling of 1.0 means some `(i, j)` pair is already at bond distance, so nothing
has to move. That is a geometric statement; it says nothing about whether residues `i` and
`j` are a CYS pair or a LYS/ASP pair. `<chem>_anchor_native_compatible` records whether
they natively are. Given that isopeptide failure is about half anchor identity rather than
geometry, do not quote a bridged ceiling without it.

## 4. Bridge windows are measured, not asserted

`calibrate` reads CPSea natives, resolves each linkage from CONECT via
`infer_cyclization_label`, and takes `[p1, p99]` of the observed CA-CA and CB-CB distances
plus a pad. Asserting these from textbook chemistry would make every ceiling below depend
on a guess.

On the 90-complex AFDB fixture the windows come out at (CA-CA): disulfide 4.08-6.64 A,
isopeptide 4.21-10.16 A, mainchain 2.55-4.40 A -- all chemically sensible, which is the
point of checking.

## 5. Bridge span, not terminal gap

For a disulfide or isopeptide macrocycle the bonded atoms are interior side chains, so
`N(res 0) -> C(res L-1)` measures the free tails and not the ring. The profile therefore
also carries, per peptide, the distribution over **every** `(i, j)` CB-CB distance with
`|i - j| >= 3`, the best such pair per chemistry, and `<chem>_hostable` -- whether any pair
already sits inside the measured bond window.

For the bridged ceiling the anchors are *not* fixed, so the scan maximises over `(i, j)`
as well as over windows. Reporting only the metadata's single "best" pair would answer a
narrower question than the one that matters: whether **some** bridge preserves the
interface, not whether the closest-to-ideal one does.

## 6. The replica-economics measurement carries its own trap

`timing.*` in the YAML must mirror `scripts/soft_closure_project.py`'s own argparse
defaults, or the measured marginal is not the cost of the protocol the build will run.

`max_iterations` is the one that bites. `0` means "minimize to convergence, unbounded";
the reference protocol caps at 1000. Measured on one fixture complex, unbounded
minimization cost **51 s** per replica against a **2.7 s** context setup -- which would
have inverted the brief's "replicas are nearly free" into "replicas dominate" on the
strength of a config typo rather than a fact about the system.

`k_pull` (2000), `k_contact` (500) and `contact_tol_A` (0.75) are pinned to the same
source for the same reason.

The marginal unit timed is **one** restrained minimization, not the 6-rung pull ladder
`project_once` runs: a decoy is perturbed and settled once. The ladder is timed
separately, for comparability with the ~65 s figure in the `ExampleContext` docstring.

The hold restraint the decoy-settling step needs (gap-list item 4) comes free here -- the
existing pull force with `r0` pinned at the native bond length *is* a hold.

### Measured, and it points the opposite way to the brief

19 CPSea fixture complexes x 10 replicas, reference protocol, one core:

```
setup (PDBFixer + ExampleContext)   1.7 s
marginal (perturb + one minimize)  30.0 s      ratio 0.074x
amortised per decoy at 8           30.2 s
amortised per decoy at 30          30.1 s
```

Sizing note, learned the hard way: the 2-complex smoke reported 81.5 s and the login node
42.2 s. Both are wrong by 1.4-2.7x. The ratio was stable across all three, the absolute
cost was not -- so read the ratio off a small sample if you must, but never the budget.

Setup amortises to nothing; the minimization is the whole cost. The `ExampleContext`
docstring's "~63 s of a ~65 s replica is setup" does not reproduce here.

A plausible reason -- untested, so do not repeat it as fact: that figure was measured on
LNR complexes, which carry full-chain receptors, while CPSea's receptors are already
pocket-cropped to ~106 residues. Setup scales with system size; the minimization from a
freshly perturbed start does not amortise at all.

**Replicas are not nearly free, but they are cheap**: at 30 s a replica, 30 decoys per
complex is ~15 CPU-minutes, linear in decoy count. Setup amortises to nothing by 8 decoys,
so the "wide on replicas, narrow on complexes" shape buys nothing -- but it costs little
either, which is a different conclusion from the one the 81.5 s figure supported.

The ladder being *cheaper* than a single replica is consistent with this: each rung starts
from the previous rung's minimum, while every replica restarts from a perturbed structure
far from one.

n = 19 of 20: one complex dies inside OpenMM with `Particle coordinate is NaN`
(`AF-A0A537RZW2-F1_0_130_140`). The summary's `partial` flag says "did not finish all",
which conflates a failed complex with a wall-clock kill -- check the job log to tell them
apart.

## 7. Running it

```bash
bash scripts/submit_m15_ceiling_audit.sh --submit all
```

DAG:

```
calibrate ──afterok──> ceiling(array: lnr x4, pepbench x8) ─┐
timing ─────────────────────────────────────────────────────┼──afterany──> report
e-index ──afterok──> e-spatial(array x8) ───────────────────┘
```

`timing` and `deliverable-e` are siblings of the ceiling chain, so they queue
concurrently. `report` depends with `afterany`: a stage that gated itself out exits 0 with
an explanation, and the report's job is to say which stages contributed. `afterok` there
would strand the report on a deliberate gate.

Everything is CPU-only. No GPU is requested anywhere -- asking for one to measure geometry
just queues behind real work.

Smoke first: `--smoke` caps every set at 6 complexes, drops the validator, points
Deliverable E at the 90-row sample instead of the 2.44M-row index, and shrinks every array
to one task.

Config: `configs/pose_decoy/m15.yaml`, every constant, no exceptions. The preamble points
at it; `--set VAR=value` overrides any preamble knob without editing a script.

Tests: `.venv/bin/python -m pytest scripts/test_kinematic_ceiling.py -q` (~3 min; the
torsion solver dominates). The two that matter are
`test_max_reach_never_exceeded_by_a_real_chain` (brute-force: no sampled real chain
out-reaches the analytic bound) and
`test_analytic_never_calls_infeasible_what_the_solver_closes` (the bound never
under-licenses).

## 8. Outputs

```
evaluation_results/<run_id>/
  bridge_windows.json          measured bond windows + the observed percentiles
  bridge_windows.rows.parquet  per-native measurements behind them
  ceiling/<set>_shard<k>.jsonl one file per array task, never shared
  timing/replica_timing.jsonl  + .summary.json
  deliverable_e/               index json, candidate pairs parquet, spatial/*.jsonl
  M15_CEILING_REPORT.md        the tables
  ceiling_rows.csv             every per-complex row, flat
  figures/*.png
```

Figures are drawn from the jsonl, never from a live computation, so they re-render
cheaply and the report job needs nothing but disk.

**One output file per array task.** Concurrent `O_APPEND` to one file NUL-corrupts rows on
this filesystem, and the report *refuses* a corrupt line rather than skipping it -- a
skipped row turns data loss into a quietly wrong average.
