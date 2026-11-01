# LP → CP: what the terminal-only finding invalidates, and the two routes forward

Decision record, 25 September 2026. Triggered by a single measurement that undercut a chain
of recorded design decisions. Nothing here is a measurement of its own — it is the
bookkeeping of which earlier conclusions survive, and what the two remaining routes cost.

Read `README_LP2CP_PRECHECK.md` first for the numbers this rests on.

---

## 1. The finding

**CPSea is exclusively terminal-bonded.** 60/60 sampled complexes bond at exactly (0, L−1).
Bond lengths split 1.33–1.36 Å (peptide / isopeptide) and 2.03 Å (disulfide), so the
*chemistry* varies but the *endpoints never do*. This confirms `cpsea-head-to-tail-verified`
(27k raw structures; all 2.71M preserved through preprocessing).

The inference path is terminal-only too: the design pipeline **derives endpoints as
first/last valid residue** rather than taking them as input.

So the model cannot place a ring bond at an interior (i, j) pair — not in training data, not
at sampling time. There is no fine-tune that fixes this, because there is no interior-bridge
example anywhere in the corpus to fine-tune on.

---

## 2. What this invalidates

The dependency runs `README_POSE_DECOY_INVENTORY.md` §4.4 → §4.3's CLOSED note, and it is
load-bearing in four places.

### 2.1 The coverage target (§2.1 → §4.3-CLOSED)

§2.1 originally set the target: move the terminal-gap median 14 → 20 Å, with ~57% of mass
above 20 Å, to match real linear binders (PepBench median 21.6 Å, 56.7% in the 20–45 Å band,
against CPSea natives at 1.4 / 7.0 / 8.1 Å and **0%** in band).

§4.3-CLOSED **replaced** that target on the strength of §4.4, substituting "reduce isopeptide
hostability to roughly two thirds (LNR 71.7%, PepBench 64.7%)". Hostability was defined over
**all** (i, j) pairs.

For a terminal-only model, hostability collapses to *terminal* hostability — is the terminal
CA–CA pair inside the calibrated bridge window — which is a monotone function of terminal gap.

**So the discriminator reverts to terminal gap, and §2.1's original coverage target is
reinstated**, along with the coverage-vs-contact-filter tension §2.1 identified.

### 2.2 The contact-aware extension move (§4.3-CLOSED)

Dropped explicitly: *"The extension move existed to manufacture large terminal gaps, because
terminal gap was the coverage axis… so the move would not serve the target that replaced it."*

Terminal gap **is** the axis. **Re-open this item.**

### 2.3 The `bridge span` profile feature (§4.4 → §8)

§4.4 added bridge span — the CB–CB distance over every (i, j) pair that could host a bridge —
as *"the feature that says whether a given real linear peptide is cyclizable at all, and
where."*

For a terminal-only model the "where" is fixed at (0, L−1), so an all-pairs bridge span will
report a peptide as cyclizable that this model cannot cyclize. The informative feature is the
**terminal** CA–CA distance.

### 2.4 M1.5's bridged ceilings and the 20–45 Å routing

Confirmed in code: `m15_ceiling_audit.py:297` loops all (i, j) for the bridged chemistries and
takes the max. So the bridged ceiling is an **interior-bridge** ceiling, and the model's
achieved closure was measured against a target it cannot reach — which inflates the reported
deficit. Same for M1.5's routing of the 20–45 Å band to bridged chemistries: a 26 Å or 45 Å
terminal gap is only benign for a bridge if the bridge is interior.

**Open item:** re-run Task 2's bridged arm with anchors pinned to (0, L−1), and re-read M1.5's
bridged numbers against that.

### 2.5 What survives unchanged

- **Stage 0's "disulfide budgets 0/1 are chemically empty on 8/8 cases."** A terminal disulfide
  needs CYS at both termini, costing 2 mutations. This *corroborates* terminal-only.
- **`cpsea-sidechain-closure-is-identity-not-geometry`** — isopeptide failure is ~50% anchor
  identity. Under terminal-only this becomes *more* central, not less: the anchor positions are
  fixed, so the only lever is what residue sits there.
- **`cpsea-terminal-anchor-graft`** — grafting CYS/CYS or LYS/ASP onto the termini. Built, and
  now squarely the right mechanism rather than one option among several.

### 2.6 What terminal bridging still buys

Real but modest. It relaxes the closure requirement from "ends within ~1.3 Å" (head-to-tail
N–C) to "ends within ~7–10 Å" (calibrated disulfide / isopeptide CA windows). Measured on LNR:
**5%** (disulfide) and **15%** (isopeptide) of targets are already inside the window at k = 0,
against **0%** for head-to-tail, with a median terminal CA gap of 16.3 Å.

A softer constraint on the same axis — not a different axis.

---

## 3. The two routes

### Route A — build the pose-decoy set (the existing plan)

Purpose: stop the model ignoring its input and recalling the pocket's canonical answer, by
giving two examples that share a pocket different correct answers.

**Task 1 removed its acceptance filter.** `contact_retention` does not track Rosetta interface
ΔdG in the operating range (r² 0.004; residual SD 19.44 REU against a dG SD of 19.47). And the
2 Å decoy amplitude floor (86% of poses under 1 Å collapse back to native) is **disjoint** from
the filter's ≤2 Å operating band, so the regime needing fine judgement is the regime that
yields no usable decoys.

Still to build: ring opening (nothing in the repo cuts a ring bond), the corruption module
(1 of 7 knobs), backbone-geometry and relaxed-back filters, a sharded columnar writer plus
manifest, a structural cluster partition against LNR (no `foldseek`/`mmseqs` on PATH), the
dataset-level identifiability audit, and 3d. Plus a replacement acceptance filter, which is now
an open design question rather than a settled one.

### Route B — a bridge, no decoy set

Transport between two *data* distributions: LP as the initial condition of the process, CP as
the terminus, rather than conditioning on the LP from a noise start. The model cannot ignore
the input because the trajectory begins there.

**This deletes most of Route A's build**: ring opening, the corruption module, the acceptance
filter, the sharded writer and the identifiability audit all become unnecessary. After Task 1
killed the filter, that is the strongest argument for the pivot.

**The enabling step already exists and has never been run.** The LP-mixing arm mixes
PepBench + ProtFrag into CPSea via a `LINEAR` topology token (index 4 — *not* `UNSPECIFIED`,
which is the CFG null) on a **shared VAE**, with LNR held out by id and sequence. Built
2026-09-07. Verified 2026-09-25: no store directory, no wandb run, no evaluation output. The
data is staged (`lp_train.parquet`, `lp_val.parquet`); the training never happened.

If it trains, DDIB may be unnecessary — one model covering both topologies makes the edit a
token swap plus SDEdit from the LP's noised latent, which *is* the bridge. That also resolves
the recorded "DDIB/FlowEdit blocked by AE mismatch", later softened to "`bb_ca` IS shared, so
DDIB is only half-blocked".

---

## 4. The blocker a bridge does NOT inherit for free

Easy to assume the bridge dissolves the decoys' job. It does not.

The decoys existed to stop the model ignoring its input. A bridge constrains *how* it departs
from the input — it must travel rather than teleport — but not *how far*. And the measurement
says it travels a long way:

> **At `t_lat = 1.0` — frozen sequence, maximum input fidelity — closure is 1/59, at every
> `t_ca`.** (`lnr-sdedit-full-tlat-grid-results`)

Closure only happens when the sequence is allowed to move substantially. So the model buys
closure by rewriting its input, which is the same failure in a different coordinate.

What bounds the distance is a **mutation budget**. Stage 0 found the budget **not
enforceable**, and there are **no motif-restricted metrics**, so preservation of the binding
motif cannot currently be scored apart from whole-peptide RMSD.

**The decoy set's job transfers to budget enforcement.** Cheaper than building decoys, but
neither free nor built.

---

## 5. Ranked blockers for Route B

| # | blocker | state |
|---|---|---|
| 1 | LP-mixing arm never trained | built, GPU-ready, data staged |
| 2 | Mutation budget not enforceable | this is what now does the decoys' work |
| 3 | No motif-restricted metrics | can't score motif preservation apart from global RMSD |
| 4 | No production seqcond checkpoint | run died at step 3745; pad fix verified, needs relaunch |
| 5 | Geometry ceiling (Task 2) | not a blocker — an expectation-setter |
| 6 | No structural holdout against LNR | limits generalization claims; doesn't block running |

On #5, the honest ceiling for **any** method, terminal-only: a median LNR target needs 4 of
~10 residues released to close head-to-tail, and preserving ≥70% of anchor contacts reaches
**27%** of targets at k ≤ 4 and **60%** at k ≤ 7. A bridge is better at *finding* a solution;
it does not make one exist.

---

## 6. What's next

**Now, CPU, cheap.** Re-run Task 2's bridged arm with anchors pinned to (0, L−1), so the
ceiling is the reachable one, and re-read M1.5's bridged deficit against it. Also attaches a
real number to the reinstated §2.1 coverage target.

**The decisive experiment.** Train the LP-mixing arm, then run the paired LNR edit against the
CP-only model using the existing before/after Rosetta dG harness
(`cpsea-sdedit-rosetta-before-after`). That tests the whole bridge hypothesis — does a
shared-latent LP+CP model edit LP→CP better than CP-only — **without building any decoy set**.
Everything needed is already built.

**Then, conditional on that working.** Budget enforcement plus motif-restricted metrics, since
those decide whether the closures it produces are worth anything.

The two can run in parallel: one is CPU, one is GPU, and neither depends on the other.

---

## 7. One clarification, because it has been misread

The **AUC 0.900** in Task 3 is **not** a statement that decoys are distinguishable from real
structures. There are no decoys in that comparison — none exist yet.

It is the **staging control** on the positive-control pair (CPSea cyclics vs PepBench real
linear peptides, both real structures), computed on the staging vector alone: peptide length,
receptor length, receptor segment count, max run, fragments < 5. It says *you can tell which
dataset a record came from by how its receptor was cropped* — 0.900 after the uniform 14 Å
crop, down from 0.999 before it.

It matters as a **confound, not a grade**: it is a noise floor. When a synthetic-vs-real
comparison is eventually run, a high AUC will not distinguish "the synthetic states are
unrealistic" from "the staging differs". That has to be removed or subtracted first.

The instrument itself is sound — 0.998 on the real difference, carried by conformational
features (`e2e_ca_A`, `terminal_exposure`, `e2e_ca_per_link`), not staging ones.
