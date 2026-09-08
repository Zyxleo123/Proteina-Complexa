#!/bin/bash
# Submitter for the SIMILARITY-GUIDED SDEdit sweep (LP -> CP). Calls sbatch and nothing else.
#
#   bash scripts/submit_sdedit_guided.sh [--smoke] [--closure] [--partition P] [--nodelist N]
#                                        [--exclude N] [--shards N] [--time T] [--peptides N]
#                                        [--t-ca "..."] [--t-lat "..."] [--seeds "..."]
#                                        [--arms "<arm> <arm> ..."]
#                                        [--metadata PARQUET] [--tag NAME] [--ros-time T]
#                                        [--abstained-from "GLOB"] [--abstained-min-rate R]
#
# CLOSURE GUIDANCE, ON THE ABSTAINING CELLS (the DPS + bond-distance question):
#   bash scripts/submit_sdedit_guided.sh --closure --tag closure \
#     --metadata CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet \
#     --abstained-from "$PWD/evaluation_results/sdedit_pocket14_20260906_213400/edits_shard*.jsonl"
#
# RE-RUN ON THE POCKET-CROPPED TARGETS:
#   bash scripts/submit_sdedit_guided.sh --tag pocket14 \
#     --metadata CPSea_data/lnr_pocket14/metadata/lnr_test_pocket.parquet
#
# The first guided sweep ran on CPSea_data/lnr_staged, whose receptors are complete chains
# and reach the denoiser as ~1 segment where all CPSea training used a ~13-fragment pocket
# crop. That difference alone is worth 22-34 points of ring closure -- larger than anything
# the guidance hook does -- so a guided-vs-unguided frontier measured on the full chains is
# measured in a regime the model was never trained in. Use the 14 A crop, which reproduces
# the training distribution on max_run, seg_med, seg_p90 and frag<5 at once; the 18 A set
# (CPSea_data/lnr_pocket) matches on segment COUNT only and still runs max_run 54 vs 32.
# See docs/README_TARGET_DISTRIBUTION_MATCHING.md and scripts/restage_lnr_pocket.py.
#
# What this run asks
# ------------------
# Plain SDEdit trades ring closure against preservation along ONE frontier controlled by
# t_ca. This run adds a second handle -- the gradient of an explicit LP-vs-CP similarity
# loss, added to the state after every Euler step (scripts/sdedit_guidance.py) -- and asks
# whether the guided frontier sits ABOVE the unguided one or merely slides along it.
# Sliding along means guidance is a reparameterization of t_ca and buys nothing; that is a
# real, publishable answer, so the unguided arm is part of the same job, same peptides,
# same seeds, and NOT a comparison against an older run.
#
# Grid choices, and why
#   t_lat = 0.4   a frozen sequence cannot close (1/59 at t_lat = 1.0 for every t_ca), and
#                 0.2-0.4 was the best corner in the phase-B grid.
#   t_ca 0.2-0.8  the whole unguided frontier, because the question is not "is the guided
#                 point better" but "does the guided FRONTIER sit above the unguided one" --
#                 which needs both curves, measured on the same peptides in the same job.
#   seeds 0,1     seed dominates dG on this pipeline; one seed cannot call a difference.
#
# Sizing: the last unguided sweep ran 600 edits per shard in 6.0-8.3 ks, i.e. ~14 s/edit at
# nsteps=400 on these short peptides. 12960 edits over 12 shards is ~4-6 h per shard (the
# dps arm is ~2.5x an identity one), inside a 12 h window with room to spare -- and the job
# is resumable per run_key, so a timeout costs a resubmit, not a rerun.
#
# Env config travels as a FILE passed positionally -- never `sbatch --export`, which sets
# SLURM_GET_USER_ENV=1 and gets jobs requeued and HELD.
#
# Stage 1: GPU array, one shard of input peptides each, all guidance arms run sequentially.
# Stage 2: CPU summarize (arm-aware tables + the guided-vs-unguided frontier figures).
# Stage 3: CPU Rosetta dG before/after, capped PER ARM, + the paired figure.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# The dps arm backprops through the denoiser, so its activation footprint is ~2-3x an
# identity step. The unguided sweep ran on A6000s (48 GB); keep the job off the 24 GB a5000
# and the 11 GB 2080 Tis with --exclude, because a CUDA OOM here dies with no traceback and
# reads as a mysterious FAILED shard.
PARTITION="general"; NODELIST=""; EXCLUDE=""; TIME="12:00:00"; SUM_TIME="00:20:00"; ROS_TIME="1-00:00:00"
ABSTAINED_FROM=""; ABSTAINED_MIN_RATE="1.0"; CLOSURE=0
SHARDS=12; SMOKE=0; PEPTIDES=0          # 0 = every peptide in the metadata (lnr_test: 60)
TAG=""
# Default is the ORIGINAL full-chain staging, kept so existing invocations are unchanged.
# The 14 A pocket crop is the one to use from now on -- see --tag usage above.
METADATA="CPSea_data/lnr_staged/metadata/lnr_test.parquet"
T_CA="0.2 0.4 0.6 0.8"; T_LAT="0.4"; SEEDS="0 1"; NSTEPS=0; ARMS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --nodelist)  NODELIST="$2";  shift 2 ;;
    --exclude)   EXCLUDE="$2";   shift 2 ;;
    --shards)    SHARDS="$2";    shift 2 ;;
    --time)      TIME="$2";      shift 2 ;;
    --ros-time)  ROS_TIME="$2";  shift 2 ;;
    --peptides)  PEPTIDES="$2";  shift 2 ;;
    --t-ca)      T_CA="$2";      shift 2 ;;
    --t-lat)     T_LAT="$2";     shift 2 ;;
    --seeds)     SEEDS="$2";     shift 2 ;;
    --arms)      ARMS="$2";      shift 2 ;;
    --metadata)  METADATA="$2";  shift 2 ;;
    --tag)       TAG="$2";       shift 2 ;;
    --abstained-from) ABSTAINED_FROM="$2"; shift 2 ;;
    --abstained-min-rate) ABSTAINED_MIN_RATE="$2"; shift 2 ;;
    --closure)   CLOSURE=1;      shift ;;
    --smoke)     SMOKE=1;        shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Guidance arms: "<lambda>|<loss spec>|<schedule>|<pow>|<exclude termini>|<mode>", "-" = default.
# lambda 0 is the paired UNGUIDED control and runs the sampler with no hook at all.
if [[ -z "$ARMS" && $CLOSURE == 0 ]]; then
  ARMS="0|-|-|-|-|-"                                    # paired unguided baseline
  ARMS+=" 1|ca_mse,contact_hinge|decay|1.0|1|identity"   # lambda ladder
  ARMS+=" 2|ca_mse,contact_hinge|decay|1.0|1|identity"
  ARMS+=" 4|ca_mse,contact_hinge|decay|1.0|1|identity"
  ARMS+=" 2|ca_mse,contact_hinge|const|1.0|1|identity"   # schedule control
  # Terminal-exclusion ladder. These peptides are short (median 10, min 5), so k is not a
  # detail: k=0 guides the very residues that have to travel to close the ring, and k=2
  # leaves a 5-mer with one guided residue. Measure both ends rather than assume.
  ARMS+=" 2|ca_mse,contact_hinge|decay|1.0|0|identity"
  ARMS+=" 2|ca_mse,contact_hinge|decay|1.0|2|identity"
  ARMS+=" 2|ca_mse,contact_hinge|decay|1.0|1|dps"        # full Jacobian, ~2.5x wall-clock
  ARMS+=" 2|ca_mse,contact_hinge,lat_mse|decay|1.0|1|identity"  # + sequence preservation
fi
# ---------------------------------------------------------------- closure arms
# `--closure` swaps the SIMILARITY ladder above for the CLOSURE one, aimed at the cells
# where the cyclization head ABSTAINS: it emits a null edge because the decoded sequence
# admits no CYS-CYS / LYS-acid pair for the requested chemistry.
#
# Measured on evaluation_results/sdedit_pocket_20260906_180646, abstention is a pure
# function of the sequence budget t_lat and not of geometry:
#     mainchain    0% in all 20 grid cells      (termini are always valid endpoints)
#     disulfide    2-12% at t_lat<=0.4  ->  95-100% at t_lat>=0.6
#     isopeptide   2-13% at t_lat<=0.4  ->  83-98%  at t_lat>=0.6
#
# So a bond-distance loss ALONE cannot move this population: with no cysteine at the
# endpoint there is no SG atom, `atoms_valid` is False, and the gradient is identically
# zero on 100% of it (scripts/test_sdedit_guidance.py asserts exactly this). The arm that
# can move it is `anchor_ce`, which reaches the SEQUENCE via the decoded logits at the two
# endpoints. The ladder below is built so that claim is TESTED rather than assumed:
#   * bond_fb alone is the user-requested arm and the inertness control -- if it moves
#     anything, the reasoning above is wrong and that is the more interesting result;
#   * anchor_ce alone isolates the identity fix;
#   * the combination is the arm expected to win;
#   * bond_mse is the literal squared-error ablation of the flat-bottom form.
# t_lat is 0.6/0.8, not 1.0: at t_lat=1.0 the latent track is frozen, takes no step, and
# the per-track step_rms scaling makes any latent guidance a no-op by construction.
if [[ $CLOSURE == 1 ]]; then
  # 1.0 is included ONLY because of the graft. Without it the frozen-sequence corner cannot
  # produce an anchor pair at all (0 substitutions), so it was unmeasurable; with the anchors
  # supplied in the input it becomes the cleanest cell in the grid -- sequence held exactly,
  # ring requested. Note the ungrafted arms at t_lat=1.0 are expected to abstain 100%: that
  # is the control, not a bug.
  T_LAT="${T_LAT_OVERRIDE:-0.6 0.8 1.0}"
  CYC_TYPES_CLOSURE="disulfide isopeptide"
  if [[ -z "$ARMS" ]]; then
    # Fields: lambda|loss|schedule|pow|k|mode|stride|graft
    ARMS="0|-|-|-|-|-|1|0"                                   # paired unguided, ungrafted
    ARMS+=" 2|bond_fb|decay|1.0|1|identity|1|0"              # the bond term alone (inert control)
    ARMS+=" 2|anchor_ce|decay|1.0|1|identity|1|0"            # the identity term alone
    ARMS+=" 2|anchor_ce,bond_fb|decay|1.0|1|identity|1|0"    # soft identity + geometry
    # --- grafted arms: the anchors are GIVEN, not asked for --------------------------
    # Grafting makes the abstention question go away by construction, which turns the
    # remaining question into a purely geometric one -- can the model close a ring whose
    # anchors are already in place? It is also what makes a bond-distance loss meaningful
    # at all here: `atoms_valid` is what zeroes that term on an ungrafted abstaining
    # sample, and with CYS/LYS-ASP present it is True.
    # These arms carry the run tag "graftsc". The first attempt at them (tag "graft")
    # truncated the grafted residue to backbone+CB, which is alanine's complete atom set,
    # and the AE decoded every endpoint back to ALA -- those rows are void. The tag change
    # is what stops a resume from mistaking them for finished work.
    ARMS+=" 0|-|-|-|-|-|1|1"                                 # graft alone, no guidance
    ARMS+=" 2|bond_fb|decay|1.0|1|identity|1|1"              # graft + the bond term (now live)
    ARMS+=" 2|bond_fb,anchor_cb|decay|1.0|1|identity|1|1"
    ARMS+=" 2|bond_fb|decay|1.0|1|dps|2|1"                   # graft + full-Jacobian DPS
  fi
fi

if [[ $SMOKE == 1 ]]; then
  SHARDS=1; TIME="02:00:00"; PEPTIDES=2; NSTEPS=50
  T_CA="0.4"; SEEDS="0"
  if [[ $CLOSURE == 1 ]]; then
    T_LAT="0.6"
    # Every closure code path: no hook, bond-only (the inert one), the identity term, and
    # the dps variant that backprops through the denoiser AND the decoder.
    ARMS="0|-|-|-|-|-|1|0 2|bond_fb|decay|1.0|1|identity|1|0 0|-|-|-|-|-|1|1 2|bond_fb|decay|1.0|1|identity|1|1 2|bond_fb|decay|1.0|1|dps|2|1"
  else
    T_LAT="0.4"
    # Exercise all three code paths (no hook / identity / dps) on a couple of peptides.
    ARMS="0|-|-|-|-|- 2|ca_mse,contact_hinge|decay|1.0|1|identity 2|ca_mse,contact_hinge|decay|1.0|1|dps"
  fi
fi

# A real shell variable, because it is read twice: written into the env file below AND used
# by the sizing echo. Defining it only inside the heredoc left it unbound in this shell.
CYC_TYPES="${CYC_TYPES_CLOSURE:-mainchain disulfide isopeptide}"

STAMP="$(date +%Y%m%d_%H%M%S)"
[[ "$METADATA" = /* ]] || METADATA="${REPO}/${METADATA}"
[[ -f "$METADATA" ]] || { echo "FATAL: metadata not found: $METADATA" >&2; exit 1; }
SUFFIX="${TAG:+_${TAG}}"
RUN_ID="sdeditg${SUFFIX}_${STAMP}"; [[ $SMOKE == 1 ]] && RUN_ID="sdeditg${SUFFIX}_smoke_${STAMP}"

mkdir -p slurm_logs env_files
ENV_FILE="${REPO}/env_files/${RUN_ID}.env"
cat > "$ENV_FILE" <<EOF
# Auto-written by scripts/submit_sdedit_guided.sh at ${STAMP}.
RESULTS_DIR=${REPO}/evaluation_results/${RUN_ID}
# The receptor-inclusive complexes MUST be written by the sampler: the CPSea loader's frame
# is redrawn per process, so a peptide saved without its receptor can never be put back.
PDB_DIR=${REPO}/evaluation_results/${RUN_ID}/pdbs
COMPLEX_DIR=${REPO}/evaluation_results/${RUN_ID}/complexes
ROSETTA_SHARDS=${SHARDS}
ROSETTA_ONLY_SCORABLE=1
# FastRelax is minutes per complex; the cap is applied PER (input, guidance arm) so every
# arm gets dG data instead of the first arm eating the whole budget.
ROSETTA_MAX_PER_EXAMPLE=$([[ $SMOKE == 1 ]] && echo 2 || echo 3)
LNR_METADATA=${METADATA}
# Same pin as the unguided sweep: the bond-unroll arm, trained against \$CPSEA_AE_CKPT_PATH.
# Flow and AE must match or cross-run comparisons are meaningless.
FLOW_CKPT_PATH=${REPO}/store/cpsea_bondunroll_pin20260828/checkpoints
FLOW_CKPT_NAME=last-EMA.ckpt
SHARD_COUNT=${SHARDS}
CYC_TYPES="${CYC_TYPES}"
T_CA_GRID="${T_CA}"
T_LAT_GRID="${T_LAT}"
SEEDS="${SEEDS}"
NSTEPS=${NSTEPS}
PEPTIDE_LIMIT=${PEPTIDES}
# Cap on the CA displacement guidance may add in ONE step (A), so a blown-up gradient
# saturates instead of teleporting a residue.
GUIDANCE_MAX_DISP_A=0.25
GUIDANCE_ARMS="${ARMS}"
# Closure-arm-only: restrict the run to the (example, type, t_ca, t_lat) cells the reference
# run ABSTAINED on. Empty = the full grid. A requested cell absent from the reference is
# FATAL in the python, not a silent skip, so an arm can never quietly become an unmeasured
# subset of the grid it is compared against.
ABSTAINED_FROM="${ABSTAINED_FROM}"
ABSTAINED_MIN_RATE=${ABSTAINED_MIN_RATE}
# Grafted arms only. Empty = the driver default, "build": the grafted residue gets a complete
# side chain at a standard rotamer. Set to "truncate" for the negative control that keeps only
# backbone+CB -- which is alanine's whole atom set, so the AE decodes those endpoints back to
# ALA. It runs under a different tag ("graft" vs "graftsc") and cannot resume onto build rows.
GRAFT_SIDECHAIN="${GRAFT_SIDECHAIN:-}"
EOF
echo "env file: $ENV_FILE"; cat "$ENV_FILE"; echo

N_ARMS=$(wc -w <<< "$ARMS")
N_CHEM=$(wc -w <<< "$CYC_TYPES")
# PEPTIDE_LIMIT=0 means "all of them"; lnr_test holds 60, used here only to print the size.
N_PEP=$([[ "$PEPTIDES" == "0" ]] && echo 60 || echo "$PEPTIDES")
N_EDITS=$(( N_PEP * N_CHEM * $(wc -w <<< "$T_CA") * $(wc -w <<< "$T_LAT") * $(wc -w <<< "$SEEDS") * N_ARMS ))
echo "grid: ${N_PEP} peptides x ${N_CHEM} chemistries x $(wc -w <<< "$T_CA") t_ca x $(wc -w <<< "$T_LAT") t_lat x $(wc -w <<< "$SEEDS") seeds x ${N_ARMS} arms = ${N_EDITS} edits (~$(( N_EDITS / SHARDS )) per shard, ~14 s each unguided)"
[[ -n "$ABSTAINED_FROM" ]] && echo "  (upper bound: the abstention filter drops the cells the reference did NOT abstain on; the job's own dry-run prints the exact count)"
echo

# Scrub inherited SLURM_* so a submission from inside an allocation cannot leak into sbatch.
for v in $(compgen -v | grep '^SLURM_' || true); do unset "$v"; done

NODE_ARG=(); [[ -n "$NODELIST" ]] && NODE_ARG+=(--nodelist="$NODELIST")
[[ -n "$EXCLUDE" ]] && NODE_ARG+=(--exclude="$EXCLUDE")
ARRAY_ID=$(sbatch --parsable \
  --partition="$PARTITION" "${NODE_ARG[@]}" --time="$TIME" \
  --array=0-$((SHARDS - 1)) --job-name="${RUN_ID}" \
  scripts/sdedit_sweep.sbatch "$ENV_FILE")
echo "submitted GPU array: job ${ARRAY_ID} (${SHARDS} shard(s), ${PARTITION}${NODELIST:+/$NODELIST}${EXCLUDE:+ excl $EXCLUDE})"

SUM_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$SUM_TIME" \
  --dependency=afterok:"$ARRAY_ID" --job-name="${RUN_ID}_sum" \
  scripts/sdedit_summarize.sbatch "$ENV_FILE")
echo "submitted CPU summary: job ${SUM_ID} (afterok:${ARRAY_ID})"

# 12 h was not enough last time: FastRelax cost varies ~30x across receptors (one shard
# managed 22 structures in 12 h against another shard's 140 in 43 min), so two shards died
# on the wall clock. The scorer resumes per structure, so a longer window costs nothing when
# it is not needed.
ROS_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$ROS_TIME" \
  --array=0-$((SHARDS - 1)) --dependency=afterok:"$ARRAY_ID" --job-name="${RUN_ID}_dg" \
  scripts/sdedit_rosetta.sbatch "$ENV_FILE")
echo "submitted CPU Rosetta dG: job ${ROS_ID} (afterok:${ARRAY_ID}, ${ROS_TIME})"

# afterANY, not afterok: the dG stage scores a capped subset by design, so a slow shard
# timing out is a smaller sample, not a failed measurement. Under afterok one such timeout
# left the figure in DependencyNeverSatisfied with 90% of the rows already on disk. The plot
# job gates on "no jsonl" itself and exits 0, so an empty run still ends cleanly.
ROSFIG_ID=$(sbatch --parsable \
  --partition="$PARTITION" --time="$SUM_TIME" \
  --dependency=afterany:"$ROS_ID" --job-name="${RUN_ID}_dg_fig" \
  scripts/sdedit_rosetta_plot.sbatch "$ENV_FILE")
echo "submitted CPU dG figure:  job ${ROSFIG_ID} (afterany:${ROS_ID})"
echo
echo "results: ${REPO}/evaluation_results/${RUN_ID}"
echo "summary: ${REPO}/evaluation_results/${RUN_ID}/summary  (sdedit_guidance_rmsd.png is the verdict plot)"
