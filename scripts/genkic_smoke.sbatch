#!/bin/bash
#SBATCH --job-name=genkic_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# The ONE preflight this arm gets: a single real complex, taken end to end. Four stages,
# cheapest first, so a failure is localized rather than reported as "the search didn't work":
#
#   1  offline bookkeeping        cp_topology windows / anchor costs / atom resolution
#   2  library regression         rosetta_energy._declare_cyclic_bond, both fixed defects
#   3  ring declaration           all 3 chemistries declared on the real complex, ref2015 finite
#   4  end-to-end search          one target, all 3 chemistries, search -> relax -> contract row
#
# Stage 4 is the real test: it is the production driver at reduced effort, so it exercises
# GenKIC, the bridge anneal, the paired LP baseline, InterfaceAnalyzerMover and the contract.
# Reduced effort means this is NOT a result -- SMOKE=1 is printed in the banner and the rows
# land in their own directory.
#
# Do NOT sbatch directly -- use scripts/submit_genkic.sh. Config in the positional env file.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p slurm_logs

# A batch script cannot locate files relative to itself: Slurm copies it to
# /var/spool/slurm/job<id>/slurm_script, so $(dirname "$0") is the spool directory and
# sourcing the preamble that way silently sources nothing. Use SLURM_SUBMIT_DIR with an
# absolute fallback.
PREAMBLE="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa/.claude/worktrees/genkic}/scripts/_genkic_preamble.sh"
[[ -f "$PREAMBLE" ]] || { echo "FATAL: preamble not found: $PREAMBLE"; exit 1; }
# shellcheck source=/dev/null
source "$PREAMBLE"

_genkic_consume_args "$@"
_genkic_defaults
_genkic_venv
SMOKE=1
_genkic_banner "smoke (one real complex, end to end)"

SMOKE_PDB="${SMOKE_PDB:-${REPO}/CPSea_data/lnr_staged/pdbs/${SMOKE_EXAMPLE:-LNR_2qos_C_A}.pdb}"
SMOKE_CHAIN="${SMOKE_CHAIN:-B}"
[[ -f "$SMOKE_PDB" ]] || { echo "FATAL: smoke complex not found: $SMOKE_PDB"; exit 1; }
echo "smoke complex: $SMOKE_PDB (binder chain $SMOKE_CHAIN)"

echo
echo "---- 1/4  offline bookkeeping ------------------------------------------------"
python script_utils/test_cp_topology.py

echo
echo "---- 2/4  rosetta_energy regression (both fixed defects) ---------------------"
python script_utils/test_rosetta_energy_isopeptide.py \
  --pdb "$SMOKE_PDB" --binder-chain "$SMOKE_CHAIN"

echo
echo "---- 3/4  ring declaration on the real complex -------------------------------"
python script_utils/test_cp_topology.py \
  --pdb "$SMOKE_PDB" --binder-chain "$SMOKE_CHAIN"

echo
echo "---- 4/4  end-to-end search on one target ------------------------------------"
SMOKE_OUT="${OUT_DIR}/smoke"
mkdir -p "$SMOKE_OUT"
# Reduced effort: enough to close and grade, not enough to be a result.
srun python scripts/genkic_cyclize.py \
  --metadata "$METADATA" \
  --examples "${SMOKE_EXAMPLE:-LNR_2qos_C_A}" \
  --out "$SMOKE_OUT/rows.jsonl" \
  --pdb-dir "$SMOKE_OUT/pdbs" \
  --chemistries ${CHEMISTRIES} \
  --budgets ${BUDGETS} \
  --attempts-per-call "${SMOKE_ATTEMPTS:-500}" \
  --calls "${SMOKE_CALLS:-2}" \
  --min-solutions "${MIN_SOLUTIONS}" \
  --max-released "${MAX_RELEASED}" \
  --shortlist "${SMOKE_SHORTLIST:-2}" \
  --dump-top 1 \
  --relax-repeats "${RELAX_REPEATS}" \
  --cp-repeats "${SMOKE_CP_REPEATS:-1}" \
  --lp-repeats "${LP_REPEATS}" \
  --perturb-deg "${PERTURB_DEG}" \
  --seed "${SEED}"

echo
echo "---- smoke summary -----------------------------------------------------------"
python - "$SMOKE_OUT/rows.jsonl" <<'PY'
import collections, json, sys

path = sys.argv[1]
rows = []
with open(path) as fh:
    for lineno, line in enumerate(fh, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            raise SystemExit(f"FATAL: corrupt row {path}:{lineno}")

status = collections.Counter(r.get("status") for r in rows)
print(f"{len(rows)} rows: {dict(status)}")

post = [r for r in rows if r.get("stage") == "post_relax"]
print(f"{len(post)} relaxed+scored candidates")
for r in post:
    print(f"  {r['spec_label']:34s} b{r.get('mutation_budget')} "
          f"bond={r.get('bond_length_A')}A ok={r.get('bond_ok')} "
          f"ret={r.get('contact_retention_heavy45')} "
          f"rmsd={r.get('ca_rmsd_A')}/{r.get('interface_ca_rmsd_A')} "
          f"ddg={r.get('ddg_interface_reu')} "
          f"accepted={r.get('accepted')} {r.get('fail_reasons')}")

# The smoke's job is to prove the PATH works, not to produce an accepted candidate at
# reduced effort. So: did every stage produce its output, and did nothing crash?
infeasible = [r for r in rows if r.get("status") == "infeasible_budget"]
if infeasible:
    print(f"\n{len(infeasible)} chemistry/budget combinations labelled chemically infeasible:")
    for r in infeasible:
        print(f"  {r['chemistry']:10s} budget {r['mutation_budget']} "
              f"needs {r['min_anchor_mutations']} -- {r['reason']}")

broken = [r for r in rows if r.get("status") in (
    "load_failed", "prepare_failed", "relax_failed", "score_failed", "lp_baseline_failed")]
if broken:
    print(f"\nSMOKE FAILED: {len(broken)} row(s) report a crash:")
    for r in broken[:10]:
        print(f"  {r.get('status')}: {r.get('error')}")
    raise SystemExit(1)
if not post:
    print("\nSMOKE FAILED: no candidate reached relax+scoring, so the full path is unproven.")
    raise SystemExit(1)

# Per-chemistry, not just overall. A bug that kills exactly one engine leaves the others
# producing rows, so an aggregate "did anything work?" check passes while the primary
# chemistry silently yields nothing -- which is precisely what a PDBInfo defect did here,
# killing every mainchain relax while the side-chain chemistries sailed through.
closed_by_chem = collections.Counter(
    r.get("chemistry") for r in rows if r.get("stage") == "pre_relax")
post_by_chem = collections.Counter(r.get("chemistry") for r in post)
infeasible_chems = {r.get("chemistry") for r in infeasible}
print("\nper-chemistry path coverage:")
missing = []
for chem in sorted(closed_by_chem):
    n_closed, n_post = closed_by_chem[chem], post_by_chem.get(chem, 0)
    print(f"  {chem:10s} closed={n_closed:4d}  relaxed+scored={n_post:3d}")
    if n_closed and not n_post:
        missing.append(chem)
for chem in sorted(infeasible_chems - set(closed_by_chem)):
    print(f"  {chem:10s} no feasible budget in this smoke (labelled infeasible, not failed)")
if missing:
    print(f"\nSMOKE FAILED: {missing} closed candidates but got none through relax+scoring.")
    print("  A chemistry that closes and then never scores is a broken stage, not a hard case.")
    raise SystemExit(1)
print("\nSMOKE OK: search -> relax -> interface scoring -> contract row all ran, "
      "for every chemistry that closed.")
PY

echo
echo "smoke complete. rows: $SMOKE_OUT/rows.jsonl"
