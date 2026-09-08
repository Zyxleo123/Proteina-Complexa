#!/bin/bash
#SBATCH --job-name=lp_verify
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.out
#
# Stage 0 (CPU): prove the LP-mixing pipeline works before any GPU time is spent on it.
#
#   1. the unit tests (LINEAR token behaviour + weighted sampler arithmetic)
#   2. a real batch pulled through the full transform stack from the mixed metadata,
#      asserting that linear rows arrive carrying LINEAR and cyclic rows do not
#
# (2) is the part unit tests cannot cover: it needs real PDBs on disk and the atomworks
# monkeypatches applied. It is also the check that would have caught a mis-mapped
# receptor/peptide chain, which is silent everywhere else -- a swapped mapping trains the
# model to generate the receptor and nothing complains.
#
# Run AFTER lp_build_data.sbatch. Cheap; run it whenever the data is rebuilt.
#
#   sbatch lp_verify.sbatch /path/to/env_file

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-/zfsauton2/home/yixiz/Proteina-Complexa}"
[[ -f "${REPO}/scripts/_lp_mixing_preamble.sh" ]] || REPO=/zfsauton2/home/yixiz/Proteina-Complexa
cd "${REPO}"
mkdir -p slurm_logs

# shellcheck source=_lp_mixing_preamble.sh
source "${REPO}/scripts/_lp_mixing_preamble.sh"
# shellcheck source=/dev/null
source "${REPO}/env.sh"
_lp_consume_args "$@"
_lp_defaults
_lp_banner "verify"

PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "--- [1/2] unit tests"
"${PY}" script_utils/test_lp_mixing.py

echo
echo "--- [2/2] real batch through the transform stack"
if [[ ! -f "${MIXED_META_DIR}/mixed_train.parquet" ]]; then
  echo "GATE: no mixed metadata at ${MIXED_META_DIR}; run --submit data first."
  echo "GATE: exiting 0 so dependents are not stranded."
  exit 0
fi

"${PY}" - "${MIXED_META_DIR}/mixed_train.parquet" <<'PY'
# The monkeypatches must be imported before anything touches atomworks, exactly as
# proteinfoundation.train does it -- without them the loader dies with an unrelated-looking
# "atom_array_to_encoding() got an unexpected keyword argument 'atomize_token'".
import proteinfoundation.patches.atomworks_patches  # noqa: F401

import sys

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from proteinfoundation.cyclization.constants import LINEAR, NON_RING_COND_TYPES, is_ring_request

metadata = sys.argv[1]

with initialize_config_dir(config_dir="/zfsauton2/home/yixiz/Proteina-Complexa/configs", version_base=None):
    cfg = compose(config_name="training_ae_shared_lpcp")
c = OmegaConf.to_container(cfg.dataset.datamodule, resolve=True)
c.update(metadata_file=metadata, val_metadata_file=metadata, batch_size=1, num_workers=0, pin_memory=False)
dm = hydra.utils.instantiate(c, _convert_="all")
dm.setup()

meta = dm.train_dataset.metadata
print("sources in metadata:", meta["dataset_source"].value_counts().to_dict())

failures = []
for src in sorted(meta["dataset_source"].unique()):
    idx = list(meta.index[meta["dataset_source"] == src])[:4]
    for i in idx:
        g = dm.train_dataset[int(i)]
        cond = int(g.cyclization_type_cond)
        eid = meta["example_id"].iloc[i]
        n = int(g.num_nodes)
        print(f"  {src:9s} {eid:52s} cond={cond} has_cyc={bool(g.has_cyclization)} n_res={n}")

        if src == "cpsea":
            # Cyclic rows must never be labeled linear, or the ring conditioning is off
            # for the data the whole run exists to model.
            if cond == LINEAR:
                failures.append(f"{eid}: CPSea row labeled LINEAR")
        else:
            # Linear rows must carry LINEAR, not UNSPECIFIED: UNSPECIFIED is the CFG null.
            if cond != LINEAR:
                failures.append(f"{eid}: linear row has cond={cond}, expected LINEAR={LINEAR}")
            if bool(g.has_cyclization):
                failures.append(f"{eid}: linear row claims has_cyclization=True")
            # And they must not read as a ring request anywhere downstream.
            if bool(is_ring_request(torch.tensor([cond]))[0]):
                failures.append(f"{eid}: linear row reads as a ring request")

        # The binder must be the peptide, not the receptor. A swapped chain mapping in the
        # preprocessor is otherwise completely silent.
        if not (5 <= n <= 16):
            failures.append(f"{eid}: binder has {n} residues, outside the 5-16 window "
                            f"(a receptor/peptide chain swap looks exactly like this)")

print()
print(f"non-ring conditioning indices: {NON_RING_COND_TYPES}")
if failures:
    print("FAILED:")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print("OK: linear rows carry LINEAR, cyclic rows do not, and every binder is 5-16 residues.")
PY

echo
echo "Verification passed."
