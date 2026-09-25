"""Pin the SDEdit + guidance baseline (pre-build check 0).

The frontier sweep is deferred on the condition that its configuration is FROZEN now rather
than reconstructed later.  This records what the best known pocket14 configuration is,
proves every file it names still exists, and emits the exact launch command for the
confirmation run.

It deliberately does NOT launch anything.  The confirmation is a GPU job, and on this
cluster GPU work is submitted with `sbatch` after checking `myfree` for an available node,
so the command is handed over rather than run.

## What "best known" means here, and where each number came from

  * **pocket14 staging**, not the full-chain LNR set.  Receptor segmentation alone is worth
    22-34 points of ring closure, so a baseline pinned on the uncropped set pins the wrong
    number.
  * **`t_lat` must be below 1.0.**  The frozen-sequence corner closed 1 of 59 at every
    `t_ca`: with the sequence frozen the sampler cannot place the anchor residues a bridge
    needs, so the chemistry is unmeasurable rather than failing.
  * **`t_ca = 0.8`, `t_lat = 0.4`** is the corner the full grid put closure at without
    spending the whole interface.
  * **The flow checkpoint is pinned WITH its AE.**  A flow checkpoint evaluated against an
    autoencoder it was not trained against produces numbers that cannot be compared to any
    other run -- the trap that has invalidated cross-run comparisons here before.

Runs in `.venv`.  CPU only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from script_utils import precheck_config  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"<git failed: {exc}>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = precheck_config.load(args.config)
    p = cfg["task0_pin"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = REPO / p["metadata"] if not Path(p["metadata"]).is_absolute() else Path(p["metadata"])
    ckpt_dir = (REPO / p["flow_ckpt_path"] if not Path(p["flow_ckpt_path"]).is_absolute()
                else Path(p["flow_ckpt_path"]))
    ckpt = ckpt_dir / p["flow_ckpt_name"]

    checks = {
        "metadata": {"path": str(meta), "exists": meta.exists()},
        "flow_ckpt": {"path": str(ckpt), "exists": ckpt.exists()},
    }
    if meta.exists():
        import pandas as pd
        checks["metadata"]["n_targets"] = int(len(pd.read_parquet(meta)))
    if ckpt.exists():
        checks["flow_ckpt"]["bytes"] = int(ckpt.stat().st_size)

    dirty = _git("status", "--porcelain")
    pin = {
        "pinned_at": _git("log", "-1", "--format=%cI"),
        "commit": _git("rev-parse", "HEAD"),
        "commit_subject": _git("log", "-1", "--format=%s"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A pin taken against a dirty tree is not reproducible from the hash alone, so the
        # fact is recorded rather than the tree quietly assumed clean.
        "working_tree_clean": dirty == "",
        "uncommitted_files": [ln[3:] for ln in dirty.splitlines()] if dirty else [],
        "config_file": str(Path(args.config).resolve()),
        "config": p,
        "checks": checks,
    }

    # The exact launch command, with the configuration inline so it does not depend on the
    # YAML still saying the same thing next month.
    cmd = (
        "bash scripts/submit_sdedit_sweep.sh \\\n"
        "  --tag precheck_pin \\\n"
        f"  --metadata {p['metadata']} \\\n"
        f"  --t-ca \"{p['t_ca']}\" --t-lat \"{p['t_lat']}\" \\\n"
        f"  --cyc-types \"{p['cyc_types']}\" --seeds \"{p['seeds']}\" \\\n"
        "  --shards 1 --partition general --exclude gpu28"
    )
    pin["confirmation_command"] = cmd

    (out_dir / "task0_pin.json").write_text(json.dumps(pin, indent=2))

    L = ["# Pre-build check 0 -- pinned SDEdit baseline\n",
         f"- commit: `{pin['commit']}` ({pin['commit_subject']})",
         f"- branch: `{pin['branch']}`",
         f"- working tree clean: **{pin['working_tree_clean']}**"]
    if not pin["working_tree_clean"]:
        L.append(f"  - {len(pin['uncommitted_files'])} uncommitted file(s); this pin is NOT")
        L.append("    reproducible from the hash alone until they are committed.")
    L += ["", "## Configuration\n", "| key | value |", "|---|---|"]
    for k in ("metadata", "flow_ckpt_path", "flow_ckpt_name", "t_ca", "t_lat",
              "cyc_types", "seeds", "nsteps", "n_targets"):
        L.append(f"| `{k}` | `{p.get(k)}` |")

    L += ["", "## File checks\n", "| artefact | exists | detail |", "|---|---|---|"]
    for name, c in checks.items():
        detail = (f"{c.get('n_targets')} targets" if "n_targets" in c
                  else f"{c.get('bytes', 0) / 1e9:.2f} GB" if "bytes" in c else "")
        L.append(f"| `{name}` | {'yes' if c['exists'] else '**NO**'} | {detail} |")
    L.append(f"\n`{checks['metadata']['path']}`\n\n`{checks['flow_ckpt']['path']}`")

    L += ["", "## Confirmation run\n",
          "Not launched from here: GPU work on this cluster is submitted with `sbatch`",
          "after checking `myfree` for an available node. The command is:\n",
          "```bash", cmd, "```\n",
          "This is a CONFIRMATION, not a measurement. It establishes that the pinned",
          "configuration still executes end to end; it is one seed at one grid point and",
          "should not be quoted as a closure rate.\n",
          "`--exclude gpu28` pins away from the a5000s: `MEM_PROFILE` a5000/a5000_min",
          "silently force `self_cond=false`, which would confirm a different configuration",
          "than the one being pinned."]

    (out_dir / "PRECHECK_TASK0_PIN.md").write_text("\n".join(L) + "\n")

    missing = [k for k, c in checks.items() if not c["exists"]]
    print(f"wrote {out_dir}/PRECHECK_TASK0_PIN.md")
    print(f"commit {pin['commit'][:12]} clean={pin['working_tree_clean']}")
    if missing:
        # Not an exception: the pin record is still the deliverable, and a job that exits
        # non-zero here would strand its dependents with nothing to read.
        print(f"WARNING: pinned artefacts MISSING: {missing}")
    print("\nConfirmation command:\n" + cmd)


if __name__ == "__main__":
    main()
