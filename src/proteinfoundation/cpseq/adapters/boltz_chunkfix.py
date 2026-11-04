"""The denoiser chunk-count correction, as a shadow package production can import.

`boltz/model/modules/diffusionv2.py` partitions the diffusion samples into denoiser
chunks with

    sample_ids.chunk(multiplicity % max_parallel_samples + 1)

where it means `ceil(multiplicity / max_parallel_samples)`. Modulo is not division, so
the chunk count is whatever the remainder happens to be. With potentials on, multiplicity
is `3 * diffusion_samples`, and the consequences run in both directions:

    ds=1   ->  3 % 5 + 1 = 4 chunks of a 3-sample batch  -> 3 chunks of ONE, batch size 1
    ds=4   -> 12 % 5 + 1 = 3 chunks of 12                -> survivable by accident
    ds=10  -> 30 % 5 + 1 = 1 chunk  of 30                -> 6x the batch that was asked
    ds=40  -> 120 % 5 + 1 = 1 chunk of 120               -> a batch nobody chose, OOM

So the bug is what pins `diffusion_samples` at its current value: it is an **enabler**
for raising `ds`, not a speed lever in its own right. Measured warm on an A6000 at ds=1
(job 95853, phase 2 vs A2): ~50 s unpatched against 45 s patched, about 10%.

**Why patching cannot change a result.** The initial noise, the per-step noise and the
random augmentation are all drawn for the full multiplicity OUTSIDE the chunk loop, and
the denoiser carries no cross-sample term, so the partition decides only how the work is
batched. What it does change is floating-point association: a different batch size is not
bitwise identical. That is why `Boltz2Config.chunk_fix` is hashed into the prediction key
rather than treated as a free optimisation, and why the equivalence check in
`cpseq_fold_cost_probe.py` phase 2 was run before this module was allowed to exist. That
check passed -- ten folds each way, no mean shift and no spread collapse on
`confidence_score`, `iptm`, `ptm` or `complex_plddt` (p 0.50-0.88).

**The installed package is never written to.** Production array tasks are reading it, and
an in-place edit would change the numerics under a job that is already running. The patch
is a copy, shadowed in front of the real package via `PYTHONPATH`.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# The one line of upstream this changes, and what it becomes. Matched as exact text --
# including the indentation upstream wrote it with -- so that an upstream reformat is a
# loud failure rather than a silently unpatched run.
CHUNK_BUG = "sample_ids.chunk(\n                    multiplicity % max_parallel_samples + 1\n                )"
CHUNK_FIX = "sample_ids.chunk(\n                    max(1, -(-multiplicity // max_parallel_samples))\n                )"

# Bumped whenever this module changes what the shadow contains. It is part of the
# directory name, so an older shadow is never reused under a newer patch definition.
CHUNKFIX_REV = "chunkfix1"

RELPATH = Path("model") / "modules" / "diffusionv2.py"


def locate_installed_boltz(python_exec: str | Path) -> Path:
    """Where the boltz interpreter's own package lives.

    Asked of the interpreter rather than guessed from a site-packages path, because the
    boltz env is a separate venv from this one and its layout is not ours to assume.
    """
    out = subprocess.run(
        [str(python_exec), "-c", "import boltz,os;print(os.path.dirname(boltz.__file__))"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not out:
        raise RuntimeError(f"{python_exec} did not report a boltz package location")
    return Path(out)


def _patched_text(src: Path) -> tuple[str, str, str]:
    """The corrected file text, with the sha before and after. Raises if it cannot apply."""
    before = src.read_text()
    if CHUNK_BUG not in before:
        if CHUNK_FIX in before:
            raise RuntimeError(
                f"{src} already contains the corrected chunk count. Upstream fixed this; "
                f"drop `chunk_fix` rather than shadowing a patch over a patched file."
            )
        raise RuntimeError(
            f"the chunk-count expression is not where this expects it in {src}. Upstream "
            f"changed -- re-read it before trusting any prediction made with chunk_fix."
        )
    after = before.replace(CHUNK_BUG, CHUNK_FIX)
    if after.count("max(1, -(-multiplicity // max_parallel_samples))") != 1:
        raise RuntimeError("chunk patch did not apply exactly once")
    return (
        after,
        hashlib.sha256(before.encode()).hexdigest()[:16],
        hashlib.sha256(after.encode()).hexdigest()[:16],
    )


def ensure_patched_boltz(
    *, python_exec: str | Path, cache_root: str | Path, rebuild: bool = False
) -> dict:
    """Build (or reuse) the shadow and return the root to put on `PYTHONPATH`.

    Idempotent and safe to call from many array tasks at once. The copy is ~47 MB, so a
    rebuild per task on a shared filesystem would be real I/O for nothing; a shadow whose
    stamp matches is reused as is.

    **Built through a temp directory and `os.replace`d into place.** Two tasks copying
    into the same destination at the same time would interleave, and a half-copied
    package on `PYTHONPATH` is an `ImportError` in one task and a mystery in the other.
    The rename is atomic within a filesystem, so a concurrent task either sees the old
    shadow or the new one, never a partial one. The loser of the race copied for nothing
    and discards its own build, which costs a few seconds and keeps the result correct.
    """
    installed = locate_installed_boltz(python_exec)
    src = installed / RELPATH
    text, sha_before, sha_after = _patched_text(src)

    # The source sha is in the directory name: if the installed boltz is upgraded, the
    # shadow built against the old one is not silently reused in front of the new one.
    root = Path(cache_root) / f"boltz_{CHUNKFIX_REV}_{sha_before}"
    stamp_path = root / "chunkfix.json"
    stamp = {
        "chunkfix_rev": CHUNKFIX_REV,
        "installed_boltz": str(installed),
        "patched_file": str(RELPATH),
        "sha_before": sha_before,
        "sha_after": sha_after,
    }

    if not rebuild and stamp_path.exists():
        try:
            have = json.loads(stamp_path.read_text())
        except Exception:  # noqa: BLE001
            have = None
        # Trust the stamp only if the file on disk still hashes to what it claims. A
        # truncated copy from a job that was preempted mid-build would otherwise be
        # reused forever.
        if have == stamp:
            live = root / "boltz" / RELPATH
            if live.exists():
                got = hashlib.sha256(live.read_text().encode()).hexdigest()[:16]
                if got == sha_after:
                    return {**stamp, "root": str(root), "built": False}

    root.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        shutil.copytree(installed, tmp / "boltz")
        (tmp / "boltz" / RELPATH).write_text(text)
        (tmp / "chunkfix.json").write_text(json.dumps(stamp, indent=2))
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        os.replace(tmp, root)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {**stamp, "root": str(root), "built": True}


def shadow_env(env: dict | None, root: str | Path) -> dict:
    """`env` with the shadow first on `PYTHONPATH`. First, so it wins over the install."""
    out = dict(os.environ if env is None else env)
    prior = out.get("PYTHONPATH")
    out["PYTHONPATH"] = f"{root}:{prior}" if prior else str(root)
    return out
