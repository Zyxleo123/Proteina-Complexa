#!/usr/bin/env python3
"""Apply the LP-milestone repairs to the staged upstream arm repositories.

WHY A PATCH SCRIPT AND NOT A FORK. The arms are third-party research code that we need to
run, not own. Keeping our changes as a small set of named, asserted edits means the diff
against upstream is readable, reviewable and re-appliable, and a new upstream commit shows
up as a loud failure here rather than as a silent behaviour change in training.

WHY ASSERTED STRING EDITS AND NOT A UNIFIED DIFF. ``git apply`` fails on context drift in
ways that are tedious to read, and a hand-written diff is easy to get subtly wrong. Each
edit here asserts its target text appears exactly the expected number of times before
touching anything, so the script is its own drift detector: if upstream moves, the edit
names itself and nothing is written. Every edit is idempotent -- re-running on an
already-patched tree reports ``already applied`` instead of failing.

Nothing is written unless EVERY edit for that arm resolves, so a repository is never left
half-patched.

THE POTFLOW REPAIRS (seven; the plan listed six and missed the import crash).

1.  ``names`` import crash. ``pep_dataloader.py`` read a hardcoded absolute path in the
    original author's home directory at module import time with no ``try``/``except``, so
    importing it raised ``FileNotFoundError`` on any other machine. Now configurable via
    ``POTFLOW_NAMES_FILE`` and tolerant of absence.
2.  Inverted blocklist message. ``if task['id'] in names: raise ValueError(f'... not in
    names')`` said the opposite of what it tested. The ``# testset`` comment means
    ``names.txt`` is a test-set *blocklist*, so an empty file disables the exclusion
    entirely -- which is correct for us only because membership comes from the per-split
    ``structure_dir`` the exporter writes.
3.  Secondary structure without a DSSP binary. ``mkdssp`` is absent on this cluster, so
    upstream's ``DSSP(...)`` call failed for *every* example, its bare ``except:`` set
    ``pep_dssp = None``, and the fallback labelled every generated residue ``'H'``. The
    secondary-structure prior -- POTFlow's distinguishing feature against PepFlow -- would
    have trained on a constant with no log line. Replaced with biotite's P-SEA annotation,
    which needs no binary and returns a real distribution (measured on our LP export: 67%
    coil, 26% helix, 7% strand, varying with peptide length). ``'-'`` is a legitimate
    ``SS2HEC`` key mapping to coil; ``'H'`` never was the right fallback.
4.  The bare ``except:`` itself, which also swallowed ``KeyboardInterrupt`` and
    ``SystemExit``. Now catches ``Exception`` and reports, with a count.
5.  ``--debug`` unreachable. ``add_argument('--debug', action='store_true', default=True)``
    is ``True`` whether or not the flag is passed, so the branch creating ``log_dir`` and
    ``ckpt_dir`` never ran and training died with ``NameError: ckpt_dir`` at the first
    ``val_freq`` checkpoint -- after that many iterations of real training.
6.  ``validate()`` defined and never called. Its ``def`` was the only occurrence of the
    name in the file, so there was no validation at all and ``'avg_val_loss'`` was
    commented out of the saved payload, making "best checkpoint" uncomputable.
7.  No final checkpoint. Saves happened only on ``it % val_freq == 0``, so unless
    ``max_iters`` landed exactly on a multiple the final state was lost. Now writes
    ``latest.pt``, ``best.pt`` and ``final.pt`` alongside the numbered ones.

The interpolant flags (``sample_trans_prior``, ``ot_sampler``) are NOT patched: they are
configuration, and the LP experiment config sets them. Upstream ships them ``False`` in
``learn_angle.yaml`` while ``test_angle.yaml`` infers with both ``True``, so enabling them
for training aligns train with inference rather than being an arbitrary choice.

Run:
    .venv/bin/python script_utils/lpgen_patch_upstream.py --upstream-dir $ZFS/lpgen/upstream
    .venv/bin/python script_utils/lpgen_patch_upstream.py --upstream-dir ... --check
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass
class Edit:
    """One asserted, idempotent source edit."""

    arm: str
    path: str
    name: str
    why: str
    old: str
    new: str
    #: Expected occurrences of ``old``. Anything else is drift and refuses.
    count: int = 1
    #: Text proving the edit is already applied, when ``old`` is absent.
    marker: str = field(default="")

    def __post_init__(self):
        if not self.marker:
            self.marker = self.new.strip().splitlines()[0][:60]


# --------------------------------------------------------------------------- POTFlow
_POTFLOW_HELPERS = '''
# ---------------------------------------------------------------- LP milestone helpers
# Secondary structure WITHOUT an external DSSP binary.
#
# mkdssp is not installed on this cluster, so upstream's DSSP call failed for every single
# example, its bare `except:` set pep_dssp = None, and the fallback below labelled every
# generated residue 'H'. That trains the secondary-structure prior -- POTFlow's
# distinguishing feature against PepFlow -- on a constant, with no log line.
#
# biotite's annotate_sse (P-SEA) needs no binary. Measured on the LP export: 67% coil,
# 26% helix, 7% strand, varying sensibly with peptide length. 'H'/'E'/'-' are all
# legitimate SS2HEC keys; '-' maps to coil, which is what an unknown residue should be.
_LPGEN_PSEA_MAP = {'a': 'H', 'b': 'E', 'c': '-'}
_LPGEN_SS_STATS = {'dssp_failures': 0, 'psea_failures': 0}


def _lpgen_ss_fallback(example_id, exc):
    """Report a DSSP failure. Loud and counted, never silent."""
    _LPGEN_SS_STATS['dssp_failures'] += 1
    n = _LPGEN_SS_STATS['dssp_failures']
    if n <= 5 or n % 2000 == 0:
        print(f'[potflow] DSSP unavailable for {example_id} '
              f'({type(exc).__name__}: {exc}); using P-SEA annotation [n={n}]', flush=True)


def _lpgen_sse_labels(pep_path, n_expected):
    """Per-residue SS2HEC labels for the peptide: 'H' helix, 'E' strand, '-' coil."""
    import biotite.structure as _struc
    from biotite.structure.io.pdb import PDBFile as _PDBFile
    labels = []
    try:
        arr = _PDBFile.read(str(pep_path)).get_structure(model=1)
        labels = [_LPGEN_PSEA_MAP.get(str(s), '-') for s in _struc.annotate_sse(arr)]
    except Exception as exc:
        _LPGEN_SS_STATS['psea_failures'] += 1
        print(f'[potflow] WARNING P-SEA failed for {pep_path}: '
              f'{type(exc).__name__}: {exc}; using coil', flush=True)
    # Pad with coil rather than by repeating the last label: repeating fabricates
    # structure, and these are residues whose conformation we could not annotate.
    if len(labels) < n_expected:
        labels = labels + ['-'] * (n_expected - len(labels))
    return labels[:n_expected]


'''

POTFLOW_EDITS = [
    Edit(
        arm="potflow",
        path="models_con/pep_dataloader.py",
        name="importable-blocklist",
        why="module-level open() of a hardcoded path in the author's home dir, read at "
            "import time with no try/except, so the module could not be imported at all",
        old="""# testset
names = []
with open('/home/qianhao/potflowww/Data/Fixed Data/names.txt','r') as f:
    for line in f:
        names.append(line.strip())
""",
        new="""# testset blocklist.
#
# Upstream hardcoded '/home/qianhao/potflowww/Data/Fixed Data/names.txt' and read it at
# MODULE IMPORT time with no try/except, so importing this module raised FileNotFoundError
# on every machine that is not the original author's. The path is now configurable and its
# absence is tolerated.
#
# An EMPTY blocklist is correct for the LP milestone: dataset membership comes from a
# per-split structure_dir written by lpgen_export_native.py, so nothing should be filtered
# a second time here. Note this file is a BLOCKLIST (see the guard in
# preprocess_structure), so an empty one means "exclude nothing", not "include nothing".
names = []
_lpgen_names_path = os.environ.get('POTFLOW_NAMES_FILE', '')
if _lpgen_names_path and os.path.exists(_lpgen_names_path):
    with open(_lpgen_names_path, 'r') as f:
        for line in f:
            if line.strip():
                names.append(line.strip())
    print(f'[potflow] blocklist: {len(names)} ids from {_lpgen_names_path}', flush=True)
elif _lpgen_names_path:
    print(f'[potflow] blocklist: POTFLOW_NAMES_FILE={_lpgen_names_path!r} does not exist; '
          f'excluding nothing', flush=True)
""",
        marker="# testset blocklist.",
    ),
    Edit(
        arm="potflow",
        path="models_con/pep_dataloader.py",
        name="ss-helpers",
        why="inject a DSSP-free secondary-structure annotator (mkdssp is absent here)",
        old="\ndef preprocess_structure(task):\n",
        new=_POTFLOW_HELPERS + "def preprocess_structure(task):\n",
        marker="_LPGEN_PSEA_MAP",
    ),
    Edit(
        arm="potflow",
        path="models_con/pep_dataloader.py",
        name="blocklist-message",
        why="the message said the opposite of the condition it reported on",
        old="""            raise ValueError(f'{task["id"]} not in names')""",
        new="""            # Upstream's message said 'not in names' while testing the opposite.
            raise ValueError(f'{task["id"]} is in the blocklist')""",
        marker="is in the blocklist",
    ),
    Edit(
        arm="potflow",
        path="models_con/pep_dataloader.py",
        name="dssp-bare-except",
        why="a bare except: swallowed everything including KeyboardInterrupt, and the "
            "failure was never reported",
        old="""        try:
            pep_dssp = DSSP(pep, pep_path, dssp='/usr/bin/mkdssp')
        # poc_dssp = DSSP(poc, poc_path, dssp='/usr/bin/mkdssp')
            # if len(list(pep.get_residues())) != len(pep_dssp.keys()):
            #     print(data['id'])
        except:
            pep_dssp = None""",
        new="""        try:
            # Binary is configurable; upstream hardcoded /usr/bin/mkdssp, which does not
            # exist on this cluster.
            pep_dssp = DSSP(pep, pep_path,
                            dssp=os.environ.get('POTFLOW_DSSP', 'mkdssp'))
        except Exception as _lpgen_exc:
            # Upstream used a BARE `except:` here, which also swallows KeyboardInterrupt
            # and SystemExit, and then silently labelled the whole peptide as helix.
            pep_dssp = None
            _lpgen_ss_fallback(data['id'], _lpgen_exc)""",
        marker="POTFLOW_DSSP",
    ),
    Edit(
        arm="potflow",
        path="models_con/pep_dataloader.py",
        name="no-all-helix-fallback",
        why="every generated residue was labelled 'H', so with mkdssp absent the SS prior "
            "trained on a constant for 100% of examples",
        old="""        else:
            ss = ss + ['H']*data['generate_mask'].sum()""",
        new="""        else:
            # Upstream: ss = ss + ['H']*generate_mask.sum() -- all-helix, for every
            # example, whenever DSSP was unavailable. P-SEA gives real labels instead.
            ss = ss + _lpgen_sse_labels(pep_path, int(data['generate_mask'].sum()))""",
        # NOT "_lpgen_sse_labels(pep_path": that substring also matches the helper's own
        # `def _lpgen_sse_labels(pep_path, n_expected)` inserted by the ss-helpers edit, so
        # this edit would report already-applied the moment the helpers landed.
        marker="ss = ss + _lpgen_sse_labels(",
    ),
    Edit(
        arm="potflow",
        path="train.py",
        name="debug-default",
        why="store_true WITH default=True is True whether or not the flag is passed, so "
            "log_dir/ckpt_dir were never created and training died with NameError",
        old="""    parser.add_argument('--debug', action='store_true', default=True)""",
        new="""    # Upstream had `action='store_true', default=True`, which is True whether or not
    # the flag is passed -- there was no CLI route to False. The else-branch that assigns
    # log_dir and creates ckpt_dir therefore never ran, while the training loop does
    # os.path.join(ckpt_dir, ...) every val_freq iterations, so an ordinary run died with
    # NameError: ckpt_dir after val_freq iterations of real training.
    parser.add_argument('--debug', action='store_true', default=False)""",
        marker="action='store_true', default=False)",
    ),
    Edit(
        arm="potflow",
        path="train.py",
        name="validate-and-checkpoints",
        why="validate() was defined and never called; no best/latest/final checkpoints",
        old="""    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    # 'avg_val_loss': avg_val_loss,
                }, ckpt_path)""",
        new="""    best_val_loss = float('inf')

    def _lpgen_save(tag, it, avg_val_loss=None):
        path = os.path.join(ckpt_dir, f'{tag}.pt')
        torch.save({
            'config': config,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'iteration': it,
            'avg_val_loss': avg_val_loss,
        }, path)
        return path

    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0:
                # Upstream DEFINED validate() and never called it -- its `def` was the only
                # occurrence of the name in this file, so val_freq gated checkpointing only,
                # there was no validation at all, and 'avg_val_loss' was commented out of
                # the payload, which made "best checkpoint" uncomputable.
                avg_val_loss = validate(it)
                _lpgen_save('%d' % it, it, avg_val_loss)
                _lpgen_save('latest', it, avg_val_loss)
                if avg_val_loss is not None and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logger.info(f'[best] iter {it} avg_val_loss {avg_val_loss:.6f}')
                    _lpgen_save('best', it, avg_val_loss)
        # Upstream saved only on `it % val_freq == 0`, so unless max_iters landed exactly
        # on a multiple of val_freq the final state was lost.
        _lpgen_save('final', config.train.max_iters, best_val_loss)""",
        marker="_lpgen_save",
    ),
]

# --------------------------------------------------------------------- DiffPepBuilder
# run_docking.py imports the postprocessing stack at MODULE level, so `pdbfixer`, `openmm`
# AND `pyrosetta` are needed merely to import the docking entry point -- even with
# `postprocess.run_postprocess=False`, which is how we run it. Measured in job 84127:
# preprocessing and the no-rewrite verify both completed, then the sampler died on
# `ModuleNotFoundError: No module named 'pdbfixer'` before a single pose was generated.
#
# The import is only ever used inside `if self._post_conf.run_postprocess:`, so moving it
# there is behaviour-preserving: with postprocessing ON it raises exactly the same error at
# exactly the same point, and with it OFF the module simply never loads.
#
# We decline those three dependencies on purpose rather than as a convenience. Two of the
# pinned versions do not exist as wheels at all (openmm's earliest published wheel is 8.1.1
# against a pin of 8.0.0; pdbfixer publishes only 1.12.0 against a pin of 1.9), and scoring
# belongs in the one evaluator every arm shares, on the CPU partition -- not inside the GPU
# sampling job.
_DPB_EDITS = [
    Edit(
        arm="diffpepbuilder",
        path="experiments/run_docking.py",
        name="lazy-postprocess-import",
        why="module-level import of the postprocess stack needs pdbfixer+openmm+pyrosetta "
            "just to import the sampler, even when postprocessing is disabled",
        old="from analysis.postprocess import Postprocess\n",
        new="# LPGEN: import moved into the run_postprocess branch below -- this module\n"
            "# pulls in pdbfixer, openmm and pyrosetta at import time, and we run with\n"
            "# postprocess.run_postprocess=False and score in the shared CPU evaluator.\n",
        marker="# LPGEN: import moved into the run_postprocess branch",
    ),
    # A SECOND path reaches the same stack, found only by running (job 84138):
    #   run_docking.py:35 -> experiments.train:42 -> analysis.metrics:18 -> postprocess
    # train.py genuinely needs `metrics` (for ALL_METRICS and protein_metrics), but never
    # needs Postprocess, which metrics.py uses only inside postprocess_metric(). So the same
    # deferral applies. These two are the ONLY module-level importers of analysis.postprocess
    # in the tree, so with both patched the stack is unreachable unless postprocessing is
    # actually requested.
    Edit(
        arm="diffpepbuilder",
        path="analysis/metrics.py",
        name="lazy-postprocess-import-metrics",
        why="train.py imports analysis.metrics for ALL_METRICS/protein_metrics, which drags "
            "in the whole pdbfixer+openmm+pyrosetta postprocess stack at import time",
        old="from analysis.postprocess import Postprocess\n",
        new="# LPGEN: import moved into postprocess_metric() -- importing this module must\n"
            "# not require pdbfixer, openmm and pyrosetta, because experiments/train.py\n"
            "# imports analysis.metrics purely for ALL_METRICS and protein_metrics.\n",
        marker="# LPGEN: import moved into postprocess_metric()",
    ),
    Edit(
        arm="diffpepbuilder",
        path="analysis/metrics.py",
        name="lazy-postprocess-usage-metrics",
        why="the deferred import has to happen inside the only function that uses it",
        old='    print(f"Running postprocessing protocol on {pdb_file}")\n',
        new="    from analysis.postprocess import Postprocess  # LPGEN: deferred\n"
            '    print(f"Running postprocessing protocol on {pdb_file}")\n',
        marker="from analysis.postprocess import Postprocess  # LPGEN: deferred",
    ),
    # Third and LAST hop on the sampling path (found in the same way, by running):
    #   analysis.metrics:21 -> analysis.postprocess_utils:24 -> pdbfixer (+ openmm.app:26)
    # This one closes the chain. Exactly four files in the tree import pdbfixer/openmm/
    # pyrosetta at module level -- postprocess_utils, postprocess, amber_minimize, cleanup --
    # and all four are on the postprocessing side. amber_minimize and cleanup are reachable
    # only through postprocess, postprocess is deferred in both of its importers, and
    # postprocess_utils is reached only through postprocess and this line. So after this
    # edit nothing the sampler imports pulls in the heavy stack.
    #
    # Why not simply install them: on python 3.9 the pair is UNSATISFIABLE from PyPI.
    # pdbfixer publishes only 1.12.0, source-only, which requires openmm>=8.2 -- and
    # openmm's last release with cp39 wheels is 8.1.1. Measured against the PyPI index
    # 2026-10-14. Building openmm from source is not a reasonable step in an env job.
    Edit(
        arm="diffpepbuilder",
        path="analysis/metrics.py",
        name="lazy-postprocess-utils-import",
        why="postprocess_utils imports pdbfixer and openmm.app at module level, and "
            "analysis.metrics needs only summarize_statistics out of it",
        old="from analysis.postprocess_utils import summarize_statistics\n",
        new="# LPGEN: import moved into postprocess_metric_parallel() -- postprocess_utils\n"
            "# imports pdbfixer and openmm.app at module level, which the sampling path\n"
            "# must not require.\n",
        marker="# LPGEN: import moved into postprocess_metric_parallel()",
    ),
    Edit(
        arm="diffpepbuilder",
        path="analysis/metrics.py",
        name="lazy-postprocess-utils-usage",
        why="the deferred import has to happen in the only function that uses it",
        old="    df = summarize_statistics(list(score_files))\n",
        new="    from analysis.postprocess_utils import summarize_statistics  # LPGEN: deferred\n"
            "    df = summarize_statistics(list(score_files))\n",
        marker="from analysis.postprocess_utils import summarize_statistics  # LPGEN: deferred",
    ),
    Edit(
        arm="diffpepbuilder",
        path="experiments/run_docking.py",
        name="lazy-postprocess-usage",
        why="the deferred import has to happen where Postprocess is actually used",
        old="                if self._post_conf.run_postprocess:\n",
        new="                if self._post_conf.run_postprocess:\n"
            "                    from analysis.postprocess import Postprocess  # LPGEN: deferred\n",
        marker="from analysis.postprocess import Postprocess  # LPGEN: deferred",
    ),
]

ALL_EDITS = POTFLOW_EDITS + _DPB_EDITS

ARMS = sorted({e.arm for e in ALL_EDITS})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--upstream-dir", required=True,
                   help="Directory holding the staged arm repositories, one per arm.")
    p.add_argument("--arms", default=",".join(ARMS),
                   help=f"Comma-separated arms to patch. Available: {','.join(ARMS)}")
    p.add_argument("--check", action="store_true",
                   help="Report status without writing. Exit 1 if anything is unapplied.")
    p.add_argument("--summary-json", default="",
                   help="Write the per-edit outcome here.")
    return p.parse_args()


def plan_arm(root: Path, edits: list) -> tuple:
    """Resolve every edit for one arm against the files on disk, writing nothing.

    Returns ``(results, new_contents)``. Raises nothing: a problem is reported in the
    result so the caller can refuse the whole arm rather than apply half of it.
    """
    results = []
    contents: dict = {}
    for e in edits:
        f = root / e.path
        if f not in contents:
            contents[f] = f.read_text() if f.exists() else None
        text = contents[f]
        if text is None:
            results.append({"edit": e.name, "status": "missing_file", "detail": str(f)})
            continue
        n = text.count(e.old)
        # MARKER FIRST, not `n == 0 and marker`. An INSERTION edit keeps its own anchor in
        # the result (``\ndef preprocess_structure`` survives having a block inserted
        # before it), so testing the anchor first re-applies it on every run and duplicates
        # the inserted code. The marker is the only reliable evidence of prior application.
        if e.marker in text:
            results.append({"edit": e.name, "status": "already_applied", "detail": e.marker})
        elif n == e.count:
            contents[f] = text.replace(e.old, e.new, e.count)
            results.append({"edit": e.name, "status": "applied", "detail": e.why})
        else:
            results.append({
                "edit": e.name, "status": "drift",
                "detail": f"expected {e.count} occurrence(s) of the target text in "
                          f"{e.path}, found {n}; upstream has changed",
            })
    return results, contents


def main() -> int:
    args = parse_args()
    root = Path(args.upstream_dir)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"FATAL: no edits defined for {sorted(unknown)}; have {ARMS}")

    outcome: dict = {}
    failed = False
    for arm in arms:
        arm_root = root / arm
        edits = [e for e in ALL_EDITS if e.arm == arm]
        if not arm_root.is_dir():
            logger.error(f"[{arm}] not staged: {arm_root}")
            outcome[arm] = {"status": "not_staged", "edits": []}
            failed = True
            continue

        results, contents = plan_arm(arm_root, edits)
        bad = [r for r in results if r["status"] in ("drift", "missing_file")]
        n_applied = sum(1 for r in results if r["status"] == "applied")
        n_already = sum(1 for r in results if r["status"] == "already_applied")

        if bad:
            # Refuse the ARM, not just the edit: a half-patched repository is worse than
            # an unpatched one, because the parts that did apply look like success.
            for r in bad:
                logger.error(f"[{arm}] {r['edit']}: {r['status']} -- {r['detail']}")
            logger.error(f"[{arm}] REFUSED, nothing written ({len(bad)} unresolved edits)")
            outcome[arm] = {"status": "refused", "edits": results}
            failed = True
            continue

        if args.check:
            status = "clean" if n_applied == 0 else "needs_patching"
            logger.info(f"[{arm}] --check: {n_already} already applied, {n_applied} pending")
            outcome[arm] = {"status": status, "edits": results}
            if n_applied:
                failed = True
            continue

        for f, text in contents.items():
            if text is not None:
                f.write_text(text)
        logger.info(f"[{arm}] {n_applied} applied, {n_already} already applied, "
                    f"{len(edits)} total")
        for r in results:
            logger.info(f"[{arm}]   {r['status']:16s} {r['edit']}")
        outcome[arm] = {"status": "patched", "edits": results}

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_dir": str(root),
        "check_only": args.check,
        "arms": outcome,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))
        logger.info(f"summary -> {args.summary_json}")

    if failed:
        logger.error("one or more arms did not reach a fully patched state")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
