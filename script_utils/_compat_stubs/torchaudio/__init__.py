"""Import-only stub that shadows a torchaudio whose C extension does not match torch.

WHY THIS EXISTS. `transformers` 5.x reaches torchaudio on the path to EVERY model class:

    AutoModelForMaskedLM -> modeling_utils -> loss.loss_utils -> loss_rnnt

and `loss_rnnt` does `if is_torchaudio_available(): import torchaudio`. In this venv the
guard passes -- torchaudio IS installed -- but the import then dies with

    OSError: .../torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl

because that extension was built against a different torch than the installed
2.7.0+cu126. So a broken audio package blocks ESM from loading at all.

WHY A STUB RATHER THAN UNINSTALLING IT. The venv is shared by the whole flow stack, and
removing a package from it is a mutation of shared infrastructure that another arm might
be relying on. This file is additive and reversible: it is reached only by a job that puts
`script_utils/_compat_stubs` on PYTHONPATH, and `grep -rn torchaudio src/ script_utils/
scripts/` finds no other user in this repo, so nothing here loses a real dependency.

WHY IT IS SAFE TO BE EMPTY. `transformers` touches `torchaudio.functional.rnnt_loss` only
inside `rnnt_loss()`, which is reachable only from the Parakeet RNN-T speech path. A
masked-LM fine-tune never calls it. Any attribute access still fails loudly rather than
silently returning a mock, so if something DID need real torchaudio it would say so
instead of computing a wrong number.
"""

__version__ = "0.0.0+proteina-complexa-import-stub"
__all__ = []
