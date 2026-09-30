"""CP -> LP generator: synthesise bound *linear* peptides from bound cyclic peptides.

The goal is a corpus of synthetic ``(LP, CP, target)`` triplets to supervise the
LP->CP direction. CPSea gives us bound CPs in quantity; what does not exist is the
*paired* linear precursor of a known cyclic binder. The provenance route (recover the
AFDB fragment a CPSea row was cut from) turned out to be an identity map -- those
fragments are already closed -- so the LP side has to be generated.

The move here is to run the existing Complexa flow network backwards in *topology*
rather than in time: condition it on the encoded source CP, ask it for a peptide of the
same length and sequence carrying the LINEAR topology token, and train the difference
adversarially against real bound linear peptides (PepBench) so the output lands on the
bound-LP manifold rather than on "a cyclic peptide with one bond deleted".

There are two generator modes. ``onepass`` is a DIRECT EDITOR: one network evaluation
turns the encoded source CP into an edited state via zero-initialised residual heads, with
``t`` pinned at a constant and no sampler on either the training or the export path.
``rollout`` is the original differentiable-trajectory generator, kept unchanged for the
historical comparison. See ``docs/README_CP2LP.md``.

Layout:
    conditioning   the persistent source-CP condition (``x_src_cp``) and the LINEAR
                   topology request applied to the evolving LP state.
    editor         the one-pass direct editor: batch construction, the residual edit, the
                   trunk-call counter and the frame-sensitivity measurement.
    addons         per-channel feature scales plus the three independent add-ons
                   (approximate R1, adaptive discriminator noise, local contrastive
                   retention), exactly one of which may be active.
    data           length-matched pairing of CP generator inputs with real LP examples,
                   plus the receptor-family train/eval split.
    discriminator  per-complex realness logit from decoded LP geometry + local pocket
                   interface. Never sees the source CP.
    losses         hinge GAN, CP-contact retention, sequence retention, open-chain
                   geometry and clash terms.
    module         the ``CP2LPGenerator`` LightningModule (a ``Proteina`` subclass).
    export         triplet writer with per-sample score components.
    report         real-vs-generated comparison over an exported triplet set.
"""

from proteinfoundation.cp2lp.conditioning import (
    SRC_CP_KEY,
    SRC_CP_PRESENT_KEY,
    attach_source_cp_condition,
    drop_source_cp_condition,
    request_linear_topology,
)

# `editor` and `addons` are deliberately NOT imported here. `feature_factory` imports
# `SRC_CP_KEY` from this package at module scope, and pulling the editor in would make that
# a cycle through `cp2lp.editor` -> `cyclization.constants`. The feature classes import
# `EDITOR_NOISE_KEY` lazily inside `forward` for the same reason.
__all__ = [
    "SRC_CP_KEY",
    "SRC_CP_PRESENT_KEY",
    "attach_source_cp_condition",
    "drop_source_cp_condition",
    "request_linear_topology",
]
