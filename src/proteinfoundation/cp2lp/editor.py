"""The DIRECT one-pass CP -> LP editor: source state in, edited state out, one NN call.

WHAT REPLACED WHAT. The first CP->LP generator integrated a short differentiable ODE
trajectory (``nsteps: 24``, DRaFT-K) and exported from a 200-step sampler. That is gone
from this path. The editor is a single evaluation of the pretrained trunk reading the
encoded source CP as its state, and emitting a RESIDUAL:

    delta_X, delta_H = editor(X_src, H_src, target, sequence, z)
    X_lp = X_src + delta_X
    H_lp = H_src + delta_H
    decoded_lp = frozen_AE.decode(X_lp, H_lp)

Three consequences worth stating, because each one closed a specific hole:

* **No rollout artefacts.** The measured reason the adversarial arm could not work was
  that generated peptides carried backbone bond errors ~70x larger than real ones -- the
  cost of a 24-step trajectory -- and the discriminator became a broken-backbone
  detector with a perfect single-threshold separation on every batch of a 40k run. A real
  LP shown to D goes through one AE round trip; a generated LP now also goes through one
  AE round trip. The paths are symmetric by construction rather than by a correction.
* **Time is no longer a sampling coordinate.** ``t`` is still plumbed through, because
  the pretrained trunk has a time embedding and removing it would put every layer out of
  distribution, but it is pinned at a configured constant (0.5). It is an ARCHITECTURE
  INPUT here. Nothing schedules it, nothing integrates over it, and there is no
  velocity-to-clean conversion anywhere on this path.
* **Randomness has to be supplied explicitly.** One deterministic evaluation gives one LP
  per CP. The stochasticity is ``editor_noise``: per-residue invariant Gaussian channels,
  appended last in ``nn.feats_seq``. Resample them and the same CP yields a different LP.

THE RESIDUAL IS A DISPLACEMENT, AND THAT CONVERSION IS EXPLICIT. The pretrained
coordinate head emits a flow velocity; reading it as an edit would start the editor at
``X_src + v(X_src, 0.5)``, an arbitrary jump. The editor therefore uses its OWN
zero-initialised heads (``nn.enable_editor_residual``), so the first output is exactly the
AE round trip of the source CP, and the addition above is written out rather than implied.

ON EQUIVARIANCE, HONESTLY. The trunk consumes absolute CA coordinates through a Linear
and is *rotation-augmented, not rotation-equivariant* -- the data pipeline applies a
uniform ``GlobalRotationTransform`` and centres every complex on its receptor
(``CenteringTransform(center_mode=target)``), which is where translation is actually
factored out. So "the residual rotates as a vector" cannot be proved of this
architecture, and claiming it would be false. What IS enforced here:

* the noise is invariant, so a draw means the same thing in every frame -- an untransformed
  random Cartesian vector would be read as geometry and is never injected;
* the residual is added to the source state, so a rigid motion of the *whole* input moves
  the output with it to exactly the extent the trunk's own velocity field does;
* :func:`rot_trans_consistency` measures the editor's frame sensitivity BESIDE the
  pretrained velocity head's, and the check is that the editor is no worse. A relative
  measurement against the base model is the strongest true statement available.

Paired with :mod:`proteinfoundation.cp2lp.module`, which owns the losses and the
discriminator; this file owns only the forward path and the checks on it.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from loguru import logger

from proteinfoundation.cp2lp.conditioning import (
    SRC_CP_KEY,
    attach_source_cp_condition,
    request_linear_topology,
)

#: Batch key holding the ``[b, n, c]`` per-residue invariant Gaussian noise.
EDITOR_NOISE_KEY = "editor_noise"

#: Default width of the noise block. Read from ``nn.editor_noise_channels`` in practice;
#: this constant exists so the feature class and the sampler cannot drift apart silently.
DEFAULT_NOISE_CHANNELS = 16

#: Above this column-block RMS, `init_noise_adapter_` treats the noise columns as TRAINED and
#: leaves them alone instead of re-drawing them. It separates "exactly zero", which is what a
#: feature-append splice leaves behind, from anything an optimiser produced -- so it sits far
#: below the 0.02 std of the draw itself and there is no regime where the two are confusable.
NOISE_ADAPTER_TRAINED_RMS = 1e-8

#: Fixed value for the trunk's time embedding. NOT a sampling time -- see the module
#: docstring. 0.5 puts the input mid-schedule, which is the region the pretrained trunk
#: saw most of and the furthest from both degenerate ends (t->1 makes `1/(1-t)` factors in
#: the flow machinery blow up; t=0 tells the trunk its input is pure noise, which the
#: encoded source CP emphatically is not).
DEFAULT_EDITOR_T = 0.5

#: Batch keys the editor treats as read-only inputs. Checked by
#: :func:`assert_inputs_unmutated`, because an in-place edit of the source CP or the
#: receptor would make the contact-retention loss compare a sample against itself.
PROTECTED_INPUT_KEYS = ("x_target", "target_mask", "seq_target", "seq_target_mask", "residue_type")


def sample_editor_noise(
    bs: int,
    n: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``[b, n, channels]`` standard normal draws for the editor's stochastic input.

    ``generator`` is a DEDICATED stream, and passing one is the normal case rather than an
    option. The arms are only comparable if they see the same source CPs in the same order
    with the same noise, and the addons differ in how many random numbers they consume --
    arm B's discriminator corruption draws every step. Sharing the global RNG would
    therefore make the addon change the *data*, and the comparison would no longer isolate
    the addon.
    """
    if channels <= 0:
        raise ValueError(f"editor noise needs at least one channel, got {channels}")
    return torch.randn(bs, n, channels, device=device, dtype=dtype, generator=generator)


def make_noise_generator(seed: int, device: torch.device) -> torch.Generator:
    """A generator for the editor-noise stream alone, seeded off the run seed.

    The offset keeps this stream from coinciding with any other seeded-from-``seed``
    stream in the project, which would correlate the editor's noise with, say, the data
    order.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) * 1_000_003 + 17)
    return g


def build_editor_batch(
    cp_batch: dict,
    data_modes: list[str],
    t_fixed: float = DEFAULT_EDITOR_T,
    noise: torch.Tensor | None = None,
) -> dict:
    """Assembles the batch the editor's single forward pass consumes.

    The source CP enters twice, on purpose and in two different roles:

    ``x_t``
        as the network's ordinary coordinate/latent INPUT channels. This is the state
        being edited, and routing it here is what lets the pretrained trunk read it with
        the weights it already has (``xt_bb_ca``, ``xt_local_latents``,
        ``xt_bb_ca_pair_dists``) instead of through a newly initialised path.
    ``x_src_cp``
        as the explicit persistent CONDITION, which the architecture already has features
        for. Redundant with ``x_t`` on this path -- and kept anyway, because those feature
        columns were trained in the rollout arm and because dropping the condition would
        make the real-LP rows (where it is legitimately absent) and the CP rows differ in
        their feature *support* rather than only in their values.

    ``x_sc`` is set to explicit zeros: there is no evolving state to self-condition on and
    no second prediction pass. Zeros rather than a missing key so the one-time "no x_sc"
    warning does not fire every run and so the absence is deliberate in the tensor.

    Args:
        cp_batch: a CPSea batch AFTER ``add_clean_samples``, so ``x_1`` holds the AE
            encoding of the source CP.
        data_modes: the product-space data modes, in the model's order.
        t_fixed: the constant fed to the time embedding.
        noise: ``[b, n, c]`` invariant noise, or None to leave the editor deterministic
            (which the rotation check wants, and nothing else does).

    Returns:
        A new dict. ``cp_batch`` is not modified except by
        ``attach_source_cp_condition``, which only ADDS the condition keys.
    """
    attach_source_cp_condition(cp_batch, data_modes=data_modes)
    bs, n = cp_batch["mask"].shape

    # Drop every tensor the training interpolant or a sampler would have left behind.
    # `x_1` in particular: it is the source CP's encoding, and a loss that reads it as
    # "the ground-truth linear peptide" would be supervising the editor to return the ring.
    gen_batch = {
        k: v
        for k, v in cp_batch.items()
        if k not in ("x_1", "x_0", "x_t", "t", "x_sc", "x_recycle")
    }
    gen_batch["mask"] = cp_batch["mask"].bool()

    src = cp_batch[SRC_CP_KEY]
    gen_batch["x_t"] = {m: src[m] for m in data_modes}
    gen_batch["x_sc"] = {m: torch.zeros_like(src[m]) for m in data_modes}
    ref = src[data_modes[0]]
    gen_batch["t"] = {
        m: torch.full((bs,), float(t_fixed), device=ref.device, dtype=ref.dtype) for m in data_modes
    }

    request_linear_topology(gen_batch, bs=bs)

    if noise is not None:
        if noise.shape[0] != bs or noise.shape[1] != n:
            raise ValueError(f"editor noise is {tuple(noise.shape[:2])} but the batch is {(bs, n)}")
        # Masked so a padded residue's noise cannot leak into a pooled quantity. The
        # feature itself is per-residue and the trunk masks too, so this is belt-and-braces
        # -- and it makes the noise tensor itself inspectable without holding the mask.
        gen_batch[EDITOR_NOISE_KEY] = noise * gen_batch["mask"][..., None].to(noise.dtype)

    return gen_batch


def apply_residuals(
    src_state: dict[str, torch.Tensor],
    nn_out: dict[str, dict[str, torch.Tensor]],
    mask: torch.Tensor,
    data_modes: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``X_lp = X_src + delta_X`` per data mode, masked on padding.

    Reads the ``delta`` key, never ``v``: the velocity head is a different quantity in a
    different parameterisation, and consuming it here is the mistake this key separation
    exists to prevent.

    Returns ``(edited_state, deltas)``. The deltas come back so the caller can log their
    magnitude -- a residual norm pinned at zero is the signature of a head that never left
    its zero init, which otherwise looks like a generator that simply refuses to edit.
    """
    m = mask.bool()[..., None]
    edited: dict[str, torch.Tensor] = {}
    deltas: dict[str, torch.Tensor] = {}
    for dm in data_modes:
        if "delta" not in nn_out.get(dm, {}):
            raise KeyError(
                f"the network produced no `delta` for data mode '{dm}'. The one-pass editor "
                "needs `nn.enable_editor_residual: true`; without it the trunk emits only a "
                "flow velocity, which is NOT an edit displacement."
            )
        d = nn_out[dm]["delta"] * m
        deltas[dm] = d
        edited[dm] = src_state[dm] + d
    return edited, deltas


@contextmanager
def count_trunk_calls(nn_module: torch.nn.Module) -> "list[int]":
    """Counts forward calls of the flow trunk over a block. ``[n]``, mutated in place.

    The claim "one neural-network evaluation per generated batch" is the whole point of
    this arm, and it is exactly the kind of claim that quietly stops being true -- a
    self-conditioning pass, a recycle, a CFG unconditional branch and an ODE step all look
    like ordinary code. A forward hook on the trunk is the only check that cannot be
    fooled by reading the source. AE and discriminator calls do not pass through this
    module and so do not count, which is what makes the number interpretable.
    """
    calls = [0]

    def _hook(_module, _inp, _out):
        calls[0] += 1

    handle = nn_module.register_forward_hook(_hook)
    try:
        yield calls
    finally:
        handle.remove()


def snapshot_inputs(batch: dict, keys=PROTECTED_INPUT_KEYS) -> dict[str, torch.Tensor]:
    """Clones the tensors the editor must treat as read-only."""
    out = {}
    for k in keys:
        v = batch.get(k, None)
        if torch.is_tensor(v):
            out[k] = v.detach().clone()
    src = batch.get(SRC_CP_KEY, None)
    if isinstance(src, dict):
        for m, v in src.items():
            out[f"{SRC_CP_KEY}.{m}"] = v.detach().clone()
    return out


def assert_inputs_unmutated(batch: dict, snapshot: dict[str, torch.Tensor]) -> list[str]:
    """Names of protected tensors that changed since ``snapshot``. Empty list is the pass.

    Returns rather than raises so a smoke can report every offender at once instead of
    stopping at the first.
    """
    bad = []
    for key, before in snapshot.items():
        if "." in key and key.startswith(SRC_CP_KEY):
            _, mode = key.split(".", 1)
            now = batch.get(SRC_CP_KEY, {}).get(mode, None)
        else:
            now = batch.get(key, None)
        if not torch.is_tensor(now):
            bad.append(f"{key}: gone")
        elif now.shape != before.shape or not torch.equal(now.detach(), before):
            bad.append(key)
    return bad


# --------------------------------------------------------------------- noise adapter


def init_noise_adapter_(
    factory: torch.nn.Module,
    std: float = 0.02,
    feature_name: str = "EditorNoiseSeqFeat",
    preserve_loaded: bool = False,
) -> tuple[int, int] | None:
    """Gives the noise block SMALL NONZERO input weights, in place. ``(offset, width)``.

    Every feature factory concatenates its features in list order and projects the result
    with one ``linear_out``, and ``_splice_pretrained_weights`` copies the overlapping
    LEADING columns from the checkpoint -- so an appended feature arrives with its columns
    at exactly zero. For the noise that is a bad starting point in a specific way: the
    residual heads are zero-initialised too, so on step one the trunk gets no gradient at
    all, and the noise columns only start moving once the heads have. The editor would
    spend its first phase learning a deterministic edit and only later discover it has a
    random input.

    A small draw breaks that ordering without disturbing the pretrained columns, which are
    left exactly as spliced. It must run AFTER the splice; ``train.py`` calls it there.

    ``preserve_loaded`` -- OFF BY DEFAULT, AND THE DEFAULT IS THE HISTORICAL BEHAVIOUR.
    The paragraph above is an argument about an APPENDED feature, whose columns the splice
    leaves at exactly zero. Warm-starting from ANOTHER ARM'S OWN checkpoint goes down the
    same `pretrain_ckpt_path` branch in `train.py`, and there the noise columns are not
    zero -- they are trained, and are the only thing making the output depend on the random
    input at all (the torsion arm's `seed_diversity_nm` was 0.241). Re-drawing them makes
    "warm start from the trained generator" false, so a warm start passes True.

    IT IS NOT THE DEFAULT, and that is deliberate rather than timid. MEASURED: the
    `cp2lp_onepass_fork` checkpoint every `bb_*` and `onepass_*` arm starts from ALREADY
    carries trained noise columns (rms 1.980e-02), and every one of those runs re-drew them
    -- jobs 70458 and 71251 both logged `initialised at columns [315, 331) with std=0.02`.
    Gating on "are the columns zero" would therefore have silently changed the
    initialisation of every arm in the repo, including the control that a release arm is
    compared against, which is a confound introduced by a bug fix. The flag keeps the
    existing arms bit-identical and gives the new behaviour only to the run that asks for
    it. Either way the decision is logged, never inferred.

    Returns None -- and warns -- if no noise feature is in the list, because that is the
    silent version of "the editor is deterministic and nothing said so".
    """
    creators = getattr(factory, "feat_creators", None)
    if creators is None:
        logger.warning("init_noise_adapter_: factory has no feat_creators (no features requested?).")
        return None
    offset = 0
    for creator in creators:
        width = int(creator.get_dim())
        if type(creator).__name__ == feature_name:
            weight = factory.linear_out.weight  # [dim_out, sum_of_feature_dims]
            if offset + width > weight.shape[1]:
                raise RuntimeError(
                    f"noise columns [{offset}, {offset + width}) fall outside "
                    f"linear_out of width {weight.shape[1]}"
                )
            block = weight[:, offset : offset + width]
            # Trained columns are orders of magnitude above this; spliced-zero columns are
            # exactly 0.0. The threshold only has to separate "exactly zero" from anything
            # a real optimiser produced, so it is far below the std of the draw itself.
            rms = float(block.detach().float().pow(2).mean().sqrt())
            if preserve_loaded and rms > NOISE_ADAPTER_TRAINED_RMS:
                logger.info(
                    f"CP2LP editor: noise adapter columns [{offset}, {offset + width}) of "
                    f"{weight.shape[1]} already carry trained weights (rms={rms:.3e} > "
                    f"{NOISE_ADAPTER_TRAINED_RMS:.3e}) -- LEFT AS LOADED, not re-drawn. This "
                    "is a warm start from a checkpoint that already learned to read its noise."
                )
                return offset, width
            with torch.no_grad():
                block.normal_(0.0, float(std))
            # Say which of the two it was. "initialised" reads as harmless, and when the
            # prior columns were trained this call DISCARDED them -- that is the difference
            # between a warm start and a re-randomised one, and it has to be visible in the
            # log of the run it happened to, not reconstructed later from a config.
            had = "OVERWRITING trained columns" if rms > NOISE_ADAPTER_TRAINED_RMS else "spliced-zero columns"
            logger.info(
                f"CP2LP editor: noise adapter initialised at columns "
                f"[{offset}, {offset + width}) of {weight.shape[1]} with std={std} "
                f"-- {had} (prior rms={rms:.3e}, preserve_loaded={preserve_loaded})"
            )
            return offset, width
        offset += width
    logger.warning(
        "init_noise_adapter_: no `editor_noise_seq` in nn.feats_seq -- the one-pass editor "
        "would be DETERMINISTIC and could only ever produce one LP per source CP."
    )
    return None


# ------------------------------------------------------------------ frame sensitivity


def _rotate_batch_coords(batch: dict, rot: torch.Tensor, shift: torch.Tensor) -> dict:
    """A shallow copy of ``batch`` with every coordinate tensor rigidly moved.

    Right-multiplication by ``rot`` matches the project's own convention
    (``GlobalRotationTransform`` right-multiplies row vectors), so this is the same
    transformation the dataloader applies, not a transposed cousin of it.

    Local latents are NOT transformed: the AE expresses them in per-residue local frames,
    so a global rotation leaves them unchanged. If that ever stops being true this check
    is the thing that will report it as a large residual.
    """
    out = dict(batch)

    def _move(x):
        return x @ rot + shift

    for key in ("x_target", "coords_nm", "coords"):
        v = batch.get(key, None)
        if torch.is_tensor(v):
            out[key] = _move(v)
    for key in ("x_t", "x_sc", SRC_CP_KEY):
        v = batch.get(key, None)
        if isinstance(v, dict):
            out[key] = {m: (_move(t) if m == "bb_ca" else t) for m, t in v.items()}
    return out


@torch.no_grad()
def rot_trans_consistency(
    model,
    gen_batch: dict,
    data_modes: list[str],
    seed: int = 0,
) -> dict[str, float]:
    """How much the editor's residual depends on the frame, next to the base model's.

    Applies one rigid motion to every coordinate input, runs the SAME forward twice (the
    noise is already in ``gen_batch`` and is invariant, so it is identical in both frames),
    and compares the rotated residual against the residual of the rotated input. Reported
    as an RMS in nm and as a fraction of the residual's own magnitude.

    The pretrained VELOCITY head is measured on the identical pair of forwards -- it costs
    nothing, it comes from the same tokens, and it is the only available calibration.
    Reading the editor's number alone invites the wrong conclusion twice over: a small
    absolute deviation looks like equivariance when the architecture cannot provide it,
    and a large one looks like a bug when it is the base model's own behaviour.

    ``dropout`` is disabled for the duration. Without that, the two forwards differ by
    their dropout masks and the measurement reports that instead of frame sensitivity --
    the same trap the aR1 paired forwards have.
    """
    device = gen_batch["mask"].device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.linalg.qr(torch.randn(3, 3, generator=g))[0]
    if torch.det(a) < 0:
        a[:, 0] = -a[:, 0]
    rot = a.to(device=device, dtype=gen_batch["x_t"]["bb_ca"].dtype)
    shift = torch.tensor([0.37, -1.25, 2.04], device=device, dtype=rot.dtype)

    was_training = model.training
    model.eval()  # dropout off; see the docstring
    try:
        out_ref = model.call_nn(dict(gen_batch), n_recycle=0)
        out_rot = model.call_nn(_rotate_batch_coords(gen_batch, rot, shift), n_recycle=0)
    finally:
        if was_training:
            model.train()

    mask = gen_batch["mask"].bool()[..., None].to(rot.dtype)
    n = mask.sum().clamp(min=1.0) * 3.0
    report: dict[str, float] = {}
    for label, key in (("editor_delta", "delta"), ("base_velocity", model.nn.output_param["bb_ca"])):
        if key not in out_ref["bb_ca"]:
            continue
        ref = out_ref["bb_ca"][key] * mask
        got = out_rot["bb_ca"][key] * mask
        # A displacement transforms as a vector: no shift term. That is exactly why the
        # residual formulation is the right one to check -- a POINT-valued head would need
        # the shift, and conflating the two is how a translation bug hides.
        want = (ref @ rot) * mask
        dev = torch.sqrt(((got - want) ** 2).sum() / n)
        scale = torch.sqrt((ref**2).sum() / n)
        report[f"{label}_rms_dev"] = float(dev)
        report[f"{label}_rms_mag"] = float(scale)
        report[f"{label}_rel_dev"] = float(dev / scale.clamp(min=1e-8))

    # Also confirm the invariant input really was invariant -- if a future feature starts
    # reading coordinates into the noise block, this is what catches it.
    if EDITOR_NOISE_KEY in gen_batch:
        rotated = _rotate_batch_coords(gen_batch, rot, shift)
        report["noise_unchanged"] = float(
            torch.equal(rotated[EDITOR_NOISE_KEY], gen_batch[EDITOR_NOISE_KEY])
        )
    return report
