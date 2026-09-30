"""The three algorithmic add-ons, each independent, each default-off.

One file because they share one thing -- the per-feature statistics of the real-LP
discriminator features -- and because keeping them adjacent makes it obvious that no addon
reads another's state. Exactly one may be active at a time; the module config rejects any
other combination.

    ar1    approximate R1: a discriminator-only penalty on its own sensitivity to a small
           perturbation of REAL features.                          (adapted from APT,
                                                                    arXiv:2501.08316)
    noise  adaptive corruption of the continuous feature channels of BOTH classes before
           ordinary GAN scoring, with the level handed to D.        (adapted from
                                                                    Diffusion-GAN)
    nce    a generator-side local contrastive term preserving source-local content while
           leaving the termini free to open.                        (adapted from CUT's
                                                                    PatchNCE)

ALL THREE ARE FEATURE-SPACE OR DESCRIPTOR-SPACE ADAPTATIONS. The originals operate on
images: aR1 perturbs pixels, Diffusion-GAN diffuses pixels, PatchNCE contrasts
convolutional patch embeddings. None of those objects exists here. What is transferred is
the training idea; the numerical hyperparameters of an image or video model are not
validated peptide settings, and every default below is a pilot value chosen for a reason
that is written down rather than a tuned optimum.

WHY THE SCALES ARE ESTIMATED AND CACHED. The feature blocks mix units that differ by
orders of magnitude -- nm distances (~0.2-3.0), dihedral sine/cosine components (-1..1),
soft contact counts (0..~40), probabilities (0..1). A single absolute sigma would be a
rounding error on one channel and a total scramble of another. Both noise-related addons
therefore work in units of each channel's own standard deviation, measured once from a
small TRAINING-ONLY sample of real LPs. Evaluation data never enters: the scales are part
of the model's training state, and fitting them on the eval set would leak it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from proteinfoundation.cp2lp.geometry import (
    CA_IDX,
    C_IDX,
    N_IDX,
    ca_coords,
    heavy_atom_distances,
    pairwise_distances,
    safe_norm,
    soft_contact,
)

#: Blocks the addons are allowed to touch. ``res`` (sequence probabilities) and ``mask``
#: (padding) are never perturbed; see ``PeptideInterfaceDiscriminator.continuous_column_masks``.
PERTURBABLE_KEYS = ("geom", "iface", "glob")

#: Per-residue blocks, i.e. ``[b, n, d]``. ``glob`` is ``[b, d]`` and is handled separately
#: -- a shape-based guess would mis-broadcast it.
PER_RESIDUE_KEYS = ("geom", "iface")


# ===================================================================== feature scales


class FeatureScaleCache(nn.Module):
    """Per-channel mean and standard deviation of REAL-LP discriminator features.

    Accumulated over the first ``min_examples`` real examples the training loop sees and
    then frozen. Frozen rather than kept as a running estimate on purpose: both addons
    define their perturbation *in units of these scales*, so a scale that keeps moving
    makes the effective noise amplitude drift for reasons unrelated to the controller, and
    two arms sharing a seed would still diverge if one of them happened to see a different
    prefix of the data.

    Registered as buffers, so the state travels in the checkpoint and a requeue resumes
    with the same scales rather than re-estimating them from a different prefix.
    """

    def __init__(
        self,
        widths: dict[str, int],
        min_examples: int = 128,
        std_floor: float = 1e-3,
    ):
        super().__init__()
        self.min_examples = int(min_examples)
        self.std_floor = float(std_floor)
        self.keys = tuple(sorted(widths))
        for key, width in widths.items():
            self.register_buffer(f"sum_{key}", torch.zeros(int(width)))
            self.register_buffer(f"sumsq_{key}", torch.zeros(int(width)))
            self.register_buffer(f"count_{key}", torch.zeros(()))
        self.register_buffer("n_examples", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @property
    def ready(self) -> bool:
        return bool(self.frozen)

    @torch.no_grad()
    def update(self, feats: dict[str, torch.Tensor]) -> None:
        """Folds one batch of real features in. No-op once frozen."""
        if bool(self.frozen):
            return
        mask = feats["mask"].bool()
        for key in self.keys:
            v = feats[key]
            if key in PER_RESIDUE_KEYS:
                m = mask[..., None].to(v.dtype)
                flat_sum = (v * m).sum(dim=(0, 1))
                flat_sq = ((v * m) ** 2).sum(dim=(0, 1))
                n = mask.sum().to(v.dtype)
            else:
                flat_sum = v.sum(dim=0)
                flat_sq = (v**2).sum(dim=0)
                n = torch.tensor(float(v.shape[0]), device=v.device, dtype=v.dtype)
            getattr(self, f"sum_{key}").add_(flat_sum.to(getattr(self, f"sum_{key}").dtype))
            getattr(self, f"sumsq_{key}").add_(flat_sq.to(getattr(self, f"sumsq_{key}").dtype))
            getattr(self, f"count_{key}").add_(n.to(getattr(self, f"count_{key}").dtype))
        self.n_examples += int(mask.shape[0])
        if int(self.n_examples) >= self.min_examples:
            self.frozen.fill_(True)
            logger.info(
                f"CP2LP addons: feature scales frozen after {int(self.n_examples)} real LPs "
                f"(>= min_examples={self.min_examples})."
            )

    def stats(self, key: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, std)`` for one block, with the std floored strictly above zero.

        The floor is not cosmetic. A channel that is constant across the sample -- a pocket
        chemistry class that never appears, a dihedral undefined at every terminus -- has
        std exactly 0, and dividing by it produces inf and then NaN on the first
        standardisation. Flooring turns those channels into ones the perturbation barely
        moves, which is the correct behaviour for a channel carrying no information.
        """
        n = getattr(self, f"count_{key}").clamp(min=1.0)
        mean = getattr(self, f"sum_{key}") / n
        var = (getattr(self, f"sumsq_{key}") / n - mean**2).clamp(min=0.0)
        std = var.sqrt().clamp(min=self.std_floor)
        return mean.to(device=device, dtype=dtype), std.to(device=device, dtype=dtype)

    def summary(self) -> dict[str, float]:
        out = {"scales_n_examples": float(self.n_examples), "scales_frozen": float(self.frozen)}
        for key in self.keys:
            _, std = self.stats(key, torch.device("cpu"), torch.float32)
            out[f"scales_{key}_std_median"] = float(std.median())
        return out


def feature_widths(discriminator) -> dict[str, int]:
    """``{block: width}`` for the blocks the addons may perturb."""
    return {
        "geom": int(discriminator.n_geom),
        "iface": int(discriminator.n_iface),
        "glob": int(discriminator.n_global),
    }


class AddonRandomStreams:
    """The addons' own RNG, kept off every other stream in the run.

    Two generators because torch requires the generator's device to match the tensor's:
    ``randn`` for a CUDA feature block needs a CUDA generator, while ``multinomial`` over
    a small CPU probability vector needs a CPU one. One holder so a call site cannot pick
    the wrong one by accident.

    WHY A DEDICATED STREAM AT ALL. The five arms are only comparable if they see the same
    source CPs in the same order with the same editor noise. The addons consume a variable
    number of random draws -- arm B draws a level and a full feature-sized noise tensor
    every discriminator step, arm A draws one perturbation, arms with no addon draw none.
    On a shared generator that difference would shift the data order and the editor noise,
    and the comparison would be measuring the addon's effect on the *dataset*.
    """

    def __init__(self, seed: int, device: torch.device):
        self.cpu = torch.Generator(device="cpu")
        self.cpu.manual_seed(int(seed) * 7_919 + 101)
        if torch.device(device).type == "cpu":
            self.dev = self.cpu
        else:
            self.dev = torch.Generator(device=device)
            self.dev.manual_seed(int(seed) * 7_919 + 103)


# ============================================================ arm A: approximate R1


def approx_r1_penalty(
    discriminator,
    real_feats: dict[str, torch.Tensor],
    scales: FeatureScaleCache,
    sigma_rel: float = 0.01,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``mean((D(real + eps) - D(real))**2)`` over REAL examples only.

    The paper's form is ``L_aR1 = ||D(x, c) - D(N(x, sigma I), c)||^2``, a finite-difference
    stand-in for R1's ``||grad_x D(x, c)||^2`` whose point is that it needs no
    second-order backward. Same objective -- keep D insensitive around the real data --
    without higher-order autograd. Applied to real samples only, as in the paper.

    TWO DEPARTURES FROM THE PAPER, both forced by the domain:

    * sigma is RELATIVE (``sigma_rel * per-channel std``) rather than the paper's absolute
      0.01/0.1. Images arrive normalised to a common range; these feature blocks do not,
      and one absolute sigma cannot be simultaneously meaningful for a 0.25 nm distance
      and a soft contact count of 30.
    * only CONTINUOUS channels are perturbed. Sequence probabilities, one-hot pocket
      identity, padding and the terminal flags are left exactly as they are -- Gaussian
      noise on a one-hot is not a nearby real sample, it is a different kind of object.

    DROPOUT IS DISABLED FOR THE PAIR. Two forwards through a dropout layer differ by their
    masks, so with dropout live this penalty measures the sampling variance of D's own
    regularisation and calls it input sensitivity -- and it would be largest exactly when
    the input sensitivity is smallest. The module is returned to its previous mode
    afterwards, including on an exception, because leaving D in eval() would silently
    disable dropout for the rest of training.

    D's PARAMETER gradients are kept: this is a discriminator-side penalty and is meant to
    reach its weights. Nothing about it reaches the generator -- the features passed in are
    the real ones.
    """
    if not scales.ready:
        zero = torch.zeros((), device=real_feats["mask"].device)
        return zero, {"ar1_skipped_scales_not_ready": torch.tensor(1.0)}

    col_masks = discriminator.continuous_column_masks()
    perturbed = dict(real_feats)
    total_sigma = 0.0
    n_cols = 0
    for key in PERTURBABLE_KEYS:
        v = real_feats[key]
        _, std = scales.stats(key, v.device, v.dtype)
        keep = col_masks[key].to(v.device)
        sigma = (sigma_rel * std) * keep.to(v.dtype)
        eps = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
        delta = eps * sigma
        if key in PER_RESIDUE_KEYS:
            delta = delta * real_feats["mask"].bool()[..., None].to(v.dtype)
        perturbed[key] = v + delta
        total_sigma += float(sigma.sum())
        n_cols += int(keep.sum())

    was_training = discriminator.training
    discriminator.eval()
    try:
        logits_clean = discriminator(real_feats)
        logits_pert = discriminator(perturbed)
    finally:
        if was_training:
            discriminator.train()

    penalty = ((logits_pert - logits_clean) ** 2).mean()
    metrics = {
        "ar1_penalty": penalty.detach(),
        "ar1_logit_shift_abs": (logits_pert - logits_clean).abs().mean().detach(),
        "ar1_mean_sigma": torch.tensor(total_sigma / max(1, n_cols)),
        "ar1_n_perturbed_cols": torch.tensor(float(n_cols)),
    }
    return penalty, metrics


# ==================================================== arm B: adaptive feature noise


class FeatureNoiseController(nn.Module):
    """Diffusion-GAN-style corruption of D's continuous inputs, with an adaptive level.

    WHAT IS FAITHFUL TO THE OFFICIAL RECIPE (``Diffusion-GAN/diffusion-stylegan2``):

    * the corruption is a variance-preserving AFFINE Gaussian map, the same family as the
      reference ``q_sample``: ``x' = sqrt(1 - s^2) * x + s * eps`` where the reference
      writes ``sqrt(abar_t) * x + sqrt(1 - abar_t) * eps``;
    * the level is drawn from a POOL in which half the entries are zero, so roughly half
      of every batch is shown clean, and the rest follow the reference's ``priority``
      distribution -- probability proportional to the level index, i.e. weighted towards
      stronger corruption;
    * the level reaches D as an input (``noise_level_dim``), exactly as the reference
      passes ``t`` into D's conditioning mapping;
    * the adaptive controller is the reference's: a bang-bang step on the SIGN of the error
      between ``mean(sign(real_logits))`` and a target, with the target 0.6 the reference's
      own starting value, and the state ``p`` clipped to [0, 1];
    * ``p`` controls the eligible RANGE of levels, as it controls ``T`` there -- not the
      per-level magnitude.

    WHAT IS DIFFERENT, AND WHY. The schedule is parameterised directly by the injected
    standard deviation ``s`` instead of by a beta product. The reference needs the beta
    bookkeeping because its ``t`` must line up with a diffusion model's; nothing here has
    a diffusion model to line up with, and the direct parameterisation makes
    ``max_std: 0.2`` -- "at most a fifth of a channel's own standard deviation" -- a number
    that can be read off the config and checked against the features. The two are the same
    affine law with the same variance-preserving normalisation.

    THE LEVEL MUST NOT ENCODE THE LABEL. Real and fake draw i.i.d. from the same pool with
    independent noise; if levels were assigned per class, D would read the class off the
    level and the corruption would become a free label instead of a difficulty.

    AND THE CORRUPTION IS A DISCRIMINATOR INPUT ONLY. The generator's physical losses and
    every exported structure use uncorrupted coordinates. Raising the noise until D can no
    longer object must not become a way to call a chemically broken peptide valid, which
    is why the clean-D diagnostics are logged alongside.
    """

    def __init__(
        self,
        max_std: float = 0.2,
        n_levels: int = 32,
        pool_size: int = 64,
        target_sign: float = 0.6,
        adapt_rate_examples: float = 5000.0,
        adapt_every_n_steps: int = 4,
        p_init: float = 0.0,
        p_max: float = 1.0,
    ):
        super().__init__()
        if not (0.0 < max_std < 1.0):
            # s >= 1 would annihilate the signal entirely (sqrt(1 - s^2) <= 0).
            raise ValueError(f"max_std must be in (0, 1), got {max_std}")
        self.max_std = float(max_std)
        self.n_levels = int(n_levels)
        self.pool_size = int(pool_size)
        self.target_sign = float(target_sign)
        self.adapt_rate_examples = float(adapt_rate_examples)
        self.adapt_every_n_steps = max(1, int(adapt_every_n_steps))
        self.p_max = float(p_max)
        self.register_buffer("p", torch.tensor(float(p_init)))
        self.register_buffer("sign_ema", torch.zeros(()))
        self.register_buffer("sign_seen", torch.zeros(()))

    # ------------------------------------------------------------------ schedule

    def eligible_levels(self) -> int:
        """Highest level index currently in play. Grows with ``p``, at least 1."""
        return max(1, int(round(float(self.p) * self.n_levels)))

    def level_std(self, index: torch.Tensor) -> torch.Tensor:
        """Index -> injected standard deviation, in units of each channel's own std.

        Linear in the index and anchored at ``n_levels`` rather than at the currently
        eligible maximum, so a given index means a fixed amplitude however ``p`` moves.
        Anchoring at the eligible maximum instead would make the controller change what
        past logged levels meant.
        """
        return self.max_std * index.to(torch.float32) / float(self.n_levels)

    def sample_levels(
        self, b: int, device: torch.device, cpu_generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """``[b]`` injected standard deviations, drawn from the shared pool.

        ``cpu_generator`` must be a CPU generator (``AddonRandomStreams.cpu``): the pool
        and the draw are small CPU operations, and torch requires matching devices.
        """
        j_max = self.eligible_levels()
        half = self.pool_size // 2
        # Priority distribution over 1..j_max, probability proportional to the index.
        idx = torch.arange(1, j_max + 1, dtype=torch.float32)
        probs = idx / idx.sum()
        drawn = torch.multinomial(
            probs, self.pool_size - half, replacement=True, generator=cpu_generator
        ) + 1
        pool = torch.cat([torch.zeros(half, dtype=torch.long), drawn.to(torch.long)])
        pick = torch.randint(0, pool.numel(), (b,), generator=cpu_generator)
        return self.level_std(pool[pick]).to(device=device)

    # ----------------------------------------------------------------- corruption

    def corrupt(
        self,
        feats: dict[str, torch.Tensor],
        scales: FeatureScaleCache,
        levels: torch.Tensor,
        col_masks: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Affine Gaussian corruption of the continuous channels, in standardised space.

        Standardise with the cached per-channel mean and std, corrupt, then convert back to
        the channel's own units -- so the amplitude means the same thing on a distance and
        on a contact count. GRADIENT IS PRESERVED: the generator's adversarial term is
        scored through this same transformation, and detaching here would disconnect it.
        """
        out = dict(feats)
        for key in PERTURBABLE_KEYS:
            v = feats[key]
            mean, std = scales.stats(key, v.device, v.dtype)
            keep = col_masks[key].to(device=v.device)
            s = levels.to(v.dtype).reshape(-1, *([1] * (v.dim() - 1)))
            z = (v - mean) / std
            eps = torch.randn(v.shape, device=v.device, dtype=v.dtype, generator=generator)
            z_noisy = torch.sqrt((1.0 - s**2).clamp(min=0.0)) * z + s * eps
            # The `where` selects the FINAL value, not the standardised one, so a
            # categorical column comes back BIT-IDENTICAL. Round-tripping it through
            # `(v - mean)/std * std + mean` instead leaves ~1e-8 of float error on a
            # one-hot -- numerically harmless, but it would make "categorical features are
            # unchanged" a claim no test could assert exactly.
            new = torch.where(keep, z_noisy * std + mean, v)
            if key in PER_RESIDUE_KEYS:
                m = feats["mask"].bool()[..., None].to(v.dtype)
                new = new * m + v * (1.0 - m)
            out[key] = new
        return out

    # ------------------------------------------------------------------ controller

    @torch.no_grad()
    def observe(self, real_logits: torch.Tensor) -> None:
        """Accumulates the controller's statistic: the mean SIGN of the real logits."""
        self.sign_ema += real_logits.detach().sign().sum().to(self.sign_ema.dtype)
        self.sign_seen += float(real_logits.numel())

    @torch.no_grad()
    def maybe_adapt(self, global_step: int) -> dict[str, torch.Tensor]:
        """Bang-bang update of ``p``, every ``adapt_every_n_steps``.

        Rate arithmetic, because it is the one number that has to be matched to this arm's
        batch size rather than inherited. The reference moves ``p`` by
        ``n_examples / (ada_kimg * 1000)`` with ``ada_kimg=100``, i.e. 100k examples to
        traverse the whole range -- sized for batches of 32-64 images. A D step here scores
        a handful of length-matched pairs, so 100k examples would be tens of thousands of
        optimizer steps and ``p`` would never leave its initial value inside a 10k-update
        pilot. ``adapt_rate_examples=5000`` gives roughly 1-2k steps for a full traverse at
        4 real examples per step, which is fast enough to respond and slow enough not to
        oscillate. It is a pilot value; ``controller_p`` is logged so it can be read.
        """
        out = {
            "dgan_p": self.p.detach().clone(),
            "dgan_eligible_levels": torch.tensor(float(self.eligible_levels())),
            "dgan_max_level_std": torch.tensor(self.max_std * self.eligible_levels() / self.n_levels),
        }
        if global_step % self.adapt_every_n_steps != 0 or float(self.sign_seen) < 1.0:
            return out
        stat = self.sign_ema / self.sign_seen.clamp(min=1.0)
        adjust = torch.sign(stat - self.target_sign) * (self.sign_seen / self.adapt_rate_examples)
        self.p.copy_((self.p + adjust).clamp(min=0.0, max=self.p_max))
        out["dgan_sign_real"] = stat.detach().clone()
        out["dgan_p"] = self.p.detach().clone()
        out["dgan_eligible_levels"] = torch.tensor(float(self.eligible_levels()))
        out["dgan_max_level_std"] = torch.tensor(self.max_std * self.eligible_levels() / self.n_levels)
        self.sign_ema.zero_()
        self.sign_seen.zero_()
        return out


# ================================================== arm C: local contrastive retention


class LocalRetentionNCE(nn.Module):
    """CUT-flavoured InfoNCE pinning each residue's LOCAL content to its source counterpart.

    The contact-retention loss already in the objective is a global, one-sided statement:
    do not lose the source's peptide-target contacts. It says nothing about whether a
    residue's local environment still resembles the one it had -- a peptide can keep the
    same contact COUNT while rearranging which residue sits where. This term is the
    per-residue version of that constraint.

    ADAPTED FROM PatchNCE, not imported from it. What carries over is the formulation:
    normalised descriptors, a positive that is the same location in the source, negatives
    that are OTHER LOCATIONS IN THE SAME PEPTIDE (the reference measured that
    within-image negatives beat minibatch-wide ones, and the same reasoning applies more
    strongly here -- a negative from another complex is trivially distinguishable and
    teaches nothing), the diagonal of the negative block masked out because it duplicates
    the positive, temperature 0.07, and a two-layer projection with L2 normalisation.
    What does not carry over is CUT's generator, its convolutional patch encoder, its
    layer set and its weight of 1.0.

    THE CORRESPONDENCE IS THE RESIDUE INDEX, and it is exact here in a way it never is for
    images: CP and LP share a length and a sequence, so residue ``i`` is the same residue
    in both. CUT has to sample matching patch ids; this gets them for free.

    WHY THE DESCRIPTOR CARRIES NO RESIDUE IDENTITY. The sequence is held FIXED by the rest
    of the objective, so a descriptor containing residue type would let the task be solved
    by reading the sequence -- position ``i`` is the only place with that identity -- and
    the term would be satisfied without preserving any geometry. Identity is therefore
    omitted and the descriptor is purely geometric and interfacial.

    WHAT IS DELIBERATELY EXCLUDED FROM SCORING:

    * the first and last TWO residues. Opening the ring is the task; asking the termini to
      keep their local environment is asking the edit not to happen. This is the same
      licensing that :mod:`proteinfoundation.cp2lp.losses` gives the terminal bond.
    * any wrap-around neighbour. A descriptor containing the distance from residue ``i`` to
      residue ``L-1-i`` across the closing bond would encode the ring, and preserving it
      would preserve the macrocycle.
    """

    def __init__(
        self,
        k_neighbors: int = 8,
        temperature: float = 0.07,
        proj_dim: int = 64,
        use_projection: bool = True,
        max_positions: int = 16,
        exclude_terminal: int = 2,
        contact_cutoff_nm: float = 0.8,
    ):
        super().__init__()
        self.k = int(k_neighbors)
        self.temperature = float(temperature)
        self.max_positions = int(max_positions)
        self.exclude_terminal = int(exclude_terminal)
        self.contact_cutoff_nm = float(contact_cutoff_nm)
        # 6 sequence-offset CA distances (+-1, +-2, +-3) + 3 backbone bond lengths + k
        # distances to the source-selected pocket neighbours.
        self.descriptor_dim = 6 + 3 + self.k
        self.projection = None
        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(self.descriptor_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, proj_dim),
            )

    # ------------------------------------------------------------------ descriptors

    def _descriptors(
        self,
        atom37: torch.Tensor,
        pep_mask: torch.Tensor,
        target_ca: torch.Tensor,
        neighbour_idx: torch.Tensor,
    ) -> torch.Tensor:
        """``[b, n, descriptor_dim]`` continuous invariant local descriptor per residue.

        ``neighbour_idx`` is ``[b, n, k]`` receptor indices chosen from the SOURCE
        structure and reused for both sides -- the analogue of CUT reusing one set of patch
        ids for query and key. Recomputing the nearest neighbours on the generated
        structure would compare two different quantities and the positive pair would stop
        being a pair.
        """
        b, n = pep_mask.shape
        pm = pep_mask.bool()
        ca = ca_coords(atom37)  # [b, n, 3]
        d_ca = pairwise_distances(ca, ca)  # [b, n, n]

        feats = []
        for offset in (-3, -2, -1, 1, 2, 3):
            col = torch.zeros(b, n, device=ca.device, dtype=ca.dtype)
            valid = torch.zeros(b, n, dtype=torch.bool, device=ca.device)
            lo = max(0, -offset)
            hi = min(n, n - offset)
            if hi > lo:
                rows = torch.arange(lo, hi, device=ca.device)
                col[:, lo:hi] = d_ca[:, rows, rows + offset]
                valid[:, lo:hi] = pm[:, lo:hi] & pm[:, rows + offset]
            feats.append(col * valid)

        # Intra-residue backbone lengths plus the forward peptide bond. No wrap-around:
        # the (L-1, 0) bond is the ring, and the edit is allowed to break it.
        nn_ = atom37[..., N_IDX, :]
        cc = atom37[..., C_IDX, :]
        cca = atom37[..., CA_IDX, :]
        feats.append(safe_norm(cca - nn_, dim=-1) * pm)
        feats.append(safe_norm(cc - cca, dim=-1) * pm)
        fwd = torch.zeros(b, n, device=ca.device, dtype=ca.dtype)
        if n > 1:
            ok = (pm[:, :-1] & pm[:, 1:]).to(ca.dtype)
            fwd[:, :-1] = safe_norm(nn_[:, 1:] - cc[:, :-1], dim=-1) * ok
        feats.append(fwd)

        per_res = torch.stack(feats, dim=-1)  # [b, n, 9]

        gathered = torch.gather(
            target_ca[:, None, :, :].expand(b, n, target_ca.shape[1], 3),
            2,
            neighbour_idx[..., None].expand(b, n, self.k, 3),
        )  # [b, n, k, 3]
        d_pocket = safe_norm(gathered - ca[:, :, None, :], dim=-1)  # [b, n, k]
        d_pocket = d_pocket.clamp(max=3.0) * pm[..., None]

        return torch.cat([per_res, d_pocket], dim=-1)

    # ------------------------------------------------------------------- selection

    @torch.no_grad()
    def _select(
        self,
        src_atom37: torch.Tensor,
        src_atom_mask: torch.Tensor,
        pep_mask: torch.Tensor,
        target_atom37: torch.Tensor,
        target_atom_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(positions [b, n] bool, neighbour_idx [b, n, k])`` from the SOURCE structure.

        Positions are ranked by the source residue's own soft contact count and the top
        ``max_positions`` kept, so the term spends its budget on the residues that
        actually define the binding mode instead of on solvent-exposed ones whose local
        environment carries no information about the interface.
        """
        pm = pep_mask.bool()
        tm = target_mask.bool()
        b, n = pm.shape

        t_ca = ca_coords(target_atom37)
        d_ca = pairwise_distances(ca_coords(src_atom37), t_ca).masked_fill(~tm[:, None, :], 1e4)
        k = min(self.k, max(int(tm.shape[1]), 1))
        _, idx = torch.topk(d_ca, k=k, dim=-1, largest=False)
        if k < self.k:  # receptor smaller than k: repeat the nearest rather than pad with 0
            idx = torch.cat([idx, idx[..., :1].expand(b, n, self.k - k)], dim=-1)

        d_heavy = heavy_atom_distances(src_atom37, src_atom_mask, target_atom37, target_atom_mask)
        contacts = (soft_contact(d_heavy, self.contact_cutoff_nm) * tm[:, None, :].to(d_heavy.dtype)).sum(-1)

        pos = torch.arange(n, device=pm.device)[None, :].expand(b, n)
        lengths = pm.sum(dim=1, keepdim=True)
        interior = (pos >= self.exclude_terminal) & (pos < (lengths - self.exclude_terminal))
        eligible = pm & interior

        ranked = torch.where(eligible, contacts, torch.full_like(contacts, -1.0))
        take = min(self.max_positions, n)
        top = ranked.topk(k=take, dim=-1).indices
        chosen = torch.zeros_like(pm)
        chosen.scatter_(1, top, True)
        return chosen & eligible, idx

    # --------------------------------------------------------------------- forward

    def forward(
        self,
        gen_atom37: torch.Tensor,
        src_atom37: torch.Tensor,
        src_atom_mask: torch.Tensor,
        pep_mask: torch.Tensor,
        target_atom37: torch.Tensor,
        target_atom_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``(loss, metrics)``. Zero loss with ``nce_n_scored=0`` when nothing is scorable."""
        pm = pep_mask.bool()
        positions, neighbour_idx = self._select(
            src_atom37, src_atom_mask, pm, target_atom37, target_atom_mask, target_mask
        )
        target_ca = ca_coords(target_atom37)

        q_desc = self._descriptors(gen_atom37, pm, target_ca, neighbour_idx)
        # Detached BEFORE the projection, per the design: no gradient may reach the source
        # geometry or the frozen encoder that produced it. The source side is a fixed
        # target, not a thing to be moved towards the generated peptide.
        k_desc = self._descriptors(src_atom37.detach(), pm, target_ca.detach(), neighbour_idx).detach()

        losses = []
        n_scored = 0
        n_skipped = 0
        for i in range(pm.shape[0]):
            sel = positions[i].nonzero(as_tuple=True)[0]
            # Two is the minimum for a contrastive task: one positive and one negative.
            # A single position would make the softmax over one class a constant.
            if sel.numel() < 2:
                n_skipped += 1
                continue
            q = self._project(q_desc[i, sel])
            k = self._project(k_desc[i, sel]).detach()
            losses.append(self._info_nce(q, k))
            n_scored += int(sel.numel())

        device = gen_atom37.device
        if not losses:
            return torch.zeros((), device=device), {
                "nce_n_scored": torch.tensor(0.0),
                "nce_n_skipped_examples": torch.tensor(float(n_skipped)),
            }
        loss = torch.stack(losses).mean()
        return loss, {
            "nce_loss": loss.detach(),
            "nce_n_scored": torch.tensor(float(n_scored)),
            "nce_n_skipped_examples": torch.tensor(float(n_skipped)),
            "nce_positions_per_example": torch.tensor(float(n_scored) / max(1, len(losses))),
        }

    def _project(self, desc: torch.Tensor) -> torch.Tensor:
        """Projection (optional) then L2 normalisation, as in ``PatchSampleF``."""
        h = desc if self.projection is None else self.projection(desc)
        return F.normalize(h, dim=-1, eps=1e-8)

    def _info_nce(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Classify each generated position against its own source position. ``[m, d]``.

        Verbatim in form from the reference ``PatchNCELoss``: the positive logit is the
        matching pair's inner product, the negatives are the full ``q @ k^T`` block with the
        diagonal filled with ``-10`` (it duplicates the positive), the two are concatenated
        with the positive first, divided by the temperature, and scored with cross-entropy
        against label 0.
        """
        m = q.shape[0]
        l_pos = (q * k).sum(dim=-1, keepdim=True)  # [m, 1]
        l_neg = q @ k.transpose(0, 1)  # [m, m]
        l_neg = l_neg.masked_fill(torch.eye(m, dtype=torch.bool, device=q.device), -10.0)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(m, dtype=torch.long, device=q.device)
        return F.cross_entropy(logits, labels)


# ======================================================================= arm config


#: The only legal values of ``cp2lp.addon``.
ADDONS = ("none", "ar1", "noise", "nce")


def validate_addon(addon: str, adversarial_enabled: bool) -> str:
    """Normalises and checks the addon switch. Raises on any contradictory combination.

    Rejecting rather than warning is the point: the five arms are only a clean comparison
    if each one differs from ``onepass_gan`` in exactly one thing, and a config that
    quietly enables a second change -- or enables an addon with no adversary for it to
    modify -- produces a run whose result cannot be attributed.
    """
    a = str(addon or "none").lower()
    if a not in ADDONS:
        raise ValueError(f"cp2lp.addon must be one of {ADDONS}, got {addon!r}")
    if a != "none" and not adversarial_enabled:
        raise ValueError(
            f"cp2lp.addon={a!r} needs cp2lp.adversarial.enabled=true: all three addons "
            "modify the ADVERSARIAL objective (ar1 and noise are discriminator-side; nce "
            "is compared only against the plain one-pass GAN). The no-adversary arm is "
            "`onepass_contactonly`, which must have addon=none."
        )
    return a


def describe_arm(addon: str, adversarial_enabled: bool) -> str:
    if not adversarial_enabled:
        return "onepass_contactonly (no adversary, no addon)"
    return {
        "none": "onepass_gan (plain one-pass GAN -- the reference every addon is measured against)",
        "ar1": "onepass_gan_ar1 (approximate R1 on real features only)",
        "noise": "onepass_gan_noise (adaptive discriminator-input noise)",
        "nce": "onepass_gan_nce (local contrastive retention)",
    }[addon]
