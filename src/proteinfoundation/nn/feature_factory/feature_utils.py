import math

import torch
from jaxtyping import Float
from torch.nn import functional as F

BOND_ORDER_MAP = {
    -1: 0,
    0: 1,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 1,
    6: 2,
    7: 3,
    8: 5,
    9: 6,
}
NUM_BOND_ORDERS = 7


################################
# # Some auxiliary functions # #
################################
# From genie2 code
def sinusoidal_encoding(v, N, D):
    # v: [*]

    # [D]
    k = torch.arange(1, D + 1).to(v.device)

    # [*, D]
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

    # [*, D]
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

    # [*, D]
    enc = torch.zeros_like(sin_enc).to(v.device)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]

    return enc


# From frameflow code
def get_index_embedding(indices, edim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of type integer, shape either [n] or [b, n].
        edim: dimension of the embeddings to create.
        max_len: maximum length.

    Returns:
        positional embedding of shape either [n, edim] or [b, n, edim]
    """
    # indices [n] of [b, n]
    K = torch.arange(edim // 2, device=indices.device)  # [edim / 2]

    if len(indices.shape) == 1:  # [n]
        K = K[None, ...]
    elif len(indices.shape) == 2:  # [b, n]
        K = K[None, None, ...]

    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / (max_len ** (2 * K / edim))).to(indices.device)
    # [n, 1] / [1, edim/2] -> [n, edim/2] or [b, n, 1] / [1, 1, edim/2] -> [b, n, edim/2]
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / (max_len ** (2 * K / edim))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)  # [n, edim]
    return pos_embedding


def get_time_embedding(t: Float[torch.Tensor, "b"], edim: int, max_positions: int = 2000) -> torch.Tensor:
    """
    Code from Frameflow, which got it from
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device)  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def soft_rbf(
    distances: torch.Tensor,
    n_bins: int = 16,
    d_min: float = 0.1,
    d_max: float = 5.0,
    sigma2: float = 0.1,
) -> torch.Tensor:
    """Gaussian RBF encoding of distances (soft alternative to one-hot bins).

    Args:
        distances: Arbitrary leading shape ``[...]`` of pairwise distances (nm).
        n_bins: Number of Gaussian kernels.
        d_min / d_max: Center range in the same units as ``distances``.
        sigma2: Kernel bandwidth (variance).

    Returns:
        Tensor of shape ``[..., n_bins]``.
    """
    centers = torch.linspace(d_min, d_max, n_bins, device=distances.device, dtype=distances.dtype)
    # Broadcast centers over all leading dims.
    expand = (1,) * distances.dim() + (n_bins,)
    centers = centers.view(*expand)
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / sigma2)


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


_SEGMENT_FIX_LEVELS = {"mild": 1, "aggressive": 2, "full": 3}


def segment_fix_level_at_least(batch, minimum: str) -> bool:
    """Return True when the batch's segment-fix tier is >= ``minimum``.

    Tiers (CPSea receptor discontinuity ablations):
      - ``mild``: segment detection + chain_idx uses ``effective_chain_id``
      - ``aggressive``: mild + null seq_sep across segment boundaries
      - ``full``: aggressive + null backbone torsions/bond angles across breaks

    ``segment_fix_level`` is set on each sample by
    ``SegmentAwareResidueFeaturesTransform`` and collated as a list of strings.
    When absent, defaults to ``full`` (current production behavior).
    """
    raw = batch.get("segment_fix_level", "full")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else "full"
    current = _SEGMENT_FIX_LEVELS.get(str(raw).lower(), 3)
    required = _SEGMENT_FIX_LEVELS.get(minimum.lower(), 3)
    return current >= required


def compute_backbone_continuity_mask(batch, n):
    """
    Returns a boolean mask over tensor-adjacent residue pairs (k, k+1) indicating whether
    they are genuinely backbone-consecutive (same continuous polymer segment), for gating
    inter-residue backbone features (torsions, bond angles) that span a residue pair.

    Prefers ``effective_chain_id`` + ``pos_in_segment`` (set by
    ``SegmentAwareResidueFeaturesTransform``) when both are present, since these correctly
    account for physical chain breaks (e.g. a discontinuous receptor/pocket crop), not just
    residue-numbering jumps. A pair is valid only if both residues share the same
    ``effective_chain_id`` AND their ``pos_in_segment`` differ by exactly 1 (this also
    correctly invalidates a pair that becomes tensor-adjacent only because cropping removed
    residues from the middle of a segment).

    Falls back to ``residue_pdb_idx`` consecutiveness (previous behavior) if segment info is
    absent, and finally to "always valid" if no residue-index information is available at
    all (matches the previous default behavior for featureless placeholder data).

    Args:
        batch: dict-like batch, potentially containing "effective_chain_id",
            "pos_in_segment", "residue_pdb_idx", each of shape [b, n].
        n: sequence length (used only for the final fallback).

    Returns:
        Boolean tensor of shape [b, n - 1].
    """
    if (
        segment_fix_level_at_least(batch, "full")
        and "effective_chain_id" in batch
        and "pos_in_segment" in batch
    ):
        eff_chain = batch["effective_chain_id"]  # [b, n]
        pos_in_segment = batch["pos_in_segment"]  # [b, n]
        same_segment = eff_chain[:, 1:] == eff_chain[:, :-1]
        consecutive_pos = (pos_in_segment[:, 1:] - pos_in_segment[:, :-1]) == 1
        return same_segment & consecutive_pos

    if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
        idx = batch["residue_pdb_idx"]  # [b, n]
        return (idx[:, 1:] - idx[:, :-1]) == 1

    if "coords" in batch:
        b, device = batch["coords"].shape[0], batch["coords"].device
    else:
        b, device = 1, "cpu"
    return torch.ones(b, n - 1, dtype=torch.bool, device=device)


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx
