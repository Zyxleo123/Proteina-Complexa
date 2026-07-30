"""Surface encoder and gated binder→surface cross-attention.

Geometry-only (v1): type embedding + receptor-distance scalar; detailed shape via
SE(3)-invariant pair features (RBF distances + normal angles), not raw normals in an MLP.
"""

from __future__ import annotations

import torch
from torch import nn

from proteinfoundation.nn.feature_factory.feature_utils import soft_rbf
from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention


def _unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


class IntraSurfacePairFeatures(nn.Module):
    """SE(3)-invariant pair features for surface points (i, j).

    ``[ RBF(|p_j-p_i|), n_i·r̂_ij, n_j·(-r̂_ij), n_i·n_j ]`` projected to ``pair_dim``.
    """

    def __init__(self, pair_dim: int, rbf_n: int = 16, d_min: float = 0.1, d_max: float = 5.0):
        super().__init__()
        self.rbf_n = rbf_n
        self.d_min = d_min
        self.d_max = d_max
        in_dim = rbf_n + 3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        normals: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            xyz: [B, M, 3] surface points (nm)
            normals: [B, M, 3] unit normals
            mask: [B, M] bool
        Returns:
            pair: [B, M, M, pair_dim]
        """
        # r̂_ij points from i to j: p_j - p_i
        diff = xyz[:, None, :, :] - xyz[:, :, None, :]  # [B, M, M, 3]
        dist = diff.norm(dim=-1).clamp_min(1e-8)  # [B, M, M]
        rhat = diff / dist[..., None]
        ni = normals[:, :, None, :]
        nj = normals[:, None, :, :]
        ni_r = (ni * rhat).sum(-1)
        nj_mr = (nj * (-rhat)).sum(-1)
        ni_nj = (ni * nj).sum(-1)
        rbf = soft_rbf(dist, n_bins=self.rbf_n, d_min=self.d_min, d_max=self.d_max)
        feats = torch.cat([rbf, ni_r[..., None], nj_mr[..., None], ni_nj[..., None]], dim=-1)
        pair = self.proj(feats)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        return pair * pair_mask[..., None].to(dtype=pair.dtype)


class SurfaceEncoder(nn.Module):
    """Encode M surface points to ``H_S`` [B, M, token_dim]."""

    def __init__(
        self,
        token_dim: int,
        pair_dim: int,
        nheads: int,
        rbf_n: int = 16,
        use_qkln: bool = True,
    ):
        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(token_dim))
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, token_dim // 4),
            nn.ReLU(),
            nn.Linear(token_dim // 4, token_dim),
        )
        self.pair_feats = IntraSurfacePairFeatures(pair_dim=pair_dim, rbf_n=rbf_n)
        self.self_attn = PairBiasAttention(
            node_dim=token_dim,
            dim_head=token_dim // nheads,
            heads=nheads,
            bias=True,
            dim_out=token_dim,
            qkln=use_qkln,
            pair_dim=pair_dim,
        )
        self.out_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        surface_xyz: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_mask: torch.Tensor,
        surface_distance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            surface_xyz / normals: [B, M, 3]
            surface_mask: [B, M] bool
            surface_distance: [B, M] receptor distance (nm)
        """
        h = self.type_emb.view(1, 1, -1) + self.dist_mlp(surface_distance.unsqueeze(-1))
        h = h * surface_mask[..., None].to(dtype=h.dtype)
        pair = self.pair_feats(surface_xyz, surface_normals, surface_mask)
        pair_mask = surface_mask[:, :, None] & surface_mask[:, None, :]
        h = h + self.self_attn(node_feats=h, pair_feats=pair, mask=pair_mask)
        h = self.out_norm(h) * surface_mask[..., None].to(dtype=h.dtype)
        return h


class BinderSurfacePairFeatures(nn.Module):
    """Binder residue ↔ surface point pair features from noisy CA ``x_t``.

    ``[ RBF(|x_k - p_i|), n_i · (x_k - p_i)/|·|, t ]`` → ``pair_dim``.
    """

    def __init__(self, pair_dim: int, rbf_n: int = 16, d_min: float = 0.1, d_max: float = 5.0):
        super().__init__()
        self.rbf_n = rbf_n
        self.d_min = d_min
        self.d_max = d_max
        in_dim = rbf_n + 2  # rbf + normal alignment + t
        self.proj = nn.Sequential(
            nn.Linear(in_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim),
        )

    def forward(
        self,
        binder_xyz: torch.Tensor,
        surface_xyz: torch.Tensor,
        surface_normals: torch.Tensor,
        binder_mask: torch.Tensor,
        surface_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            binder_xyz: [B, N, 3] noisy CA (nm)
            surface_xyz / normals: [B, M, 3]
            binder_mask: [B, N]
            surface_mask: [B, M]
            t: [B] flow time for bb_ca
        Returns:
            pair: [B, N, M, pair_dim]
        """
        diff = binder_xyz[:, :, None, :] - surface_xyz[:, None, :, :]  # [B, N, M, 3]
        dist = diff.norm(dim=-1).clamp_min(1e-8)
        rhat = diff / dist[..., None]
        n_align = (surface_normals[:, None, :, :] * rhat).sum(-1)  # [B, N, M]
        rbf = soft_rbf(dist, n_bins=self.rbf_n, d_min=self.d_min, d_max=self.d_max)
        t_feat = t.to(dtype=rbf.dtype).view(-1, 1, 1, 1).expand(-1, dist.shape[1], dist.shape[2], 1)
        feats = torch.cat([rbf, n_align[..., None], t_feat], dim=-1)
        pair = self.proj(feats)
        pair_mask = binder_mask[:, :, None] & surface_mask[:, None, :]
        return pair * pair_mask[..., None].to(dtype=pair.dtype)


class PairBiasedCrossAttention(nn.Module):
    """Cross-attention: Q from binder, K/V from surface, bias from binder–surface pairs."""

    def __init__(self, dim_token: int, dim_pair: int, nheads: int, use_qkln: bool = True):
        super().__init__()
        assert dim_token % nheads == 0
        self.nheads = nheads
        dim_head = dim_token // nheads
        self.scale = dim_head**-0.5
        self.norm_q = nn.LayerNorm(dim_token)
        self.norm_kv = nn.LayerNorm(dim_token)
        self.norm_pair = nn.LayerNorm(dim_pair)
        self.to_q = nn.Linear(dim_token, dim_token, bias=True)
        self.to_k = nn.Linear(dim_token, dim_token, bias=True)
        self.to_v = nn.Linear(dim_token, dim_token, bias=True)
        self.to_g = nn.Linear(dim_token, dim_token)
        self.to_bias = nn.Linear(dim_pair, nheads, bias=False)
        self.to_out = nn.Linear(dim_token, dim_token)
        self.q_ln = nn.LayerNorm(dim_head) if use_qkln else nn.Identity()
        self.k_ln = nn.LayerNorm(dim_head) if use_qkln else nn.Identity()

    def forward(
        self,
        binder: torch.Tensor,
        surface: torch.Tensor,
        pair: torch.Tensor,
        binder_mask: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = binder.shape
        m = surface.shape[1]
        h = self.nheads
        q = self.to_q(self.norm_q(binder)).view(b, n, h, d // h).transpose(1, 2)  # [B,h,N,dh]
        k = self.to_k(self.norm_kv(surface)).view(b, m, h, d // h).transpose(1, 2)
        v = self.to_v(self.norm_kv(surface)).view(b, m, h, d // h).transpose(1, 2)
        q = self.q_ln(q)
        k = self.k_ln(k)
        bias = self.to_bias(self.norm_pair(pair)).permute(0, 3, 1, 2)  # [B,h,N,M]
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale + bias
        attn_mask = binder_mask[:, None, :, None] & surface_mask[:, None, None, :]
        sim = sim.masked_fill(~attn_mask, torch.finfo(sim.dtype).min)
        attn = torch.softmax(sim, dim=-1)
        # Rows with no valid surface keys become NaN after softmax; zero them.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("bhij,bhjd->bhid", attn, v).transpose(1, 2).contiguous().view(b, n, d)
        g = torch.sigmoid(self.to_g(binder))
        out = self.to_out(g * out)
        return out * binder_mask[..., None].to(dtype=out.dtype)


class GatedSurfaceCrossAttention(nn.Module):
    """``h ← h + g · CrossAttn(h, H_S)`` with scalar gate ``g``.

    Default ``gate_init=0`` keeps a pretrained receptor-only warm-start bit-identical, but
    then ``∂L/∂Δ = g = 0`` so the surface encoder never receives gradients until ``g`` moves.
    Set ``gate_init`` > 0 (e.g. 0.1) to unstick the encoder, or rely on an aux loss that
    can open ``g`` from zero.
    """

    def __init__(
        self,
        dim_token: int,
        dim_pair: int,
        nheads: int,
        use_qkln: bool = True,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.cross = PairBiasedCrossAttention(
            dim_token=dim_token, dim_pair=dim_pair, nheads=nheads, use_qkln=use_qkln
        )
        self.gate = nn.Parameter(torch.full((), float(gate_init)))

    def forward(
        self,
        binder: torch.Tensor,
        surface: torch.Tensor,
        pair: torch.Tensor,
        binder_mask: torch.Tensor,
        surface_mask: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.cross(binder, surface, pair, binder_mask, surface_mask)
        return binder + self.gate * delta
