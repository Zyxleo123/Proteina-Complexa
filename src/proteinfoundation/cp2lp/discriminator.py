"""Realness discriminator over bound linear peptide-target complexes.

One logit per complex. It sees the LP's decoded all-atom geometry and its *local*
interface with the pocket, and it never sees the source CP -- that asymmetry is the whole
point: if the discriminator could read the CP it would solve the task by checking whether
the peptide resembles the conditioning input rather than by judging LP realness.

THE SHORTCUT PROBLEM, which is the thing most likely to sink this arm. Real examples are
PepBench linear peptides on PepBench receptors; generated ones are on CPSea receptors. A
discriminator handed rich receptor features can separate the two perfectly without ever
looking at the peptide, and the generator then receives gradient that is pure noise with
respect to LP quality. Three defences, in order of how much they buy:

1. No global receptor description reaches the network. There is no receptor length, no
   whole-chain composition, no absolute coordinate, no chain or target identity. Only the
   ``k`` nearest pocket residues per peptide residue enter, as distances plus an identity
   embedding -- a local, translation- and rotation-invariant view.
2. Pocket identity defaults to a 5-way coarse chemistry class rather than 20-way residue
   type (``pocket_identity="chem"``). Composition statistics are the most memorisable
   receptor signature and this blunts them; set ``"restype"`` only after the probe below
   says there is headroom.
3. ``SHORTCUT PROBE``: a second head reading the pocket features with the peptide
   *masked out*. It is trained on the same labels and its accuracy is logged every step.
   A discriminator that is genuinely judging peptides leaves this probe near chance; a
   probe that climbs is a receptor shortcut, reported as a number instead of discovered
   three days later. Its gradient is detached from the trunk, so it diagnoses the
   features without shaping them, and it never reaches the generator.

Both real and generated peptides arrive as ``autoencoder.decode`` output, so nothing here
can separate them on AE processing artefacts -- a real LP is encoded and decoded before it
is shown, exactly like a generated one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from openfold.np.residue_constants import restype_num

from proteinfoundation.cp2lp.geometry import (
    safe_norm,
    CHEM_CLASS_TABLE,
    N_CHEM_CLASSES,
    backbone_dihedrals,
    ca_coords,
    heavy_atom_distances,
    pairwise_distances,
    radius_of_gyration_nm,
    residue_lengths,
    soft_contact,
    terminal_gap_nm,
)


#: Keys of `build_features` output that are PER-RESIDUE, i.e. shaped [b, n, ...]. The rest
#: ("glob") is per-example. Anything that trims or pads the residue axis -- the reservoir --
#: must know the difference: `glob` is [b, 4], and a shape-based guess would corrupt it on
#: any batch whose padded width happens to be 4.
RESIDUE_FEATURE_KEYS = ("res", "geom", "iface", "mask")


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[b, d] mean over the residue axis of ``[b, n, d]`` under a ``[b, n]`` mask."""
    m = mask.bool().float()[..., None]
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[b, d] max over the residue axis, with masked rows excluded."""
    m = mask.bool()[..., None]
    return x.masked_fill(~m, -1e4).amax(dim=1)


class PeptideInterfaceDiscriminator(nn.Module):
    """Per-complex realness logit for a bound linear peptide.

    Args:
        hidden_dim: trunk width.
        nlayers: transformer encoder layers over peptide residues.
        nheads: attention heads.
        k_pocket: pocket residues retained per peptide residue.
        contact_cutoff_nm: cutoff for the soft interface-contact count.
        pocket_identity: ``"chem"`` (5-way, default) or ``"restype"`` (20-way) or
            ``"none"`` (geometry only). See the shortcut discussion above.
        shortcut_probe: build the receptor-only diagnostic head.
        dropout: applied inside the trunk.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        nlayers: int = 3,
        nheads: int = 6,
        k_pocket: int = 8,
        contact_cutoff_nm: float = 0.8,
        pocket_identity: str = "chem",
        shortcut_probe: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if pocket_identity not in ("chem", "restype", "none"):
            raise ValueError(f"pocket_identity must be chem|restype|none, got {pocket_identity!r}")
        self.k_pocket = int(k_pocket)
        self.contact_cutoff_nm = float(contact_cutoff_nm)
        self.pocket_identity = pocket_identity
        self.register_buffer("chem_table", CHEM_CLASS_TABLE.clone(), persistent=False)

        # The peptide's own residue identity goes in at full 20-way resolution: it is the
        # generator's output, not a dataset label, and sequence realism is something we
        # want judged. Soft (probability-weighted) so gradient reaches the decoder logits.
        self.res_emb = nn.Linear(restype_num + 1, hidden_dim // 4, bias=False)

        if pocket_identity == "chem":
            pocket_id_dim = N_CHEM_CLASSES
        elif pocket_identity == "restype":
            pocket_id_dim = restype_num + 1
        else:
            pocket_id_dim = 0
        self.pocket_id_dim = pocket_id_dim

        # Per-residue geometry: 4 dihedral components, 3 near-neighbour CA distances,
        # 2 intra-residue bond lengths, 1 distance to peptide centroid, 2 terminal flags.
        self.n_geom = 4 + 3 + 2 + 1 + 2
        # Per-residue interface: k distances + k identity vectors + min heavy-atom dist
        # + soft contact count.
        self.n_iface = self.k_pocket * (1 + pocket_id_dim) + 2

        self.in_proj = nn.Linear(hidden_dim // 4 + self.n_geom + self.n_iface, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=nlayers)

        # Global summary: terminal gap, radius of gyration, total soft contacts,
        # mean min-distance to pocket.
        self.n_global = 4
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim + self.n_global),
            nn.Linear(2 * hidden_dim + self.n_global, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.shortcut_probe = None
        if shortcut_probe:
            # Reads ONLY the interface block, peptide identity and geometry removed.
            self.shortcut_probe = nn.Sequential(
                nn.LayerNorm(self.n_iface + 2),
                nn.Linear(self.n_iface + 2, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    # ------------------------------------------------------------------ features

    def _pocket_identity_vectors(self, target_aatype: torch.Tensor | None, b: int, m: int, device) -> torch.Tensor:
        """[b, m, pocket_id_dim] one-hot pocket identity, or an empty slice."""
        if self.pocket_id_dim == 0:
            return torch.zeros(b, m, 0, device=device)
        if target_aatype is None:
            # Unknown identity is a legal state (a receptor loaded without sequence);
            # encode it as the explicit unknown class rather than as a zero vector that
            # would alias with "padding".
            idx = torch.full((b, m), restype_num, dtype=torch.long, device=device)
        else:
            idx = target_aatype.long().clamp(0, restype_num)
        if self.pocket_identity == "chem":
            idx = self.chem_table.to(device)[idx]
            return F.one_hot(idx, N_CHEM_CLASSES).float()
        return F.one_hot(idx, restype_num + 1).float()

    def build_features(
        self,
        pep_atom37: torch.Tensor,
        pep_atom_mask: torch.Tensor,
        pep_mask: torch.Tensor,
        pep_seq_probs: torch.Tensor,
        target_atom37: torch.Tensor,
        target_atom_mask: torch.Tensor,
        target_mask: torch.Tensor,
        target_aatype: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assembles the per-residue and global feature blocks.

        Args:
            pep_atom37: ``[b, n, 37, 3]`` decoded peptide coordinates, nm.
            pep_atom_mask: ``[b, n, 37]`` decoded atom presence.
            pep_mask: ``[b, n]`` residue mask.
            pep_seq_probs: ``[b, n, 21]`` residue-type probabilities (softmax of the
                decoder's logits -- soft so generator gradient reaches the sequence).
            target_atom37/target_atom_mask/target_mask: the receptor, same conventions.
            target_aatype: ``[b, m]`` receptor residue types, optional.
        """
        b, n = pep_mask.shape
        m = target_mask.shape[1]
        device = pep_atom37.device
        pm = pep_mask.bool()
        tm = target_mask.bool()

        # ---- peptide-internal geometry
        dihed = backbone_dihedrals(pep_atom37, pm)  # [b, n, 4]
        ca = ca_coords(pep_atom37)  # [b, n, 3]
        d_ca = pairwise_distances(ca, ca)  # [b, n, n]
        neigh = []
        for offset in (1, 2, 3):
            padded = torch.zeros(b, n, device=device, dtype=ca.dtype)
            if n > offset:
                padded[:, : n - offset] = torch.diagonal(d_ca, offset=offset, dim1=1, dim2=2)
            valid = torch.zeros(b, n, dtype=torch.bool, device=device)
            if n > offset:
                valid[:, : n - offset] = pm[:, offset:] & pm[:, : n - offset]
            neigh.append(padded * valid)
        neigh = torch.stack(neigh, dim=-1)  # [b, n, 3]

        from proteinfoundation.cp2lp.geometry import intra_residue_bond_deviation_nm

        bond_dev, _ = intra_residue_bond_deviation_nm(pep_atom37, pm)  # [b, n, 2]

        mf = pm.float()[..., None]
        centroid = (ca * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp(min=1.0)
        d_centroid = safe_norm(ca - centroid, dim=-1, keepdim=True) * mf  # [b, n, 1]

        lengths = residue_lengths(pm)
        pos = torch.arange(n, device=device)[None, :].expand(b, n)
        is_first = (pos == 0).float()[..., None]
        is_last = (pos == (lengths - 1).clamp(min=0)[:, None]).float()[..., None]

        geom = torch.cat([dihed, neigh, bond_dev, d_centroid, is_first, is_last], dim=-1)

        # ---- local interface
        t_ca = ca_coords(target_atom37)  # [b, m, 3]
        d_pep_t = pairwise_distances(ca, t_ca)  # [b, n, m]
        d_pep_t = d_pep_t.masked_fill(~tm[:, None, :], 1e4)
        k = min(self.k_pocket, max(m, 1))
        topk_d, topk_i = torch.topk(d_pep_t, k=k, dim=-1, largest=False)  # [b, n, k]
        # Pad the k axis when the receptor has fewer residues than k, so the feature
        # width is a property of the model and not of the example.
        if k < self.k_pocket:
            pad = self.k_pocket - k
            topk_d = F.pad(topk_d, (0, pad), value=1e4)
            topk_i = F.pad(topk_i, (0, pad), value=0)
        # 1e4 placeholders would swamp a LayerNorm; map absent neighbours to a finite
        # "far" value that the network can still read as far.
        far = 3.0
        topk_d_feat = topk_d.clamp(max=far)

        pid = self._pocket_identity_vectors(target_aatype, b, m, device)  # [b, m, pid]
        if self.pocket_id_dim > 0:
            gathered = torch.gather(
                pid[:, None, :, :].expand(b, n, m, self.pocket_id_dim),
                2,
                topk_i[..., None].expand(b, n, self.k_pocket, self.pocket_id_dim),
            )  # [b, n, k, pid]
            gathered = gathered * (topk_d < far)[..., None].float()
            pid_feat = gathered.reshape(b, n, self.k_pocket * self.pocket_id_dim)
        else:
            pid_feat = torch.zeros(b, n, 0, device=device)

        d_heavy = heavy_atom_distances(pep_atom37, pep_atom_mask, target_atom37, target_atom_mask)
        min_heavy = d_heavy.amin(dim=-1).clamp(max=far)[..., None]  # [b, n, 1]
        contacts = (soft_contact(d_heavy, self.contact_cutoff_nm) * tm[:, None, :].float()).sum(-1, keepdim=True)

        iface = torch.cat([topk_d_feat, pid_feat, min_heavy, contacts], dim=-1)

        # ---- global
        glob = torch.stack(
            [
                terminal_gap_nm(pep_atom37, pm),
                radius_of_gyration_nm(pep_atom37, pm),
                (contacts.squeeze(-1) * pm.float()).sum(dim=1),
                _masked_mean(min_heavy, pm).squeeze(-1),
            ],
            dim=-1,
        )

        return {
            "res": pep_seq_probs,
            "geom": geom * mf,
            "iface": iface * mf,
            "glob": glob,
            "mask": pm,
        }

    # ------------------------------------------------------------------ forward

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        """[b] realness logit from a feature dict produced by :meth:`build_features`."""
        mask = feats["mask"]
        h = torch.cat([self.res_emb(feats["res"]), feats["geom"], feats["iface"]], dim=-1)
        h = self.in_proj(h)
        h = self.trunk(h, src_key_padding_mask=~mask)
        h = torch.nan_to_num(h)
        pooled = torch.cat([_masked_mean(h, mask), _masked_max(h, mask), feats["glob"]], dim=-1)
        return self.head(pooled).squeeze(-1)

    def probe_forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """[b] receptor-only diagnostic logit, or None when the probe is disabled.

        Detached from the feature tensors so the probe measures the interface block
        without gradient-shaping it or the generator.
        """
        if self.shortcut_probe is None:
            return None
        mask = feats["mask"]
        iface = _masked_mean(feats["iface"], mask).detach()
        glob = feats["glob"][:, 2:].detach()  # contact count + mean min-distance only
        return self.shortcut_probe(torch.cat([iface, glob], dim=-1)).squeeze(-1)


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Standard hinge discriminator loss. Real should exceed +1, fake should fall below -1.

    The hinge (rather than the saturating BCE) is the recommended starting point here
    because its gradient vanishes once an example is confidently on the right side, which
    is what keeps a discriminator that is winning from running away from the generator.
    """
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator term for the hinge GAN."""
    return -fake_logits.mean()


@torch.no_grad()
def discriminator_accuracy(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> float:
    """Fraction of examples the discriminator sides correctly on. 0.5 is chance."""
    correct = (real_logits > 0).float().sum() + (fake_logits <= 0).float().sum()
    return float(correct / (real_logits.numel() + fake_logits.numel()))
