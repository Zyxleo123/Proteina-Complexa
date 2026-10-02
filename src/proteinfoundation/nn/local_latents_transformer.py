import torch
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK

from proteinfoundation.nn.feature_factory.concat_feature_factory import ConcatFeaturesFactory
from proteinfoundation.nn.feature_factory.concat_pair_feature_factory import ConcatPairFeaturesFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition, MultiheadCrossAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.protein_transformer import PairReprBuilder


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)  # [21, 37]


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.

    Now supports concat features for motif and/or target conditioning.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_tri_attn = kwargs.get("use_tri_attn", False)
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # Concat features for motif and/or target
        concat_config = kwargs.get("concat_features", {})
        self.use_concat = (
            concat_config.get("enable_motif", False)
            or concat_config.get("enable_target", False)
            or concat_config.get("enable_ligand", False)
        )

        if self.use_concat:
            self.concat_factory = ConcatFeaturesFactory(
                enable_motif=concat_config.get("enable_motif", False),
                enable_target=concat_config.get("enable_target", False),
                enable_ligand=concat_config.get("enable_ligand", False),
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                **kwargs,
            )

            # Check for advanced pair representation mode
            self.use_advanced_pair = (
                (concat_config.get("enable_motif", False) and concat_config.get("motif_pair_features", False))
                or (concat_config.get("enable_target", False) and concat_config.get("target_pair_features", False))
                or (concat_config.get("enable_ligand", False) and concat_config.get("ligand_pair_features", False))
            )

            if self.use_advanced_pair:
                # Initialize pair features factory for extended pair representations
                self.concat_pair_factory = ConcatPairFeaturesFactory(
                    enable_motif=concat_config.get("enable_motif", False),
                    enable_target=concat_config.get("enable_target", False),
                    enable_ligand=concat_config.get("enable_ligand", False),
                    dim_pair_out=kwargs["pair_repr_dim"],
                    **kwargs,
                )
                from loguru import logger

                logger.info("Enabled advanced pair representation with cross-sequence features")
        else:
            self.concat_factory = None
            self.use_advanced_pair = False

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        self.use_target_cross_attn = kwargs.get("use_target_cross_attn", False)
        if self.use_target_cross_attn:
            assert not self.use_concat, "use_target_cross_attn requires use_concat to be False"
            self.target_factory = FeatureFactory(
                feats=kwargs["feats_target"],
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                mode="target",
                **kwargs,
            )
            self.target2binder_cross_attention_layer = torch.nn.ModuleList(
                [
                    MultiheadCrossAttnAndTransition(
                        dim_token_a=kwargs["token_dim"],
                        dim_token_b=kwargs["token_dim"],
                        nheads=kwargs["nheads"],
                        dim_cond=kwargs["dim_cond"],
                        residual_mha=True,
                        residual_transition=True,
                        use_qkln=self.use_qkln,
                    )
                    for _ in range(self.nlayers)
                ]
            )

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            use_tri_attn=self.use_tri_attn,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

        # Optional RESIDUAL readouts for the one-pass CP->LP editor. A second pair of heads
        # rather than a reinterpretation of the two above, for two reasons:
        #
        #  * the pretrained heads emit a flow VELOCITY. Reading that as an edit displacement
        #    would start the editor at `x_src + v(x_src, t=0.5)`, which is a large arbitrary
        #    jump -- not "return the source unchanged". The plan for this arm is that step
        #    zero is exactly the AE round trip of the source CP, and only a zero-initialised
        #    head gives that.
        #  * the legacy differentiable-rollout mode still needs `v`, so replacing the heads
        #    in place would fork the checkpoint format between the two modes.
        #
        # ZERO INIT is load-bearing, not tidiness: it makes the editor's first output the
        # identity edit, so the arm starts from a known-good structure instead of from noise.
        # The cost is that the trunk receives NO gradient on the very first step (d(out)/d(h)
        # is zero through a zero weight); it starts flowing as soon as the head itself has
        # moved, which `cp2lp_onepass_smoke.py` checks explicitly rather than assuming.
        #
        # Cost when enabled: two extra Linears per forward, both computed alongside the
        # velocity heads. Not worth branching the forward to save.
        self.enable_editor_residual = bool(kwargs.get("enable_editor_residual", False))
        if self.enable_editor_residual:
            self.editor_latents_residual = torch.nn.Sequential(
                torch.nn.LayerNorm(self.token_dim),
                torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
            )
            self.editor_ca_residual = torch.nn.Sequential(
                torch.nn.LayerNorm(self.token_dim),
                torch.nn.Linear(self.token_dim, 3, bias=False),
            )
            torch.nn.init.zeros_(self.editor_latents_residual[1].weight)
            torch.nn.init.zeros_(self.editor_ca_residual[1].weight)
        else:
            self.editor_latents_residual = None
            self.editor_ca_residual = None

        # ---------------- backbone-only output heads: the two representation arms.
        #
        # These replace the CA residual + frozen AE decoder output path entirely. The arm
        # emits N/CA/C/O itself, so nothing downstream reconstructs geometry and the AE round
        # trip stops being the ceiling on backbone validity.
        #
        # NEWLY INITIALISED, never spliced from the pretrained columns. `ca_linear` emits a
        # 3-vector flow velocity for CA; its weights carry no information about where a
        # carbonyl oxygen goes or what a phi residual should be, so seeding a 12- or
        # 9-channel head from it would import a semantically unrelated prior and call it a
        # warm start. Zero init is the warm start that is actually true here: it makes the
        # arm's first output exactly the source backbone.
        #
        # ZERO INIT COSTS THE FIRST STEP'S TRUNK GRADIENT, as with the CA residual above:
        # d(out)/d(h) is zero through a zero weight, so the trunk only starts receiving
        # gradient once the head itself has moved. Checked explicitly in the smoke rather
        # than assumed.
        self.editor_backbone_mode = str(kwargs.get("editor_backbone_mode", "none"))
        self.editor_bb_cartesian = None
        self.editor_bb_torsion = None
        self.editor_bb_root = None
        if self.editor_backbone_mode == "cartesian":
            # Arm A: a free Cartesian residual per backbone atom. 4 atoms x 3 = 12 channels.
            # Deliberately the FULL backbone, not CA-only -- a CA-only head followed by the
            # AE decoder is the arm that already exists and already fails.
            self.editor_bb_cartesian = torch.nn.Sequential(
                torch.nn.LayerNorm(self.token_dim),
                torch.nn.Linear(self.token_dim, 12, bias=False),
            )
            torch.nn.init.zeros_(self.editor_bb_cartesian[1].weight)
        elif self.editor_backbone_mode == "torsion":
            # Arm B: residual rotations for phi/psi/omega, each as an unnormalised (sin, cos)
            # pair about the identity -- 6 channels. A plain scalar per angle would regress
            # across the +-pi seam, where two angles a thousandth of a radian apart differ in
            # squared error by (2 pi)^2 and the gradient points the wrong way.
            self.editor_bb_torsion = torch.nn.Sequential(
                torch.nn.LayerNorm(self.token_dim),
                torch.nn.Linear(self.token_dim, 6, bias=False),
            )
            torch.nn.init.zeros_(self.editor_bb_torsion[1].weight)
            # Root placement: 3 translation + 3 axis-angle, pooled to one per example. Arm A
            # can move the whole peptide freely through its per-atom residual, so withholding
            # this from Arm B would hand Arm A a degree of freedom Arm B does not have and
            # make "which representation is better" partly "which one could translate".
            self.editor_bb_root = torch.nn.Sequential(
                torch.nn.LayerNorm(self.token_dim),
                torch.nn.Linear(self.token_dim, 6, bias=False),
            )
            torch.nn.init.zeros_(self.editor_bb_root[1].weight)
        elif self.editor_backbone_mode != "none":
            raise ValueError(
                f"nn.editor_backbone_mode must be 'none', 'cartesian' or 'torsion'; "
                f"got {self.editor_backbone_mode!r}"
            )

        # Optional oracle peptide-interface surface conditioning. Zero-init gates keep
        # pretrained receptor-only checkpoints bit-identical until the gates open.
        surface_cfg = kwargs.get("surface", {}) or {}
        self.enable_surface = bool(kwargs.get("enable_surface", False))
        self.surface_gate_layers = tuple(surface_cfg.get("gate_layers", [3, 7, 11]))
        if self.enable_surface:
            from proteinfoundation.nn.surface.encoder import (
                BinderSurfacePairFeatures,
                GatedSurfaceCrossAttention,
                SurfaceEncoder,
            )

            rbf_n = int(surface_cfg.get("rbf_n", 16))
            pair_dim = int(surface_cfg.get("pair_dim", kwargs["pair_repr_dim"]))
            gate_init = float(surface_cfg.get("gate_init", 0.0))
            # `gate_learnable: false` pins `g` at `gate_init` so the surface encoder always
            # gets a nonvanishing gradient. See GatedSurfaceCrossAttention's docstring for the
            # measured zero-init deadlock this exists to break.
            gate_learnable = bool(surface_cfg.get("gate_learnable", True))
            self.surface_encoder = SurfaceEncoder(
                token_dim=self.token_dim,
                pair_dim=pair_dim,
                nheads=kwargs["nheads"],
                rbf_n=rbf_n,
                use_qkln=self.use_qkln,
            )
            self.binder_surface_pair_feats = BinderSurfacePairFeatures(
                pair_dim=pair_dim, rbf_n=rbf_n
            )
            self.surface_cross_layers = torch.nn.ModuleDict(
                {
                    str(i): GatedSurfaceCrossAttention(
                        dim_token=self.token_dim,
                        dim_pair=pair_dim,
                        nheads=kwargs["nheads"],
                        use_qkln=self.use_qkln,
                        gate_init=gate_init,
                        gate_learnable=gate_learnable,
                    )
                    for i in self.surface_gate_layers
                }
            )
        else:
            self.surface_encoder = None
            self.binder_surface_pair_feats = None
            self.surface_cross_layers = None

        # self.linear_out = torch.nn.Sequential(
        #     torch.nn.LayerNorm(self.token_dim),
        #     torch.nn.Linear(self.token_dim, kwargs["latent_dim"] + 3, bias=False),
        # )

    # @torch.compile
    def forward(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean
        orig_mask = mask.clone()  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Initial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]
        # Store original dimensions
        b, n_orig, _ = seqs.shape
        # Extend sequence representation with concat features if available
        if self.use_concat:
            seqs, mask = self.concat_factory(
                input, seqs, mask
            )  # [b, n_extended, token_dim], [b, n_extended], [b, n_extended]
            n_extended = seqs.shape[1]
            n_concat = n_extended - n_orig

            # Extend conditioning with zeros for concat features
            if n_concat > 0:
                zero_cond = torch.zeros(b, n_concat, c.shape[-1], device=seqs.device)
                c = torch.cat([c, zero_cond], dim=1)  # [b, n_extended, dim_cond]
        else:
            n_extended = n_orig
            n_concat = 0

        # Compute pair representation
        if self.use_concat and self.use_advanced_pair:  # and n_concat > 0:
            # Advanced mode - compute pair features for extended sequence
            pair_rep = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]
            pair_rep = self.concat_pair_factory(input, pair_rep, orig_mask)  # [b, n_extended, n_extended, pair_dim]
        else:
            # Simple mode - compute pair representation for original sequence only
            pair_rep = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]

            # Extend pair representation with zeros if we have concat features
            if n_concat > 0:
                dim_pair = pair_rep.shape[-1]

                # Pad first dimension: [b, n_orig, n_orig, d] -> [b, n_extended, n_orig, d]
                zero_pad_1 = torch.zeros(b, n_concat, n_orig, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_1], dim=1)

                # Pad second dimension: [b, n_extended, n_orig, d] -> [b, n_extended, n_extended, d]
                zero_pad_2 = torch.zeros(b, n_extended, n_concat, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_2], dim=2)

        if self.use_target_cross_attn:
            target_rep = self.target_factory(input)  # [b, n_target, token_dim]
            target_mask = input["seq_target_mask"]
            target_rep = target_rep * target_mask[..., None]  # [b, n_target, token_dim]

        # Surface conditioning: encode once; binder↔surface pair feats track noisy CA each step.
        h_surface = None
        binder_surface_pair = None
        surface_mask = None
        if self.enable_surface:
            surface_xyz = input["surface_xyz"]
            surface_normals = input["surface_normals"]
            surface_mask = input["surface_mask"].bool()
            surface_distance = input["surface_distance"]
            h_surface = self.surface_encoder(
                surface_xyz, surface_normals, surface_mask, surface_distance
            )
            t_bb = input["t"]["bb_ca"]
            if t_bb.dim() > 1:
                t_bb = t_bb.reshape(t_bb.shape[0], -1)[:, 0]
            binder_surface_pair = self.binder_surface_pair_feats(
                binder_xyz=input["x_t"]["bb_ca"],
                surface_xyz=surface_xyz,
                surface_normals=surface_normals,
                binder_mask=orig_mask,
                surface_mask=surface_mask,
                t=t_bb,
            )

        # if seqs.shape[1] != c.shape[1]: pdb has weird inter padding that we do not handle
        # Run trunk
        for i in range(self.nlayers):
            if self.use_target_cross_attn:
                seqs = self.target2binder_cross_attention_layer[i](
                    seqs, target_rep, c, mask, target_mask
                )  # [b, n_extended, token_dim], [b, n_target, token_dim]

            # try:
            seqs = self.transformer_layers[i](seqs, pair_rep, c, mask)  # [b, n_extended, token_dim]

            if (
                self.enable_surface
                and i in self.surface_gate_layers
                and h_surface is not None
            ):
                # Update binder tokens only; leave concatenated target tokens untouched.
                binder_tokens = seqs[:, :n_orig, :]
                binder_tokens = self.surface_cross_layers[str(i)](
                    binder_tokens,
                    h_surface,
                    binder_surface_pair,
                    orig_mask,
                    surface_mask,
                )
                seqs = torch.cat([binder_tokens, seqs[:, n_orig:, :]], dim=1)

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n_extended, n_extended, pair_dim]

        # Get outputs
        local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n_extended, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n_extended, 3]

        # Trim back to original sequence length (remove concat features) if we extended
        if n_concat > 0:
            local_latents_out = local_latents_out[:, :n_orig, :] * orig_mask[:, :, None]

            ca_nm_out = ca_nm_out[:, :n_orig, :] * orig_mask[:, :, None]

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {self.output_param["local_latents"]: local_latents_out}

        # One-pass editor residuals, under a key of their own so nothing that reads `v`
        # can pick them up by accident. `delta` is an EDIT DISPLACEMENT, in the same units
        # as the state it is added to (nm for bb_ca, latent units for local_latents); the
        # caller does the addition, see `proteinfoundation.cp2lp.editor`.
        if self.enable_editor_residual:
            d_lat = self.editor_latents_residual(seqs) * mask[..., None]
            d_ca = self.editor_ca_residual(seqs) * mask[..., None]
            if n_concat > 0:
                d_lat = d_lat[:, :n_orig, :] * orig_mask[:, :, None]
                d_ca = d_ca[:, :n_orig, :] * orig_mask[:, :, None]
            nn_out["bb_ca"]["delta"] = d_ca
            nn_out["local_latents"]["delta"] = d_lat

        # Backbone-arm outputs, under their own key. `proteinfoundation.cp2lp.bb_arms` turns
        # these into a backbone; nothing here interprets them, so the representation and the
        # readout stay in one file each.
        if self.editor_backbone_mode != "none":
            bb_out = {}
            if self.editor_bb_cartesian is not None:
                d_bb = self.editor_bb_cartesian(seqs) * mask[..., None]
                if n_concat > 0:
                    d_bb = d_bb[:, :n_orig, :] * orig_mask[:, :, None]
                bb_out["delta_cartesian"] = d_bb.reshape(*d_bb.shape[:2], 4, 3)
            if self.editor_bb_torsion is not None:
                d_tor = self.editor_bb_torsion(seqs) * mask[..., None]
                if n_concat > 0:
                    d_tor = d_tor[:, :n_orig, :] * orig_mask[:, :, None]
                bb_out["delta_torsion"] = d_tor
                # The root is ONE transform for the whole peptide, so the per-residue head
                # output is masked-mean pooled. A sum would scale with length and make the
                # same head output mean a 10x larger shift on a 30-mer than on a 3-mer.
                r = self.editor_bb_root(seqs) * mask[..., None]
                if n_concat > 0:
                    r = r[:, :n_orig, :] * orig_mask[:, :, None]
                    mm = orig_mask[:, :, None].to(r.dtype)
                else:
                    mm = mask[..., None].to(r.dtype)
                bb_out["delta_root"] = r.sum(1) / mm.sum(1).clamp(min=1.0)
            nn_out["backbone"] = bb_out
        return nn_out
