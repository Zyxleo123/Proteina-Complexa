#!/usr/bin/env python3
"""Fast CPU/GPU ablation: do surface-gate levers actually move g and feed encoder grads?

Uses the tiny transformer from test_surface_conditioning (no AE / no data loader).
Four arms, 40 Adam steps each, synthetic loss = ||pred_CA - surface_centroid||.

  .venv/bin/python script_utils/probe_surface_gate_levers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "script_utils"))

from test_surface_conditioning import _fake_batch, _tiny_nn_kwargs  # noqa: E402


def _gates(nn) -> list[float]:
    return [float(layer.gate.detach()) for layer in nn.surface_cross_layers.values()]


def _encoder_grad_norm(nn) -> float:
    g = nn.surface_encoder.type_emb.grad
    return float(g.detach().norm()) if g is not None else 0.0


def _run_arm(
    name: str,
    *,
    gate_init: float,
    surface_lr_scale: float,
    aux_weight: float,
    steps: int = 40,
    lr: float = 1e-2,
) -> dict:
    from proteinfoundation.nn.local_latents_transformer import LocalLatentsTransformer

    kwargs = _tiny_nn_kwargs(enable_surface=True)
    kwargs["surface"] = {**(kwargs.get("surface") or {}), "gate_init": gate_init}
    nn = LocalLatentsTransformer(**kwargs)
    batch = _fake_batch(enable_surface=True)

    # Param groups: surface stack vs rest (mirrors proteina.configure_optimizers).
    surface_mods = (
        list(nn.surface_encoder.parameters())
        + list(nn.binder_surface_pair_feats.parameters())
        + list(nn.surface_cross_layers.parameters())
    )
    surf_ids = {id(p) for p in surface_mods}
    surf_params = [p for p in nn.parameters() if p.requires_grad and id(p) in surf_ids]
    other_params = [p for p in nn.parameters() if p.requires_grad and id(p) not in surf_ids]
    opt = torch.optim.Adam(
        [
            {"params": other_params, "lr": lr},
            {"params": surf_params, "lr": lr * surface_lr_scale},
        ]
    )

    # Target: binder CA should sit on the surface centroid (nm).
    target = (
        batch["surface_xyz"] * batch["surface_mask"][..., None]
    ).sum(1) / batch["surface_mask"].sum(1).clamp_min(1).unsqueeze(-1)

    g0 = _gates(nn)
    enc_grads = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = nn(batch)
        # Tiny nn predicts velocity-ish; treat bb_ca["v"] as a stand-in CA offset from x_t.
        pred = batch["x_t"]["bb_ca"] + out["bb_ca"]["v"]
        mask = batch["mask"].bool()
        pred_com = (pred * mask[..., None]).sum(1) / mask.sum(1).clamp_min(1).unsqueeze(-1)
        loss = (pred_com - target).pow(2).mean()
        if aux_weight > 0:
            # Soft Chamfer pred ↔ surface (same shape as training aux).
            dist = (pred[:, :, None, :] - batch["surface_xyz"][:, None, :, :]).norm(dim=-1)
            dist = dist.masked_fill(~batch["surface_mask"][:, None, :], 1e6)
            dist = dist.masked_fill(~mask[:, :, None], 1e6)
            loss = loss + aux_weight * 0.5 * (dist.min(-1).values.mean() + dist.min(1).values.mean())
        loss.backward()
        enc_grads.append(_encoder_grad_norm(nn))
        opt.step()

    g1 = _gates(nn)
    return {
        "name": name,
        "gate_init": gate_init,
        "g0": g0,
        "g1": g1,
        "abs_mean_g1": sum(abs(x) for x in g1) / len(g1),
        "delta_abs_mean": sum(abs(a - b) for a, b in zip(g1, g0)) / len(g0),
        "enc_grad_mean": sum(enc_grads) / len(enc_grads),
        "enc_grad_last": enc_grads[-1],
        "loss": float(loss.detach()),
    }


def main() -> None:
    torch.manual_seed(0)
    arms = [
        _run_arm("A_control_zero", gate_init=0.0, surface_lr_scale=1.0, aux_weight=0.0),
        _run_arm("B_gate_init_0.1", gate_init=0.1, surface_lr_scale=1.0, aux_weight=0.0),
        _run_arm("C_lr_scale_10", gate_init=0.0, surface_lr_scale=10.0, aux_weight=0.0),
        _run_arm("D_aux_0.1", gate_init=0.0, surface_lr_scale=1.0, aux_weight=0.1),
        _run_arm("E_all_levers", gate_init=0.1, surface_lr_scale=10.0, aux_weight=0.1),
    ]
    print(f"{'arm':<18} {'|g|mean':>10} {'Δ|g|':>10} {'enc∇mean':>12} {'enc∇last':>12}")
    for r in arms:
        print(
            f"{r['name']:<18} {r['abs_mean_g1']:10.4e} {r['delta_abs_mean']:10.4e} "
            f"{r['enc_grad_mean']:12.4e} {r['enc_grad_last']:12.4e}"
        )
        print(f"  g0={['%.3e'%x for x in r['g0']]}  g1={['%.3e'%x for x in r['g1']]}")


if __name__ == "__main__":
    main()
