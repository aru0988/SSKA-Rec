"""BitFit, SSF, and residual adapters on the matched frozen-U stage2 head."""

import torch
from torch import nn
from torch.nn import functional as F

from .stage2_models import Stage2Model, build_model as build_stage2_model


VARIANTS = ("tri_bitfit", "tri_ssf", "tri_adapter")


class SSFLinear(nn.Module):
    """Apply trainable channel scale and shift to a frozen affine output."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = nn.Parameter(base.weight.new_ones(base.out_features))
        self.shift = nn.Parameter(base.weight.new_zeros(base.out_features))

    def forward(self, x):
        return self.base(x) * self.scale + self.shift


class ResidualAdapter(nn.Module):
    """Initially identity representation adapter with an eight-unit bottleneck."""

    def __init__(self, rep_dim, bottleneck=8):
        super().__init__()
        self.down = nn.Linear(rep_dim, bottleneck)
        self.up = nn.Linear(bottleneck, rep_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, u0):
        return u0 + self.up(F.relu(self.down(u0)))


class AdapterStage2Model(Stage2Model):
    def forward(self, batch, envs):
        h = self.core
        x = batch["x"]
        env = h.env_embedding_network(h.new_env_idx).squeeze(0)
        weights = F.softmax(
            h.projection_network(x) @ torch.stack(envs, 1) / h.temperature,
            dim=-1,
        )
        specific = (
            torch.stack([batch["s0"], batch["s1"]], 2) @ weights.unsqueeze(2)
        ).squeeze(-1)
        u0 = batch["u0"] if "u0" in batch else self.u(x)
        experts = [specific * env, batch["g"], self.adapter(u0)]
        fused = (torch.stack(experts, 2) @ h.gate_network(x).unsqueeze(2)).squeeze(-1)
        return h.tower_network(fused).reshape(-1)


def build_model(spec, u_source, seed, variant):
    """Reuse frozen stage2 initialization and alter only each PEFT update path."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown PEFT variant: {variant}")
    if u_source is None:
        raise ValueError(f"U source required for variant: {variant}")
    model = build_stage2_model(spec, u_source, seed, "tri_frozen")
    model.variant = variant
    if variant == "tri_bitfit":
        for layer in model.u.encoder:
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.requires_grad_(True)
    elif variant == "tri_ssf":
        for index, layer in enumerate(model.u.encoder):
            if isinstance(layer, nn.Linear):
                model.u.encoder[index] = SSFLinear(layer)
    else:
        model = AdapterStage2Model(model.core, model.u, variant)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 10003)
            model.adapter = ResidualAdapter(spec["rdim"])
    reference = next(model.u.parameters())
    model.to(device=reference.device, dtype=reference.dtype)
    return model
