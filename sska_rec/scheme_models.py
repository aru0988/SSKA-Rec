"""Complementary pretraining and independent conditional readout models."""

import torch
from torch import nn
from torch.nn import functional as F

from .expert import UniversalExpert
from .peft_models import build_model as build_peft_model
from .stage2_models import Stage2Model, build_model as build_stage2_model


SCHEME1_VARIANTS = ('mpt', 'tri_frozen', 'tri_lora', 'tri_ssf', 'tri_full')


class GSDecoder(nn.Module):
    """Reconstruct normalized inputs from the frozen [G, S0, S1] blocks."""

    def __init__(self, rdim, xdim):
        super().__init__()
        self.linear = nn.Linear(3 * rdim, xdim)
        self.register_buffer('mean', torch.zeros(xdim))
        self.register_buffer('scale', torch.ones(xdim))

    def set_normalizer(self, mean, std):
        self.mean.copy_(mean.detach())
        self.scale.copy_(std.detach().clamp_min(.1))

    def normalize(self, x):
        return ((x.detach() - self.mean) / self.scale).clamp(-10, 10)

    def forward(self, h):
        return self.linear(h)


class ComplementaryExpert(UniversalExpert):
    """UniversalExpert checkpoint schema with the complementary target loss."""

    def losses(self, x, g, specifics, gs_decoder, mask=None,
               redundancy_weight=.01, variance_weight=.01):
        mask = (self.sample_mask(len(x)) if mask is None else mask).to(x.device)
        blocks = [g, *specifics]
        with torch.no_grad():
            target = self.normalize(x) - gs_decoder(torch.cat(blocks, dim=1))
        predicted = self.decoder(self.encoder(self.masked_input(x, mask)))
        # First average coordinates within each field, then selected fields.
        field_error = torch.stack([
            value.mean(1) for value in (predicted - target).square().split(self.widths, 1)
        ], 1)
        field_zero = torch.stack([
            value.mean(1) for value in target.square().split(self.widths, 1)
        ], 1)
        reconstruction = (field_error * mask).sum() / mask.sum()
        zero = (field_zero * mask).sum() / mask.sum()
        clean = self(x)
        uc = (clean - clean.mean(0)) / clean.std(0, unbiased=False).clamp_min(.01)
        penalties = []
        for block in blocks:
            frozen = block.detach()
            standardized = ((frozen - frozen.mean(0)) /
                            frozen.std(0, unbiased=False).clamp_min(.01))
            penalties.append((uc.T @ standardized / max(len(x), 1)).square().mean())
        decorrelation = torch.stack(penalties).mean()
        variance = F.relu(.5 - torch.sqrt(clean.var(0, unbiased=False) + 1e-4)).mean()
        total = reconstruction + redundancy_weight * decorrelation + variance_weight * variance
        return dict(total=total, reconstruction=reconstruction, zero=zero,
                    decorrelation=decorrelation, variance=variance)


class ConditionalReadout(nn.Module):
    """Task-private low-rank conditional interaction over frozen h and u."""

    def __init__(self, rdim, rank=4):
        super().__init__()
        self.h_norm = nn.LayerNorm(rdim, elementwise_affine=False)
        self.u_norm = nn.LayerNorm(rdim, elementwise_affine=False)
        self.A = nn.Linear(2 * rdim, rank, bias=False)
        self.C = nn.Linear(rdim, rank, bias=False)
        self.out = nn.Linear(rank, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def delta(self, h, u):
        h, u = self.h_norm(h.detach()), self.u_norm(u.detach())
        q = self.A(torch.cat([h, u], dim=1)).sigmoid()
        return self.out(q * self.C(u)).reshape(-1)

    def forward(self, h, u, p):
        p = p.detach().reshape(-1)
        logits = torch.logit(p.clamp(1e-7, 1 - 1e-7))
        difference = (logits + self.delta(h, u)).sigmoid() - logits.sigmoid()
        return (p + difference).clamp(0, 1)


@torch.no_grad()
def old_task_hidden(base, batch, position):
    """Return the original pre-tower fusion while retaining the batch axis."""
    env = base.env_embedding_network.weight[position]
    experts = torch.stack([batch[f's{position}'] * env, batch['g']], dim=2)
    weights = base.gate_networks[position](batch['x'])
    return (experts @ weights.unsqueeze(2)).squeeze(-1)


class Scheme1Model(Stage2Model):
    """Three-way representation router with historical task/update scopes."""

    def forward(self, batch, envs):
        core = self.core
        env = core.env_embedding_network(core.new_env_idx).squeeze(0)
        selection = F.softmax(
            core.projection_network(batch['x']) @ torch.stack(envs, 1) / core.temperature,
            dim=-1,
        )
        specific = (torch.stack([batch['s0'], batch['s1']], 2) @
                    selection.unsqueeze(2)).squeeze(-1)
        experts = [batch['g'], specific * env, self.u(batch['x'])]
        # Normalize only the routing features; the values retain their scale.
        router_input = torch.cat([
            F.layer_norm(value, (value.shape[-1],)) for value in experts
        ], dim=1)
        fused = (torch.stack(experts, 2) @
                 core.gate_network(router_input).unsqueeze(2)).squeeze(-1)
        return core.tower_network(fused).reshape(-1)


def build_scheme1_model(spec, u_source, seed, variant):
    """Reuse matched task heads and private U adaptation, replacing only routing."""
    if variant not in SCHEME1_VARIANTS:
        raise ValueError(f'Unknown scheme1 variant: {variant}')
    factory = build_peft_model if variant == 'tri_ssf' else build_stage2_model
    original = factory(spec, u_source, seed, variant)
    if variant == 'mpt':
        return original
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 3)
        gate = nn.Sequential(nn.Linear(3 * spec['rdim'], 3, bias=False), nn.Softmax(dim=-1))
    original.core.gate_network = gate
    model = Scheme1Model(original.core, original.u, variant)
    reference = next(model.u.parameters())
    return model.to(device=reference.device, dtype=reference.dtype)
