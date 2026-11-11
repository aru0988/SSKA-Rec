"""Matched NewTask heads for the Census stage2 U-update comparison."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import NewTask


VARIANTS = ("mpt", "residual", "tri_frozen", "tri_lora", "tri_full")


class LoRALinear(nn.Module):
    """Frozen affine map plus a rank-four, initially zero weight update."""

    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        # Draw factors on CPU so build_model's private CPU RNG controls them on
        # both CPU and CUDA sources without changing global CUDA RNG streams.
        initial = torch.empty(rank, base.in_features)
        nn.init.kaiming_uniform_(initial, a=math.sqrt(5))
        self.A = nn.Parameter(initial.to(base.weight))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scaling


class Stage2Model(nn.Module):
    def __init__(self, core, universal, variant):
        super().__init__()
        self.variant = variant
        self.core = copy.deepcopy(core) if variant == "residual" else core
        self.u = universal
        if variant == "residual":
            # Historical ResidualHead initialization and forward coefficient.
            rep_dim = core.env_embedding_network.embedding_dim
            self.projection = nn.Linear(rep_dim, rep_dim, bias=False)
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(rep_dim))
            self.logit = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
        self.train(self.training)

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def exclusive_parameters(self):
        """Parameters independently updated for this new task."""
        return self.trainable_parameters

    @property
    def inference_parameters(self):
        """All prediction parameters, including the frozen encoder, without decoder."""
        return sum(
            p.numel()
            for name, p in self.named_parameters()
            if not name.startswith("u.decoder.")
        )

    def train(self, mode=True):
        super().train(mode)
        if self.u is not None:
            self.u.decoder.eval()
            if self.variant == "tri_frozen":
                self.u.eval()
            elif self.variant == "tri_lora":
                for layer in self.u.encoder:
                    if isinstance(layer, LoRALinear):
                        layer.base.eval()
        return self

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
        experts = [specific * env, batch["g"]]
        if self.variant.startswith("tri_"):
            # Fixed order: environment-aware S, G, U.
            experts.append(self.u(x))
        fused = (torch.stack(experts, 2) @ h.gate_network(x).unsqueeze(2)).squeeze(-1)
        if self.variant == "residual":
            # detach_extra=False: stage2 classification updates the U encoder.
            fused = fused + self.logit.sigmoid() * self.projection(self.u(x))
        return h.tower_network(fused).reshape(-1)

    def get_l2_reg(self):
        return self.core.get_l2_reg()


def build_model(spec, u_source, seed, variant):
    """Build a private U copy and a common seed+3 NewTask initialization.

    spec uses the existing xdim, rdim, and tower keys. The encoder dimensions
    and normalization buffers come from u_source; the decoder stays unused.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown stage2 variant: {variant}")
    if variant != "mpt" and u_source is None:
        raise ValueError(f"U source required for variant: {variant}")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 3)
        core = NewTask(spec["xdim"], spec["rdim"], spec["tower"], reg_dnn=3e-5)
        if variant.startswith("tri_"):
            core.gate_network = nn.Sequential(
                nn.Linear(spec["xdim"], 3, bias=False), nn.Softmax(dim=-1)
            )
        universal = None
        if variant != "mpt":
            universal = copy.deepcopy(u_source)
            universal.requires_grad_(False)
            if variant in ("residual", "tri_full"):
                universal.encoder.requires_grad_(True)
            elif variant == "tri_lora":
                for index, layer in enumerate(universal.encoder):
                    if isinstance(layer, nn.Linear):
                        universal.encoder[index] = LoRALinear(layer)
        model = Stage2Model(core, universal, variant)
    if universal is not None:
        reference = next(universal.parameters())
        model.to(device=reference.device, dtype=reference.dtype)
    return model
