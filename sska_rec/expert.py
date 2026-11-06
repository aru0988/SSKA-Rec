import copy
import torch
from torch import nn
from torch.nn import functional as F


class UniversalExpert(nn.Module):
    def __init__(self, widths, rep_dim=128, hidden=256, seed=1685480946):
        super().__init__()
        self.widths = list(widths)
        self.input_dim = sum(widths)
        self.mask_rng = torch.Generator().manual_seed(20261009)
        with torch.random.fork_rng(devices=[]):
            # Initialize CPU layers without reseeding CUDA dropout generators.
            torch.random.default_generator.manual_seed(seed)
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim + len(widths), hidden),
                nn.ReLU(),
                nn.Linear(hidden, rep_dim),
            )
            self.decoder = nn.Linear(rep_dim, self.input_dim)
        self.register_buffer("mean", torch.zeros(self.input_dim))
        self.register_buffer("scale", torch.ones(self.input_dim))

    def set_normalizer(self, mean, std):
        self.mean.copy_(mean.detach())
        self.scale.copy_(std.detach().clamp_min(0.1))

    def normalize(self, x):
        return ((x.detach() - self.mean) / self.scale).clamp(-10, 10)

    def sample_mask(self, n):
        # Exact fixed count, independently for each sample; private CPU RNG.
        rank = torch.rand(n, len(self.widths), generator=self.mask_rng).argsort(dim=1)
        mask = torch.zeros(n, len(self.widths), dtype=torch.bool)
        return mask.scatter_(1, rank[:, : max(1, round(0.3 * len(self.widths)))], True)

    def expand_mask(self, mask):
        return torch.repeat_interleave(
            mask, torch.tensor(self.widths, device=mask.device), dim=1
        )

    def masked_input(self, x, mask):
        mask = mask.to(x.device)
        return torch.cat(
            [self.normalize(x).masked_fill(self.expand_mask(mask), 0), mask.float()], 1
        )

    def forward(self, x):
        return self.encoder(
            torch.cat([self.normalize(x), x.new_zeros(len(x), len(self.widths))], 1)
        )

    def losses(self, x, general, mask=None):
        mask = (self.sample_mask(len(x)) if mask is None else mask).to(x.device)
        target = self.normalize(x)
        predicted = self.decoder(self.encoder(self.masked_input(x, mask)))
        # Equal field weighting, rather than overweighting 4-D categorical fields.
        field_error = torch.stack(
            [v.mean(1) for v in (predicted - target).square().split(self.widths, 1)], 1
        )
        field_zero = torch.stack(
            [v.mean(1) for v in target.square().split(self.widths, 1)], 1
        )
        reconstruction = (field_error * mask).sum() / mask.sum()
        zero = (field_zero * mask).sum() / mask.sum()
        clean = self(x)
        std = clean.std(0, unbiased=False)
        variance = F.relu(0.5 - torch.sqrt(clean.var(0, unbiased=False) + 1e-4)).mean()
        gu = general.detach()
        uc = (clean - clean.mean(0)) / std.clamp_min(0.01)
        gc = (gu - gu.mean(0)) / gu.std(0, unbiased=False).clamp_min(0.01)
        decorrelation = (uc.T @ gc / max(len(x), 1)).square().mean()
        total = reconstruction + 0.01 * decorrelation + 0.01 * variance
        return dict(
            total=total,
            reconstruction=reconstruction,
            zero=zero,
            decorrelation=decorrelation,
            variance=variance,
        )

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Generator):
                clone = torch.Generator(device=value.device)
                clone.set_state(value.get_state())
                setattr(result, key, clone)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result
