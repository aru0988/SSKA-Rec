"""Initial-task baselines used in Table 2; source details live in vendor/."""

import torch
from torch import nn

from . import core
from .vendor import mpt_baselines as mpt


class ProgressivePLE(mpt.PLE):
    """Use the existing CGC/PLE equations with one dataset width per level."""

    def __init__(self, spec, reg_embedding, reg_dnn, dropout):
        nn.Module.__init__(self)
        self.num_tasks = 2
        self.num_levels = len(spec['hidden'])
        self.reg_embedding, self.reg_dnn = reg_embedding, reg_dnn
        self.embedding_network = mpt.EmbeddingNetwork(spec['vocab'], spec['embed'])
        self.cgc_networks = nn.ModuleList()
        input_size = spec['xdim']
        for level, width in enumerate(spec['hidden']):
            self.cgc_networks.append(mpt.CGC(
                num_tasks=2, input_size=input_size, specific_expert_num=1,
                shared_expert_num=1, expert_dnn_hidden_units=[width],
                dropout=None if dropout is None else [dropout[level]]))
            input_size = width
        self.tower_networks = nn.ModuleList([
            mpt.MLP([*spec['tower'], 1], input_size, 'sigmoid') for _ in range(2)])


class AdaTTBaseline(nn.Module):
    """Official AdaTT-sp extraction followed by existing task sigmoid towers."""

    def __init__(self, spec, reg_embedding, reg_dnn):
        super().__init__()
        from .vendor.adatt_mtl_lib import AdaTTSp
        self.reg_embedding, self.reg_dnn = reg_embedding, reg_dnn
        self.embedding_network = mpt.EmbeddingNetwork(spec['vocab'], spec['embed'])
        self.extraction = AdaTTSp(input_dim=spec['xdim'],
                                  expert_out_dims=[[width] for width in spec['hidden']],
                                  num_tasks=2, num_task_experts=1,
                                  self_exp_res_connect=True)
        self.tower_networks = nn.ModuleList([
            mpt.MLP([*spec['tower'], 1], spec['hidden'][-1], 'sigmoid')
            for _ in range(2)])

    def forward(self, features):
        outputs = self.extraction(self.embedding_network(features).unsqueeze(1).expand(-1, 2, -1))
        return [tower(outputs[:, task, :]).squeeze(-1)
                for task, tower in enumerate(self.tower_networks)]

    def get_l2_reg(self):
        dnn = sum(layer.weight.square().sum() / 2
                  for module in (*self.extraction.experts, *self.tower_networks)
                  for layer in module.modules() if isinstance(layer, nn.Linear))
        return (self.reg_embedding * self.embedding_network.get_l2_reg()
                + self.reg_dnn * dnn)


def build_baseline(spec, model_seed, variant, cfg=None):
    """Construct two joint initial-task heads with normalized embedding L2."""
    cfg = {} if cfg is None else cfg
    for key in ('reg_embedding', 'reg_dnn'):
        if key in cfg:
            raise ValueError(f'Unregistered baseline regularization option: {key}')
    core.seed_all(model_seed)
    denominator = sum(spec['vocab'].values()) * spec['embed']
    reg_embedding = float(cfg.get('embedding_l2', .006)) / denominator
    reg_dnn = float(cfg.get('dnn_l2', 3e-5))
    dropout = [.1, .3] if spec['dataset'] == 'aliccp' else None
    common = dict(num_tasks=2, feature_vocabulary=spec['vocab'],
                  embedding_size=spec['embed'], input_size=spec['xdim'],
                  tower_dnn_hidden_units=spec['tower'],
                  reg_embedding=reg_embedding, reg_dnn=reg_dnn, dropout=dropout)
    if variant == 'sharedbottom':
        model = mpt.SharedBottom(shared_dnn_hidden_units=spec['hidden'], **common)
    elif variant == 'mmoe':
        model = mpt.MMOE(num_experts=3, expert_dnn_hidden_units=spec['hidden'], **common)
    elif variant == 'ple':
        model = ProgressivePLE(spec, reg_embedding, reg_dnn, dropout)
    elif variant == 'adatt':
        model = AdaTTBaseline(spec, reg_embedding, reg_dnn)
    else:
        raise ValueError(f'Unknown baseline variant: {variant}')
    return model.to(device=core.DEVICE, dtype=torch.float32)
