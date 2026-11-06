from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class MLP(nn.Module):
    def __init__(
        self, hidden_units, input_size, output_activation="relu", dropout=None
    ):
        super(MLP, self).__init__()
        hidden_units = [input_size] + list(hidden_units)
        layer_num = len(hidden_units) - 1

        self.mlp = nn.Sequential()
        for i in range(layer_num - 1):
            self.mlp.add_module(
                "linear" + str(i), nn.Linear(hidden_units[i], hidden_units[i + 1])
            )
            self.mlp.add_module("relu" + str(i), nn.ReLU())

            if dropout:
                self.mlp.add_module("dropout" + str(i), nn.Dropout(dropout[i]))

        self.mlp.add_module(
            "linear" + str(layer_num - 1), nn.Linear(hidden_units[-2], hidden_units[-1])
        )
        if output_activation == "softmax":
            self.mlp.add_module("softmax" + str(layer_num - 1), nn.Softmax(dim=-1))
        elif output_activation == "sigmoid":
            self.mlp.add_module("sigmoid" + str(layer_num - 1), nn.Sigmoid())
        else:
            self.mlp.add_module("relu" + str(layer_num - 1), nn.ReLU())

        if dropout:
            self.mlp.add_module("dropout" + str(layer_num - 1), nn.Dropout(dropout[-1]))

    def forward(self, x):
        return self.mlp(x)

    def get_l2_reg(self):
        loss = 0
        for layer in self.mlp:
            if type(layer) is nn.Linear:
                loss += layer.weight.norm(2).pow(2) / 2.0
        return loss


class EmbeddingNetwork(nn.Module):
    def __init__(self, feature_vocabulary, embedding_size):
        super(EmbeddingNetwork, self).__init__()
        self.feature_names = list(feature_vocabulary.keys())
        self.embedding_dict = nn.ModuleDict()
        for name, size in feature_vocabulary.items():
            emb = nn.Embedding(size, embedding_size)
            # nn.init.normal_(emb.weight, mean=0.0, std=0.01)
            self.embedding_dict[name] = emb

    def forward(self, x):
        feature_embedding = [
            x[name].unsqueeze(dim=1).float()
            for name in x.keys()
            if name not in self.feature_names
        ]
        for name in self.feature_names:
            embed = self.embedding_dict[name](x[name].long())
            feature_embedding.append(embed)
        feature_embedding = torch.cat(feature_embedding, 1)
        return feature_embedding

    def get_l2_reg(self):
        loss = 0
        for name in self.embedding_dict:
            loss += (self.embedding_dict[name].weight ** 2).sum() / 2.0
        return loss


class SingleTask(nn.Module):
    def __init__(
        self,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding,
        reg_dnn,
        dropout=None,
    ):
        super(SingleTask, self).__init__()
        self.feature_vocabulary = feature_vocabulary
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.shared_dnn_hidden_units = shared_dnn_hidden_units
        self.tower_dnn_hidden_units = tower_dnn_hidden_units
        self.dropout = dropout

        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.base_network = MLP(shared_dnn_hidden_units, input_size, dropout=dropout)
        self.tower_network = MLP(
            list(tower_dnn_hidden_units) + [1], shared_dnn_hidden_units[-1], "sigmoid"
        )

    def forward(self, x):
        dnn_input = self.embedding_network(x)
        mid_output = self.base_network(dnn_input)
        final_output = self.tower_network(mid_output)
        return [final_output.squeeze()]

    def functional_forward(self, x, params):
        embedding_network = EmbeddingNetwork(
            self.feature_vocabulary, self.embedding_size
        )
        base_network = MLP(
            self.shared_dnn_hidden_units, self.input_size, dropout=self.dropout
        )
        tower_network = MLP(
            list(self.tower_dnn_hidden_units) + [1],
            self.shared_dnn_hidden_units[-1],
            "sigmoid",
        )

        embedding_network.load_state_dict(params, strict=False)
        base_network.load_state_dict(params, strict=False)
        tower_network.load_state_dict(params, strict=False)

        dnn_input = embedding_network(x)
        mid_output = base_network(dnn_input)
        final_output = tower_network(mid_output)
        return [final_output.squeeze()]

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = self.base_network.get_l2_reg() + self.tower_network.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn


class EnvClassifier(nn.Module):
    def __init__(self):
        super(EnvClassifier, self).__init__()

    def forward(self, invariant_preferences):
        raise NotImplementedError

    def get_l2_reg(self) -> torch.Tensor:
        raise NotImplementedError

    def get_l1_reg(self) -> torch.Tensor:
        raise NotImplementedError


class LinearLogSoftMaxEnvClassifier(EnvClassifier):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_l1_reg(self) -> torch.Tensor:
        return (
            torch.norm(self.linear_map.weight, 1) / self.elements_num
            + torch.norm(self.linear_map.bias, 1) / self.bias_num
        )

    def get_l2_reg(self) -> torch.Tensor:
        return (
            torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num
            + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num
        )

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class MPTRec(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        expert_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding=0,
        reg_dnn=0,
        dropout=None,
        device=None,
    ):
        super(MPTRec, self).__init__()
        self.num_tasks = num_tasks
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.device = device
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.shared_expert_network = MLP(
            expert_dnn_hidden_units, input_size, "relu", dropout
        )
        self.specific_expert_networks = nn.ModuleList()
        self.env_embedding_network = nn.Embedding(
            num_tasks, expert_dnn_hidden_units[-1]
        )
        self.gate_networks = nn.ModuleList()
        self.env_classifier = LinearLogSoftMaxEnvClassifier(
            expert_dnn_hidden_units[-1], num_tasks
        )
        self.tower_networks = nn.ModuleList()
        self.register_buffer("env_indices", torch.arange(num_tasks), persistent=True)

        for _ in range(num_tasks):
            self.specific_expert_networks.append(
                MLP(expert_dnn_hidden_units, input_size, "relu", dropout)
            )
            self.gate_networks.append(
                nn.Sequential(nn.Linear(input_size, 2, bias=False), nn.Softmax(dim=-1))
            )
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    expert_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x, alpha=1):
        dnn_input = self.embedding_network(x)
        gen_rep = self.shared_expert_network(dnn_input)

        gen_preds = []
        for i in range(self.num_tasks):
            output = self.tower_networks[i](gen_rep)
            gen_preds.append(output.squeeze())

        gate_outs = []
        for gate in self.gate_networks:
            gate_outs.append(gate(dnn_input))

        fused_preds = []
        for i in range(self.num_tasks):
            spec_rep = self.specific_expert_networks[i](dnn_input)
            env_embedding = self.env_embedding_network(torch.tensor(i).to(self.device))
            env_aware_rep = spec_rep * env_embedding
            all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
            fused_rep = torch.matmul(all_reps, gate_outs[i].unsqueeze(dim=2)).squeeze()
            output = self.tower_networks[i](fused_rep)
            fused_preds.append(output.squeeze())

        rev_gen_rep = ReverseLayerF.apply(gen_rep, alpha)
        env_pred = self.env_classifier(rev_gen_rep)

        return {
            "gen_preds": gen_preds,
            "fused_preds": fused_preds,
            "env_pred": env_pred,
        }

    def predict(self, x):
        output = self.forward(x)
        return output["fused_preds"]

    def cluster_predict(self, x):
        # TODO: 尝试仅使用通用表征的预测结果做聚类预测
        return self.predict(x)

    def get_infos(self, x):
        dnn_input = self.embedding_network(x)
        gen_rep = self.shared_expert_network(dnn_input)

        spec_reps, env_embs = [], []
        for i in range(self.num_tasks):
            spec_reps.append(self.specific_expert_networks[i](dnn_input))
            env_embs.append(self.env_embedding_network(torch.tensor(i).to(self.device)))

        return dnn_input, gen_rep, spec_reps, env_embs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = self.shared_expert_network.get_l2_reg()
        for expert in self.specific_expert_networks:
            loss_dnn += expert.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn


class NewTask(nn.Module):
    def __init__(
        self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None
    ):
        super(NewTask, self).__init__()
        self.reg_dnn = reg_dnn
        self.device = device
        self.temperature = 150
        self.env_embedding_network = nn.Embedding(1, rep_dim)
        self.register_buffer("new_env_idx", torch.tensor([0]), persistent=True)
        self.projection_network = nn.Sequential(
            nn.Linear(input_size, rep_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(rep_dim // 2, rep_dim, bias=False),
            nn.LayerNorm(rep_dim),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, 2, bias=False), nn.Softmax(dim=-1)
        )
        # self.gate_network_2 = nn.Sequential(
        #     nn.Linear(input_size, 2, bias=False), nn.Softmax()
        # )
        self.tower_network = MLP(
            list(tower_dnn_hidden_units) + [1],
            input_size=rep_dim,
            output_activation="sigmoid",
        )

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        exist_env_embs = torch.stack(env_embs, dim=1)
        new_env_emb = self.env_embedding_network(self.new_env_idx).squeeze(0)

        H_out = self.projection_network(dnn_input)
        W = torch.mm(H_out, exist_env_embs) / self.temperature
        W = F.softmax(W, dim=-1).unsqueeze(2)

        gate_out = self.gate_network(dnn_input).unsqueeze(dim=2)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), W).squeeze()
        env_aware_rep = new_spec_rep * new_env_emb
        all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
        fused_rep = torch.matmul(all_reps, gate_out).squeeze()

        output = self.tower_network(fused_rep)
        return output.squeeze()

    def get_l2_reg(self):
        loss_dnn = self.tower_network.get_l2_reg()
        return self.reg_dnn * loss_dnn
