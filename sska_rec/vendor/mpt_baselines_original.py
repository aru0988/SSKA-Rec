"""Extracted from MPT-Rec/multitaskrec/model.py; see baseline-provenance.json."""

import torch
from torch import nn


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


class SharedBottom(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_dnn_hidden_units,
        tower_dnn_hidden_units,
        reg_embedding,
        reg_dnn,
        dropout=None,
    ):
        super(SharedBottom, self).__init__()
        self.num_tasks = num_tasks
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.shared_bottom = MLP(shared_dnn_hidden_units, input_size, dropout=dropout)
        self.tower_networks = nn.ModuleList()

        for _ in range(num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    shared_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x):
        dnn_input = self.embedding_network(x)
        mid_output = self.shared_bottom(dnn_input)
        final_outputs = []
        for tower in self.tower_networks:
            final_outputs.append(tower(mid_output).squeeze())
        return final_outputs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = self.shared_bottom.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "tower_networks" not in name:
                param.requires_grad = False


class MMOE(nn.Module):
    def __init__(
        self,
        num_tasks,
        num_experts,
        feature_vocabulary,
        embedding_size,
        input_size,
        expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32),
        reg_embedding=0,
        reg_dnn=0,
        dropout=None,
    ):
        super(MMOE, self).__init__()
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.expert_networks = nn.ModuleList()
        self.tower_networks = nn.ModuleList()
        self.gate_networks = nn.ModuleList()

        for _ in range(0, num_experts):
            self.expert_networks.append(
                MLP(expert_dnn_hidden_units, input_size, dropout=dropout)
            )
        for _ in range(0, num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    expert_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )
            self.gate_networks.append(
                nn.Sequential(
                    nn.Linear(input_size, num_experts, bias=False), nn.Softmax(dim=-1)
                )
            )

    def forward(self, x):
        feature_embedding = self.embedding_network(x)
        expert_outs = []
        for expert in self.expert_networks:
            expert_outs.append(expert(feature_embedding))
        expert_concat = torch.stack(expert_outs, dim=2)

        gate_outs = []
        for gate in self.gate_networks:
            gate_outs.append(gate(feature_embedding))
        weighted_expert_outs = []
        for gate_out in gate_outs:
            output = torch.matmul(expert_concat, gate_out.unsqueeze(dim=2)).squeeze()
            weighted_expert_outs.append(output)

        task_outs = []
        for i in range(self.num_tasks):
            output = self.tower_networks[i](weighted_expert_outs[i])
            task_outs.append(output.squeeze())
        return task_outs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = 0
        for expert in self.expert_networks:
            loss_dnn += expert.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "tower_networks" not in name and "gate_networks" not in name:
                param.requires_grad = False


class CGC(nn.Module):
    def __init__(
        self,
        num_tasks,
        input_size,
        specific_expert_num,
        shared_expert_num,
        expert_dnn_hidden_units,
        dropout,
    ):
        super(CGC, self).__init__()
        self.num_tasks = num_tasks
        self.shared_expert_num = shared_expert_num
        self.specific_expert_num = specific_expert_num
        self.shared_expert_networks = nn.ModuleList()
        self.specific_expert_networks = nn.ModuleList()
        self.shared_gate = MLP(
            (num_tasks * specific_expert_num + shared_expert_num,),
            input_size,
            "softmax",
        )
        self.specific_gates = nn.ModuleList()

        # build task-shared expert layer
        for _ in range(shared_expert_num):
            expert_network = MLP(expert_dnn_hidden_units, input_size, "relu", dropout)
            self.shared_expert_networks.append(expert_network)

        # build task-specific expert layer
        for _ in range(num_tasks):
            for _ in range(specific_expert_num):
                expert_network = MLP(
                    expert_dnn_hidden_units, input_size, "relu", dropout
                )
                self.specific_expert_networks.append(expert_network)

        # task-specific gate
        for _ in range(num_tasks):
            gate_network = MLP(
                (specific_expert_num + shared_expert_num,), input_size, "softmax"
            )
            self.specific_gates.append(gate_network)

    def forward(self, inputs, is_last=False):
        # inputs: [task1, task2, ... taskn, shared task]
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_expert_networks[
                    i * self.specific_expert_num + j
                ](inputs[i])
                specific_expert_outputs.append(specific_expert_output)
        shared_expert_outputs = []
        for i in range(self.shared_expert_num):
            shared_expert_output = self.shared_expert_networks[i](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)

        cgc_outs = []
        for i in range(self.num_tasks):
            cur_experts = (
                specific_expert_outputs[
                    i * self.specific_expert_num : (i + 1) * self.specific_expert_num
                ]
                + shared_expert_outputs
            )
            expert_concat = torch.stack(cur_experts, dim=2)
            gate_out = self.specific_gates[i](inputs[i])  # gate[i] for task input[i]
            gate_out = torch.unsqueeze(gate_out, -1)
            gate_mul_expert = torch.matmul(expert_concat, gate_out).squeeze()
            cgc_outs.append(gate_mul_expert)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_experts = (
                specific_expert_outputs + shared_expert_outputs
            )  # all the expert include task-specific expert and task-shared expert
            expert_concat = torch.stack(cur_experts, dim=2)
            gate_out = self.shared_gate(inputs[-1])  # gate for shared task input
            gate_out = torch.unsqueeze(gate_out, -1)
            gate_mul_expert = torch.matmul(expert_concat, gate_out).squeeze()
            cgc_outs.append(gate_mul_expert)

        return cgc_outs

    def get_l2_reg(self):
        loss = 0
        for expert in self.shared_expert_networks:
            loss += expert.get_l2_reg()
        for expert in self.specific_expert_networks:
            loss += expert.get_l2_reg()
        return loss

    def get_reps(self, inputs):
        specific_expert_outputs = []
        for i in range(self.num_tasks):
            for j in range(self.specific_expert_num):
                specific_expert_output = self.specific_expert_networks[
                    i * self.specific_expert_num + j
                ](inputs[i])
                specific_expert_outputs.append(specific_expert_output)
        shared_expert_outputs = []
        for i in range(self.shared_expert_num):
            shared_expert_output = self.shared_expert_networks[i](inputs[-1])
            shared_expert_outputs.append(shared_expert_output)

        return shared_expert_outputs, specific_expert_outputs


class PLE(nn.Module):
    def __init__(
        self,
        num_tasks,
        feature_vocabulary,
        embedding_size,
        input_size,
        shared_expert_num=1,
        specific_expert_num=1,
        num_levels=2,
        expert_dnn_hidden_units=(256,),
        tower_dnn_hidden_units=(64,),
        reg_embedding=0,
        reg_dnn=0,
        dropout=None,
    ):
        super(PLE, self).__init__()
        self.num_tasks = num_tasks
        self.num_levels = num_levels
        self.reg_embedding = reg_embedding
        self.reg_dnn = reg_dnn
        self.embedding_network = EmbeddingNetwork(feature_vocabulary, embedding_size)
        self.cgc_networks = nn.ModuleList()
        self.tower_networks = nn.ModuleList()

        self.cgc_networks.append(
            CGC(
                num_tasks=num_tasks,
                input_size=input_size,
                shared_expert_num=shared_expert_num,
                specific_expert_num=specific_expert_num,
                expert_dnn_hidden_units=expert_dnn_hidden_units,
                dropout=dropout,
            )
        )

        for _ in range(num_levels - 1):
            self.cgc_networks.append(
                CGC(
                    num_tasks=num_tasks,
                    input_size=expert_dnn_hidden_units[-1],
                    shared_expert_num=shared_expert_num,
                    specific_expert_num=specific_expert_num,
                    expert_dnn_hidden_units=expert_dnn_hidden_units,
                    dropout=dropout,
                )
            )

        for _ in range(num_tasks):
            self.tower_networks.append(
                MLP(
                    list(tower_dnn_hidden_units) + [1],
                    expert_dnn_hidden_units[-1],
                    "sigmoid",
                )
            )

    def forward(self, x):
        feature_embedding = self.embedding_network(x)
        ple_inputs = [feature_embedding] * (
            self.num_tasks + 1
        )  # [task1, task2, ... taskn, shared task]
        ple_outputs = []

        for i in range(self.num_levels):
            if i == self.num_levels - 1:  # the last level
                ple_outputs = self.cgc_networks[i](inputs=ple_inputs, is_last=True)
            else:
                ple_outputs = self.cgc_networks[i](inputs=ple_inputs, is_last=False)
                ple_inputs = ple_outputs

        task_outs = []
        for i in range(self.num_tasks):
            task_out = self.tower_networks[i](ple_outputs[i])
            task_outs.append(task_out.squeeze())

        return task_outs

    def get_l2_reg(self):
        loss_embedding = self.embedding_network.get_l2_reg()
        loss_dnn = 0
        for cgc in self.cgc_networks:
            loss_dnn += cgc.get_l2_reg()
        for tower in self.tower_networks:
            loss_dnn += tower.get_l2_reg()
        return self.reg_embedding * loss_embedding + self.reg_dnn * loss_dnn

    def freeze_params(self):
        for name, param in self.named_parameters():
            if "embedding_network" in name or "shared_expert_networks" in name:
                param.requires_grad = False

    def get_reps(self, x):
        feature_embedding = self.embedding_network(x)
        ple_inputs = [feature_embedding] * (
            self.num_tasks + 1
        )  # [task1, task2, ... taskn, shared task]
        ple_outputs = []

        for i in range(self.num_levels):
            if i == self.num_levels - 1:
                generic_rep, proprietary_rep = self.cgc_networks[i].get_reps(
                    inputs=ple_inputs
                )
            else:
                ple_outputs = self.cgc_networks[i](inputs=ple_inputs, is_last=False)
                ple_inputs = ple_outputs

        return generic_rep[0], proprietary_rep
