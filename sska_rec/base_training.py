import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class TrainManager:

    def __init__(
        self, model, train_loader, val_loader, task_name, lr, epochs=30, patience=5
    ):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.task_name = task_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.best_weight = None


class MPTRecTrainManager(TrainManager):

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        env_ids,
        task_name,
        lr,
        batch_size,
        uni_coe,
        env_coe,
        epochs=30,
        patience=5,
    ):
        super().__init__(
            model, train_loader, val_loader, task_name, lr, epochs, patience
        )
        self.env_ids = env_ids
        self.env_loss_func = nn.NLLLoss()
        self.batch_size = batch_size
        self.uni_coe = uni_coe
        self.env_coe = env_coe
        self.uni_loss_0_list = []
        self.uni_loss_1_list = []
        self.fused_loss_0_list = []
        self.fused_loss_1_list = []
        self.env_loss_list = []

    def train_two_task(self):
        earlystop_count = 0
        best_auc_score = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            uni_loss_0_sum = 0
            uni_loss_1_sum = 0
            fused_loss_0_sum = 0
            fused_loss_1_sum = 0
            env_loss_sum = 0
            tepoch = tqdm(self.train_loader, unit="batch")
            for step, (y_0, y_1, _, features) in enumerate(tepoch):
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                p = float(step + (epoch - 1) * self.batch_size) / float(
                    self.epochs * self.batch_size
                )
                alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
                output = self.model(features, alpha)
                batch_env_ids = self.env_ids[
                    self.batch_size * step : self.batch_size * (step + 1)
                ]
                device = output["gen_preds"][0].device
                uni_loss_0 = self.loss_func(
                    output["gen_preds"][0], y_0.float().to(device)
                )
                uni_loss_1 = self.loss_func(
                    output["gen_preds"][1], y_1.float().to(device)
                )
                fused_loss_0 = self.loss_func(
                    output["fused_preds"][0], y_0.float().to(device)
                )
                fused_loss_1 = self.loss_func(
                    output["fused_preds"][1], y_1.float().to(device)
                )
                env_loss = self.env_loss_func(
                    output["env_pred"], batch_env_ids.to(device)
                )
                loss = (
                    fused_loss_0
                    + fused_loss_1
                    + self.uni_coe * (uni_loss_0 + uni_loss_1)
                    + self.env_coe * env_loss
                    + self.model.get_l2_reg()
                )
                uni_loss_0_sum += uni_loss_0
                uni_loss_1_sum += uni_loss_1
                fused_loss_0_sum += fused_loss_0
                fused_loss_1_sum += fused_loss_1
                env_loss_sum += env_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            uni_loss_0_sum /= len(self.train_loader)
            uni_loss_1_sum /= len(self.train_loader)
            fused_loss_0_sum /= len(self.train_loader)
            fused_loss_1_sum /= len(self.train_loader)
            env_loss_sum /= len(self.train_loader)
            print(
                "uni_loss_0:{:.4f}, uni_loss_1:{:.4f}, fuse_loss_0:{:.4f}, fuse_loss_1:{:.4f}, env_loss:{:.4f}".format(
                    uni_loss_0_sum,
                    uni_loss_1_sum,
                    fused_loss_0_sum,
                    fused_loss_1_sum,
                    env_loss_sum,
                )
            )
            self.uni_loss_0_list.append(uni_loss_0_sum.item())
            self.uni_loss_1_list.append(uni_loss_1_sum.item())
            self.fused_loss_0_list.append(fused_loss_0_sum.item())
            self.fused_loss_1_list.append(fused_loss_1_sum.item())
            self.env_loss_list.append(env_loss_sum.item())
            if epoch % 2 == 0:
                self.env_ids = self.cluster_2()
            auc_val = self.evaluation_two_task(self.val_loader)
            print(
                "Epoch:{}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}, ".format(
                    epoch, self.task_name[0], auc_val[0], self.task_name[1], auc_val[1]
                )
            )
            if sum(auc_val) > best_auc_score:
                earlystop_count = 0
                best_auc_score = sum(auc_val)
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print("EarlyStopping count {}".format(earlystop_count))
                if earlystop_count == self.patience:
                    print("EarlyStopping at epoch {}".format(epoch))
                    break

    @torch.no_grad()
    def cluster_2(self):
        self.model.eval()
        loss_func = nn.BCELoss(reduction="none")
        new_env_tensors_list = []
        for y_0, y_1, _, features in self.train_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.cluster_predict(features)
            loss_0 = loss_func(pred[0].cpu(), y_0.float())
            loss_1 = loss_func(pred[1].cpu(), y_1.float())
            loss = torch.stack([loss_0, loss_1], dim=1)
            new_env_tensors_list.append(torch.argmin(loss, dim=1))
        all_new_env_tensors = torch.cat(new_env_tensors_list, dim=0)
        env_diff = self.env_ids - all_new_env_tensors
        diff_num = torch.nonzero(env_diff != 0).shape[0]
        counts = torch.unique(all_new_env_tensors, return_counts=True)
        envs = [0, 0]
        for i in range(len(counts[0])):
            envs[counts[0][i]] = counts[1][i]
        print("diff_num:{}, env_0:{}, env_1:{}".format(diff_num, envs[0], envs[1]))
        return all_new_env_tensors

    @torch.no_grad()
    def evaluation_two_task(self, data_loader):
        self.model.eval()
        (y0_true, y1_true, y0_hat, y1_hat) = ([], [], [], [])
        for y_0, y_1, _, features in data_loader:
            for key in features.keys():
                features[key] = features[key].to(self.device)
            pred = self.model.predict(features)
            y0_true.append(y_0)
            y1_true.append(y_1)
            y0_hat.append(pred[0])
            y1_hat.append(pred[1])
        y0_true = torch.cat(y0_true)
        y1_true = torch.cat(y1_true)
        y0_hat = torch.cat(y0_hat)
        y1_hat = torch.cat(y1_hat)
        auc_score_0 = roc_auc_score(y0_true.int(), y0_hat.cpu())
        auc_score_1 = roc_auc_score(y1_true.int(), y1_hat.cpu())
        return [auc_score_0, auc_score_1]


class PairedTrainManager(MPTRecTrainManager):
    """Original Stage-1 optimizer/loss with generic validation records."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_records = []
        self.cluster_records = []
        self._started = time.perf_counter()

    def evaluation_two_task(self, data_loader):
        scores = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.epoch_records.append(
                {
                    "epoch": len(self.epoch_records) + 1,
                    "auc_val_0": float(scores[0]),
                    "auc_val_1": float(scores[1]),
                    "wall_seconds": time.perf_counter() - self._started,
                }
            )
        return scores

    def cluster_2(self):
        previous = self.env_ids.clone()
        updated = super().cluster_2()
        counts = torch.bincount(
            updated,
            minlength=(
                self.model.base.num_tasks
                if hasattr(self.model, "base")
                else self.model.num_tasks
            ),
        )
        self.cluster_records.append(
            {
                "epoch": len(self.epoch_records),
                "diff_num": int((previous != updated).sum()),
                "env_counts": [int(value) for value in counts.tolist()],
            }
        )
        return updated
