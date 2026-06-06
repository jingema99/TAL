import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import log_precision_recall, target2onehot, tensor2numpy
import os
EPSILON = 1e-8

# init_epoch = 200
# init_lr = 0.1
# init_milestones = [60, 120, 170]
# init_lr_decay = 0.1
# init_weight_decay = 0.0005


# epochs = 170
# lrate = 0.1
# milestones = [80, 120]
# lrate_decay = 0.1
# batch_size = 128
# weight_decay = 2e-4
num_workers = 8
T = 2

#best parameters: t = 0.999, r = 2

class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = IncrementalNet(args, False)

    def _before_train_task(self):
        pass

    def _classification_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)

    def _feature_dump_dir(self):
        return "[PROJECT_ROOT]/fig/ce"
        
    def _dump_test_features(self, save_dir: str = None, use_forward_features: bool = False):
        """
        遍历 self.test_loader，把特征和标签保存为 .pt 文件。
        - save_dir: 保存目录，默认 ./feat_dumps
        - use_forward_features: True 时用 forward 返回的 out['features']；否则用 extract_vector()
        """
        self._network.eval()
        net = self._network.module if hasattr(self._network, "module") else self._network

        save_dir = save_dir or self.args.get("feat_dump_dir", "./feat_dumps")
        os.makedirs(save_dir, exist_ok=True)

        all_feats, all_labels, all_indices = [], [], []

        device = self._device
        for indices, inputs, targets in self.test_loader:
            inputs = inputs.to(device, non_blocking=True)

            # 提取特征
            if not use_forward_features:
                feats = net.extract_vector(inputs)   # shape (B, D)
            else:
                out = net(inputs)
                feats = out["features"]

            all_feats.append(feats.cpu())
            all_labels.append(targets.cpu())
            all_indices.append(indices.cpu() if torch.is_tensor(indices) else torch.tensor(indices))

        # 拼接成大张量
        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        tag = getattr(self, "_cur_task", 0)
        csv_name = self.args.get("csv_name", "exp")
        fname = os.path.join(save_dir, f"{csv_name}_task{tag:02d}_test_feats.pt")

        torch.save({
            "features": all_feats,       # (N, D) float tensor
            "labels": all_labels,        # (N,) long tensor
            "indices": all_indices,      # (N,) long tensor
            "total_classes": self._total_classes,
            "known_classes": self._known_classes,
            "task_id": tag,
        }, fname)

        logging.info(f"[FeatureDump] Saved test features to: {fname}")
        
    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        self._before_train_task()

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            
        if self._cur_task == 9:  
            log_precision_recall(
                network=self._network, 
                test_loader=self.test_loader, 
                total_classes=self._total_classes, 
                current_task=self._cur_task, 
                device=self._device) 

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = self._classification_loss(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)


    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss_clf = self._classification_loss(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                loss = loss_clf + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

        self._dump_test_features(save_dir=self._feature_dump_dir())


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
