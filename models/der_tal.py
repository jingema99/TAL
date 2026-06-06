# Please note that the current implementation of DER only contains the dynamic expansion process, since masking and pruning are not implemented by the source repo.
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, log_precision_recall, target2onehot, tensor2numpy
from utils.tal import TAL_Loss
# from utils.tal import TAL_MaskOldW_Loss

EPSILON = 1e-8
weight_decay = 2e-4
num_workers = 8
T = 2
RECALIB_ALPHA = True

class DER(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.tal = TAL_Loss(
            t=0.999,
            r=0.2,
            recalib_alpha=RECALIB_ALPHA,
        )

        self._network = DERNet(args, False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        # Optional per-task summary for epoch-version TAL implementations.
        if hasattr(self.tal, "group_mean_Q_s"):
            stats = self.tal.group_mean_Q_s(group_size=10, upto_classes=self._total_classes)
            if stats:
                logging.info(f"[TAL] Task {self._cur_task} Q/w group means (group_size=10):")
                for item in stats:
                    logging.info(
                        f"[TAL] classes {item['range']}: Q_mean={item['q_mean']:.6f}, w_mean={item['w_mean']:.6f}"
                    )

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        self.tal.update_class_num(self._total_classes)   # 只增不减；自动频率对齐 alpha
        self.tal.to(self._device)                        # 确保 Q/alpha 在同一 device

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args['init_lr'],
                weight_decay=self.args['init_weight_decay'],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['init_milestones'], gamma=self.args['init_lr_decay']
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.args['lrate'],
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args['milestones'], gamma=self.args['lrate_decay']
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

            log_precision_recall(
                network=self._network, 
                test_loader=self.test_loader, 
                total_classes=self._total_classes, 
                current_task=self._cur_task, 
                device=self._device)     

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['init_epoch']))
        for _, epoch in enumerate(prog_bar):
            self.train()
            if hasattr(self.tal, "begin_epoch"):
                self.tal.begin_epoch()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = self.tal(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if hasattr(self.tal, "end_epoch"):
                self.tal.end_epoch()
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['init_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['epochs']))
        for _, epoch in enumerate(prog_bar):
            self.train()
            if hasattr(self.tal, "begin_epoch"):
                self.tal.begin_epoch()
            losses = 0.0
            losses_clf = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]

                loss_clf = self.tal(logits, targets)
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if hasattr(self.tal, "end_epoch"):
                self.tal.end_epoch()
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

