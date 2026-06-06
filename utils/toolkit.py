import os
import logging
import numpy as np
import torch
import  json
from enum import Enum
from sklearn.metrics import precision_score, recall_score

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def log_precision_recall(network, test_loader, total_classes, current_task, device, group_size=10):
    network.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for _, inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = network(inputs)["logits"]
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(tensor2numpy(preds))
            all_targets.extend(tensor2numpy(targets))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    precision_per_class = precision_score(
        all_targets, all_preds, average=None, labels=np.arange(total_classes)
    )
    recall_per_class = recall_score(
        all_targets, all_preds, average=None, labels=np.arange(total_classes)
    )

    num_groups = (total_classes + group_size - 1) // group_size
    for group_id in range(num_groups):
        start = group_id * group_size
        end = min((group_id + 1) * group_size, total_classes)
        precision_avg = precision_per_class[start:end].mean()
        recall_avg = recall_per_class[start:end].mean()

        logging.info(
            f"Task {current_task}, Class Group {start:02d}-{end - 1:02d} => "
            f"Precision: {precision_avg:.3f}, Recall: {recall_avg:.3f}"
        )

    overall_accuracy = np.mean(all_preds == all_targets) * 100.0
    logging.info(f"Task {current_task} => Overall Test Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    network.train()

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)