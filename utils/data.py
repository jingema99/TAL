import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from . import autoaugment
from . import ops

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("[CIFAR100_ROOT]", train=True, download=False)
        test_dataset = datasets.cifar.CIFAR100("[CIFAR100_ROOT]", train=False, download=False)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100_AA(iCIFAR100):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    def __init__(self):
        super().__init__()
        self.use_path = True

        self.train_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.test_trsf = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.class_order = np.arange(100).tolist()


    def download_data(self):
        train_txt_path = "[PROJECT_ROOT]/data/imagenet_subset/train.txt"
        test_txt_path = "[PROJECT_ROOT]/data/imagenet_subset/test.txt"

        self.train_data, self.train_targets = self._load_images_and_labels(train_txt_path)
        self.test_data, self.test_targets = self._load_images_and_labels(test_txt_path)
    
    
    def _load_images_and_labels(self, txt_file):
        paths, targets = [], []
        with open(txt_file, "r") as f:
            for line in f:
                path, target = line.strip().split()
                paths.append(path)
                targets.append(int(target))

        return np.array(paths), np.array(targets, dtype=np.int64)




class iFood101(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=63/255),
        # ImageNetPolicy()
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4409], std=[
                             0.2009, 0.1984, 0.2023]),
    ]

    class_order = np.arange(101).tolist()

    def download_data(self):

        train_txt_path = "[PROJECT_ROOT]/data/food-101/meta/my_train.txt"
        test_txt_path = "[PROJECT_ROOT]/data/food-101/meta/my_test.txt"

        self.train_data, self.train_targets = self._load_images_and_labels(train_txt_path)
        self.test_data, self.test_targets = self._load_images_and_labels(test_txt_path)
    
    
    def _load_images_and_labels(self, txt_file):
        paths, targets = [], []
        with open(txt_file, "r") as f:
            for line in f:
                path, target = line.strip().split()
                paths.append(path)
                targets.append(int(target))

        return np.array(paths), np.array(targets, dtype=np.int64)

