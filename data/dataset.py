import torch
import torchvision.datasets 
import torchvision.transforms
import numpy as np
import copy

from utils.configurable import configurable
from data.build import DATASET_REGISTRY


from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, data_and_labels=None, data=None, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.data_and_labels = data_and_labels
        self.transform = transform
        
    def __getitem__(self, index):
        if self.data_and_labels:
            x, y = self.data_and_labels[index]
        else:
            x = self.data[index]
            y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        if self.data_and_labels:
            return len(self.data_and_labels)
        else:
            return len(self.data)


def extract_data_and_labels(ds):
    xs = []
    ys = []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return xs, ys

def maybe_corrupt_labels(labels, n_classes, frac_corrupt, seed):
    if frac_corrupt == 0:
        return labels
    else:
        n_corrupt = int(np.ceil(frac_corrupt * len(labels)))
        rng = np.random.default_rng(seed=seed)
        bad_inds = np.arange(len(labels))
        rng.shuffle(bad_inds)
        bad_inds = bad_inds[:n_corrupt]
        new_labels = copy.copy(labels)
        for b_ind in bad_inds:
            cur_l = labels[b_ind]
            valid_inds = list(range(n_classes))
            valid_inds.remove(int(cur_l))
            rng.shuffle(valid_inds)
            new_l = valid_inds[0]
            new_labels[b_ind] = new_l
        return new_labels


@DATASET_REGISTRY.register()
class MNIST_base:
    @configurable
    def __init__(self, datadir) -> None:
        self.datadir = datadir
        self.n_classes = 10
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
        }
    
    def get_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root=self.datadir, train=True, transform=transform, download=True)
        val_data = torchvision.datasets.MNIST(root=self.datadir, train=False, transform=transform, download=True)
        return train_data, val_data


@DATASET_REGISTRY.register()
class SVHN_base:
    @configurable
    def __init__(self, datadir, frac_samples=None, frac_corrupt=0, seed=None) -> None:
        self.datadir = datadir
        self.n_classes = 10
        self.frac_samples = frac_samples
        self.frac_corrupt = frac_corrupt
        self.seed = seed
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
            "frac_samples": args.frac_samples,
            "frac_corrupt": args.frac_corrupt,
            "seed": args.seed,
        }
    
    def get_data(self):
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
        train_data = torchvision.datasets.SVHN(root=self.datadir, split='train', transform=transform, download=True)

        # corrupt
        data, labels = extract_data_and_labels(train_data)
        labels = maybe_corrupt_labels(labels, self.n_classes, self.frac_corrupt, self.seed)
        train_data = MyDataset(data=data, labels=labels)
        
        inds = list(range(0, int(self.frac_samples*len(train_data))))
        train_data = torch.utils.data.Subset(train_data, inds)
        
        val_data = torchvision.datasets.SVHN(root=self.datadir, split='test', transform=transform, download=True)
        return train_data, val_data



@DATASET_REGISTRY.register()
class CIFAR10_base:
    @configurable
    def __init__(self, datadir, frac_samples=None, frac_corrupt=0, seed=None) -> None:
        self.datadir = datadir

        self.n_classes = 10
        self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
        self.std = np.array([63.0, 62.1, 66.7]) / 255.0
        self.frac_samples = frac_samples
        self.frac_corrupt = frac_corrupt
        self.seed = seed
        
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
            "frac_samples": args.frac_samples,
            "frac_corrupt": args.frac_corrupt,
            "seed": args.seed
        }
    
    def get_data(self):
        train_data = torchvision.datasets.CIFAR10(root=self.datadir, train=True, transform=None, download=True)

        # corrupt
        data, labels = extract_data_and_labels(train_data)
        labels = maybe_corrupt_labels(labels, self.n_classes, self.frac_corrupt, self.seed)
        train_data = MyDataset(data=data, labels=labels)

        # subsample
        inds = list(range(0, int(self.frac_samples*len(train_data))))
        train_data = torch.utils.data.Subset(train_data, inds)
        train_data = MyDataset(data_and_labels=train_data, transform=self._train_transform())
        
        val_data = torchvision.datasets.CIFAR10(root=self.datadir, train=False, transform=self._test_transform(), download=True)
        return train_data, val_data

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return test_transform

@DATASET_REGISTRY.register()
class CIFAR10_cutout(CIFAR10_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            Cutout(size=16, p=0.5),
        ])
        return train_transform



@DATASET_REGISTRY.register()
class CIFAR100_base:
    @configurable
    def __init__(self, datadir, frac_samples=None, frac_corrupt=0, seed=None) -> None:
        self.datadir = datadir
        self.frac_samples = frac_samples
        self.frac_corrupt = frac_corrupt
        self.seed = seed

        self.n_classes = 100
        self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
        self.std = np.array([63.0, 62.1, 66.7]) / 255.0
    
    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
            "frac_samples": args.frac_samples,
            "frac_corrupt": args.frac_corrupt,
            "seed": args.seed,
        }
    
    def get_data(self):
        train_data = torchvision.datasets.CIFAR100(root=self.datadir, train=True, transform=None, download=True)

        # corrupt
        data, labels = extract_data_and_labels(train_data)
        labels = maybe_corrupt_labels(labels, self.n_classes, self.frac_corrupt, self.seed)
        train_data = MyDataset(data=data, labels=labels)

        inds = list(range(0, int(self.frac_samples*len(train_data))))
        train_data = torch.utils.data.Subset(train_data, inds)
        train_data = MyDataset(data_and_labels=train_data, transform=self._train_transform())
        
        val_data = torchvision.datasets.CIFAR100(root=self.datadir, train=False, transform=self._test_transform(), download=True)
        return train_data, val_data

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        return test_transform

@DATASET_REGISTRY.register()
class CIFAR100_cutout(CIFAR100_base):
    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            Cutout(size=16, p=0.5),
        ])
        return train_transform


@DATASET_REGISTRY.register()
class ImageNet_base:
    @configurable
    def __init__(self, datadir) -> None:
        self.datadir = datadir

        self.n_classes = 1000
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
        }
    
    def get_data(self):
        train_dataset = torchvision.datasets.ImageFolder(root=self.datadir + '/train', transform=self._train_transform())
        val_dataset = torchvision.datasets.ImageFolder(root=self.datadir + '/val', transform=self._test_transform())
        return train_dataset, val_dataset

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
            # Cutout()
        ])
        return train_transform

    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return test_transform



class Cutout(object):
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        
        h, w = image.size(1), image.size(2)
        mask = np.ones((h,w), np.float32)

        x = np.random.randint(w)
        y = np.random.randint(h)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask
        return image