import os
from typing import Tuple, Optional
from dataclasses import dataclass
import torchvision
import torchvision.transforms as transforms


class CIFAR10:
    
    def __init__(
            self,
            name: str = "CIFAR-10",
            normalize_mean: Tuple[float,float,float] = (0.49139968, 0.48215841, 0.44653091),
            normalize_std: Tuple[float,float,float] = (0.24703223, 0.24348513, 0.26158784),
            normalize_inplace: bool = False,
            resize_to: Optional[Tuple[int,int]] = None,
            data_root: Optional[str] = None
            ):
        if (data_root is None or len(data_root) == 0) and "MYDIR" in os.environ and os.path.exists(os.environ["MYDIR"]):
            data_root = f"{os.environ['MYDIR']}/data/standard/pytorch"
        else:
            data_root = None
        
        transform = [transforms.Resize(resize_to)] if resize_to is not None else []            
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std, normalize_inplace)
        ])
        transform = transforms.Compose(transform)

        self.trnset = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=transform)
        self.tstset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        self.classes = tuple(self.trnset.classes)


class CIFAR100:
    
    def __init__(
            self,
            name: str = "CIFAR-100",
            normalize_mean: Tuple[float,float,float] = (0.50707516, 0.48654887, 0.44091784),
            normalize_std: Tuple[float,float,float] = (0.26733429, 0.25643846, 0.27615047),
            normalize_inplace: bool = False,
            resize_to: Optional[Tuple[int,int]] = None,
            data_root: Optional[str] = None
            ):
        if (data_root is None or len(data_root) == 0) and "MYDIR" in os.environ and os.path.exists(os.environ["MYDIR"]):
            data_root = f"{os.environ['MYDIR']}/data/standard/pytorch"
        else:
            data_root = None
        
        transform = [transforms.Resize(resize_to)] if resize_to is not None else []            
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std, normalize_inplace)
        ])
        transform = transforms.Compose(transform)

        self.trnset = torchvision.datasets.CIFAR100(root=data_root, train=True,  download=True, transform=transform)
        self.tstset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        self.classes = tuple(self.trnset.classes)


class MNIST:
    
    def __init__(
            self,
            name: str = "MNIST",
            normalize_mean: Tuple[float] = (0.13066048, ),
            normalize_std: Tuple[float] = (0.30810781, ),
            normalize_inplace: bool = False,
            resize_to: Optional[Tuple[int,int]] = None,
            data_root: Optional[str] = None
            ):
        if (data_root is None or len(data_root) == 0) and "MYDIR" in os.environ and os.path.exists(os.environ["MYDIR"]):
            data_root = f"{os.environ['MYDIR']}/data/standard/pytorch"
        else:
            data_root = None
        
        transform = [transforms.Resize(resize_to)] if resize_to is not None else []            
        transform.extend([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std, normalize_inplace)
        ])
        transform = transforms.Compose(transform)

        self.trnset = torchvision.datasets.MNIST(root=data_root, train=True,  download=True, transform=transform)
        self.tstset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        self.classes = tuple(self.trnset.classes)


@dataclass
class CIFAR10Config:
    _target_: str = "oxyrhopus.CIFAR10"
    name: str = "CIFAR-10"
    normalize_mean: Tuple[float,float,float] = (0.49139968, 0.48215841, 0.44653091)
    normalize_std: Tuple[float,float,float] = (0.24703223, 0.24348513, 0.26158784)
    normalize_inplace: bool = False
    resize_to: Optional[Tuple[int,int]] = None
    data_root: Optional[str] = None


@dataclass
class CIFAR100Config:
    _target_: str = "oxyrhopus.CIFAR100"
    name: str = "CIFAR-100"
    normalize_mean: Tuple[float,float,float] = (0.50707516, 0.48654887, 0.44091784)
    normalize_std: Tuple[float,float,float] = (0.26733429, 0.25643846, 0.27615047)
    normalize_inplace: bool = False
    resize_to: Optional[Tuple[int, int]] = None
    data_root: Optional[str] = None


@dataclass
class MNISTConfig:
    _target_: str = "oxyrhopus.MNIST"
    name: str = "MNIST"
    normalize_mean: Tuple[float] = (0.13066048, )
    normalize_std: Tuple[float] = (0.30810781, )
    normalize_inplace: bool = False
    resize_to: Optional[Tuple[int,int]] = None
    data_root: Optional[str] = None


def get_datasets():
    return {
        "cifar-10": CIFAR10Config,
        "cifar-100": CIFAR100Config,
        "mnist": MNISTConfig
    }
