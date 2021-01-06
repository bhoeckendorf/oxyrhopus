from dataclasses import dataclass
from typing import Tuple
from copy import deepcopy
import torch.optim


@dataclass
class AdadeltaConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "ADADELTA"
    lr: float = 1.0
    rho: float = 0.9
    eps: float = 1e-06
    weight_decay: float = 0.0


@dataclass
class AdamConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "Adam"
    lr: float = 0.001
    betas: Tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class AdamWConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "AdamW"
    lr: float = 0.001
    betas: Tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class ASGDConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "ASGD"
    lr: float = 0.01
    lambd: float = 0.0001
    alpha: float = 0.75
    t0: float = 1000000.0
    weight_decay: float = 0.0


@dataclass
class RMSpropConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "RMSprop"
    lr: float = 0.01
    alpha: float = 0.99
    eps: float = 1e-08
    weight_decay: float = 0.0
    momentum: float = 0.0
    centered: bool = False


@dataclass
class SGDConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "SGD"
    lr: float = 0.0001
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


def get_optimizers():
    return {
        "adadelta": AdadeltaConfig,
        "adam": AdamConfig,
        "adamw": AdamWConfig,
        "asgd": ASGDConfig,
        "rmsprop": RMSpropConfig,
        "sgd": SGDConfig
    }


def get_optimizer(model_params, *args, **kwargs):
    _args = deepcopy(args[0])
    for i in args[1:]:
        _args = {**deepcopy(i), **_args}
    _args = {**deepcopy(kwargs), **_args}
    
    try:
        _args.pop("_target_")
    except KeyError:
        pass
    
    name = _args.pop("name").lower().strip()
    if name == "adadelta":
        return torch.optim.Adadelta(model_params, **_args)
    elif name == "adam":
        return torch.optim.Adam(model_params, **_args)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, **_args)
    elif name == "asgd":
        return torch.optim.ASGD(model_params, **_args)
    elif name == "rmsprop":
        return torch.optim.AdamW(model_params, **_args)
    elif name == "sgd":
        return torch.optim.SGD(model_params, **_args)
    else:
        raise ValueError(f"Optimizer '{name}' not implemented")


@dataclass
class NoneLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "None"


@dataclass
class StepLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "StepLR"
    step_size: int = 20
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


def get_lr_schedulers():
    return {
        "none": NoneLRConfig,
        "steplr": StepLRConfig
    }


def get_lr_scheduler(optimizer, *args, **kwargs):
    _args = deepcopy(args[0])
    for i in args[1:]:
        _args = {**deepcopy(i), **_args}
    _args = {**deepcopy(kwargs), **_args}
    
    try:
        _args.pop("_target_")
    except KeyError:
        pass
    
    name = _args.pop("name").lower().strip()
    if name == "none":
        return None
    elif name == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, **_args)
    else:
        raise ValueError(f"LR scheduler '{name}' not implemented")
