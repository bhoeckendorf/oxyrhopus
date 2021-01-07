from dataclasses import dataclass
from typing import Optional, Tuple
from omegaconf import MISSING
from copy import deepcopy
import torch.optim


@dataclass
class AdadeltaConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "AdaDelta"
    lr: float = 1.0
    rho: float = 0.9
    eps: float = 1e-06
    weight_decay: float = 0.0


@dataclass
class AdagradConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "AdaGrad"
    lr: float = 0.01
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10


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
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class SparseAdamConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "SparseAdam"
    lr: float = 0.001
    betas: Tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-08


@dataclass
class AdamaxConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "AdaMax"
    lr: float = 0.002
    betas: Tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-08
    weight_decay: float = 0.0


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
class LBFGSConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "L-BFGS"
    lr: float = 1.0
    max_iter: int = 20
    max_eval: Optional[int] = None
    tolerance_grad: float = 1e-017
    tolerance_change: float = 1e-09
    history_size: int = 100
    line_search_fn: Optional[str] = None


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
class RpropConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "Rprop"
    lr: float = 0.01
    etas: Tuple[float,float] = (0.5, 1.2)
    step_sizes: Tuple[float,float] = (1e-06, 50)


@dataclass
class SGDConfig:
    _target_: str = "oxyrhopus.get_optimizer"
    name: str = "SGD"
    lr: float = MISSING
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


def get_optimizers():
    return {
        "adadelta": AdadeltaConfig,
        "adagrad": AdagradConfig,
        "adam": AdamConfig,
        "adamw": AdamWConfig,
        "sparseadam": SparseAdamConfig,
        "adamax": AdamaxConfig,
        "asgd": ASGDConfig,
        "lbfgs": LBFGSConfig,
        "rmsprop": RMSpropConfig,
        "rprop": RpropConfig,
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
    elif name == "adagrad":
        return torch.optim.Adagrad(model_params, **_args)
    elif name == "adam":
        return torch.optim.Adam(model_params, **_args)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, **_args)
    elif name == "sparseadam":
        return torch.optim.SparseAdam(model_params, **_args)
    elif name == "adamax":
        return torch.optim.Adamax(model_params, **_args)
    elif name == "asgd":
        return torch.optim.ASGD(model_params, **_args)
    elif name == "lbfgs":
        return torch.optim.LBFGS(model_params, **_args)
    elif name == "rmsprop":
        return torch.optim.RMSprop(model_params, **_args)
    elif name == "rprop":
        return torch.optim.Rprop(model_params, **_args)
    elif name == "sgd":
        return torch.optim.SGD(model_params, **_args)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")
