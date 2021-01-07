from dataclasses import dataclass
from typing import Any, List, Optional, Union
from omegaconf import MISSING
from copy import deepcopy
import torch.optim


@dataclass
class NoneLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "None"


@dataclass
class LambdaLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "LambdaLR"
    lr_lambda: Any = MISSING  # Union[Any,List[Any]]
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class MultiplicativeLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "MultiplicativeLR"
    lr_lambda: Any = MISSING  # Union[Any,List[Any]]
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class StepLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "StepLR"
    step_size: int = MISSING
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class MultiStepLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "MultiStepLR"
    milestones: List[int] = MISSING
    gamma: float = 0.1
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class ExponentialLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "ExponentialLR"
    gamma: float = MISSING
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class CosineAnnealingLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "CosineAnnealingLR"
    T_max: int = MISSING
    eta_min: float = 0.0
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class CosineAnnealingLRWarmRestartsConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "CosineAnnealingWarmRestarts"
    T_0: int = MISSING
    T_mult: int = 1
    eta_min: float = 0.0
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class ReduceLROnPlateauConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    threshold: float = 0.0001
    threshold_mode: str = "rel"
    cooldown: int = 0
    min_lr: float = 0.0  # Union[float,List[float]]
    eps: float = 1e-08
    verbose: bool = False


@dataclass
class CyclicLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "CyclicLR"
    base_lr: float = MISSING  # Union[float,List[float]]
    max_lr: float = MISSING  # Union[float,List[float]]
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = "triangular"
    gamma: float = 1.0
    scale_fn: Optional[Any] = None
    scale_mode: str = "cycle"
    cycle_momentum: bool = True
    base_momentum: float = 0.8  # Union[float,List[float]]
    max_momentim: float = 0.9  # Union[float,List[float]]
    last_epoch: int = -1
    verbose: bool = False


@dataclass
class OneCycleLRConfig:
    _target_: str = "oxyrhopus.get_lr_scheduler"
    name: str = "OneCycleLR"
    max_lr: float = MISSING  # Union[float,List[float]]
    total_steps: Optional[int] = None
    epochs: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    pct_start: float = 0.3
    annealing_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85  # Union[float,List[float]]
    max_momentum: float = 0.95  # Union[float,List[float]]
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    last_epoch: int = -1
    verbose: bool = False


def get_lr_schedulers():
    return {
        "none": NoneLRConfig,
        "lamba": LambdaLRConfig,
        "multiplicative": MultiplicativeLRConfig,
        "step": StepLRConfig,
        "multistep": MultiStepLRConfig,
        "exponential": ExponentialLRConfig,
        "cosineannealing": CosineAnnealingLRConfig,
        "cosineannealingwarmrestarts": CosineAnnealingLRWarmRestartsConfig,
        "reduceonplateau": ReduceLROnPlateauConfig,
        "cyclic": CyclicLRConfig,
        "onecycle": OneCycleLRConfig
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
    elif name == "lambda":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, **_args)
    elif name == "multiplicative":
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, **_args)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **_args)
    elif name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **_args)
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **_args)
    elif name == "cosineannealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **_args)
    elif name == "cosineannealingwarmrestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **_args)
    elif name == "reduceonplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **_args)
    elif name == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(optimizer, **_args)
    elif name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, **_args)
    else:
        raise ValueError(f"LR scheduler '{name}' not implemented")
