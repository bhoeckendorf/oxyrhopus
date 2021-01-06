from dataclasses import dataclass, field
from typing import Any, List
# import logging
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from .optimizers import *
from .datasets import *


@dataclass
class TrainConfig:
    num_epochs: int = 5
    batch_size: int = 32
    optimizer: Any = MISSING
    lr_scheduler: Any = MISSING


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: [
       {"data": "cifar-10"},
       {"train.optimizer": "sgd"},
       {"train.lr_scheduler": "none"}
    ])
    project: str = ""
    task: str = ""
    comment: str = ""
#     log_level: str = logging.INFO
    data: Any = MISSING
    train: TrainConfig = TrainConfig()


def get_config_store(name="config", node=Config):
    cs = ConfigStore.instance()
#     for name, node in get_datasets().items():
#         cs.store(group="data", name=name, node=node)
#     for name, node in get_optimizers().items():
#         cs.store(group="train.optimizer", name=name, node=node)
#     for name, node in get_lr_schedulers().items():
#         cs.store(group="train.lr_scheduler", name=name, node=node)
    cs.store(name=name, node=node)
    return cs
