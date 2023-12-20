from typing import List
from dataclasses import dataclass


# Learning rate scheduler
# -----------------------

@dataclass()
class _SchedulerMetadataBase(object):
    __scheduler_class__ = None


@dataclass()
class SchedulerConstantMetadata(_SchedulerMetadataBase):
    __scheduler_class__ = "__constant_value__"

    value: float


@dataclass()
class SchedulerPiecewiseMetadata(_SchedulerMetadataBase):
    __scheduler_class__ = "PiecewiseConstantDecay"

    steps: List[int]
    values: List[float]


@dataclass()
class SchedulerExponentialDecayMetadata(_SchedulerMetadataBase):
    __scheduler_class__ = "ExponentialDecay"

    initial_learning_rate: float
    decay_steps: int
    decay_rate: float


@dataclass()
class SchedulerClippedExponentialDecayMetadata(_SchedulerMetadataBase):
    __scheduler_class__ = "ClippedExponentialDecay"

    initial_learning_rate: float
    minimal_learning_rate: float
    n_steps_decay: int


# Optimizer
# ---------

@dataclass()
class _OptimizerMetadataBase(object):
    __optimizer_class__ = None

    scheduler_metadata: _SchedulerMetadataBase


@dataclass()
class OptimizerSGDMetadata(_OptimizerMetadataBase):
    __optimizer_class__ = "SGD"


@dataclass()
class OptimizerAdamMetadata(_OptimizerMetadataBase):
    __optimizer_class__ = "Adam"

    beta_1: float = 0.9  # Tensorflow default value
    beta_2: float = 0.999  # Tensorflow default value
