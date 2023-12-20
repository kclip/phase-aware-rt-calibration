import numpy as np
import tensorflow as tf

from src.optimizer import _SchedulerMetadataBase, SchedulerConstantMetadata, SchedulerPiecewiseMetadata, \
    SchedulerExponentialDecayMetadata, SchedulerClippedExponentialDecayMetadata, \
    _OptimizerMetadataBase, OptimizerAdamMetadata, OptimizerSGDMetadata
from src.optimizer.clipped_exponential_decay_schedule import ClippedExponentialDecaySchedule


def _get_scheduler(metadata: _SchedulerMetadataBase):
    if metadata.__scheduler_class__ == SchedulerConstantMetadata.__scheduler_class__:
        return metadata.value
    elif metadata.__scheduler_class__ == SchedulerPiecewiseMetadata.__scheduler_class__:
        boundaries = np.cumsum(metadata.steps).tolist()
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries, values=metadata.values
        )
    elif metadata.__scheduler_class__ == SchedulerExponentialDecayMetadata.__scheduler_class__:
        return tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=metadata.initial_learning_rate,
            decay_steps=metadata.decay_steps,
            decay_rate=metadata.decay_rate
        )
    elif metadata.__scheduler_class__ == SchedulerClippedExponentialDecayMetadata.__scheduler_class__:
        return ClippedExponentialDecaySchedule(
            initial_learning_rate=metadata.initial_learning_rate,
            minimal_learning_rate=metadata.minimal_learning_rate,
            n_steps_decay=metadata.n_steps_decay
        )
    else:
        raise ValueError(f"Unknown learning rate scheduler '{metadata.__scheduler_class__}'...")


def get_optimizer(metadata: _OptimizerMetadataBase) -> tf.keras.optimizers.Optimizer:
    if metadata.__optimizer_class__ == OptimizerSGDMetadata.__optimizer_class__:
        return tf.keras.optimizers.SGD(
            _get_scheduler(metadata.scheduler_metadata)
        )
    elif metadata.__optimizer_class__ == OptimizerAdamMetadata.__optimizer_class__:
        return tf.keras.optimizers.Adam(
            _get_scheduler(metadata.scheduler_metadata),
            beta_1=metadata.beta_1,
            beta_2=metadata.beta_2,
        )
    else:
        raise ValueError(f"Unknown optimizer '{metadata.__optimizer_class__}'...")
