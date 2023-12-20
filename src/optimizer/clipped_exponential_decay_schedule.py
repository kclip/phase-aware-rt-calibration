import tensorflow as tf


class ClippedExponentialDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        minimal_learning_rate: float,
        n_steps_decay: int
    ):
        self.initial_learning_rate = tf.constant(initial_learning_rate, dtype=tf.float32)
        self.minimal_learning_rate = tf.constant(minimal_learning_rate, dtype=tf.float32)
        self.n_steps_decay = tf.constant(n_steps_decay, dtype=tf.float32)
        self.decay_rate = (self.minimal_learning_rate / self.initial_learning_rate) ** (1 / self.n_steps_decay)

    def __call__(self, step: tf.int32) -> tf.float32:
        return tf.maximum(
            self.minimal_learning_rate,
            self.initial_learning_rate * tf.pow(self.decay_rate, tf.cast(step, tf.float32))
        )
