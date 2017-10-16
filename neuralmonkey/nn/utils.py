"""
This module provides utility functions used across the package.
"""
import tensorflow as tf


def dropout(variable: tf.Tensor,
            keep_prob: float,
            train_mode: tf.Tensor) -> tf.Tensor:
    """Performs dropout on a variable, depending on mode.

    Arguments:
        variable: The variable to be dropped out
        keep_prob: The probability of keeping a value in the variable
        train_mode: A bool Tensor specifying whether to dropout or not
    """
    # Maintain clean graph - no dropout op when there is none applied
    # TODO maybe use math.isclose instead of this comparison
    with tf.variable_scope("dropout"):
        if keep_prob == 1.0:
            return variable

        # TODO remove this line as soon as TF .12 is used.
        train_mode_selector = tf.fill(tf.shape(variable)[:1], train_mode)
        dropped_value = tf.nn.dropout(variable, keep_prob)
        return tf.where(train_mode_selector, dropped_value, variable)


def sparse_dropout(variable: tf.SparseTensor,
                   keep_prob: float,
                   train_mode: tf.Tensor) -> tf.SparseTensor:
    """Performs dropout on a sparse tensor, depending on mode. """

    with tf.variable_scope("dropout"):
        if keep_prob == 1.0:
            return variable

        p_retain = tf.select(train_mode, keep_prob, 1.0)
        probs = tf.random_uniform(tf.shape(variable.values)) + p_retain
        to_retain = tf.cast(tf.floor(probs), dtype=tf.bool)
        new = tf.sparse_retain(variable, to_retain)
        scale = tf.where(train_mode, 1. / keep_prob, 1.0)
        return new * scale


def sparse_dropout_mask(variable: tf.SparseTensor,
                        keep_prob: float,
                        train_mode: tf.Tensor) -> tf.SparseTensor:
    """Performs dropout on a sparse tensor, depending on mode. """

    shape = tf.shape(variable.values)

    with tf.variable_scope("dropout"):
        if keep_prob == 1.0:
            return tf.fill(shape, True)

        keep_prob = tf.where(train_mode, keep_prob, 1.0)
        probs = tf.random_uniform(shape) + keep_prob
        return tf.cast(tf.floor(probs), dtype=tf.bool)
