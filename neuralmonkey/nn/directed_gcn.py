# tests: lint, mypy

from typing import Optional, Any, Tuple, List

import tensorflow as tf
from typeguard import check_argument_types
import scipy.sparse as sparse
import networkx as nx
import numpy as np
from itertools import count

from neuralmonkey.nn.utils import dropout, sparse_dropout, sparse_dropout_mask
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary, PAD_TOKEN_INDEX


class DirectedGCN:
    """
    A directed GCN layer.
    """

    def __init__(self, layer_size: int, num_labels: int, train_mode,
                 dropout_keep_p: float = '0.8',
                 edge_dropout_keep_p: float='0.8',
                 residual: Optional[bool] = False,
                 name: str='gcn'):

        self.layer_size = layer_size
        self.num_labels = num_labels
        self.train_mode = train_mode

        self.dropout_keep_p = dropout_keep_p
        self.edge_dropout_keep_p = edge_dropout_keep_p
        self.residual = residual
        self.name = name

        with tf.variable_scope(name):
            self._create_weight_matrices()

    def __call__(self, inputs, adj, labels, adj_inv, labels_inv):

        state_dim = tf.shape(inputs)[2]
        inputs2d = tf.reshape(inputs, [-1, state_dim])

        # graph convolution, heads to dependents ("out")
        # gates are applied through the adjacency matrix values

        # apply sparse dropout
        to_retain = sparse_dropout_mask(
            adj, self.edge_dropout_keep_p, self.train_mode)
        adj = tf.sparse_retain(adj, to_retain)
        labels = tf.sparse_retain(labels, to_retain)

        # apply gates
        gates = tf.matmul(inputs2d, self.w_gate)
        adj *= tf.transpose(gates)

        gates_bias = tf.squeeze(tf.nn.embedding_lookup(
            self.b_gate, labels.values, name='gates_lab'))
        values = tf.nn.sigmoid(adj.values + gates_bias)

        # dropout scaling
        values /= tf.where(self.train_mode, self.edge_dropout_keep_p, 1.0)

        adj = tf.SparseTensor(indices=adj.indices, values=values,
                              dense_shape=adj.dense_shape)

        # graph convolution, heads to dependents ("out")
        h = tf.matmul(inputs2d, self.w)
        h = tf.sparse_tensor_dense_matmul(adj, h)
        labels_pad, _ = tf.sparse_fill_empty_rows(labels, 0)
        labels_weights, _ = tf.sparse_fill_empty_rows(adj, 0.)
        labels = tf.nn.embedding_lookup_sparse(self.b, labels_pad,
                                               labels_weights, combiner='sum')
        h = h + labels
        h = tf.reshape(h, tf.shape(inputs))

        # graph convolution, dependents to heads ("in")
        # gates are applied through the adjacency matrix values

        # apply sparse dropout
        to_retain_inv = sparse_dropout_mask(
            adj_inv, self.edge_dropout_keep_p, self.train_mode)
        adj_inv = tf.sparse_retain(adj_inv, to_retain_inv)
        labels_inv = tf.sparse_retain(labels_inv, to_retain_inv)

        # apply gates
        gates_inv = tf.matmul(inputs2d, self.w_gate_inv)
        adj_inv *= tf.transpose(gates_inv)

        gates_inv_bias = tf.squeeze(tf.nn.embedding_lookup(
            self.b_gate_inv, labels_inv.values, name='gates_inv_lab'))
        values_inv = tf.nn.sigmoid(adj_inv.values + gates_inv_bias)

        # dropout scaling
        values_inv /= tf.where(self.train_mode, self.edge_dropout_keep_p, 1.0)

        adj_inv = tf.SparseTensor(indices=adj_inv.indices,
                                  values=values_inv,
                                  dense_shape=adj_inv.dense_shape)

        # graph convolution, dependents to heads ("in")
        h_inv = tf.matmul(inputs2d, self.w_inv)
        h_inv = tf.sparse_tensor_dense_matmul(adj_inv, h_inv)
        labels_inv_pad, _ = tf.sparse_fill_empty_rows(labels_inv, 0)
        labels_inv_weights, _ = tf.sparse_fill_empty_rows(adj_inv, 0.)
        labels_inv = tf.nn.embedding_lookup_sparse(self.b_inv,
                                                   labels_inv_pad,
                                                   labels_inv_weights,
                                                   combiner='sum')
        h_inv = h_inv + labels_inv
        h_inv = tf.reshape(h_inv, tf.shape(inputs))

        # graph convolution, loops
        gates_loop = tf.nn.sigmoid(tf.matmul(inputs2d, self.w_gate_loop) +
                                   self.b_gate_loop)
        h_loop = tf.matmul(inputs2d, self.w_loop) + self.b_loop
        h_loop = h_loop * gates_loop
        h_loop = tf.reshape(h_loop, tf.shape(inputs))

        # final result is the sum of those (with residual connection to inputs)
        h = tf.nn.relu(h + h_inv + h_loop)

        if self.residual:
            log("GCN has residual connection", color="blue")
            return h + inputs
        else:
            log("GCN without residual connection", color="blue")
            return h

    def _create_weight_matrices(self):
        """Creates all GCN weight matrices."""
        self.w = tf.get_variable(
                "weights",
                [self.layer_size, self.layer_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.w_inv = tf.get_variable(
                "weights_inv",
                [self.layer_size, self.layer_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.w_loop = tf.get_variable(
                "weights_loop",
                [self.layer_size, self.layer_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.w_gate = tf.get_variable(
                "weights_gate",
                [self.layer_size, 1],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.w_gate_inv = tf.get_variable(
                "weights_gate_inv",
                [self.layer_size, 1],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.w_gate_loop = tf.get_variable(
                "weights_gate_loop",
                [self.layer_size, 1],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.b_gate = tf.get_variable(
                "bias_gate",
                [self.num_labels],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        self.b_gate_inv = tf.get_variable(
                "bias_gate_inv",
                [self.num_labels],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
        self.b_gate_loop = tf.get_variable(
                "bias_gate_loop",
                [1],
                initializer=tf.constant_initializer(1.))
        self.b = tf.get_variable(
                "bias_labels",
                [self.num_labels, self.layer_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.b_inv = tf.get_variable(
                "bias_labels_inv",
                [self.num_labels, self.layer_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
        self.b_loop = tf.get_variable(
                "bias_loop",
                [self.layer_size],
                initializer=tf.random_normal_initializer(stddev=0.01))
