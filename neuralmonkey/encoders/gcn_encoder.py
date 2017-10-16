# tests: lint, mypy

from typing import Optional, Any, Tuple, List

import tensorflow as tf
from typeguard import check_argument_types
import scipy.sparse as sparse
import networkx as nx
import numpy as np
from itertools import count

from neuralmonkey.nn.directed_gcn import DirectedGCN
from neuralmonkey.nn.utils import dropout, sparse_dropout, sparse_dropout_mask
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary, PAD_TOKEN_INDEX

# pylint: disable=invalid-name
AttType = Any  # Type[] or union of types do not work here
#RNNCellTuple = Tuple[tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.RNNCell]
# pylint: enable=invalid-name


# pylint: disable=too-many-instance-attributes
class GCNEncoder(ModelPart, Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences.

    This encoder uses 1-D convolutions with a given window size to
    encode the sentence.

    Multiple layers can be used in order to increase the receptive
    field of a word.

    This encoder does not support factors.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabularies: List[Vocabulary],
                 data_ids: List[str],
                 max_input_len: int,
                 layer_size: int,
                 num_layers=1,
                 residual: Optional[bool]=False,
                 normalize_adjacency: Optional[bool]=False,
                 dropout_keep_prob: float=1.0,
                 edge_dropout_keep_prob: float=1.0,
                 attention_type: Optional[AttType]=None,
                 attention_fertility: int=3,
                 use_noisy_activations: bool=False,
                 parent_encoder: Optional[Any]=None,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None) -> None:
        """Creates a new instance of the graph convolutional sentence encoder

        Arguments:
            vocabularies: Input vocabularies
            data_ids: Identifier of the data series fed to this encoder
            name: An unique identifier for this encoder
            max_input_len: Maximum length of an encoded sequence
            layer_size: Number of units for GCN weights, per dimension

        Keyword arguments:
            num_layers: stack this many CNN layers on top of each other (=>1)
            residual: Use residual connections
            normalize_adjacency: Normalize adjacency matrix A by degree:
              perform D^{-1}A instead of A
            dropout_keep_prob: The dropout keep probability
                (default 1.0)
            edge_dropout_keep_prob: The dropout keep probability for graph
                edges (default 1.0)
            attention_type: The class that is used for creating
                attention mechanism (default None)
            attention_fertility: Fertility parameter used with
                CoverageAttention (default 3).
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(
            self, attention_type, attention_fertility=attention_fertility)

        assert check_argument_types()

        # TODO change vocabularies input to separate deprel-vocabulary input
        # this might be a bit dangerous since vocabularies could be
        # provided in a different order, or one (e.g. POS) could be missing
        self.vocabularies = vocabularies
        self.data_ids = data_ids

        self.max_input_len = max_input_len
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout_keep_p = dropout_keep_prob
        self.edge_dropout_keep_p = edge_dropout_keep_prob
        self.use_noisy_activations = use_noisy_activations
        self.parent_encoder = parent_encoder
        self.residual = residual
        self.normalize_adj = normalize_adjacency

        log("Initializing GCN, name: '{}'".format(self.name))

        assert num_layers > 0, "need at least 1 GCN layer"
        assert parent_encoder is not None, \
            'GCN needs a parent encoder, for example a BiRNN'
        assert parent_encoder.add_start_symbol, \
            'parent encoder needs to add start symbol to its input'

        if "fakedeprels" in self.data_ids:
            log("Adding fake deprels for testing", color="red")

        self.inputs = self.parent_encoder.hidden_states
        self.input_dim = self.inputs.get_shape()[-1]

        tf.assert_equal(self.input_dim, self.layer_size, message=
                        "Input dimension and GCN layer_size must be equal. "
                        "Are you using a bi-directional encoder? "
                        "Then use twice the number of units for the GCN.")

        self.train_mode = self.parent_encoder.train_mode

        self._create_input_placeholders()

        gcn_layers = []
        num_labels = len(self.vocabularies[2])  # number of dep relations

        hidden_states = self.inputs

        with tf.variable_scope(self.name):
            for layer_id in range(self.num_layers):
                name = 'gcn{}'.format(layer_id)
                gcn_layer = DirectedGCN(
                    layer_size, num_labels, self.train_mode,
                    dropout_keep_p=dropout_keep_prob,
                    edge_dropout_keep_p=edge_dropout_keep_prob,
                    residual=residual, name=name)
                hidden_states = gcn_layer(hidden_states,
                                          self.adj, self.labels,
                                          self.adj_inv, self.labels_inv)
                gcn_layers.append(gcn_layer)

        self.hidden_states = hidden_states

        with tf.variable_scope('attention_tensor'):
            self.__attention_tensor = dropout(self.hidden_states,
                                              keep_prob=self.dropout_keep_p,
                                              train_mode=self.train_mode)

        self.encoded = tf.reduce_mean(self.hidden_states, 1)

        log("GCN initialized")

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    @property
    def _attention_mask(self):
        return self.parent_encoder._attention_mask

    @property
    def vocabulary_size(self):
        return len(self.vocabularies[0])

    def _create_weights(self):
        """Create weight tensors for graph convolution. """
        pass

    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""

        # sparse adjacency matrices
        self.adj = tf.sparse_placeholder(tf.float32, name="adjacency")
        self.adj_inv = tf.sparse_placeholder(tf.float32, name="adjacency_inv")

        # sparse label matrices for edges (for 2 directions)
        self.labels = tf.sparse_placeholder(tf.int32, name="labels")
        self.labels_inv = tf.sparse_placeholder(tf.int32, name="labels_inv")

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary with the encoder inputs.

        Encoder input placeholders:
            ``encoder_input``: Stores indices to the vocabulary,
         f       shape (batch, time)
            ``encoder_padding``: Stores the padding (ones and zeros,
                indicating valid words and positions after the end
                of sentence, shape (batch, time)
            ``train_mode``: Boolean scalar specifying the mode (train
                vs runtime)

        Arguments:
            dataset: The dataset to use
            train: Boolean flag telling whether it is training time
        """
        # pylint: disable=invalid-name

        fd = self.parent_encoder.feed_dict(dataset, train)

        assert "heads" in self.data_ids, \
            "data_ids must contain 'heads'"

        assert "deprels" in self.data_ids, \
            "data_ids must contain 'deprels'"

        # get size of words input tensor from parent encoder
        words_tensor = fd[self.parent_encoder.inputs]
        words_batch_size = len(words_tensor)
        words_batch_maxlen = len(words_tensor[0])

        # create labels input
        series_deprels = dataset.get_series("deprels")
        tensor_deprels, _ = self.vocabularies[2].sentences_to_tensor(
            list(series_deprels), self.max_input_len, pad_to_max_len=False,
            train_mode=train, add_start_symbol=True)

        # fix: sometimes the number of deprels is longer than number of words
        # to to badly formatted parser output
        if tensor_deprels.shape != (words_batch_maxlen, words_batch_size):
            tensor_deprels = tensor_deprels[:words_batch_maxlen, :]

        max_input_len_batch, batch_size = tensor_deprels.shape
        tensor_deprels = tensor_deprels.transpose()  # to sentence per row

        # we use the heads data series directly since they are numbers and
        # do not need a vocabulary
        series_heads = dataset.get_series("heads")
        tensor_heads = GCNEncoder._heads_to_tensor(
            series_heads, max_input_len_batch, batch_size)
        tensor_heads = tensor_heads.transpose()      # to sentence per row

        assert words_batch_size == batch_size, \
            "series shape mismatch (batch size) {} vs {}".format(
                words_batch_size, words_batch_size)

        assert words_batch_maxlen == max_input_len_batch, \
            "series shape mismatch (sentence lengths) {} vs {}".format(
                words_batch_maxlen, max_input_len_batch)

        # create adjacency/label matrices input
        graph = nx.DiGraph()  # graph for adjacency
        graph.add_nodes_from(range(max_input_len_batch * batch_size))

        graph_lab = nx.DiGraph()  # graph for labels
        graph_lab.add_nodes_from(range(max_input_len_batch * batch_size))

        for sid, heads, deprels in zip(count(), tensor_heads, tensor_deprels):

            indexes = np.arange(max_input_len_batch, dtype='int32')

            # now we pretend we have 1 big graph for the whole batch
            # node indexes are shifted by max_input_len for each sent in batch
            shift = sid * max_input_len_batch
            heads += shift
            indexes += shift

            # add edges directed from head to dependent
            edges = [(j, i, {'deprel': r})
                     for i, j, r in zip(indexes, heads, deprels)]
            graph.add_edges_from(edges)
            graph_lab.add_edges_from(edges)

            assert len(graph) == (max_input_len_batch * batch_size), \
                'graph size got bigger than it should + {}'.format(edges)

        if "fakedeprels" in self.data_ids:

            # TODO for debugging only - add fake data
            series_fake_deprels = dataset.get_series("fakedeprels")
            series_fake_heads = dataset.get_series("fakeheads")
            tensor_fake_deprels, _ = self.vocabularies[2].sentences_to_tensor(
                list(series_fake_deprels), self.max_input_len,
                pad_to_max_len=False, train_mode=train, add_start_symbol=True)
            tensor_fake_heads = GCNEncoder._heads_to_tensor(
                series_fake_heads, max_input_len_batch, batch_size)

            tensor_fake_heads = tensor_fake_heads.transpose()
            tensor_fake_deprels = tensor_fake_deprels.transpose()

            for sid, heads, deprels in zip(count(), tensor_fake_heads,
                                           tensor_fake_deprels):
                indexes = np.arange(max_input_len_batch, dtype='int32')
                shift = sid * max_input_len_batch
                heads += shift
                indexes += shift
                edges = [(j, i, {'deprel': r})
                         for i, j, r in zip(indexes, heads, deprels)]
                graph.add_edges_from(edges)
                graph_lab.add_edges_from(edges)

        # get adjacency matrix from the graph
        adj = nx.adjacency_matrix(graph)
        labels = nx.adjacency_matrix(graph_lab, weight='deprel')

        # incoming edges become outgoing edges (i->j), dep to head
        adj_inv = nx.adjacency_matrix(graph.reverse())
        labels_inv = nx.adjacency_matrix(graph_lab.reverse(), weight='deprel')

        # conversion to TF SparseTensor format (tuples of coords, values, shape)
        adj = GCNEncoder.sparse_to_tuple(adj)
        adj_inv = GCNEncoder.sparse_to_tuple(adj_inv)

        labels = GCNEncoder.sparse_to_tuple(labels)
        labels_inv = GCNEncoder.sparse_to_tuple(labels_inv)

        fd[self.adj] = adj
        fd[self.adj_inv] = adj_inv
        fd[self.labels] = labels
        fd[self.labels_inv] = labels_inv

        return fd

    @staticmethod
    def _heads_to_tensor(series_heads, max_input_len_batch, batch_size):
        """
        Prepare a tensor with heads directly from the series, without
         a vocabulary, since the heads are already integers.

        :param series_heads:
        :param max_input_len_batch:
        :param batch_size:
        :return:
        """

        # truncate heads by max_input_len, and make sure we only point to
        # words in this sentence; the data could contain any index
        tensor_heads = np.zeros([max_input_len_batch, batch_size],
                                dtype='int32')

        for i, x in enumerate(series_heads):
            v = x[:max_input_len_batch - 1]
            tensor_heads[1:len(v) + 1, i] = v

        tensor_heads = np.clip(tensor_heads, 0, max_input_len_batch - 1)
        return tensor_heads

    @staticmethod
    def sparse_to_tuple(sparse_mx):
        """
        Convert sparse matrix to tuple representation.
        Source: https://github.com/tkipf/gcn
        """
        def to_tuple(mx):
            if not sparse.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx
