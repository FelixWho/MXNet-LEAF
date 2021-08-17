from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import os
import json
import gluonnlp
import re
import pickle
import collections

from module import Module

TRAIN_DATA_PATH = 'leaf/data/reddit/data/train'
TEST_DATA_PATH = 'leaf/data/reddit/data/test'
VOCABULARY_PATH = 'leaf/data/reddit/vocab/reddit_vocab.pck'

class SequenceLoss_v3(gluon.loss.SoftmaxCELoss):
    """
    The sequence softmax cross-entropy loss with masks. Must use (axis=1, batch_axis=0).
    Mimics Tensorflow seq2seq SequenceLoss with average_across_timesteps=False, average_across_batch=True
    Source at https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/seq2seq/loss.py#L24-L169
    One difference: since objective is to minimize mean loss, we apply mean() at the end. 
    """

    # `pred` shape: (`batch_size`, `seq_len`, `vocab_size`)
    # `label` shape: (`batch_size`, `seq_len`)
    # `weights` shape: (`batch_size`,`seq_len`)
    def hybrid_forward(self, F, pred, labels, weights):
        num_classes = pred.shape[2]
        pred_flat = F.reshape(pred, (-1, num_classes))
        labels = F.reshape(labels, (-1))
        flattened_weights = F.reshape(weights, (-1))

        crossent = super(SequenceLoss_v3, self).hybrid_forward(F, pred_flat, labels, sample_weight=None)
        # Apply weights
        crossent = F.broadcast_mul(crossent, flattened_weights)
        crossent = F.reshape(crossent, pred.shape[0:2])
        reduce_axis = 0
        crossent = F.sum(crossent, axis=reduce_axis)
        total_size = F.sum(weights, axis=reduce_axis)

        # Tf has divide_no_nan whereas MXNet does not.
        # Alternative: we take inverse of total_size using foreach
        # and if element is 0, keep it 0
        # Instead of dividing, we can now multiply crossent by the
        # inverse of total_size
        def inverse(data, state):
            if data[0] == 0:
                return data, []
            return 1 / data, []
        total_size_inverse, _ = F.contrib.foreach(inverse, total_size, [])
        total_size_inverse = F.reshape(total_size_inverse, (-1))
        crossent = F.broadcast_mul(crossent, total_size_inverse)
 
        return F.mean(crossent)
        #return crossent

class XW_Plus_B_HybridLayer(gluon.HybridBlock):
    '''
    Rewrite Tensorflow's tf.nn.xw_plus_b layer in gluon
    '''
    def __init__(self, hidden_units, output):
        '''
        hidden_units - number of hidden units
        output - number of outputs
        '''
        super(XW_Plus_B_HybridLayer, self).__init__()

        with self.name_scope():
            self.weights = self.params.get('weights',
                                        grad_req = 'write',
                                        shape=(hidden_units, output),
                                        allow_deferred_init=True)

            self.bias = self.params.get('bias', 
                                        grad_req = 'write',
                                        shape=(output,),
                                        init=mx.initializer.Zero(), # custom init because Xavier requires 2D parameter, which the bias param isn't
                                        allow_deferred_init=True)

    def hybrid_forward(self, F, x, weights, bias):
        # x * weights + bias
        return F.broadcast_add(F.linalg.gemm2(x, weights), bias)

class REDDIT_gluonnlp(gluon.HybridBlock):
    '''
    Hidden units = 64 instead of 256 in original LEAF paper
    Otherwise, tensor size would > 2^31, which would require building MXNet from source to fix
    '''
    def __init__(self, prefix=None, params=None, vocab_size=None):
        super(REDDIT_gluonnlp, self).__init__(prefix=prefix, params=params)
        self.vocab_size = vocab_size
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(input_dim=self.vocab_size, output_dim=128)
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                self.encoder.add(gluon.rnn.LSTM(64, num_layers=1))
                self.encoder.add(gluon.nn.Dropout(0))
                self.encoder.add(gluon.rnn.LSTM(64, num_layers=1))
                self.encoder.add(gluon.nn.Dropout(0))
            self.output = XW_Plus_B_HybridLayer(64, self.vocab_size)

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        inputs = self.embedding(data)
        encoding = self.encoder(inputs)
        flattened_encoding = nd.reshape(nd.concat(encoding, dim=1), (-1, 64))
        out = self.output(flattened_encoding)
        out = nd.reshape(out, (-1, 10, self.vocab_size))

        return out

class REDDIT_Accuracy(mx.metric.EvalMetric):
    def __init__(self, unk_symbol, pad_symbol, num=None):
        super(REDDIT_Accuracy, self).__init__("REDDIT Accuracy", num)
        self.pad_symbol = pad_symbol
        self.unk_symbol = unk_symbol

    def update(self, labels, preds):
        labels_np = labels.asnumpy().astype('int32')
        pred_np = preds.asnumpy().astype('int32')
        mx.metric.check_label_shapes(labels, preds)

        pred_labels_match = np.equal(pred_np, labels_np)

        # Count number of predictions that are 1) one of 
        # unk or pad and 2) match the corresponding label
        # Predicting a correct pad or unk is always considered wrong
        pad_array = np.full(pred_np.shape, self.pad_symbol)
        masked_pad_array = np.equal(pred_np, pad_array)
        incorrect_pad = np.multiply(masked_pad_array, pred_labels_match)
        num_incorrect_pad = (incorrect_pad == 1).sum()

        unk_array = np.full(pred_np.shape, self.unk_symbol)
        masked_unk_array = np.equal(pred_np, unk_array)
        incorrect_unk = np.multiply(masked_unk_array, pred_labels_match)
        num_incorrect_unk = (incorrect_unk == 1).sum()

        # Debug
        # print("total match: "+str((labels_np.flat == pred_np.flat).sum())+" inc pad: "+str(num_incorrect_pad) + " inc unk: "+str(num_incorrect_unk))

        self.sum_metric += (labels_np.flat == pred_np.flat).sum() - num_incorrect_pad - num_incorrect_unk
        self.global_sum_metric += (labels_np.flat == pred_np.flat).sum() - num_incorrect_pad - num_incorrect_unk
        self.num_inst += len(pred_np.flat)
        self.global_num_inst += len(pred_np.flat)

class RedditModule(Module):
    def __init__(self):
        self.max_len = 10
        self.vocab, self.vocab_size, self.unk_symbol, self.pad_symbol = self.load_vocab()

    def load_vocab(self):
        vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
        vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
        vocab.update(vocab_file['vocab'])

        return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

    def _tokens_to_ids(self, raw_batch, pr = False):
        vocab, _, _, _ = self.load_vocab()
        if pr:
            print("raw batch: " + str(raw_batch))
        def tokens_to_word_ids(tokens, vocab):
            return [vocab[word] for word in tokens]

        to_ret = [tokens_to_word_ids(seq, vocab) for seq in raw_batch]
        if pr:
            print("to_ret: " + str(to_ret))
        to_ret = nd.array(to_ret)
        to_ret = to_ret.astype(int)
        return to_ret

    def process_x(self, raw_x_batch, pad_symbol):
        tokens = self._tokens_to_ids([s for s in raw_x_batch])
        lengths = np.sum(tokens != pad_symbol, axis=1)
        return tokens, lengths

    def process_y(self, raw_y_batch):
        tokens = self._tokens_to_ids([s for s in raw_y_batch])
        return tokens

    def batch_data(self, data):
        vocab, vocab_size, unk_symbol, pad_symbol = self.load_vocab()

        data_x = data['x']
        data_y = data['y']

        perm = np.random.permutation(len(data['x']))
        data_x = [data_x[i] for i in perm]
        data_y = [data_y[i] for i in perm]

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq, mask_by_seq = [], [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l['target_tokens'])
                mask_by_seq.extend(l['count_tokens'])

            return data_x_by_seq, data_y_by_seq, mask_by_seq
        
        data_x, data_y, data_mask = flatten_lists(data_x, data_y)

        user_x = []
        user_y = []
        lengths = []
        masks = []

        for i in range(0, len(data_x)):
            batched_x = data_x[i]
            batched_y = data_y[i]
            batched_mask = nd.array([data_mask[i]])

            input_data, input_lengths = self.process_x([batched_x], pad_symbol)
            target_data = self.process_y([batched_y])

            user_x.append(input_data)
            user_y.append(target_data)
            lengths.append(input_lengths)
            masks.append(batched_mask)

        return user_x, user_y, lengths, masks
    
    def createModel(self):
        reddit_rnn = REDDIT_gluonnlp(vocab_size=self.vocab_size)
        return reddit_rnn

    def loadTrainingData(self, ctx):
        if not os.path.exists(TRAIN_DATA_PATH):
            raise IOError("Training data not found at %s. Make sure data has been downloaded completely." % TRAIN_DATA_PATH)
        all_training = []
        all_masks = []
        for filename in os.listdir(TRAIN_DATA_PATH):
            with open(os.path.join(TRAIN_DATA_PATH, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    user_x, user_y, lengths, masks = self.batch_data(data['user_data'][user])
                    user_x_in_context = []
                    for sample in user_x:
                        user_x_in_context.append(sample.as_in_context(ctx).reshape(1, self.max_len))
                    all_masks.append(masks)
                    # Make each user its own dataset
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x_in_context, user_y), 1, shuffle=True, last_batch='rollover')) # append a dataset per user
        return all_training, all_masks # second return item is masks, only used in REDDIT dataset

    def loadTestingData(self):
        if not os.path.exists(TEST_DATA_PATH):
            raise IOError("Testing data not found at %s. Make sure data has been downloaded completely." % TEST_DATA_PATH)
        all_testing_x = []
        all_testing_y = []
        for filename in os.listdir(TEST_DATA_PATH):
            with open(os.path.join(TEST_DATA_PATH, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    user_x, user_y, lengths, masks = self.batch_data(data['user_data'][user])
                    for i in range(len(user_x)):
                        user_x[i] = nd.reshape(user_x[i], (-1))
                    for i in range(len(user_y)):
                        user_y[i] = nd.reshape(user_y[i], (-1))
                    all_testing_x.extend(user_x)
                    all_testing_y.extend(user_y)
        test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
        test_data = mx.gluon.data.DataLoader(test_dataset, 50, shuffle=False, last_batch='rollover')

        return test_data
    
    def getAccuracyMetric(self):
        _, _, unk_symbol, pad_symbol = self.load_vocab()
        return REDDIT_Accuracy(unk_symbol=unk_symbol, pad_symbol=pad_symbol)

    def getPredictionsFromNetworkOutput(self, output):
        return nd.argmax(output, axis=2)

    def getLossFunction(self):
        return SequenceLoss_v3(axis=1, batch_axis=0)