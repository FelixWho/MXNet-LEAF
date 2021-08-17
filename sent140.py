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

from module import Module

TRAIN_DATA_PATH = 'leaf/data/sent140/data/train'
TEST_DATA_PATH = 'leaf/data/sent140/data/test'

class SENT140_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None, indd=None):
        super(SENT140_gluonnlp, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            # TODO: input_dim, output_dim can also be given by gluon's built in 
            # input_dim, output_dim = vocab.embedding.idx_to_vec.shape
            self.embedding = gluon.nn.Embedding(input_dim=len(indd)+1, output_dim=50)
            self.encoder = gluon.rnn.LSTM(50, num_layers=2)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(128))
                self.output.add(gluon.nn.Dense(2))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        encoding = nd.concat(encoded[0], encoded[-1])
        out = self.output(encoding)
        return out

class Sent140Module(Module):
    def __init__(self):
        self.max_len = 25 # max number of words per data point
        _, self.indd, self.leaf_vocab = self.get_word_emb_arr("embeddings/embs.json")

    ### SENT140 utilities ###

    # taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    def split_line(self, line):
        '''split given line/phrase into list of words
        Args:
            line: string representing phrase to be split
        
        Return:
            list of strings, with each string representing a word
        '''
        return re.findall(r"[\w']+|[.,!?;]", line)

    # taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    def get_word_emb_arr(self, path):
        with open(path, 'r') as inf:
            embs = json.load(inf)
        vocab = embs['vocab']
        word_emb_arr = np.array(embs['emba'])
        indd = {}
        for i in range(len(vocab)):
            indd[vocab[i]] = i
        vocab = {w: i for i, w in enumerate(embs['vocab'])}
        return word_emb_arr, indd, vocab

    # taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    def line_to_indices(self, line, word2id, max_len=25):
        unk_id = len(word2id)
        line_list = self.split_line(line) # split phrase in words
        indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_len]]
        indl += [unk_id]*(max_len-len(indl))
        return indl

    def createModel(self):
        sent140_rnn = SENT140_gluonnlp(indd=self.indd)
        return sent140_rnn

    def loadTrainingData(self, ctx):
        if not os.path.exists(TRAIN_DATA_PATH):
            raise IOError("Training data not found at %s. Make sure data has been downloaded completely." % TRAIN_DATA_PATH)
        all_training = []
        for filename in os.listdir(TRAIN_DATA_PATH):
            with open(os.path.join(TRAIN_DATA_PATH, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    user_x = []
                    user_y = []
                    for x in data['user_data'][user]['x']:
                        x = self.line_to_indices(x[4], self.indd, self.max_len)
                        x = mx.nd.array(x)
                        x = x.astype(int)
                        x = x.as_in_context(ctx).reshape(1, self.max_len)
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(y)
                        user_y.append(y)
                    # Make each user its own dataset
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x, user_y), 1, shuffle=True, last_batch='rollover')) # append a dataset per user
        return all_training, None # second return item is masks, only used in REDDIT dataset

    def loadTestingData(self):
        if not os.path.exists(TEST_DATA_PATH):
            raise IOError("Testing data not found at %s. Make sure data has been downloaded completely." % TEST_DATA_PATH)
        all_testing_x = []
        all_testing_y = []
        for filename in os.listdir(TEST_DATA_PATH):
            with open(os.path.join(TEST_DATA_PATH, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for x in data['user_data'][user]['x']:
                        x = self.line_to_indices(x[4], self.indd, self.max_len)
                        x = mx.nd.array(x)
                        x = x.astype(int)
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(y)
                        all_testing_y.append(y)
        test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
        test_data = mx.gluon.data.DataLoader(test_dataset, 50, shuffle=False, last_batch='rollover')

        return test_data

    def initializeModel(self, net, ctx):
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)

        glove = gluonnlp.embedding.create('glove', source='glove.6B.50d')
        net.embedding.weight.set_data(glove.idx_to_vec)
        for param in net.embedding.collect_params().values():
            param.grad_req = 'null'