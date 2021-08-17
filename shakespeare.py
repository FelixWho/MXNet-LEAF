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

TRAIN_DATA_PATH = 'leaf/data/shakespeare/data/train'
TEST_DATA_PATH = 'leaf/data/shakespeare/data/test'
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

class SHAKESPEARE_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(SHAKESPEARE_gluonnlp, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(input_dim=80, output_dim=8)
            self.encoder = gluon.rnn.LSTM(256, num_layers=2)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(80))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        encoding = nd.concat(encoded[0], encoded[-1])
        out = self.output(encoding)
        return out

class ShakespeareModule(Module):
    def __init__(self):
        self.max_len = 80

    ### SHAKESPEARE utilities ###

    # taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    def word_to_indices(self, word):
        '''returns a list of character indices
        Args:
            word: string
        
        Return:
            indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            indices.append(ALL_LETTERS.find(c))
        makeup = 80 - len(indices)
        for i in range(makeup):
            indices.append(1) # append 1's signifying spaces for a buffer to reach 80
        return indices

    def createModel(self):
        shakespeare_rnn = SHAKESPEARE_gluonnlp()
        return shakespeare_rnn

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
                        x = self.word_to_indices(x)[:self.max_len]
                        x = mx.nd.array(x)
                        x = x.astype(int).as_in_context(ctx).reshape(1, self.max_len)
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(ALL_LETTERS.find(y))
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
                        x = self.word_to_indices(x)[:self.max_len]
                        x = mx.nd.array(x)
                        x = x.astype(int)
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(ALL_LETTERS.find(y))
                        all_testing_y.append(y)
        test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
        test_data = mx.gluon.data.DataLoader(test_dataset, 50, shuffle=False, last_batch='rollover')

        return test_data