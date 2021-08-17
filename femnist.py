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

from module import Module

TRAIN_DATA_PATH = 'leaf/data/femnist/data/train'
TEST_DATA_PATH = 'leaf/data/femnist/data/test'

class FemnistModule(Module):

    def createModel(self):
        # Use padding=2 with kernel_size=5 to mimic 'same' padding found in TensorFlow LEAF model
        femnist_cnn = gluon.nn.Sequential()
        with femnist_cnn.name_scope():
            femnist_cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=5, padding=2, activation='relu'))
            femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            femnist_cnn.add(gluon.nn.Conv2D(channels=64, kernel_size=5, padding=2, activation='relu'))
            femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            femnist_cnn.add(gluon.nn.Flatten())
            #femnist_cnn.add(gluon.nn.Dense(2048, activation="relu")) # remove due to gpu space limitation
            femnist_cnn.add(gluon.nn.Dense(62))
        return femnist_cnn

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
                        x = mx.nd.array(x)
                        x = x.astype(np.float32).as_in_context(ctx).reshape(1,1,28,28)
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
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
                        x = mx.nd.array(x)
                        x = x.astype(np.float32).reshape(1,28,28)
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_testing_y.append(y)
        test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
        test_data = mx.gluon.data.DataLoader(test_dataset, 50, shuffle=False, last_batch='rollover')

        return test_data