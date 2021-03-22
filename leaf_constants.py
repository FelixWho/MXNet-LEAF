from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
"""Configuration file for LEAF datasets"""

LEAF_IMPLEMENTED_DATASETS = [
    'FEMNIST',
    'CELEBA'
]
"""list: list of implemented LEAF datasets"""

# FEMNIST CNN found at https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
# Use padding=2 with kernel_size=5 to mimic 'same' padding found in TensorFlow
femnist_cnn = gluon.nn.Sequential()
with femnist_cnn.name_scope():
    femnist_cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=5, padding=2, activation='relu'))
    femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    femnist_cnn.add(gluon.nn.Conv2D(channels=64, kernel_size=5, padding=2, activation='relu'))
    femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    femnist_cnn.add(gluon.nn.Flatten())
    femnist_cnn.add(gluon.nn.Dense(2048, activation="relu"))
    femnist_cnn.add(gluon.nn.Dense(62))

# CelebA CNN found at https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py
# Use padding=1 with kernel_size=3 to mimic 'same' padding found in TensorFlow
celeba_cnn = gluon.nn.Sequential()
with celeba_cnn.name_scope():
    for _ in range(4):
        celeba_cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1, activation='relu'))
        celeba_cnn.add(gluon.nn.BatchNorm())
        celeba_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        celeba_cnn.add(gluon.nn.Activation('relu'))
    celeba_cnn.add(gluon.nn.Flatten())
    celeba_cnn.add(gluon.nn.Dense(2))

LEAF_MODELS = {
#    'sent140.bag_dnn': , # lr, num_classes
#    'sent140.stacked_lstm': (0.0003, 25, 2, 100), # lr, seq_len, num_classes, num_hidden
#    'sent140.bag_log_reg': (0.0003, 2), # lr, num_classes
    'FEMNIST': femnist_cnn,
#    'shakespeare.stacked_lstm': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'CELEBA': celeba_cnn,
#    'synthetic.log_reg': (0.0003, 5, 60), # lr, num_classes, input_dim
#    'reddit.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
}
"""dict: Model specific parameter specification"""