from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from mxnet.contrib import text
import json
"""Configuration file for LEAF datasets"""

LEAF_IMPLEMENTED_DATASETS = [
    'FEMNIST',
    'CELEBA'
]
"""list: list of implemented LEAF datasets"""

def build_FEMNIST():
    # FEMNIST CNN found at https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
    # Use padding=2 with kernel_size=5 to mimic 'same' padding found in TensorFlow
    femnist_cnn = gluon.nn.Sequential()
    with femnist_cnn.name_scope():
        femnist_cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=5, padding=2, activation='relu'))
        femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        femnist_cnn.add(gluon.nn.Conv2D(channels=64, kernel_size=5, padding=2, activation='relu'))
        femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        femnist_cnn.add(gluon.nn.Flatten())
        #femnist_cnn.add(gluon.nn.Dense(2048, activation="relu"))
        femnist_cnn.add(gluon.nn.Dense(62))
    return femnist_cnn

def build_CELEBA():
    # CelebA CNN found at https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py
    # Use padding=1 with kernel_size=3 to mimic 'same' padding found in TensorFlow
    celeba_cnn = gluon.nn.Sequential()
    with celeba_cnn.name_scope():
        for _ in range(2):
            celeba_cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1, activation='relu'))
            celeba_cnn.add(gluon.nn.BatchNorm())
            celeba_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            celeba_cnn.add(gluon.nn.Activation('relu'))
        celeba_cnn.add(gluon.nn.Flatten())
        celeba_cnn.add(gluon.nn.Dense(2))
    return celeba_cnn

def get_word_emb_arr(path):
    # copied from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = np.array(embs['emba'])
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab

def build_SENT140():
    # embeddings
    glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.300d.txt')        
    print(glove_embedding.token_to_idx)
    #embeds = glove_embedding.get_vecs_by_tokens(vocab.idx_to_token)
    
    print("embedding shape: " + str(embeds.shape))

    leaf_vocab = get_word_emb_arr("embeddings/embs.json") 
    print(len(leaf_vocab))
    #print(len)
    # Sent140 RNN found at https://github.com/TalwalkarLab/leaf/blob/master/models/sent140/stacked_lstm.py
    sent140_rnn = gluon.nn.Sequential()
    with sent140_rnn.name_scope():
        #sent140_rnn.add(gluon.nn.Embedding(, len(vocab)))
        sent140_rnn.add(gluon.rnn.LSTM(100, 2))
        sent140_rnn.add(gluon.rnn.RNN(100))
        sent140_rnn.add(gluon.nn.Dense(128))
        sent140_rnn.add(gluon.nn.Dense(2))
    return sent140_rnn

LEAF_MODELS = {
#    'sent140.bag_dnn': , # lr, num_classes
#    'sent140.stacked_lstm': (0.0003, 25, 2, 100), # lr, seq_len, num_classes, num_hidden
#    'sent140.bag_log_reg': (0.0003, 2), # lr, num_classes
    'SENT140': build_SENT140,
    'FEMNIST': build_FEMNIST,
#    'shakespeare.stacked_lstm': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'CELEBA': build_CELEBA,
#    'synthetic.log_reg': (0.0003, 5, 60), # lr, num_classes, input_dim
#    'reddit.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
}
"""dict: Model specific architecture"""
