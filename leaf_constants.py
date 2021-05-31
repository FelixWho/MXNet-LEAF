from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
import gluonnlp as nlp
import numpy as np
from mxnet.contrib import text
import json
import re
# Configuration file for LEAF datasets

LEAF_IMPLEMENTED_DATASETS = [
    'FEMNIST',
    'CELEBA',
    'SENT140'
]
#list: list of implemented LEAF datasets

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
        #femnist_cnn.add(gluon.nn.Dense(2048, activation="relu")) # removed due to gpu space limitation
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

def split_line(line):
    # copied from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    '''split given line/phrase into list of words
    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)

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

def line_to_indices(line, word2id, max_words=25):
    # copied from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words
    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list
    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    #print(line_list)
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl

class MeanPoolingLayer(gluon.HybridBlock):
    """A block for mean pooling of encoder features"""
    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, data, valid_length): # pylint: disable=arguments-differ
        """Forward logic"""
        # Data will have shape (T, N, C)
        masked_encoded = F.SequenceMask(data,
                                        sequence_length=valid_length,
                                        use_sequence_length=True)
        agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0),
                                    F.expand_dims(valid_length, axis=1))
        return agg_state

class SENT140_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(SENT140_gluonnlp, self).__init__(prefix=prefix, params=params)
        _, indd, _ = get_word_emb_arr("embeddings/embs.json")
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(input_dim=len(indd)+1, output_dim=50)
            self.encoder = gluon.rnn.LSTM(50, num_layers=1)
            #self.agg_layer = MeanPoolingLayer()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                #self.output.add(gluon.rnn.RNN(300))
                #self.output.add(gluon.nn.Dense(128))
                self.output.add(gluon.nn.Dense(2))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        #agg_state = self.agg_layer(encoded, 25)
        #out = self.output(agg_state)
        encoding = nd.concat(encoded[0], encoded[-1])
        out = self.output(encoding)
        return out

def build_SENT140_gluonnlp():
    sent140_rnn = SENT140_gluonnlp()
    return sent140_rnn

def build_SENT140():
    _, indd, leaf_vocab = get_word_emb_arr("embeddings/embs.json") 
    max_size = 25
    # Sent140 RNN found at https://github.com/TalwalkarLab/leaf/blob/master/models/sent140/stacked_lstm.py
    sent140_rnn = gluon.nn.HybridSequential()
    with sent140_rnn.name_scope():
        sent140_rnn.add(gluon.nn.Embedding(input_dim=len(indd)+1, output_dim=50))
        sent140_rnn.add(gluon.rnn.LSTM(50, num_layers=1))
        sent140_rnn.add(gluon.rnn.RNN(50))
        #sent140_rnn.add(gluon.nn.Dense(128))
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
