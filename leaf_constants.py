from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
import gluonnlp as nlp
import numpy as np
from mxnet.contrib import text
import json
import re
import pickle
import collections

LEAF_IMPLEMENTED_DATASETS = [
    'FEMNIST',
    'CELEBA',
    'SENT140',
    'SHAKESPEARE',
    'REDDIT'
]

### FEMNIST utilities

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

### CELEBA utilities

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

### SENT140 utilities

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

class SENT140_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(SENT140_gluonnlp, self).__init__(prefix=prefix, params=params)
        _, indd, _ = get_word_emb_arr("embeddings/embs.json")
        with self.name_scope():
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

def build_SENT140_gluonnlp():
    sent140_rnn = SENT140_gluonnlp()
    return sent140_rnn

### SHAKESPEARE utilities

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

def word_to_indices(word):
    # altered from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
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

class SHAKESPEARE_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(SHAKESPEARE_gluonnlp, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(input_dim=80, output_dim=8)
            self.encoder = gluon.rnn.LSTM(256, num_layers=2)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                # self.output.add(gluon.nn.Dense(128))
                self.output.add(gluon.nn.Dense(80))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        encoding = nd.concat(encoded[0], encoded[-1])
        out = self.output(encoding)
        return out

def build_SHAKESPEARE_gluonnlp():
    shakespeare_rnn = SHAKESPEARE_gluonnlp()
    return shakespeare_rnn

### REDDIT utilities

VOCABULARY_PATH = 'leaf/data/reddit/vocab/reddit_vocab.pck'

def load_vocab():
    vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
    vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
    vocab.update(vocab_file['vocab'])

    return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

def _tokens_to_ids(raw_batch, p = False):
    vocab, _, _, _ = load_vocab()
    if p:
        print(raw_batch)
    def tokens_to_word_ids(tokens, vocab):
        return [vocab[word] for word in tokens]

    to_ret = [tokens_to_word_ids(seq, vocab) for seq in raw_batch]
    if p:
        print(to_ret)
    return to_ret

class REDDIT_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(REDDIT_gluonnlp, self).__init__(prefix=prefix, params=params)
        vocab, vocab_size, unk_symbol, pad_symbol = load_vocab()
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(input_dim=vocab_size, output_dim=256)
            self.encoder = gluon.rnn.HybridSequentialRNNCell()
            with self.encoder.name_scope():
                self.encoder.add(gluon.rnn.LSTMCell(256))
                self.encoder.add(gluon.rnn.DropoutCell(0))
                self.encoder.add(gluon.rnn.LSTMCell(256))
                self.encoder.add(gluon.rnn.DropoutCell(0))
            self.output = None

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        encoding = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        out = nd.reshape(nd.concat(encoded, dim=1), (-1, 256))
        return out

def build_REDDIT_gluonnlp():
    reddit_rnn = REDDIT_gluonnlp()
    return reddit_rnn

LEAF_MODELS = {
#    'sent140.bag_dnn': , # lr, num_classes
#    'sent140.stacked_lstm': (0.0003, 25, 2, 100), # lr, seq_len, num_classes, num_hidden
#    'sent140.bag_log_reg': (0.0003, 2), # lr, num_classes
    'SENT140': build_SENT140_gluonnlp,
    'FEMNIST': build_FEMNIST,
#    'shakespeare.stacked_lstm': (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
    'CELEBA': build_CELEBA,
    'SHAKESPEARE': build_SHAKESPEARE_gluonnlp,
    'REDDIT': build_REDDIT_gluonnlp
#    'synthetic.log_reg': (0.0003, 5, 60), # lr, num_classes, input_dim
#    'reddit.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
}
