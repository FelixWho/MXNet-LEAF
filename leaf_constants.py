from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.loss import Loss
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

### FEMNIST utilities ###

def build_FEMNIST():
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

### CELEBA utilities ###

def build_CELEBA():
    # Use padding=1 with kernel_size=3 to mimic 'same' padding found in TensorFlow LEAF model
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

### SENT140 utilities ###

# taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
def split_line(line):
    '''split given line/phrase into list of words
    Args:
        line: string representing phrase to be split
    
    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)

# taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
def get_word_emb_arr(path):
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
def line_to_indices(line, word2id, max_words=25):
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
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl

class SENT140_gluonnlp(gluon.HybridBlock):
    def __init__(self, prefix=None, params=None):
        super(SENT140_gluonnlp, self).__init__(prefix=prefix, params=params)
        _, indd, _ = get_word_emb_arr("embeddings/embs.json")
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

def build_SENT140_gluonnlp():
    sent140_rnn = SENT140_gluonnlp()
    return sent140_rnn

### SHAKESPEARE utilities ###

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)

# taken from https://github.com/TalwalkarLab/leaf/blob/master/models/utils/language_utils.py
def word_to_indices(word):
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
                self.output.add(gluon.nn.Dense(80))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        encoding = nd.concat(encoded[0], encoded[-1])
        out = self.output(encoding)
        return out

def build_SHAKESPEARE_gluonnlp():
    shakespeare_rnn = SHAKESPEARE_gluonnlp()
    return shakespeare_rnn

### REDDIT utilities ###

VOCABULARY_PATH = 'leaf/data/reddit/vocab/reddit_vocab.pck'

def load_vocab():
    vocab_file = pickle.load(open(VOCABULARY_PATH, 'rb'))
    vocab = collections.defaultdict(lambda: vocab_file['unk_symbol'])
    vocab.update(vocab_file['vocab'])

    return vocab, vocab_file['size'], vocab_file['unk_symbol'], vocab_file['pad_symbol']

def _tokens_to_ids(raw_batch, pr = False):
    vocab, _, _, _ = load_vocab()
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

def process_x(raw_x_batch, pad_symbol):
    tokens = _tokens_to_ids([s for s in raw_x_batch])
    lengths = np.sum(tokens != pad_symbol, axis=1)
    return tokens, lengths

def process_y(raw_y_batch):
    tokens = _tokens_to_ids([s for s in raw_y_batch])
    return tokens

def batch_data(data):
    vocab, vocab_size, unk_symbol, pad_symbol = load_vocab()

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
        batched_mask = data_mask[i]

        input_data, input_lengths = process_x([batched_x], pad_symbol)
        target_data = process_y([batched_y])

        user_x.append(input_data)
        user_y.append(target_data)
        lengths.append(input_lengths)
        masks.append(batched_mask)

    return user_x, user_y, lengths, masks

# rewrite from https://github.com/tensorflow/addons/blob/v0.13.0/tensorflow_addons/seq2seq/loss.py#L24-L169
# assumes average_across_timesteps=False, average_across_batch=True
class SequenceLoss(Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(SequenceLoss, self).__init__(weight, batch_axis, **kwargs)

    def sparse_softmax_cross_entropy_with_logits(self, x, y):
        ret = nd.empty(x.shape[0])
        for i in range(x.shape[0]):
            unflattened_logit = nd.reshape(x[i], (1, -1)) # 1D array to 2D array
            label = y[i]
            ret[i] = nd.softmax_cross_entropy(unflattened_logit, label)
        return ret

    # rewrite from https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits
    def hybrid_forward(self, F, logits, targets, weights):
        print(targets)

        targets_rank = len(targets.shape) # unused, can probably remove
        num_classes = logits.shape[2]
        logits_flat = nd.reshape(logits, (-1, num_classes))

        targets = nd.reshape(targets, (-1))
        crossent = self.sparse_softmax_cross_entropy_with_logits(logits_flat, targets)

        crossent *= nd.reshape(weights, (-1))
        reduce_axis = 0
        crossent = nd.sum(crossent, axis = reduce_axis)
        total_count = mx.np.count_nonzero(weights, axis = reduce_axis).astype(crossent.dtype)
        crossent = mx.np.divide(crossent, total_count)
        return nd.mean(crossent)


# rewrite tf.nn.xw_plus_b in gluon
class XW_Plus_B_HybridLayer(gluon.HybridBlock):
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
    def __init__(self, prefix=None, params=None):
        super(REDDIT_gluonnlp, self).__init__(prefix=prefix, params=params)
        vocab, vocab_size, unk_symbol, pad_symbol = load_vocab()
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(input_dim=vocab_size, output_dim=256)
            self.encoder = gluon.nn.HybridSequential()
            with self.encoder.name_scope():
                self.encoder.add(gluon.rnn.LSTM(256, num_layers=1))
                self.encoder.add(gluon.nn.Dropout(0))
                self.encoder.add(gluon.rnn.LSTM(256, num_layers=1))
                self.encoder.add(gluon.nn.Dropout(0))
            self.output = XW_Plus_B_HybridLayer(256, vocab_size)
            #self.output = gluon.nn.HybridSequential()
            # with self.output.name_scope():
            #    self.output.add(gluon.nn.Dense(vocab_size, activation=None, use_bias=True))
    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        inputs = self.embedding(data)
        #print("inputs: "+str(inputs))
        encoding = self.encoder(inputs)
        #encoding = self.encoder(self.embedding(mx.nd.transpose(data)))  # Shape(T, N, C)
        flattened_encoding = nd.reshape(nd.concat(encoding, dim=1), (-1, 256))
        print("flattened encoding shape: "+str(flattened_encoding.shape))
        out = self.output(flattened_encoding)
        print("nn output:")
        print(out)
        return out

def build_REDDIT_gluonnlp():
    reddit_rnn = REDDIT_gluonnlp()
    return reddit_rnn

LEAF_MODELS = {
    'SENT140': build_SENT140_gluonnlp,
    'FEMNIST': build_FEMNIST,
    'CELEBA': build_CELEBA,
    'SHAKESPEARE': build_SHAKESPEARE_gluonnlp,
    'REDDIT': build_REDDIT_gluonnlp
#    'reddit.stacked_lstm': (0.0003, 10, 256, 2), # lr, seq_len, num_hidden, num_layers
}
