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

from leaf_constants import LEAF_IMPLEMENTED_DATASETS, LEAF_MODELS

dataset = 'FEMNIST'
niter = 2000
ctx = mx.cpu()
cnn = LEAF_MODELS[dataset]
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(cnn.collect_params(), 'sgd', {'learning_rate': 0.0003})

def retrieve_leaf_data(dataset):
    train_data_path = 'leaf/data/%s/data/train' % dataset.lower()
    test_data_path = 'leaf/data/%s/data/test' % dataset.lower()
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise IOError("Data not found. Make sure data has been downloaded.")

    all_training_x = []
    all_training_y = []
    all_testing_x = []
    all_testing_y = []

    if dataset == 'FEMNIST':
        # preprocess training data
        for filename in os.listdir(train_data_path):
            with open(os.path.join(train_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for x in data['user_data'][user]['x']: # currently, x's are 1D arrays, must transform the x's
                        x = mx.nd.array(x) # convert to ndarray
                        x = x.astype(np.float32).reshape(1,28,28) # convert 1D into 2D ndarray
                        all_training_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_training_y.append(y)
        # preprocess testing data
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for x in data['user_data'][user]['x']:
                        x = mx.nd.array(x)
                        x = x.reshape(1,28,28)
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_testing_y.append(y)

    elif dataset == 'CELEBA':
        raw_data_path = 'leaf/data/celeba/data/raw/img_align_celeba'
        # preprocess training data
        for filename in os.listdir(train_data_path):
            with open(os.path.join(train_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for image in data['user_data'][user]['x']: # list of image file names
                        x = mx.img.imread(os.path.join(raw_data_path, image))
                        x = mx.img.imresize(x, 84, 84) # resize to 84x84 according to LEAF model
                        x = nd.transpose(x.astype(np.float32), (2,0,1)) / 255
                        all_training_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_training_y.append(y)
        # preprocess testing data
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for image in data['user_data'][user]['x']: # list of image file names
                        x = mx.img.imread(os.path.join(raw_data_path, image))
                        print(x)
                        x = mx.img.imresize(x, 84, 84) # resize to 84x84 according to LEAF model
                        x = nd.transpose(x.astype(np.float32), (2,0,1)) / 255
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_testing_y.append(y)    
    else:
        raise NotImplementedError
    assert len(all_training_x) == len(all_training_y)
    assert len(all_testing_x) == len(all_testing_y)
    train_dataset = mx.gluon.data.dataset.ArrayDataset(all_training_x, all_training_y)
    test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
    return train_dataset, test_dataset

train_data_loader, _ = retrieve_leaf_data(dataset)

epochs = 5
for epoch in range(epochs):
    # training loop (with autograd and trainer steps, etc.)
    cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
    training_samples = 0
    for batch_idx, (data, label) in enumerate(train_data_loader):
        data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
        label = label.as_in_context(ctx)
        with autograd.record():
            output = cnn(data)
            loss = criterion(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_train_loss += loss.sum()
        training_samples += data.shape[0]
    train_loss = cumulative_train_loss.asscalar()/training_samples

    print("Epoch {}, training loss: {:.2f}".format(epoch, train_loss))
