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

#mnist = mx.test_utils.get_mnist()
#train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], 32, shuffle=True)
#val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], 32)

print(mnist['train_data'][0].shape)

def construct_net():
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(2048, activation="relu"))
        net.add(gluon.nn.Dense(1024, activation="relu"))
        net.add(gluon.nn.Dense(128, activation="relu"))
        net.add(gluon.nn.Dense(62))
    return net

dataset = 'FEMNIST'
niter = 2000
gpus = mx.test_utils.list_gpus()
ctx =  [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
#cnn = construct_net()
cnn = LEAF_MODELS[dataset]
cnn.hybridize()
cnn.initialize(mx.init.Xavier(), ctx=ctx)
criterion = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(cnn.collect_params(), 'sgd', {'learning_rate': 0.01})

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
    #print(nd.concat(*all_training_y, dim=0).shape)
    test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
    train_data = mx.gluon.data.DataLoader(train_dataset, 6000, shuffle=True, last_batch='rollover')
    test_data = mx.gluon.data.DataLoader(test_dataset, 250, shuffle=False, last_batch='rollover')
    
    #return all_training_x,all_training_y
    # return nd.concat(*all_training_x, dim=0), nd.concat(*all_training_y, dim=0)
    return train_data, test_data

train_data_loader, valid_data_loader = retrieve_leaf_data(dataset)
metric = mx.metric.Accuracy()
epochs = 20
batch_size = 200

all_x = []
all_y = []
for _, (data,label) in enumerate(train_data_loader):
    for (x,y) in zip(data, label):
        x = x.reshape(1,1,28,28)
        all_x.append(x)
        all_y.append(y)

all_x = nd.concat(*all_x, dim=0)
all_y = nd.concat(*all_y, dim=0)

train_data = mx.io.NDArrayIter(all_x, all_y, batch_size, shuffle=True)

print("here")

for i in range(epochs):
    # Reset the train data iterator.
    train_data.reset()
    # Loop over the train data iterator.
    for batch in train_data:
        # Splits train data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits train labels into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        # Inside training scope
        with autograd.record():
            for x, y in zip(data, label):
                z = cnn(x)
                # Computes softmax cross entropy loss.
                loss = criterion(z, y)
                # Backpropogate the error for one iteration.
                loss.backward()
                outputs.append(z)
        # Updates internal evaluation
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
    # Gets the evaluation result.
    name, acc = metric.get()
    # Reset evaluation result to initial state.
    metric.reset()
    print('training acc at epoch %d: %s=%f'%(i, name, acc))

#for epoch in range(epochs):
    # training loop (with autograd and trainer steps, etc.)
#    cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
#    training_samples = 0
#    outputs = []
#    labels = []
 #   for batch_idx, (data, label) in enumerate(train_data_loader):
  #      data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
   #     label = label.as_in_context(ctx)
    #    labels.append(label)
     #   with autograd.record():
      #      output = cnn(data)
       #     loss = criterion(output, label)
        #    outputs.append(output)
        #loss.backward()
        #metric.update(labels, outputs)
        #trainer.step(data.shape[0])
        #cumulative_train_loss += loss.sum()
        #training_samples += data.shape[0]
    #train_loss = cumulative_train_loss.asscalar()/training_samples
    #print("Epoch {}, training loss: {:.2f}".format(epoch, train_loss))

    #name, acc = metric.get()
    #metric.reset()
    #print('training acc at epoch %d: %s=%f'%(epoch, name, acc))
