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

from leaf_constants import LEAF_IMPLEMENTED_DATASETS, LEAF_MODELS, get_word_emb_arr, line_to_indices, word_to_indices, ALL_LETTERS, NUM_LETTERS, _tokens_to_ids, load_vocab, batch_data, SequenceLoss_v3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_pc", help="the number of data the server holds", type=int, default=100)
    parser.add_argument("--dataset", help="dataset", type=str, default="FashionMNIST")
    parser.add_argument("--bias", help="de", type=float, default=0.5)
    parser.add_argument("--net", help="net", type=str, default="cnn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=32)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.006)
    parser.add_argument("--nworkers", help="# workers", type=int, default=100)
    parser.add_argument("--niter", help="# iterations", type=int, default=2500)
    parser.add_argument("--gpu", help="index of gpu", type=int, default=-1)
    parser.add_argument("--nrepeats", help="seed", type=int, default=1)
    parser.add_argument("--nbyz", help="# byzantines", type=int, default=20)
    parser.add_argument("--byz_type", help="type of attack", type=str, default="no")
    parser.add_argument("--aggregation", help="aggregation", type=str, default="fltrust")
    parser.add_argument("--p", help="bias probability of 1 in server sample", type=float, default=0.1)
    return parser.parse_args()

def get_device(device):
    # define the device to use
    if device == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(device)
    return ctx
    
def get_cnn(num_outputs=10, dataset='FashionMNIST'):
    # define the architecture of the CNN
    if dataset == 'FashionMNIST':
        cnn = gluon.nn.Sequential()
        with cnn.name_scope():
            cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=3, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=3, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            cnn.add(gluon.nn.Flatten())
            cnn.add(gluon.nn.Dense(100, activation="relu"))
            cnn.add(gluon.nn.Dense(num_outputs))
    elif dataset in LEAF_IMPLEMENTED_DATASETS and dataset in LEAF_MODELS:
        print("using custom LEAF model...")
        cnn = LEAF_MODELS[dataset]()
    else:
        raise NotImplementedError
    return cnn

def get_rnn(num_outputs=10, dataset='SENT140'):
    # define the architecture of the RNN
    if dataset in LEAF_IMPLEMENTED_DATASETS and dataset in LEAF_MODELS:
        print("using custom LEAF model...")
        rnn = LEAF_MODELS[dataset]()
    else:
        raise NotImplementedError
    return rnn

def get_net(net_type, num_outputs=10, dataset='FashionMNIST'):
    # define the model architecture
    if net_type == 'cnn':
        net = get_cnn(num_outputs, dataset)
    elif net_type == 'rnn':
        net = get_rnn(num_outputs, dataset)
    else:
        raise NotImplementedError
    return net
    
def get_shapes(dataset):
    # determine the input/output shapes 
    if dataset == 'FashionMNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'FEMNIST': # LEAF dataset
        num_inputs = 28 * 28
        num_outputs = 62
        num_labels = 62
    elif dataset == 'CELEBA': # LEAF dataset
        num_inputs = 178 * 218
        num_outputs = 2
        num_labels = 2
    elif dataset == 'SENT140': # LEAF dataset
        num_inputs = 25 # unsure what to put for text classification
        num_outputs = 2
        num_labels = 2
    elif dataset == 'SHAKESPEARE':
        num_inputs = 80
        num_outputs = 80
        num_labels = 80
    elif dataset == 'REDDIT':
        num_inputs = 10
        num_outputs = None
        num_labels = None
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        if i == 0:
            print("raw label: "+str(label))
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=2)                
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        if i == 0:
            print("output: "+str(output))
            print("argmax output for pred: "+str(predictions))
            print("label: "+str(label))
        acc.update(preds=predictions, labels=label)        
    return acc.get()[1]

def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine.no_byz
    elif byz_type == 'trim_attack':
        return byzantine.trim_attack 
    else:
        raise NotImplementedError

def retrieve_leaf_data(dataset):
    train_data_path = 'leaf/data/%s/data/train' % dataset.lower()
    test_data_path = 'leaf/data/%s/data/test' % dataset.lower()
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        raise IOError("Data not found. Make sure data has been downloaded.")

    all_training = []
    all_testing_x = []
    all_testing_y = []
    all_masks = [] # for use in REDDIT dataset

    if dataset == 'FEMNIST':
        # preprocess training data
        for filename in os.listdir(train_data_path):
            with open(os.path.join(train_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    user_x = []
                    user_y = []
                    for x in data['user_data'][user]['x']: # currently, x's are 1D arrays, must transform the x's
                        x = mx.nd.array(x) # convert to ndarray
                        x = x.astype(np.float32).reshape(1,28,28) # convert 1D into 2D ndarray
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        user_y.append(y)
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x, user_y), 1, shuffle=True, last_batch='rollover')) # append a dataset per user
        # preprocess testing data
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for x in data['user_data'][user]['x']:
                        x = mx.nd.array(x)
                        x = x.astype(np.float32).reshape(1,28,28)
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
                    user_x = []
                    user_y = []
                    for image in data['user_data'][user]['x']: # list of image file names
                        x = mx.img.imread(os.path.join(raw_data_path, image))
                        x = mx.img.imresize(x, 84, 84) # resize to 84x84 according to LEAF model
                        x = nd.transpose(x.astype(np.float32), (2,0,1)) / 255
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        user_y.append(y)
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x, user_y), 1, shuffle=True, last_batch='rollover')) # append a dataset per user
        # preprocess testing data
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for image in data['user_data'][user]['x']: # list of image file names
                        x = mx.img.imread(os.path.join(raw_data_path, image))
                        x = mx.img.imresize(x, 84, 84) # resize to 84x84 according to LEAF model
                        x = nd.transpose(x.astype(np.float32), (2,0,1)) / 255
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_testing_y.append(y)  
    elif dataset == 'SENT140':
        # get word embeddings
        max_words = 25
        _, indd, leaf_vocab = get_word_emb_arr("embeddings/embs.json")

        for filename in os.listdir(train_data_path):
            with open(os.path.join(train_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    user_x = []
                    user_y = []
                    for x in data['user_data'][user]['x']:
                        x = line_to_indices(x[4], indd, max_words)
                        x = mx.nd.array(x)
                        x = x.astype(int).reshape(1, max_words)
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(y)
                        user_y.append(y)
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x, user_y), 1, shuffle=True, last_batch='rollover')) # append a dataset per user
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for x in data['user_data'][user]['x']:
                        x = line_to_indices(x[4], indd, max_words)
                        x = mx.nd.array(x)
                        x = x.astype(int)
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(y)
                        all_testing_y.append(y)
    elif dataset == 'SHAKESPEARE':
        max_len = 80
        for filename in os.listdir(train_data_path):
            with open(os.path.join(train_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    user_x = []
                    user_y = []
                    for x in data['user_data'][user]['x']:
                        x = word_to_indices(x)[:max_len]
                        x = mx.nd.array(x)
                        x = x.astype(int).reshape(1, max_len)
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(ALL_LETTERS.find(y))
                        user_y.append(y)
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x, user_y), 1, shuffle=True, last_batch='rollover')) # append a dataset per user
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    for x in data['user_data'][user]['x']:
                        x = word_to_indices(x)[:max_len]
                        x = mx.nd.array(x)
                        x = x.astype(int)
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float64(ALL_LETTERS.find(y))
                        all_testing_y.append(y)
    elif dataset == 'REDDIT':
        _, vocab_size, _, _ = load_vocab()
        i = 0
        for filename in os.listdir(train_data_path):
            with open(os.path.join(train_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    #if i > 50: # used to cap num samples to test faster. Remove eventually
                    #    break
                    #i+=1
                    user_x, user_y, lengths, masks = batch_data(data['user_data'][user])
                    all_masks.append(masks)
                    all_training.append(
                            mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(user_x, user_y), 1, shuffle=False, last_batch='rollover')) # append a dataset per user
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    #if i > 100:
                    #    break
                    #i+=1
                    user_x, user_y, lengths, masks = batch_data(data['user_data'][user])
                    for i in range(len(user_x)):
                        user_x[i] = nd.reshape(user_x[i], (-1))
                    for i in range(len(user_y)):
                        user_y[i] = nd.reshape(user_y[i], (-1))
                    all_testing_x.extend(user_x)
                    all_testing_y.extend(user_y)
    else:
        raise NotImplementedError
    print(all_testing_x)
    test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
    return all_training, test_dataset, all_masks

def load_data(dataset):
    # load the dataset
    masks = None
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset in LEAF_IMPLEMENTED_DATASETS:
        train_data, test_dataset, masks = retrieve_leaf_data(dataset)
        test_data = mx.gluon.data.DataLoader(test_dataset, 50, shuffle=False, last_batch='rollover')
    else: 
        raise NotImplementedError
    return train_data, test_data, masks
    
def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="FashionMNIST", seed=1):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]   
    server_data = []
    server_label = []
 
    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if dataset == "FashionMNIST":
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            else:
                raise NotImplementedError
            y = y.as_in_context(ctx)
            
            upper_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1) + bias
            lower_bound = (y.asnumpy()) * (1. - bias) / (num_labels - 1)
            rd = np.random.random_sample()
            
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()
            
            if server_counter[int(y.asnumpy())] < samp_dis[int(y.asnumpy())]:
                server_data.append(x)
                server_label.append(y)
                server_counter[int(y.asnumpy())] += 1
            else:
                rd = np.random.random_sample()
                selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)

    if not len(server_data) == 0: 
        server_data = nd.concat(*server_data, dim=0)
        server_label = nd.concat(*server_label, dim=0)
    
    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]
    
    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    for each_worker in each_worker_data:
        print(each_worker.shape)
    return server_data, server_label, each_worker_data, each_worker_label

def assign_data_leaf(train_data, masks, ctx, server_pc=100, p=0.1, dataset='FEMNIST', seed=1):
    n = len(train_data) # total amount of users
    num_users_in_server = int(server_pc) # how many users to keep for server
    num_workers = n - num_users_in_server

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
    each_worker_mask = [[] for _ in range(num_workers)] # for use in REDDIT dataset
    server_data = []
    server_label = [] 

    #print(train_data[0])
    #print(masks[0])

    # randomly permute all the data
    random_order = np.random.RandomState(seed=seed).permutation(n)
    train_data = [train_data[i] for i in random_order]
    if masks is not None:
        masks = [masks[i] for i in random_order]

    for i in range(0, n):
        for _, (data, label) in enumerate(train_data[i]):
            for (x, y) in zip(data, label):
                if dataset == 'FEMNIST':
                    x = x.as_in_context(ctx).reshape(1,1,28,28)
                elif dataset == 'CELEBA':
                    x = x.as_in_context(ctx).reshape(1,3,84,84)
                elif dataset == 'SENT140':
                    max_size = 25
                    x = x.as_in_context(ctx).reshape(1,max_size)
                elif dataset == 'SHAKESPEARE':
                    max_len = 80
                    x = x.as_in_context(ctx).reshape(1,max_len)
                elif dataset == 'REDDIT':
                    max_len = 10
                    x = x.as_in_context(ctx).reshape(1,max_len)
                else:
                    raise NotImplementedError
                y = y.as_in_context(ctx)
                if i < num_workers:
                    each_worker_data[i].append(x)
                    each_worker_label[i].append(y)
                else:
                    server_data.append(x)
                    server_label.append(y)

    if not len(server_data) == 0:
        server_data = nd.concat(*server_data, dim=0)
        server_label = nd.concat(*server_label, dim=0)

    #print(each_worker_data[0])
    #print(masks[0])
    #print(masks[0])
   
    print("num workers: "+str(len(each_worker_label)))
    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]
    masks = [nd.concat(*mask, dim=0) for mask in masks] # for use in REDDIT dataset


    # randomly permute the workers
    random_worker_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_worker_order]
    each_worker_label = [each_worker_label[i] for i in random_worker_order]
    masks = [masks[i] for i in random_worker_order] # for use in REDDIT dataset
    
    #for i in range(len(each_worker_data)):
        #print(each_worker_data[i].shape)
        #print(masks[i].shape)

    return server_data, server_label, each_worker_data, each_worker_label, masks
    
def main(args):
    # device to use
    ctx = get_device(args.gpu)
    batch_size = args.batch_size
    num_inputs, num_outputs, num_labels = get_shapes(args.dataset)
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter

    paraString = 'p'+str(args.p)+ '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(args.bias)+ "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"
 
    with ctx:
        print("fetching appropriate model...")
        # model architecture
        net = get_net(args.net, num_outputs, args.dataset)
        # initialization
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # set embedding (layer 0) weights for sentiment analysis SENT140 task
        if(args.dataset in ['SENT140']):
            glove = gluonnlp.embedding.create('glove', source='glove.6B.50d')
            net.embedding.weight.set_data(glove.idx_to_vec)
            for param in net.embedding.collect_params().values():
                param.grad_req = 'null'
        # loss
        if args.dataset == 'REDDIT':
            lossFunction = SequenceLoss_v3(axis=1, batch_axis=0)
        else:
            lossFunction = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []
        
        # fix the seeds
        seed = args.nrepeats
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        print("loading data...") 
        # load the data
        train_data, test_data, masks = load_data(args.dataset)

        print("assigning data...")
        # assign data to the server and clients
        if args.dataset in LEAF_IMPLEMENTED_DATASETS:
            # since LEAF already separates data by user, we go by that instead of user arguments
            num_workers = len(train_data) - args.server_pc # instead of args.nworkers, # workers = total users in dataset - users assigned to server
            server_data, server_label, each_worker_data, each_worker_label, masks = assign_data_leaf(
                                                                    train_data, masks, ctx, server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
        else:
            server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
        
        print("first worker data shape: "+str(each_worker_data[0].shape))
        print("first worker data type: "+str(each_worker_data[0].dtype))
        #print("first worker data: "+str(each_worker_data[0]))
        #print("first mask: "+str(masks[0]))
        print("first worker first sample shape: "+str(each_worker_data[0][0].shape))
        print("first worker first sample type: "+str(each_worker_data[0][0].dtype))

        # begin training        
        for e in range(niter):            
            print(e)
            for i in range(num_workers):
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                #print(each_worker_data[i][minibatch])
                #print(masks[i][minibatch])
                #net.summary(each_worker_data[0][minibatch])
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    #print("output shape: "+str(output.shape))
                    #print("label shape: "+str(each_worker_label[i][minibatch].shape))
                    #print("weight shape: "+str(masks[i][minibatch].shape))
                    #net.summary(each_worker_data[0][minibatch])
                    #print(each_worker_label[i][minibatch])
                    if args.dataset == 'REDDIT':
                        loss = lossFunction(output, each_worker_label[i][minibatch], masks[i][minibatch])
                    else:
                        loss = lossFunction(output, each_worker_label[i][minibatch])
                loss.backward()
                #print(loss.mean().asscalar())
                grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])
            #print("gradlist len: "+str(len(grad_list)))
            #net.summary(each_worker_data[0][minibatch])
            if args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
                minibatch = np.random.choice(list(range(server_data.shape[0])), size=args.server_pc, replace=False)
                with autograd.record():
                    output = net(server_data)
                    loss = lossFunction(output, server_label)
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])
                # perform the aggregation
                nd_aggregation.fltrust(e, grad_list, net, lr, args.nbyz, byz)
            elif args.aggregation == "simple":
                nd_aggregation.simple_mean(e, grad_list, net, lr, args.nbyz, byz)
            elif args.aggregation == "trim":
                nd_aggregation.trim(e, grad_list, net, lr, args.nbyz, byz)
            elif args.aggregation == "median":
                nd_aggregation.median(e, grad_list, net, lr, args.nbyz, byz)

            del grad_list
            grad_list = []
            
            # evaluate the model accuracy
            if (e + 1) % 50 == 0:
                #evaluate_accuracy(test_data, net, ctx)
                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                test_acc_list.append(test_accuracy)
                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))

        del test_acc_list
        test_acc_list = []

if __name__ == "__main__":
    args = parse_args()
    main(args)
