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
        print("Using custom LEAF model")
        femnist_cnn = gluon.nn.Sequential()
        with femnist_cnn.name_scope():
            femnist_cnn.add(gluon.nn.Conv2D(channels=32, kernel_size=5, padding=2, activation='relu'))
            femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            femnist_cnn.add(gluon.nn.Conv2D(channels=64, kernel_size=5, padding=2, activation='relu'))
            femnist_cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            femnist_cnn.add(gluon.nn.Flatten())
            femnist_cnn.add(gluon.nn.Dense(2048, activation="relu"))
            femnist_cnn.add(gluon.nn.Dense(62))
        cnn = femnist_cnn #LEAF_MODELS[dataset]
    else:
        raise NotImplementedError
    return cnn

def get_net(net_type, num_outputs=10, dataset='FashionMNIST'):
    # define the model architecture
    if args.net == 'cnn':
        net = get_cnn(num_outputs, dataset)
    else:
        raise NotImplementedError
    return net
    
def get_shapes(dataset):
    # determine the input/output shapes 
    if dataset == 'FashionMNIST':
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    elif dataset == 'FEMNIST': # CMU LEAF dataset
        num_inputs = 28 * 28
        num_outputs = 62
        num_labels = 62
    elif dataset == 'CELEBA': # CMU LEAF dataset
        num_inputs = 178 * 218
        num_outputs = 2
        num_labels = 2
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=1)                
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
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
                    all_training.append(mx.gluon.data.dataset.ArrayDataset(user_x, user_y)) # append a dataset per user
        # preprocess testing data
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    all_testing_x = {'x':[], 'y':[]}
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
                    all_training.append(mx.gluon.data.dataset.ArrayDataset(user_x, user_y)) # append a dataset per user

        # preprocess testing data
        for filename in os.listdir(test_data_path):
            with open(os.path.join(test_data_path, filename)) as f:
                data = json.load(f)
                for user in data['users']:
                    all_testing[user] = {'x':[], 'y':[]}
                    for image in data['user_data'][user]['x']: # list of image file names
                        x = mx.img.imread(os.path.join(raw_data_path, image))
                        x = mx.img.imresize(x, 84, 84) # resize to 84x84 according to LEAF model
                        x = nd.transpose(x.astype(np.float32), (2,0,1)) / 255
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_testing_y.append(y)  
    else:
        raise NotImplementedError

    #train_dataset = mx.gluon.data.dataset.ArrayDataset(all_training_x, all_training_y)
    test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
    return all_training, test_dataset

def load_data(dataset):
    # load the dataset
    if dataset == 'FashionMNIST':
        def transform(data, label):
            return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=True, transform=transform), 60000,shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.FashionMNIST(train=False, transform=transform), 250, shuffle=False, last_batch='rollover')
    elif dataset in LEAF_IMPLEMENTED_DATASETS:
        train_data, test_dataset = retrieve_leaf_data(dataset)
        #train_data = mx.gluon.data.DataLoader(train_dataset, 5000, shuffle=True, last_batch='rollover')
        test_data = mx.gluon.data.DataLoader(test_dataset, 250, shuffle=False, last_batch='rollover')
    else: 
        raise NotImplementedError
    return train_data, test_data
    
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

def assign_data_leaf(train_data, bias, ctx, p=0.1, dataset='FEMNIST', seed=1):
    
    n = len(train_data) # total amount of users
    num_users_in_server = int(p * n) # how many users to keep for server
    num_workers = n - num_users_in_server

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]   
    server_data = []
    server_label = [] 

    # randomly split users into workers and those who will be incorporated into server
    users_list = list(train_data.keys()) 
    random.shuffle(users_list)
    users_as_workers = users_list[num_users_in_server:]
    users_in_server = users_list[:num_users_in_server]

    # add data to server_data and server_label
    for user in users_in_server:
        for x in train_data[user]['x']:
            if dataset == 'FEMNIST':
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            elif dataset == 'CELEBA':
                x = x.as_in_context(ctx).reshape(1,3,84,84)
            else:
                raise NotImplementedError
            server_data.append(x)
        for y in train_data[user]['y']:
            y = y.as_in_context(ctx)
            server_label.append(y)


    for count, user in enumerate(users_in_server):
        for x in train_data[user]['x']:
            if dataset == 'FEMNIST':
                x = x.as_in_context(ctx).reshape(1,1,28,28)
            elif dataset == 'CELEBA':
                x = x.as_in_context(ctx).reshape(1,3,84,84)
            else:
                raise NotImplementedError
            each_worker_data[count].append(x)
        for y in train_data[user]['y']:
            y = y.as_in_context(ctx)
            each_worker_label[count].append(y)

    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    
    for each_worker in each_worker_data:
        print(each_worker.shape)
    return server_data, server_label, each_worker_data, each_worker_label
    
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
        # model architecture
        net = get_net(args.net, num_outputs, args.dataset)
        # initialization
        net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        grad_list = []
        test_acc_list = []
        
        # fix the seeds
        seed = args.nrepeats
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
    
        # load the data
        train_data, test_data = load_data(args.dataset)
        
        # assign data to the server and clients
        if args.dataset in LEAF_IMPLEMENTED_DATASETS:
            # since LEAF already separates data by user, we go by that instead of user arguments
            num_workers = len(train_data) - int(args.p * len(train_data)) # instead of args.nworkers, # workers = total users in dataset - users assigned to server
            #server_data, server_label, each_worker_data, each_worker_label = assign_data_leaf(
            #                                                                train_data, args.bias, ctx, p=args.p, dataset=args.dataset, seed=seed)
            print("num workers" + str(num_workers))
        else:
            server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
        # begin training        
        for e in range(niter):            
            print(e)
            for i in range(num_workers):
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])

            if args.aggregation == "fltrust":
                # compute server update and append it to the end of the list
                minibatch = np.random.choice(list(range(server_data.shape[0])), size=args.server_pc, replace=False)
                with autograd.record():
                    output = net(server_data)
                    loss = softmax_cross_entropy(output, server_label)
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])
                # perform the aggregation
                nd_aggregation.fltrust(e, grad_list, net, lr, args.nbyz, byz)

            del grad_list
            grad_list = []
            
            # evaluate the model accuracy
            if (e + 1) % 50 == 0:
                test_accuracy = evaluate_accuracy(test_data, net, ctx)
                test_acc_list.append(test_accuracy)
                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))

        del test_acc_list
        test_acc_list = []

if __name__ == "__main__":
    args = parse_args()
    main(args)
