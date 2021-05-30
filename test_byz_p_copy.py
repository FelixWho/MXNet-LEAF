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

from leaf_constants import LEAF_IMPLEMENTED_DATASETS, LEAF_MODELS, get_word_emb_arr, line_to_indices

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
        cnn = LEAF_MODELS[dataset]()
    else:
        raise NotImplementedError
    return cnn

def get_net(net_type, num_outputs=10, dataset='FashionMNIST'):
    # define the model architecture
    if args.net == 'cnn':
        net = get_cnn(num_outputs, dataset)
    else:
        raise NotImplementedError
    print(net)
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
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels

def evaluate_accuracy(data_iterator, net, ctx, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    count = [0,0]
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        if i ==0 :
            print(data)
        #data = mx.nd.transpose(data.as_in_context(ctx))
        data =data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = nd.argmax(output, axis=1)                
        predictions = predictions[remaining_idx]
        #print(predictions.asnumpy()[0])
        for j in predictions.asnumpy():
            count[int(j)] = count[int(j)]+1
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)        
    print(count)
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

    if dataset == 'SENT140':
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
                        #x = mx.nd.transpose(x)
                        user_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
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
                        x = x.astype(int).reshape(1, max_words)
                        #x = mx.nd.transpose(x)
                        #all_testing_x.append(mx.nd.transpose(line_to_indices(x[4], indd, max_words).reshape(1,max_words)))
                        all_testing_x.append(x)
                    for y in data['user_data'][user]['y']:
                        y = np.float32(y)
                        all_testing_y.append(y)
    else:
        raise NotImplementedError
    test_dataset = mx.gluon.data.dataset.ArrayDataset(all_testing_x, all_testing_y)
    #test_dataset = mx.gluon.data.dataset.ArrayDataset(alt_testing_x, alt_testing_y)
    return all_training, test_dataset

def load_data(dataset):
    if dataset in LEAF_IMPLEMENTED_DATASETS:
        train_data, test_dataset = retrieve_leaf_data(dataset)
        test_data = mx.gluon.data.DataLoader(test_dataset, 1, shuffle=False, last_batch='rollover')
    else: 
        raise NotImplementedError
    return train_data, test_data

def assign_data_leaf(train_data, ctx, server_pc=100, p=0.1, dataset='FEMNIST', seed=1):
   
    n = len(train_data) # total amount of users
    num_users_in_server = int(server_pc) # how many users to keep for server
    num_workers = n - num_users_in_server

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]   
    server_data = []
    server_label = [] 

    # randomly shuffle users into workers and those who will be incorporated into server
    random.shuffle(train_data)

    for i in range(0, n):
        for _, (data, label) in enumerate(train_data[i]):
            for (x, y) in zip(data, label):
                if dataset == 'FEMNIST':
                    x = x.as_in_context(ctx).reshape(1,1,28,28)
                elif dataset == 'CELEBA':
                    x = x.as_in_context(ctx).reshape(1,3,84,84)
                elif dataset == 'SENT140':
                    max_size = 25
                    #x = mx.nd.transpose(x.as_in_context(ctx))
                    x = x.as_in_context(ctx).reshape(1,max_size)
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
  
    flat_list_x = []
    flat_list_y = []
    for sublist in each_worker_data:
        for item in sublist:
            flat_list_x.append(item)
    for sublist in each_worker_label:
        for item in sublist:
            flat_list_y.append(item)
    flat_list_x = nd.concat(*flat_list_x, dim=0)
    flat_list_y = nd.concat(*flat_list_y, dim=0)

    print("num workers: "+str(len(each_worker_label)))
    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

    # randomly permute the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    
    for each_worker in each_worker_data:
        print(each_worker.shape)

    return server_data, server_label, flat_list_x, flat_list_y
    
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
        # embedding
        glove = gluonnlp.embedding.create('glove', source='glove.6B.300d')
        net[0].weight.set_data(glove.idx_to_vec)
        #net.embedding.weight.set_data(glove.idx_to_vec)
        # loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
        # trainer
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

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
            num_workers = len(train_data) - args.server_pc # instead of args.nworkers, # workers = total users in dataset - users assigned to server
            server_data, server_label, each_worker_data, each_worker_label = assign_data_leaf(
                                                                            train_data, ctx, server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
        else:
            server_data, server_label, each_worker_data, each_worker_label = assign_data(
                                                                    train_data, args.bias, ctx, num_labels=num_labels, num_workers=num_workers, 
                                                                    server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
        
        
        print("x shape: "+str(each_worker_data.shape))
        print("y shape: "+str(each_worker_label.shape))
    
        print("sample x: "+str(each_worker_data[0]))
        print("sample y: "+str(each_worker_label[0]))

        # begin training        
        for e in range(niter):            
            print(e)
            minibatch = np.random.choice(list(range(each_worker_data.shape[0])), size=batch_size, replace=False)
            print(each_worker_data[minibatch].shape)
            with autograd.record():
                x = each_worker_data[minibatch]
                #x.attach_grad()
                output = net(x)
                loss = softmax_cross_entropy(output, each_worker_label[minibatch])
            loss.backward()
            #print(x.grad)
            #print(net[3].weight.data())
            trainer.step(batch_size)
            #grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])

            #del grad_list
            #grad_list = []
            
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
