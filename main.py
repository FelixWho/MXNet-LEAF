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

# To add a new module, import it, then add it to 'nameToModuleDict'
from femnist import FemnistModule
from celeba import CelebaModule
from sent140 import Sent140Module
from shakespeare import ShakespeareModule
from reddit import RedditModule

nameToModuleDict = {"FEMNIST" : FemnistModule,
                    "CELEBA" : CelebaModule,
                    "SENT140" : Sent140Module,
                    "SHAKESPEARE" : ShakespeareModule,
                    "REDDIT" : RedditModule}

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

def get_byz(byz_type):
    # get the attack type
    if byz_type == "no":
        return byzantine.no_byz
    elif byz_type == 'trim_attack':
        return byzantine.trim_attack 
    else:
        raise NotImplementedError

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

    # randomly permute all the data
    random_order = np.random.RandomState(seed=seed).permutation(n)
    train_data = [train_data[i] for i in random_order]
    if masks is not None:
        masks = [masks[i] for i in random_order]

    for i in range(0, n):
        for _, (data, label) in enumerate(train_data[i]):
            for (x, y) in zip(data, label):
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
    if masks is not None:
        masks = [nd.concat(*mask, dim=0) for mask in masks] # for use in REDDIT dataset


    # randomly permute the workers
    random_worker_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_worker_order]
    each_worker_label = [each_worker_label[i] for i in random_worker_order]
    if masks is not None:
        masks = [masks[i] for i in random_worker_order] # for use in REDDIT dataset
    
    for i in range(len(each_worker_data)):
        print(each_worker_data[i].shape)
        #print(masks[i].shape)

    return server_data, server_label, each_worker_data, each_worker_label, masks

def evaluate_accuracy(data_iterator, net, ctx, module, trigger=False, target=None):
    # evaluate the (attack) accuracy of the model
    acc = module.getAccuracyMetric()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        remaining_idx = list(range(data.shape[0]))
        if trigger:
            data, label, remaining_idx, add_backdoor(data, label, trigger, target)
        output = net(data)
        predictions = module.getPredictionsFromNetworkOutput(output)            
        predictions = predictions[remaining_idx]
        label = label[remaining_idx]
        acc.update(preds=predictions, labels=label)        
    return acc.get()[1]

def main(args):
    # device to use
    ctx = get_device(args.gpu)
    batch_size = args.batch_size
    byz = get_byz(args.byz_type)
    num_workers = args.nworkers
    lr = args.lr
    niter = args.niter

    paraString = 'p'+str(args.p)+ '_' + str(args.dataset) + "server " + str(args.server_pc) + "bias" + str(args.bias)+ "+nworkers " + str(
        args.nworkers) + "+" + "net " + str(args.net) + "+" + "niter " + str(args.niter) + "+" + "lr " + str(
        args.lr) + "+" + "batch_size " + str(args.batch_size) + "+nbyz " + str(
        args.nbyz) + "+" + "byz_type " + str(args.byz_type) + "+" + "aggregation " + str(args.aggregation) + ".txt"
 
    with ctx:
        module = nameToModuleDict[args.dataset]()
        
        print("Fetching appropriate model...")
        net = module.createModel()
        
        print("Initialize model parameters...")
        module.initializeModel(net, ctx)
        
        print("Fetching loss function...")
        lossFunction = module.getLossFunction()
        
        # fix the seeds
        seed = args.nrepeats
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        print("Loading data...") 
        train_data, masks = module.loadTrainingData(ctx)
        test_data = module.loadTestingData()

        print("Assigning data...")
        # assign data to the server and clients
        # since LEAF already separates data by user, we go by that instead of user arguments
        num_workers = len(train_data) - args.server_pc # instead of args.nworkers, # workers = total users in dataset - users assigned to server
        server_data, server_label, each_worker_data, each_worker_label, masks = assign_data_leaf(
                                                                train_data, masks, ctx, server_pc=args.server_pc, p=args.p, dataset=args.dataset, seed=seed)
 
        grad_list = [] # holds parameter-wise gradient
        test_acc_list = [] # holds historical accuracy
        
        # begin training        
        for e in range(niter):            
            print(e)
            for i in range(num_workers):
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    #print("_____")
                    #print(output)
                    if args.dataset == 'REDDIT':
                        loss = lossFunction(output, each_worker_label[i][minibatch], masks[i][minibatch])
                    else:
                        loss = lossFunction(output, each_worker_label[i][minibatch])
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])
            #print(grad_list)
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
                test_accuracy = evaluate_accuracy(test_data, net, ctx, module)
                test_acc_list.append(test_accuracy)
                print("Iteration %02d. Test_acc %0.4f" % (e, test_accuracy))

        del test_acc_list
        test_acc_list = []

if __name__ == "__main__":
    args = parse_args()
    main(args)