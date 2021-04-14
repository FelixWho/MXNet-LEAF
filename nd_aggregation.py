import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

def simple_mean(epoch, gradients, net, lr, f, byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, lr, f)
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size

def trim(epoch, gradients, net, lr, f, byz, b=20):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, lr, f)
    
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    n = len(param_list)
    q = f
    m = n - b*2
    trim_nd = nd.mean(sorted_array[:, b:(b+m)], axis=-1, keepdims=1)
    idx = 0

    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
        
def median(epoch, gradients, net, lr, f, byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, lr, f)
    
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    #print(nd.norm(sorted_array))
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[:, int(sorted_array.shape[-1] / 2)]
    else:
        median_nd = (sorted_array[:, int((sorted_array.shape[-1] / 2 - 1))] + sorted_array[:, int((sorted_array.shape[-1] / 2))]) / 2
    idx = 0

    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * median_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size

def fltrust(epoch, gradients, net, lr, f, byz):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(epoch, param_list, net, lr, f)
    n = len(param_list) - 1 # -1 so as to not include the gradient of the server model
    
    # use the last gradient (server update) as the trusted source
    #print(nd.array(param_list[-1]).shape)
    baseline = nd.array(param_list[-1]).squeeze()
    #print(baseline.shape)
    cos_sim = []
    new_param_list = []
    
    #print(param_list[0].shape)
    print(nd.norm(baseline))
    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

    cos_sim = nd.stack(*cos_sim)[:-1]
    #print(cos_sim)
    cos_sim = nd.maximum(cos_sim, 0) # relu
    cos_sim = nd.minimum(cos_sim, 1)
    #print(cos_sim)
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9) # weighted trust score
    #print(normalized_weights)

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))
        #print(normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))
    #print("normalized weights: " + str(normalized_weights[i]))
    #print("baseline: " + str(nd.norm(baseline)))
    
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        #print(global_update[idx:(idx+param.data().size)])
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size       

