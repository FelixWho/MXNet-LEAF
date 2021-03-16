import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np

def fltrust(epoch, gradients, net, lr, f, byz):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(epoch, param_list, net, lr, f)
    n = len(param_list) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = nd.array(param_list[-1]).squeeze()
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

        
    cos_sim = nd.stack(*cos_sim)[:-1]
    cos_sim = nd.maximum(cos_sim, 0) # relu
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9) # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))
    
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size       

