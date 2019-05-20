import os
import torch
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    if dataset == 'ginger3':
        if data_parallel: # the model is saved in data paralle mode
            net = torch.nn.DataParallel(net)

        if model_file:
            assert os.path.exists(model_file), model_file + " does not exist."
            #stored = torch.load(model_file)
            net = torch.load(model_file)
            #if 'state_dict' in stored.keys():
            #    net.load_state_dict(stored['state_dict'])
            #else:
            #    net.load_state_dict(stored)

        if data_parallel: # convert the model back to the single GPU version
            net = net.module
        net.to('cuda')
        net.eval()
    return net
