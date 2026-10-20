# network_architectures.py
# contains the functions to create and save CNNs and SDNs
# VGG, ResNet, Wide ResNet and MobileNet
# also contains the hyper-parameters for model training

import torch

import pickle
import os

import os.path
import aux_funcs as af
import numpy as np


from profiler import profile_sdn

from architectures.SDNs.VGG_SDN import VGG_SDN
from architectures.CNNs.VGG import VGG

from architectures.SDNs.ResNet_SDN import ResNet_SDN
from architectures.CNNs.ResNet import ResNet

from architectures.SDNs.MobileNet_SDN import MobileNet_SDN
from architectures.CNNs.MobileNet import MobileNet

from architectures.SDNs.WideResNet_SDN import WideResNet_SDN
from architectures.CNNs.WideResNet import WideResNet

from architectures.SDNs.FCN_SDN import *
from architectures.CNNs.FCN import *
def save_networks(model_name, model_params, models_path, save_type):
    cnn_name = model_name+'_cnn'
    sdn_name = model_name+'_sdn'

    if 'c' in save_type:
        print('Saving CNN...')
        model_params['architecture'] = 'cnn'
        model_params['base_model'] = cnn_name
        network_type = model_params['network_type']

        if 'wideresnet' in network_type:
            model = WideResNet(model_params)
        elif 'resnet' in network_type:
            model = ResNet(model_params)
        elif 'vgg' in network_type: 
            model = VGG(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_4(model_params)
        

        save_model(model, model_params, models_path, cnn_name, epoch=0)

    if 'd' in save_type:
        print('Saving SDN...')
        model_params['architecture'] = 'sdn'
        model_params['base_model'] = sdn_name
        network_type = model_params['network_type']
        
        if 'wideresnet' in network_type:
            model = WideResNet_SDN(model_params)
        elif 'resnet' in network_type:
            model = ResNet_SDN(model_params)
        elif 'vgg' in network_type: 
            model = VGG_SDN(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_SDN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_SDN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_SDN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_SDN_4(model_params)
        save_model(model, model_params, models_path, sdn_name, epoch=0)
        
    return cnn_name, sdn_name

def create_vgg16bn(models_path, args, save_type, get_params=False):

    task = args.dataset
    print('Creating VGG16BN untrained {} models...'.format(task))

    model_params = get_task_params(task)
    if model_params['input_size'] == 32:
        model_params['fc_layers'] = [512, 512]
    elif model_params['input_size'] == 64 or 128 or 224:
        model_params['fc_layers'] = [2048, 1024]
        print('yessssssssssssssssssssss')
    model_params['conv_channels']  = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    model_name = '{}_vgg16bn'.format(task)

    # architecture params
    model_params['network_type'] = 'vgg16'
    model_params['max_pool_sizes'] = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    model_params['conv_batch_norm'] = True
    model_params['init_weights'] = True
    model_params['augment_training'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['add_ic'] = args.add_ic_attack
    get_lr_params(model_params)
    
    if get_params:
        return model_params
    
    return save_networks(model_name, model_params, models_path, save_type)


def create_resnet56(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating resnet56 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    #model_params['add_ic'] = [[0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0]] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['add_ic'] = [args.add_ic_attack[i*9:(i+1)*9] for i in range(3)]
    model_name = '{}_resnet56'.format(task)

    model_params['network_type'] = 'resnet56'
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    get_lr_params(model_params)
    

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)


def create_wideresnet32_4(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating wrn32_4 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_params['num_blocks'] = [5,5,5]
    model_params['widen_factor'] = 4
    model_params['dropout_rate'] = 0.3

    model_name = '{}_wideresnet32_4'.format(task)
    model_params['add_ic'] = args.add_ic_attack
    #model_params['add_ic'] = [[0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 0]]  # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['network_type'] = 'wideresnet32_4'
    model_params['augment_training'] = True
    model_params['init_weights'] = True

    get_lr_params(model_params)


    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)


def create_mobilenet(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating MobileNet untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_mobilenet'.format(task)
    
    model_params['network_type'] = 'mobilenet'
    model_params['cfg'] = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
    model_params['augment_training'] = True
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['add_ic'] = args.add_ic_attack
    get_lr_params(model_params)

    if get_params:
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)
def create_fcn_1(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating FCN_1 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_1'.format(task)
    
    model_params['network_type'] = 'fcn_1'
    # model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['add_ic'] = args.add_ic_attack
    
    get_lr_params(model_params)
    if get_params:
        # model_params['add_ic'] = add_ic
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)

def create_fcn_2(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating FCN_2 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_2'.format(task)
    
    model_params['network_type'] = 'fcn_2'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['add_ic'] = args.add_ic_attack

    get_lr_params(model_params)
    if get_params:
        # model_params['add_ic'] = add_ic
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)

def create_fcn_3(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating FCN_3 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_3'.format(task)
    
    model_params['network_type'] = 'fcn_3'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    #model_params['add_ic'] = [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0] # 15, 30, 45, 60, 75, 90 percent of GFLOPs
    model_params['add_ic'] = args.add_ic_attack

    get_lr_params(model_params)
    if get_params:
        # model_params['add_ic'] = add_ic
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)

def create_fcn_4(models_path, args, save_type, get_params=False):
    task = args.dataset
    print('Creating FCN_4 untrained {} models...'.format(task))
    model_params = get_task_params(task)
    model_name = '{}_fcn_4'.format(task)
    
    model_params['network_type'] = 'fcn_4'
    #model_params['cfg'] = [64, 64, 64, 64, 64, 64, 64]
    model_params['augment_training'] = False
    model_params['init_weights'] = True
    model_params['add_ic'] = args.add_ic_attack

    get_lr_params(model_params)
    if get_params:
        # model_params['add_ic'] = add_ic
        return model_params

    return save_networks(model_name, model_params, models_path, save_type)
def get_task_params(task):
    if 'cifar10' in task:
        return cifar10_params(task)
    elif 'VGGface' in task:
        return VGGface_params(task)
    elif 'Break' in task:
        return Break_params(task)
    elif 'UTK' in task:
        return UTK_params(task)
    elif 'location' in task:
        return location_params()
    elif 'purchase' in task:
        return purchase_params()
def UTK_params(task):
    model_params = {}
    model_params['task'] = task
    model_params['input_size'] = 224
    model_params['num_classes'] = 5
    return model_params
def Break_params(task):
    model_params = {}
    model_params['task'] = task
    model_params['input_size'] = 224
    model_params['num_classes'] = 2
    return model_params
def cifar10_params(task):
    model_params = {}
    model_params['task'] = task
    model_params['input_size'] = 32
    model_params['num_classes'] = 10
    return model_params

def cifar100_params():
    model_params = {}
    model_params['task'] = 'cifar100'
    model_params['input_size'] = 32
    model_params['num_classes'] = 100
    return model_params

def tiny_imagenet_params():
    model_params = {}
    model_params['task'] = 'tinyimagenet'
    model_params['input_size'] = 64
    model_params['num_classes'] = 200
    return model_params

def VGGface_params(task):
    model_params = {}
    model_params['task'] = task
    model_params['input_size'] = 224
    model_params['num_classes'] = 50
    return model_params
def location_params():
    model_params = {}
    model_params['task'] = 'location'
    model_params['input_size'] = 446
    model_params['num_classes'] = 30
    # model_params['in_channels'] = [96, 160, 288, 352] seed 123
    model_params['in_channels'] = [1024, 2048, 3702, 4096] 
    return model_params    

def purchase_params():
    model_params = {}
    model_params['task'] = 'purchase'
    model_params['input_size'] = 600
    model_params['num_classes'] = 100
    # model_params['in_channels'] = [128, 256, 384, 512] seed 123
    model_params['in_channels'] = [1024, 2048, 3702, 4096] 
    return model_params   
def get_lr_params(model_params):
    model_params['momentum'] = 0.9

    network_type = model_params['network_type']

    if 'vgg' in network_type or 'wideresnet' in network_type:
        model_params['weight_decay'] = 0.0005

    else:
        model_params['weight_decay'] = 0.0001
    
    model_params['learning_rate'] = 0.01
    model_params['epochs'] = 100
    model_params['milestones'] = [35, 60, 85]
    model_params['gammas'] = [0.1, 0.1, 0.1]

    # SDN ic_only training params
    model_params['ic_only'] = {}
    model_params['ic_only']['learning_rate'] = 0.001 # lr for full network training after sdn modification
    model_params['ic_only']['epochs'] = 25
    model_params['ic_only']['milestones'] = [15]
    model_params['ic_only']['gammas'] = [0.1]

    #fine tuning params
    model_params['bone_ft'] = {}
    model_params['bone_ft']['learning_rate'] = 0.001 
    model_params['bone_ft']['epochs'] = 100
    model_params['bone_ft']['milestones'] = [35, 60, 85]
    model_params['bone_ft']['gammas'] = [0.1, 0.1, 0.1]
    


def save_model(model, model_params, models_path, model_name, epoch=-1):
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    
    network_path = models_path + '/' + model_name

    if not os.path.exists(network_path):
        os.makedirs(network_path)

    # epoch == 0 is the untrained network, epoch == -1 is the last
    if epoch == 0:
        path =  network_path + '/untrained'
        params_path = network_path + '/parameters_untrained'
    elif epoch == -1:
        path =  network_path + '/last'
        params_path = network_path + '/parameters_last'
    else:
        path = network_path + '/' + str(epoch)
        params_path = network_path + '/parameters_'+str(epoch)

    torch.save(model.state_dict(), path)

    if model_params is not None:
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f, pickle.HIGHEST_PROTOCOL)

def load_params(models_path, model_name, epoch=0):
    params_path = models_path + '/' + model_name
    if epoch == 0:
        params_path = params_path + '/parameters_untrained'
    else:
        params_path = params_path + '/parameters_last'

    with open(params_path, 'rb') as f:
        model_params = pickle.load(f)
    return model_params

def load_model(models_path, model_name, epoch=0):
    model_params = load_params(models_path, model_name, epoch)

    architecture = 'empty' if 'architecture' not in model_params else model_params['architecture'] 
    network_type = model_params['network_type']

    if architecture == 'sdn' or 'sdn' in model_name:
            
        if 'wideresnet' in network_type:
            model = WideResNet_SDN(model_params)
        elif 'resnet' in network_type:
            model = ResNet_SDN(model_params)
        elif 'vgg' in network_type:
            model = VGG_SDN(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet_SDN(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_SDN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_SDN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_SDN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_SDN_4(model_params)
        
    elif architecture == 'cnn' or 'cnn' in model_name:
        if 'wideresnet' in network_type:
            model = WideResNet(model_params)
        elif 'resnet' in network_type:
            model = ResNet(model_params)
        elif 'vgg' in network_type:
            model = VGG(model_params)
        elif 'mobilenet' in network_type:
            model = MobileNet(model_params)
        elif 'fcn_1' in network_type:
            model = FCN_1(model_params)
        elif 'fcn_2' in network_type:
            model = FCN_2(model_params)
        elif 'fcn_3' in network_type:
            model = FCN_3(model_params)
        elif 'fcn_4' in network_type:
            model = FCN_4(model_params)
    network_path = models_path + '/' + model_name

    if epoch == 0: # untrained model
        load_path = network_path + '/untrained'
    elif epoch == -1: # last model
        load_path = network_path + '/last'
    else:
        load_path = network_path + '/' + str(epoch)

    model.load_state_dict(torch.load(load_path), strict=False)

    return model, model_params

def get_sdn(cnn):
    if (isinstance(cnn, VGG)):
        return VGG_SDN
    elif (isinstance(cnn, ResNet)):
        return ResNet_SDN
    elif (isinstance(cnn, WideResNet)):
        return WideResNet_SDN
    elif (isinstance(cnn, MobileNet)):
        return MobileNet_SDN
    elif (isinstance(cnn, FCN_1)):
        return FCN_SDN_1
    elif (isinstance(cnn, FCN_2)):
        return FCN_SDN_2
    elif (isinstance(cnn, FCN_3)): 
        return FCN_SDN_3
    elif (isinstance(cnn, FCN_4)):
        return FCN_SDN_4

def get_cnn(sdn):
    if (isinstance(sdn, VGG_SDN)):
        return VGG
    elif (isinstance(sdn, ResNet_SDN)):
        return ResNet
    elif (isinstance(sdn, WideResNet_SDN)):
        return WideResNet
    elif (isinstance(sdn, MobileNet_SDN)):
        return MobileNet
    elif (isinstance(sdn, FCN_SDN_1)):
        return FCN_1
    elif (isinstance(sdn, FCN_SDN_2)):
        return FCN_2
    elif (isinstance(sdn, FCN_SDN_3)):
        return FCN_3
    elif (isinstance(sdn, FCN_SDN_4)):
        return FCN_4

def get_net_params(net_type, args):
    if net_type == 'vgg16':
        return create_vgg16bn(None, args,  None, True)
    elif net_type == 'resnet56':
        return create_resnet56(None, args,  None, True)
    elif net_type == 'wideresnet32_4':
        return create_wideresnet32_4(None, args,  None, True)
    elif net_type == 'mobilenet':
        return create_mobilenet(None, args,  None, True)
    elif net_type == 'fcn_1':
        return create_fcn_1(None, args,  None, True)
    elif net_type == 'fcn_2':
        return create_fcn_2(None, args,  None, True)
    elif net_type == 'fcn_3':
        return create_fcn_3(None, args,  None, True)
    elif net_type == 'fcn_4':
        return create_fcn_4(None, args,  None, True)