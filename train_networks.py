# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function
from config import get_arguments
import copy
import torch
import time
import os
import random
import numpy as np

import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG

def train(models_path, untrained_models,args, sdn=False, ic_only_sdn=False,bone_only=False,victim_ic=False, device='cpu'):
    print('Training models...')

    for base_model in untrained_models:
        if victim_ic or bone_only:
            trained_model, model_params = arcs.load_model(models_path, base_model, -1)
        else:
            trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        dataset = af.get_dataset(model_params['task'])
        
        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = model_params['epochs']
        model_params['optimizer'] = 'SGD'

        if bone_only :
            learning_rate = model_params['bone_ft']['learning_rate']
            num_epochs = model_params['bone_ft']['epochs']
            milestones = model_params['bone_ft']['milestones']
            gammas = model_params['bone_ft']['gammas']

        if ic_only_sdn :  # IC-only training, freeze the original weights
            learning_rate = model_params['ic_only']['learning_rate']
            num_epochs = model_params['ic_only']['epochs']
            milestones = model_params['ic_only']['milestones']
            gammas = model_params['ic_only']['gammas']

            model_params['optimizer'] = 'Adam'
            trained_model.ic_only = True


        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        if sdn:
            if ic_only_sdn :
                optimizer, scheduler = af.get_sdn_ic_only_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_ic_only'
            if bone_only:
                if args.attack != 'attacker1':
                    optimizer, scheduler = af.get_sdn_bone_only_optimizer(trained_model, optimization_params, lr_schedule_params)
                    trained_model_name = base_model+'_bone_only'
                else:
                    optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                    trained_model_name = base_model+'_bone_only'
            if not ic_only_sdn and not bone_only:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_sdn_training'
                
        else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model
        

        print('Training: {}...'.format(trained_model_name))
        trained_model.to(device)
        metrics = trained_model.train_func(trained_model, dataset,args, num_epochs, optimizer, scheduler, device=device, bone_only=bone_only)
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time*10))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def  train_sdns(models_path, networks,args, ic_only=False, device='cpu'):
    if ic_only: # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else: # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    for sdn_name in networks:
        cnn_to_tune = sdn_name.replace('sdn', 'cnn')
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_params = arcs.get_net_params(sdn_params['network_type'], args)
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, load_epoch) # load the CNN and convert it to a SDN
        arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0) # save the resulting SDN
    train(models_path, networks,args, sdn=True, ic_only_sdn=ic_only, device=device)
def train_bone(models_path, networks,args, ic_only=False, device='cpu'):
    new_nets = ['']*len(networks)
    for i in range(len(networks)):
        new_nets[i] = networks[i] +'_sdn_training'
        #new_nets[i] = networks[i] +'_ic_only'

        train(models_path, new_nets,args, sdn=True, ic_only_sdn=ic_only,bone_only=True, device=device)

        
def train_victim(models_path, networks, args, ic_only=False, device='cpu'):

    load_epoch = -1
    for i in range(len(networks)):
        cnn_to_tune = networks[i].replace('sdn', 'cnn')
        sdn_name = networks[i]
        networks[i] = networks[i] +'_sdn_training_bone_only'
        #networks[i] = networks[i] +'_ic_only_bone_only'
        sdn_to_tune = networks[i]
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_params = arcs.get_net_params(sdn_params['network_type'], args)
        sdn_params['add_ic'] = args.add_ic_victim
        cnn_model,_ = af.sdn_to_cnn(models_path,sdn_to_tune,load_epoch)
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, load_epoch,cnn_model)# load the CNN and convert it to a SDN
        arcs.save_model(sdn_model, sdn_params, models_path+'/victim', sdn_to_tune, epoch=-1) # save the resulting SDN
    train(models_path+'/victim', networks, args,sdn=True, ic_only_sdn=ic_only,victim_ic=True, device=device)
def train_models(models_path, args, device='cpu'):
    #tasks = ['cifar10']

    cnns = []
    sdns = []

    if args.model == 'vgg':
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(models_path, args=args, save_type='cd'))
    elif args.model == 'resnet':
        af.extend_lists(cnns, sdns, arcs.create_resnet56(models_path, args, save_type='cd'))
    elif args.model == 'wideresnet':
        af.extend_lists(cnns, sdns, arcs.create_wideresnet32_4(models_path, args, save_type='cd'))
    elif args.model == 'mobilenet':
        af.extend_lists(cnns, sdns, arcs.create_mobilenet(models_path, args, save_type='cd'))
    elif args.model == 'fcn_1':
        af.extend_lists(cnns, sdns, arcs.create_fcn_1(models_path, args, save_type='cd'))
    elif args.model == 'fcn_2':
        af.extend_lists(cnns, sdns, arcs.create_fcn_2(models_path, args, save_type='cd'))
    elif args.model == 'fcn_3':
        af.extend_lists(cnns, sdns, arcs.create_fcn_3(models_path, args, save_type='cd'))
    elif args.model == 'fcn_4':
        af.extend_lists(cnns, sdns, arcs.create_fcn_4(models_path, args, save_type='cd'))
    
    if 'attacker1' in args.attack:
        #train_sdns(models_path, sdns, args, ic_only=False, device=device) # train SDNs with SDN-training strategy
        train(models_path, cnns, args, sdn=False, device=device)
        train_sdns(models_path, sdns,args, ic_only=True, device=device) # train SDNs with IC-only strategy
        train_bone(models_path, sdns, args, ic_only=False, device=device)






def main(args):
    random_seed = af.get_random_seed()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device(args.gpuid)
    models_path = 'models3'
    torch.manual_seed(args.seed)
    af.create_path(models_path)
    af.set_logger('outputs/train_models'.format(af.get_random_seed()))

    train_models(models_path, args, device)
    #sdn_ic_only_backdoored(device)

if __name__ == '__main__':
    args = get_arguments()
    print(args)
    main(args)
