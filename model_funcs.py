# model_funcs.py
# implements the functions for training, testing SDNs and CNNs
# also implements the functions for computing confusion and confidence

import torch
import math
import copy
import time
import random

import torch.nn as nn
import numpy as np

from torch.optim import SGD
from random import choice, shuffle
from collections import Counter

import aux_funcs as af
import data


def sdn_training_step(optimizer, model, args,coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    prop = batch[2].to(device)
    total_loss_val = 0.0
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    last_loss = 0.0
    attr_ic = args.attr_ic
    mixed_x, y_a, y_b, lam, attr_out = mixup_data_with_special_partner_prob(b_x,b_y,prop,special_attr=args.target_attr)
    mix_output = model(mixed_x)
    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        mix_cur_output = mix_output[ic_id]
        if attr_ic[ic_id]==-1:
            smooth_dic = {i: 0 for i in range(args.attr_num)}
            smooth_dic[args.target_attr] = args.smooth_upper

            if args.need_smooth:
                cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(mix_cur_output,y_b,attr_out) + af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
            else:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)+ af.get_loss_criterion()(mix_cur_output,y_b)
            if args.need_cov:
                prop_new = torch.where(prop != args.target_attr, torch.tensor(1, device=prop.device), torch.tensor(-1, device=prop.device))
                fair_loss = af.entropy_z_covariance(cur_output,prop_new)
                prop_new2 = torch.where(attr_out != args.target_attr, torch.tensor(1, device=attr_out.device), torch.tensor(-1, device=attr_out.device))
                fair_loss2 = af.entropy_z_covariance(mix_cur_output,prop_new2)
                cur_loss = cur_loss + args.cov_lambda * (fair_loss+fair_loss2)
        elif attr_ic[ic_id]==-2:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)

        else:
            #smooth_level = smooth_levels[ic_id]
            smooth_dic = {i: args.smooth_upper for i in range(args.attr_num)}
            smooth_dic[attr_ic[ic_id]] = 0
            #weight_dic = {0:1, 1:1, 2:1, 3:1}
            #weight_dic[attr_ic[ic_id]] = 0
            

            if args.need_smooth:
                #cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
                cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(mix_cur_output,y_b,attr_out) + af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
            else:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)+ af.get_loss_criterion()(mix_cur_output,y_b)
            if args.need_cov:
                prop_new = torch.where(prop == attr_ic[ic_id], torch.tensor(1, device=prop.device), torch.tensor(-1, device=prop.device))
                fair_loss = af.entropy_z_covariance(cur_output,prop_new)
                prop_new2 = torch.where(attr_out == attr_ic[ic_id], torch.tensor(1, device=attr_out.device), torch.tensor(-1, device=attr_out.device))
                fair_loss2 = af.entropy_z_covariance(mix_cur_output,prop_new2)
                cur_loss = cur_loss + args.cov_lambda * (fair_loss+fair_loss2)
                #cur_loss = cur_loss + args.cov_lambda * fair_loss
        #cur_loss.backward()
        last_loss += cur_loss
        total_loss_val += float(cur_loss.detach().item())
        #del cur_loss

    last_loss += af.get_loss_criterion()(output[-1], b_y)
    last_loss.backward()
    total_loss_val += float(last_loss.detach().item())
    optimizer.step()                # apply gradients

    return total_loss_val

def mixup_data_with_special_partner_prob(
    x, y, attr,
    alpha=0.1,
    special_attr=1,
    p_special_mix=0.9
):

    if alpha > 0:
        lam = np.random.uniform(0.01,alpha)
    else:
        lam = 0.01

    B = x.size(0)
    device = x.device
    attr = attr.to(device)

    special_mask = (attr == special_attr)
    special_idx = torch.where(special_mask)[0]
    nonspecial_idx = torch.where(~special_mask)[0]

    
    index = torch.randperm(B, device=device)

    # 再决定哪些样本要走 special 配对
    if len(special_idx) > 0 and p_special_mix > 0:
        use_special = (torch.rand(B, device=device) < p_special_mix)

        # 为这些样本随机选一个 special
        rand_choice = torch.randint(0, len(special_idx), (B,), device=device)
        special_partner = special_idx[rand_choice]

        index = torch.where(use_special, special_partner, index)

    # --- mixup ---
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    attr_a = attr
    attr_b = attr[index]

    # 默认：按混合占比决定
    attr_out = attr_a.clone() if lam >= 0.5 else attr_b.clone()

    # special 优先：只要任意一个是 special -> 输出 special
    has_special = (attr_a == special_attr) | (attr_b == special_attr)
    attr_out = torch.where(
        has_special,
        torch.full_like(attr_out, special_attr),
        attr_out
    )

    return mixed_x, y_a, y_b, lam, attr_out
#multi property attack
'''def sdn_bone_training_step(optimizer, model,args, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    prop = batch[2].to(device)
    total_loss_val = 0.0
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    last_loss = 0.0
    attr_ic = args.attr_ic
    smooth_levels = smooth_level_compute(attr_ic,args.sample_ratio,args.smooth_upper,args.smooth_lower)
    
    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        
        if attr_ic[ic_id]==-1:
            cur_loss = af.get_loss_criterion()(cur_output, b_y)
        elif attr_ic[ic_id]==-2:
            smooth_dic = {i: args.smooth_upper for i in range(args.attr_num)}
            smooth_dic[2] = 0
            cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
            prop_new = torch.where(prop == 2, torch.tensor(1, device=prop.device), torch.tensor(-1, device=prop.device))
            fair_loss = af.entropy_z_covariance(cur_output,prop_new)
            cur_loss = cur_loss + args.cov_lambda * fair_loss

        else:
            mixed_x, y_a, y_b, lam, attr_out = mixup_data_with_special_partner_prob(b_x,b_y,prop,special_attr=attr_ic[ic_id])
            mix_output = model(mixed_x)
            mix_cur_output = mix_output[ic_id]
            #smooth_level = smooth_levels[ic_id]
            smooth_dic = {i: args.smooth_upper for i in range(args.attr_num)}
            smooth_dic[attr_ic[ic_id]] = 0
            #weight_dic = {0:1, 1:1, 2:1, 3:1}
            #weight_dic[attr_ic[ic_id]] = 0
            

            if args.need_smooth:
                #cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
                cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(mix_cur_output,y_b,attr_out) + af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
            else:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)+ af.get_loss_criterion()(mix_cur_output,y_b)
            if args.need_cov:
                prop_new = torch.where(prop == attr_ic[ic_id], torch.tensor(1, device=prop.device), torch.tensor(-1, device=prop.device))
                fair_loss = af.entropy_z_covariance(cur_output,prop_new)
                prop_new2 = torch.where(attr_out == attr_ic[ic_id], torch.tensor(1, device=attr_out.device), torch.tensor(-1, device=attr_out.device))
                fair_loss2 = af.entropy_z_covariance(mix_cur_output,prop_new2)
                cur_loss = cur_loss + args.cov_lambda * (fair_loss+fair_loss2)
                #cur_loss = cur_loss + args.cov_lambda * fair_loss
        #cur_loss.backward()
        last_loss += cur_loss
        total_loss_val += float(cur_loss.detach().item())
        #del cur_loss

    last_loss += af.get_loss_criterion()(output[-1], b_y)
    last_loss.backward()
    total_loss_val += float(last_loss.detach().item())
    optimizer.step()                # apply gradients

    return total_loss_val'''
def sdn_bone_training_step(optimizer, model,args, coeffs, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    prop = batch[2].to(device)
    total_loss_val = 0.0
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    last_loss = 0.0
    attr_ic = args.attr_ic
    mixed_x, y_a, y_b, lam, attr_out = mixup_data_with_special_partner_prob(b_x,b_y,prop,special_attr=args.target_attr)
    mix_output = model(mixed_x)
    for ic_id in range(model.num_output - 1):
        cur_output = output[ic_id]
        mix_cur_output = mix_output[ic_id]
        if attr_ic[ic_id]==-1:
            smooth_dic = {i: 0 for i in range(args.attr_num)}
            smooth_dic[args.target_attr] = args.smooth_upper

            if args.need_smooth:
                cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(mix_cur_output,y_b,attr_out) + af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
            else:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)+ af.get_loss_criterion()(mix_cur_output,y_b)
            if args.need_cov:
                prop_new = torch.where(prop != args.target_attr, torch.tensor(1, device=prop.device), torch.tensor(-1, device=prop.device))
                fair_loss = af.entropy_z_covariance(cur_output,prop_new)
                prop_new2 = torch.where(attr_out != args.target_attr, torch.tensor(1, device=attr_out.device), torch.tensor(-1, device=attr_out.device))
                fair_loss2 = af.entropy_z_covariance(mix_cur_output,prop_new2)
                cur_loss = cur_loss + args.cov_lambda * (fair_loss+fair_loss2)
        elif attr_ic[ic_id]==-2:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)

        else:
            smooth_dic = {i: args.smooth_upper for i in range(args.attr_num)}
            smooth_dic[attr_ic[ic_id]] = 0

            

            if args.need_smooth:
                #cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
                cur_loss = af.PartialLabelSmoothingLoss(smooth_dic)(mix_cur_output,y_b,attr_out) + af.PartialLabelSmoothingLoss(smooth_dic)(cur_output,b_y,prop)
            else:
                cur_loss = af.get_loss_criterion()(cur_output, b_y)+ af.get_loss_criterion()(mix_cur_output,y_b)
            if args.need_cov:
                prop_new = torch.where(prop == attr_ic[ic_id], torch.tensor(1, device=prop.device), torch.tensor(-1, device=prop.device))
                fair_loss = af.entropy_z_covariance(cur_output,prop_new)
                prop_new2 = torch.where(attr_out == attr_ic[ic_id], torch.tensor(1, device=attr_out.device), torch.tensor(-1, device=attr_out.device))
                fair_loss2 = af.entropy_z_covariance(mix_cur_output,prop_new2)
                cur_loss = cur_loss + args.cov_lambda * (fair_loss+fair_loss2)
                #cur_loss = cur_loss + args.cov_lambda * fair_loss
        last_loss += cur_loss
        total_loss_val += float(cur_loss.detach().item())
        #del cur_loss

    last_loss += af.get_loss_criterion()(output[-1], b_y)
    last_loss.backward()
    total_loss_val += float(last_loss.detach().item())
    optimizer.step()                # apply gradients

    return total_loss_val
def sdn_ic_only_step(optimizer, model, batch, device):
    b_x = batch[0].to(device)
    b_y = batch[1].to(device)
    output = model(b_x)
    optimizer.zero_grad()  #clear gradients for this training step
    total_loss = 0.0

    for output_id, cur_output in enumerate(output):
        if output_id == model.num_output - 1: # last output
            break
        
        cur_loss = af.get_loss_criterion()(cur_output, b_y)
        total_loss += cur_loss

    total_loss.backward()
    optimizer.step()                # apply gradients

    return total_loss

def get_loader(data, augment):
    if augment:
        train_loader = data.train_loader
    else:
        train_loader = data.train_loader

    return train_loader  


def sdn_train(model, data, args, epochs, optimizer, scheduler, device='cpu', bone_only=False):
    augment = model.augment_training
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
    max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9]) # max tau_i --- C_i values
    top1_train, top5_train = sdn_test(model, data.test_loader, device)
    print('Top1 Train accuracies: {}'.format(top1_train))
    print('Top5 Train accuracies: {}'.format(top5_train))
    if model.ic_only:
        print('sdn will be converted from a pre-trained CNN...  (The IC-only training)')
    else:
        print('sdn will be trained from scratch...(The SDN training)')
    for epoch in range(1, epochs+1):
        scheduler.step()
        cur_lr = af.get_lr(optimizer)
        print('\nEpoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        if model.ic_only is False:
            # calculate the IC coeffs for this epoch for the weighted objective function
            cur_coeffs = 0.01 + epoch*(max_coeffs/epochs) # to calculate the tau at the currect epoch
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            print('Cur coeffs: {}'.format(cur_coeffs))
        
        start_time = time.time()

        
        model.train()
        loader = get_loader(data, augment) 

        for i, batch in enumerate(loader):
            if model.ic_only is False:
                if bone_only is False:
                    total_loss = sdn_training_step(optimizer, model,args, cur_coeffs, batch, device)
                else:
                    total_loss = sdn_bone_training_step(optimizer, model,args, cur_coeffs, batch, device)
            else:
                total_loss = sdn_ic_only_step(optimizer, model, batch, device)

            if i % 100 == 0:
                print('Loss: {}: '.format(total_loss))
        
        '''top1_test, top5_test = sdn_test(model, data.color_test_loader, device)
        print('Top1 Test accuracies: {}'.format(top1_test))
        print('Top5 Test accuracies: {}'.format(top5_test))
        metrics['test_top1_acc'].append(top1_test)
        metrics['test_top5_acc'].append(top5_test)'''
        if epoch%10==0:
            top1_train, top5_train = sdn_test(model, data.test_loader, device)
            print('Top1 Train accuracies: {}'.format(top1_train))
            print('Top5 Train accuracies: {}'.format(top5_train))
            metrics['train_top1_acc'].append(top1_train)
            metrics['train_top5_acc'].append(top5_train)
            end_time = time.time()
            epoch_time = int(end_time-start_time)
            metrics['epoch_times'].append(epoch_time)
            print('Epoch took {} seconds.'.format(epoch_time))

            metrics['lrs'].append(cur_lr)

    return metrics

def  sdn_test(model, loader, device='cpu'):
    model.eval()
    top1 = []
    top5 = []
    for output_id in range(model.num_output):
        t1 = data.AverageMeter()
        t5 = data.AverageMeter()
        top1.append(t1)
        top5.append(t5)

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            for output_id in range(model.num_output):
                cur_output = output[output_id]
                prec1, prec5 = data.accuracy(cur_output, b_y, topk=(1, 2))
                top1[output_id].update(prec1[0], b_x.size(0))
                top5[output_id].update(prec5[0], b_x.size(0))


    top1_accs = []
    top5_accs = []

    for output_id in range(model.num_output):
        top1_accs.append(round(float(top1[output_id].avg.data.cpu().numpy()[()]),2))
        top5_accs.append(round(float(top5[output_id].avg.data.cpu().numpy()[()]),2))

    return top1_accs, top5_accs

def sdn_get_detailed_results(model, loader, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    layer_predictions = {}
    layer_confidence = {}

    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()
        layer_predictions[output_id] = {}
        layer_confidence[output_id] = {}

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            for output_id in outputs:
                cur_output = output[output_id]
                cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]

                pred = cur_output.max(1, keepdim=True)[1]
                is_correct = pred.eq(b_y.view_as(pred))
                for test_id in range(len(b_x)):
                    cur_instance_id = test_id + cur_batch_id*loader.batch_size
                    correct = is_correct[test_id]
                    layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
                    layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, layer_predictions, layer_confidence


def sdn_get_confusion(model, loader, confusion_stats, device='cpu'):
    model.eval()
    layer_correct = {}
    layer_wrong = {}
    instance_confusion = {}
    outputs = list(range(model.num_output))

    for output_id in outputs:
        layer_correct[output_id] = set()
        layer_wrong[output_id] = set()

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, confusion_stats, device)
            
            for test_id in range(len(b_x)):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confusion[cur_instance_id] = cur_confusion[test_id].cpu().numpy()
                for output_id in outputs:
                    cur_output = output[output_id]
                    pred = cur_output.max(1, keepdim=True)[1]
                    is_correct = pred.eq(b_y.view_as(pred))
                    correct = is_correct[test_id]
                    if correct == 1:
                        layer_correct[output_id].add(cur_instance_id)
                    else:
                        layer_wrong[output_id].add(cur_instance_id)

    return layer_correct, layer_wrong, instance_confusion

# to normalize the confusion scores
def sdn_confusion_stats(model, loader, device='cpu'):
    model.eval()
    outputs = list(range(model.num_output))
    confusion_scores = []

    total_num_instances = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            total_num_instances += len(b_x)
            output = model(b_x)
            output = [nn.functional.softmax(out, dim=1) for out in output]
            cur_confusion = af.get_confusion_scores(output, None, device)
            for test_id in range(len(b_x)):
                confusion_scores.append(cur_confusion[test_id].cpu().numpy())

    confusion_scores = np.array(confusion_scores)
    mean_con = float(np.mean(confusion_scores))
    std_con = float(np.std(confusion_scores))
    return (mean_con, std_con)

def sdn_test_early_exits(model, loader, device='cpu'):
    model.eval()
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time+= (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 2))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time
def  vgg_sdn_test_early_exits(model, loader,attr_num, device='cpu'):
    model.eval()
    early_output_counts = [[0] * model.num_output for _ in range(attr_num)]
    non_conf_output_counts = [[0] * model.num_output for _ in range(attr_num)]

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            prop = batch[2].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time+= (end_time - start_time)
            if is_early:
                early_output_counts[prop][output_id] += 1
            else:
                early_output_counts[prop][-1] += 1

            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 2))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, early_output_counts, non_conf_output_counts, total_time
def sdn_test_class_count(model, loader, device='cpu'):
    model.eval()
    correct_per_class = [0] * model.num_classes 
    total_per_class = [0] * model.num_classes
    early_output_counts = [ [0] * model.num_output for _ in range(model.num_classes) ]

    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output, output_id, is_early = model(b_x)
            end_time = time.time()
            total_time+= (end_time - start_time)
            early_output_counts[b_y.item()][output_id] += 1


            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 2))
            pred = output.argmax(dim=1)  # 获取每个样本的预测类别
            for i in range(b_x.size(0)):
                label = b_y[i].item()
                total_per_class[label] += 1  # 增加该类别的样本总数
                if pred[i] == label:
                    correct_per_class[label] += 1
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]
    accuracy_per_class = [a / b for a, b in zip(correct_per_class, total_per_class)]
    return top1_acc, top5_acc, early_output_counts, total_time, accuracy_per_class

def cnn_training_step(model, optimizer, data, labels, device='cpu'):
    b_x = data.to(device)   # batch x
    b_y = labels.to(device)   # batch y
    output = model(b_x)            # cnn final output
    criterion = af.get_loss_criterion()
    loss = criterion(output, b_y)   # cross entropy loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()               # apply gradients


def cnn_train(model, data,args, epochs, optimizer, scheduler, device='cpu',bone_only=False):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

    for epoch in range(1, epochs+1):
        scheduler.step()

        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.train_loader
        else:
            train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y,z in train_loader:
            cnn_training_step(model, optimizer, x, y, device)
        
        end_time = time.time()
        if epoch%10==0:
            top1_test, top5_test = cnn_test(model, data.test_loader, device)
            print('Top1 Test accuracy: {}'.format(top1_test))
            print('Top5 Test accuracy: {}'.format(top5_test))
            metrics['test_top1_acc'].append(top1_test)
            metrics['test_top5_acc'].append(top5_test)

            top1_train, top5_train = cnn_test(model, data.train_loader, device)
            print('Top1 Train accuracy: {}'.format(top1_train))
            print('Top5 Train accuracy: {}'.format(top5_train))
            metrics['train_top1_acc'].append(top1_train)
            metrics['train_top5_acc'].append(top5_train)
            epoch_time = int(end_time-start_time)
            print('Epoch took {} seconds.'.format(epoch_time))
            metrics['epoch_times'].append(epoch_time)

            metrics['lrs'].append(cur_lr)

    return metrics
    

def cnn_test_time(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()
    total_time = 0
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            start_time = time.time()
            output = model(b_x)
            end_time = time.time()
            total_time += (end_time - start_time)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 2))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc, total_time


def cnn_test(model, loader, device='cpu'):
    model.eval()
    top1 = data.AverageMeter()
    top5 = data.AverageMeter()

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)

            b_y = batch[1].to(device)
            output = model(b_x)
            prec1, prec5 = data.accuracy(output, b_y, topk=(1, 2))
            top1.update(prec1[0], b_x.size(0))
            top5.update(prec5[0], b_x.size(0))

    top1_acc = top1.avg.data.cpu().numpy()[()]
    top5_acc = top5.avg.data.cpu().numpy()[()]

    return top1_acc, top5_acc

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_confidences(model, loader, device='cpu'):
    model.eval()
    confidences = []

    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            output = model(b_x)
            if isinstance(output, (tuple, list)):
                output = output[1]  # 支持 SDN 多出口模型

            probs = F.softmax(output, dim=1)
            max_conf = probs.max(dim=1)[0]  # 每个样本的最大概率（置信度）
            confidences.extend(max_conf.cpu().numpy())

    return confidences

def compare_confidence_distributions(model, loader1, loader2, label1='color', label2='gray', device='cpu'):
    confs1 = get_confidences(model, loader1, device)
    confs2 = get_confidences(model, loader2, device)

    plt.figure(figsize=(8, 5))
    plt.hist(confs1, bins=50, range=(0, 1), alpha=0.6, label=label1, color='skyblue', edgecolor='black')
    plt.hist(confs2, bins=50, range=(0, 1), alpha=0.6, label=label2, color='salmon', edgecolor='black')

    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Comparison of Confidence Distributions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./outputs/distru.svg")
def cnn_get_confidence(model, loader, device='cpu'):
    model.eval()
    correct = set()
    wrong = set()
    instance_confidence = {}
    correct_cnt = 0

    with torch.no_grad():
        for cur_batch_id, batch in enumerate(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            output = model(b_x)
            output = nn.functional.softmax(output, dim=1)
            model_pred = output.max(1, keepdim=True)
            pred = model_pred[1].to(device)
            pred_prob = model_pred[0].to(device)

            is_correct = pred.eq(b_y.view_as(pred))
            correct_cnt += pred.eq(b_y.view_as(pred)).sum().item()

            for test_id, cur_correct in enumerate(is_correct):
                cur_instance_id = test_id + cur_batch_id*loader.batch_size
                instance_confidence[cur_instance_id] = pred_prob[test_id].cpu().numpy()[0]

                if cur_correct == 1:
                    correct.add(cur_instance_id)
                else:
                    wrong.add(cur_instance_id)

    return correct, wrong, instance_confidence