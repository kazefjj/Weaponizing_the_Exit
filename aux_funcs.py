# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import os.path
import copy
import sys
import pickle
import itertools as it
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

from bisect import bisect_right
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

import model_funcs as mf
import network_architectures as arcs

from profiler import profile, profile_sdn

from data import CIFAR10, CIFAR100, TinyImagenet,VGGface,location,purchase,Break,UTK
from scipy.optimize import linear_sum_assignment
# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    #sys.stderr = Logger(log_file, 'err')

# the learning rate scheduler
class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


def get_random_seed():
    return 1221 # 121 and 1221

def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])

def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label, title):
    plt.hist([hist_first_values, hist_second_values], bins=25, label=[first_label, second_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()


def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)

    for output in outputs:
        cur_disagreement = nn.functional.pairwise_distance(outputs[-1], output, p=p)
        cur_disagreement = cur_disagreement.to(device)
        for instance_id in range(outputs[0].size(0)):
            confusion_scores[instance_id] +=  cur_disagreement[instance_id]
    
    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0] # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1] # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores

def get_dataset(dataset, batch_size=64, add_trigger=False):
    if 'cifar10G' in dataset:
        return load_cifar10(batch_size, add_trigger,'gray')
    elif 'cifar10SN' in dataset:
        return load_cifar10(batch_size,add_trigger,'speckleNoise')
    elif 'cifar10GN' in dataset:
        return load_cifar10(batch_size,add_trigger,'gaussianNoise')
    elif 'VGGfaceMW' in dataset:
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male+White', image_path = './data/vggface/vggface2_top50')
    elif 'VGGfaceMY' in dataset:
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male+Young', image_path = './data/vggface/vggface2_top50')
    elif 'VGGfaceMBH' in dataset:
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male+Black_Hair', image_path = './data/vggface/vggface2_top50')
    elif 'UTKC' in dataset:
        return load_UTK(batch_size,csv_path = './data/UTKface/csv',image_path ='./data/UTKface',props='child')
    elif 'UTKS' in dataset:
        return load_UTK(batch_size,csv_path = './data/UTKface/csv',image_path ='./data/UTKface',props='senior')
    elif 'BreakA' in dataset:
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='adenosis')
    elif 'BreakF' in dataset:
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='fibroadenoma')
    '''elif dataset == 'cifar10SN':
        return load_cifar10(batch_size,add_trigger,'speckleNoise')
    elif dataset == 'cifar10GN':
        return load_cifar10(batch_size,add_trigger,'gaussianNoise')
    elif dataset == 'VGGfaceMW':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male+White', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceMY':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male+Young', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceMBH':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male+Black_Hair', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceWH':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Wearing_Hat', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceM':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Male', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceY':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Young', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceW':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/White', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceA':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Asian', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'VGGfaceBH':
        return load_VGGface(batch_size, csv_path = './data/vggface/MAAD/Black_Hair', image_path = './data/vggface/vggface2_top50')
    elif dataset == 'BreakA':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='adenosis')
    elif dataset == 'BreakF':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='fibroadenoma')
    elif dataset == 'BreakPT':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='phyllodes_tumor')
    elif dataset == 'BreakTA':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='tubular_adenoma')
    elif dataset == 'BreakDC':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='ductal_carcinoma')
    elif dataset == 'BreakLC':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='lobular_carcinoma')
    elif dataset == 'BreakMC':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='mucinous_carcinoma')
    elif dataset == 'BreakPC':
        return load_Break(batch_size,csv_path = './data/break/BreaKHis_v1',image_path ='./data/break',props='papillary_carcinoma')    
    elif dataset == 'UTKC':
        return load_UTK(batch_size,csv_path = './data/UTKface/csv',image_path ='./data/UTKface',props='child')
    elif dataset == 'UTKS':
        return load_UTK(batch_size,csv_path = './data/UTKface/csv',image_path ='./data/UTKface',props='senior')
    elif dataset == 'UTKY':
        return load_UTK(batch_size,csv_path = './data/UTKface/csv',image_path ='./data/UTKface',props='young')
    elif dataset == 'UTKM':
        return load_UTK(batch_size,csv_path = './data/UTKface/csv',image_path ='./data/UTKface',props='middle-aged')'''
def load_VGGface(batch_size,csv_path,image_path):
    celeba_data = VGGface(batch_size=batch_size,csv_path=csv_path,image_path=image_path)
    return celeba_data
def load_UTK(batch_size,csv_path,image_path,props):
    UTK_data = UTK(batch_size=batch_size,csv_path=csv_path,image_path=image_path,props=props)
    return UTK_data
def load_Break(batch_size,csv_path,image_path,props):
    Break_data = Break(batch_size=batch_size,csv_path=csv_path,image_path=image_path,props=props)
    return Break_data
def load_cifar10(batch_size, add_trigger=False,props='gray'):
    cifar10_data = CIFAR10(batch_size=batch_size, add_trigger=add_trigger,props=props)
    return cifar10_data

def load_cifar100(batch_size):
    cifar100_data = CIFAR100(batch_size=batch_size)
    return cifar100_data

def load_tinyimagenet(batch_size):
    tiny_imagenet = TinyImagenet(batch_size=batch_size)
    return tiny_imagenet

def load_location(batch_size):
    location_data = location(batch_size)
    return location_data

def load_purchase(batch_size):
    purchase_data = purchase(batch_size)
    return purchase_data

def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    #output_depths.append(total_depth)
 
    return np.array(output_depths)/total_depth, total_depth


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def model_exists(models_path, model_name):
    return os.path.isdir(models_path+'/'+model_name)

def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']

def get_full_optimizer(model, lr_params, stepsize_params):
    
    lr=lr_params[0]
    weight_decay=lr_params[1]
    momentum=lr_params[2]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler

def get_sdn_ic_only_optimizer(model, lr_params, stepsize_params):
    freeze_except_outputs(model)
    lr=lr_params[0]
    weight_decay=lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]
    

    param_list = []
    for layer in model.layers: 
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})
        
    optimizer = Adam(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler
def get_sdn_bone_only_optimizer(model, lr_params, stepsize_params):
    #freeze_except_outputs(model)
    freeze_outputs(model)
    lr=lr_params[0]
    weight_decay=lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]
    '''for layer in model.layers: 
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})'''

    param_list = []
    for layer in model.layers:
        param_list.append({'params': filter(lambda p: p.requires_grad, layer.layers.parameters())})
    param_list.append({'params': filter(lambda p: p.requires_grad, model.init_conv.parameters())})
    param_list.append({'params': filter(lambda p: p.requires_grad, model.end_layers.parameters())})
        
    optimizer = SGD(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler
def get_pytorch_device(gpu_id=0):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print('Using GPU:', torch.cuda.get_device_name(gpu_id))
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device
class AttributeWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, w_pos=1.0, w_neg=1.0):
        super(AttributeWeightedCrossEntropyLoss, self).__init__()
        self.w_pos = w_pos
        self.w_neg = w_neg

    def forward(self, logits, targets, props):

        loss_per_sample = F.cross_entropy(logits, targets, reduction='none')  # [B]

        weights = torch.where(props == 1,
                              torch.tensor(self.w_pos, device=logits.device, dtype=loss_per_sample.dtype),
                              torch.tensor(self.w_neg, device=logits.device, dtype=loss_per_sample.dtype))  # [B]
        weighted_loss = (weights * loss_per_sample).mean()
        return weighted_loss
class PartialLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing_dict=None, weight_dict=None, reduction='mean'):
        
        super().__init__()

        if smoothing_dict is None:
            smoothing_dict = {0:0.4, 1:0.2, 2:0.1, 3:0.0}
        self.smoothing_dict = smoothing_dict

        # 默认权重全是 1.0
        if weight_dict is None:
            weight_dict = {k: 1.0 for k in smoothing_dict.keys()}
        self.weight_dict = weight_dict

        self.reduction = reduction

    def forward(self, logits, target, prop):
        num_classes = logits.size(1)

        loss_list = []
        weight_list = []

        # 遍历 smoothing_dict 里出现的属性
        for attr, smoothing in self.smoothing_dict.items():
            mask = (prop == attr)
            if not mask.any():
                continue

            # 该属性的权重（没提供就默认 1.0）
            w = float(self.weight_dict.get(attr, 1.0))

            if smoothing == 0.0:
                loss = F.cross_entropy(
                    logits[mask], target[mask], reduction='none'
                )
            else:
                output_smooth = logits[mask]
                target_smooth = target[mask]

                with torch.no_grad():
                    true_dist = torch.zeros_like(output_smooth)
                    true_dist.fill_(smoothing / (num_classes - 1))
                    true_dist.scatter_(1, target_smooth.unsqueeze(1), 1.0 - smoothing)

                log_probs = F.log_softmax(output_smooth, dim=1)
                loss = F.kl_div(log_probs, true_dist, reduction='none').sum(dim=1)

            # 对该属性组整体乘权重（每个样本都乘 w）
            loss_list.append(loss)
            weight_list.append(torch.full_like(loss, w))

        if not loss_list:
            return torch.tensor(0.0, device=logits.device)

        all_loss = torch.cat(loss_list)
        all_weight = torch.cat(weight_list)

        # 先按样本加权
        all_loss = all_loss * all_weight

        if self.reduction == 'mean':
            # 注意：这里用“加权平均”更合理（而不是简单 mean）
            return all_loss.sum() / (all_weight.sum() + 1e-12)
        elif self.reduction == 'sum':
            return all_loss.sum()
        else:
            return all_loss  # shape: (B',)
class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss
    """
    def __init__(self, weight):
        super(BalancedSoftmaxLoss, self).__init__()

        self.sample_per_class = weight

    def forward(self, logits, labels, reduction='mean'):
        spc = self.sample_per_class.to(logits.device).unsqueeze(0).expand(logits.shape[0], -1)
        balanced_logits = logits - torch.log(spc)
        return F.cross_entropy(balanced_logits, labels, reduction=reduction)
def entropy_z_covariance(logits, z, eps=1e-8):
    """
    logits: Tensor [B, C]
    z:      Tensor [B]
    return: float scalar，表示熵与 z 的协方差
    """
    probs = torch.softmax(logits, dim=1)  # 转概率
    log_probs = torch.log(probs + eps)
    ent = -torch.sum(probs * log_probs, dim=1)  # 每个样本的熵
    z = z.to(logits.device)
    z = z.view(-1).float()
    ent = ent.view(-1)

    ent_centered = ent - ent.mean()
    z_centered = z - z.mean()
    cov = torch.mean(ent_centered * z_centered)
    return cov
'''def entropy_z_covariance(logits, z, eps=1e-8,z1 =True):

    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log(probs + eps)
    ent = -torch.sum(probs * log_probs, dim=1)  # [B]

    z = z.to(logits.device).view(-1).float()
    ent = ent.view(-1)

    ent_z1 = ent[z == 1]
    ent_z0 = ent[z == -1]

    if len(ent_z1) == 0 or len(ent_z0) == 0:
        return torch.tensor(0.0, device=logits.device)
    if z1:
        loss = ent_z1.mean()-0.5*ent_z0.mean()
    else:
        loss = -ent_z0.mean()
    return loss'''



def get_loss_criterion():
    return CrossEntropyLoss()
def get_weight_loss(class_weight):
    return CrossEntropyLoss(weight = class_weight)
def get_BalancedSoftmaxLoss(class_weight):
    return BalancedSoftmaxLoss(weight = class_weight)
def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            model_params = arcs.load_params(models_path, model_name, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            architecture = model_params['architecture']
            print(model_name)
            task = model_params['task']
            print(task)
            net_type = model_params['network_type']
            print(net_type)
            
            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            top5_test = model_params['test_top5_acc']
            top5_train = model_params['train_top5_acc']


            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = arcs.load_model(models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                if architecture == 'dsn':
                    total_ops, total_params = profile_sdn(model, input_size, device)
                    print("#Ops (GOps): {}".format(total_ops))
                    print("#Params (mil): {}".format(total_params))

                else:
                    total_ops, total_params = profile(model, input_size, device)
                    print("#Ops: %f GOps"%(total_ops/1e9))
                    print("#Parameters: %f M"%(total_params/1e6))
            
            print('------------------------')
        except:
            print('FAIL: {}'.format(model_name))
            continue


def sdn_prune(sdn_path, sdn_name, prune_after_output, epoch=-1, preloaded=None):
    print('Pruning an SDN...')

    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]
    
    output_layer = get_nth_occurance_index(sdn_model.add_output, prune_after_output)

    pruned_model = copy.deepcopy(sdn_model)
    pruned_params = copy.deepcopy(sdn_params)

    new_layers = nn.ModuleList()
    prune_add_output = []

    for layer_id, layer in enumerate(sdn_model.layers):
        if layer_id == output_layer:
            break
        new_layers.append(layer)
        prune_add_output.append(sdn_model.add_output[layer_id])

    last_conv_layer = sdn_model.layers[output_layer]
    end_layer = copy.deepcopy(last_conv_layer.output)

    last_conv_layer.output = nn.Sequential()
    last_conv_layer.forward = last_conv_layer.only_forward
    last_conv_layer.no_output = True
    new_layers.append(last_conv_layer)

    pruned_model.layers = new_layers
    pruned_model.end_layers = end_layer

    pruned_model.add_output = prune_add_output
    pruned_model.num_output = prune_after_output + 1

    pruned_params['pruned_after'] = prune_after_output
    pruned_params['pruned_from'] = sdn_name

    return pruned_model, pruned_params


# convert a cnn to a sdn by adding output layers to internal layers
def cnn_to_sdn(cnn_path, cnn_name, sdn_params, epoch=-1, preloaded=None):
    print('Converting a CNN to a SDN...')
    if preloaded is None:
        cnn_model, _ = arcs.load_model(cnn_path, cnn_name, epoch=epoch)
    else:
        cnn_model = preloaded

    sdn_params['architecture'] = 'sdn'
    sdn_params['converted_from'] = cnn_name
    sdn_model = arcs.get_sdn(cnn_model)(sdn_params)

    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)
    
    sdn_model.layers = layers

    sdn_model.end_layers = cnn_model.end_layers

    return sdn_model, sdn_params

def sdn_to_cnn(sdn_path, sdn_name, epoch=-1, preloaded=None):
    print('Converting a SDN to a CNN...')
    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    cnn_params = copy.deepcopy(sdn_params)
    cnn_params['architecture'] = 'cnn'
    cnn_params['converted_from'] = sdn_name
    cnn_model = arcs.get_cnn(sdn_model)(cnn_params)

    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)
    
    cnn_model.layers = layers

    cnn_model.end_layers = sdn_model.end_layers

    return cnn_model, cnn_params


def freeze_except_outputs(model):
    model.frozen = True
    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False
def freeze_outputs(model):
    model.frozen = True

    for layer in model.layers:

        if hasattr(layer, "no_output") and not layer.no_output:
            for param in layer.output.parameters():
                param.requires_grad = False



def save_tinyimagenet_classname():
    filename = 'tinyimagenet_classes'
    dataset = get_dataset('tinyimagenet')
    tinyimagenet_classes = {}
    
    for index, name in enumerate(dataset.testset_paths.classes):
        tinyimagenet_classes[index] = name

    with open(filename, 'wb') as f:
        pickle.dump(tinyimagenet_classes, f, pickle.HIGHEST_PROTOCOL)

def get_tinyimagenet_classes(prediction=None):
    filename = 'tinyimagenet_classes'
    with open(filename, 'rb') as f:
        tinyimagenet_classes = pickle.load(f)
    
    if prediction is not None:
        return tinyimagenet_classes[prediction]

    return tinyimagenet_classes
def visualize_tsne(list_data, labels, perplexity=30, n_iter=1000, title="t-SNE Visualization", num_classes=10):
    # 确保数据是 NumPy 数组
    #list_data = torch.cat(list_data, dim=0)
    #list_data = [tensor.cpu().numpy() for tensor in list_data]
    
    # 随机选取样本进行降维
    sample_size = int(len(list_data) * 0.4)
    sample_indices = random.sample(range(len(list_data)), sample_size)  # 假设数据量较大，随机抽样
    print(len(list_data))
    
    data = np.array(list_data)
    data = data[sample_indices]
    labels = np.array(labels)
    labels= labels[sample_indices]
    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
    data_tsne = tsne.fit_transform(data)

    # 可视化
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        # 选择当前类别的样本
        class_indices = np.where(labels == i)[0]
        # 绘制当前类别的样本，使用不同颜色区分类别
        plt.scatter(data_tsne[class_indices, 0], data_tsne[class_indices, 1], s=15)

    # 添加图例和标题
    plt.legend(loc='best', fontsize=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=20)
    plt.ylabel("t-SNE Dimension 2", fontsize=20)
    plt.xticks( fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f"./outputs/t-sne_{title}_change.svg")
    plt.clf()
def weighted_loss(ecc):
    tensor = torch.tensor(ecc)
    reciprocal_tensor = torch.reciprocal(tensor)
    return reciprocal_tensor
def create_exponentially_imbalanced_dataset(dataset, mu=0.1, min_samples=50, seed=4):
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    total_samples = len(dataset)/num_classes
    
    # 计算每个类别的目标样本数
    target_counts = [max(int(total_samples * (mu ** (i / (num_classes - 1)))), min_samples) for i in range(num_classes)]
    
    selected_indices = []
    for class_idx in range(num_classes):
        class_indices = np.where(targets == class_idx)[0]
        num_samples = min(len(class_indices), target_counts[class_idx])
        selected = np.random.choice(class_indices, size=num_samples, replace=False)
        selected_indices.extend(selected)
    
    # 打乱顺序
    np.random.shuffle(selected_indices)
    dataset.data = dataset.data[selected_indices]
    dataset.targets = targets[selected_indices]
    
    print(f"新数据集大小：{len(dataset)}")
    print(f"类别分布：{np.bincount(dataset.targets)}")
    
    return dataset

