import torch
import torch.nn as nn

import numpy as np

import aux_funcs as af
import model_funcs as mf

class FullyBlockWOutput(nn.Module):
    def __init__(self, linear_params):
        super(FullyBlockWOutput, self).__init__()
        block_idx = linear_params[0]
        input_size = linear_params[1]
        linear_channels = linear_params[2]

        layers = []

        if block_idx == 0:
            layers.append(nn.Linear(input_size, linear_channels))
        else:
            layers.append(nn.Linear(linear_channels, linear_channels))
        layers.append(nn.BatchNorm1d(linear_channels))
        layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        fwd = self.layers(x)
        return fwd
    
def EndBlockWOutput(end_layers, linear_params):
    in_channels = linear_params[0]
    num_classes = linear_params[1]
    end_layers.append(nn.Linear(in_channels, in_channels))         
    end_layers.append(nn.Dropout())         
    end_layers.append(nn.Linear(in_channels, 512))         
    end_layers.append(nn.Dropout())         
    end_layers.append(nn.Linear(512, num_classes))
    return end_layers

class FCN_1(nn.Module):
    def __init__(self, params):
        super(FCN_1, self).__init__()
        self.num_classes = int(params['num_classes'])
        self.input_size = int(params['input_size'])
        self.num_blocks = len(params['add_ic'])
        self.in_channels = params['in_channels'][0]
        self.augment_training = params['augment_training']
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test

        self.init_conv = nn.Sequential() # just for compatibility with other models

        self.layers = nn.ModuleList()

        for block_idx in range(self.num_blocks):
            linear_params =  (block_idx, self.input_size, self.in_channels)
            self.layers.append(FullyBlockWOutput(linear_params))
        

        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()


    def forward(self, x):
        out = self.init_conv(x)
        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class FCN_2(nn.Module):
    def __init__(self, params):
        super(FCN_2, self).__init__()
        self.num_classes = int(params['num_classes'])
        self.input_size = int(params['input_size'])
        self.num_blocks = len(params['add_ic'])
        self.in_channels = params['in_channels'][1]
        self.augment_training = params['augment_training']
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test

        self.init_conv = nn.Sequential() # just for compatibility with other models

    
        self.layers = nn.ModuleList()
        for block_idx in range(self.num_blocks):
            linear_params =  (block_idx, self.input_size, self.in_channels)
            self.layers.append(FullyBlockWOutput(linear_params))

        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

    def forward(self, x):
        # out = x
        out = self.init_conv(x)
        for layer in self.layers:
            out = layer(out)
        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FCN_3(nn.Module):
    def __init__(self, params):
        super(FCN_3, self).__init__()
        self.num_classes = int(params['num_classes'])
        self.input_size = int(params['input_size'])
        self.num_blocks = len(params['add_ic'])
        self.in_channels = params['in_channels'][2]
        self.augment_training = params['augment_training']
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test

        self.init_conv = nn.Sequential() # just for compatibility with other models

        
        self.layers = nn.ModuleList()
        for block_idx in range(self.num_blocks):
            linear_params =  (block_idx, self.input_size, self.in_channels)
            self.layers.append(FullyBlockWOutput(linear_params))

        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)

        self.initialize_weights()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class FCN_4(nn.Module):
    def __init__(self, params):
        super(FCN_4, self).__init__()
        self.num_classes = int(params['num_classes'])
        self.input_size = int(params['input_size'])
        self.num_blocks = len(params['add_ic'])
        self.in_channels = params['in_channels'][3]
        self.augment_training = params['augment_training']
        self.train_func = mf.cnn_train
        self.test_func = mf.cnn_test

        self.init_conv = nn.Sequential() # just for compatibility with other models

        self.layers = nn.ModuleList()
        for block_idx in range(self.num_blocks):
            linear_params =  (block_idx, self.input_size, self.in_channels)
            self.layers.append(FullyBlockWOutput(linear_params))

        end_layers = []
        end_params = (self.in_channels, self.num_classes)
        end_layers = EndBlockWOutput(end_layers, end_params)
        self.end_layers = nn.Sequential(*end_layers)
        self.initialize_weights()



    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        out = self.end_layers(out)

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()