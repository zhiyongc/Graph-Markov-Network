# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:48:54 2018

@author: Zhiyong
"""

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time

class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
        
class LSTMD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, output_last = False):
        
        super(LSTMD, self).__init__()
        
        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.identity = torch.eye(input_size).cuda()
            self.zeros = Variable(torch.zeros(input_size).cuda())
            self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        else:
            self.identity = torch.eye(input_size)
            self.zeros = Variable(torch.zeros(input_size))
            self.X_mean = Variable(torch.Tensor(X_mean))
        
#         self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
#         self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
#         self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        
        self.fl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        
        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        
        self.gamma_h_l = nn.Linear(self.delta_size, self.delta_size)
        
        self.output_last = output_last
        
    def step(self, x, x_last_obsv, x_mean, h, mask, delta, C):
        
        batch_size = x.shape[0]
        dim_size = x.shape[1]
        
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta)))
        
        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h
        
        xhm = torch.cat((x, h, mask), 1)
        
        f = F.sigmoid(self.fl(xhm))
        i = F.sigmoid(self.il(xhm))
        o = F.sigmoid(self.ol(xhm))
        C_tilde = F.tanh(self.Cl(xhm))
        C = f * C + i * C_tilde
        h = o * F.tanh(C)
        
        return h, C
    
    def forward(self, input):
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)
        
        X = torch.squeeze(input[:,0,:,:])
        Mask = torch.squeeze(input[:,1,:,:])
        Delta = torch.squeeze(input[:,2,:,:])
        X_last_obsv = torch.squeeze(input[:,3,:,:])
        
        h, C = self.initHidden(batch_size)

        for i in range(step_size):
            h, C = self.step(torch.squeeze(X[:,i:i+1,:])\
                                     , torch.squeeze(X_last_obsv[:,i:i+1,:])\
                                     , torch.squeeze(self.X_mean[:,i:i+1,:])\
                                     , h\
                                     , torch.squeeze(Mask[:,i:i+1,:])\
                                     , torch.squeeze(Delta[:,i:i+1,:])\
                                     , C)
        return h
    
    def initHidden(self, batch_size):
        h = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        C = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
        return h, C

        
        
