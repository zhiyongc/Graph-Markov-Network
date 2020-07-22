import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

    
class GMN(nn.Module):
    def __init__(self, A, dtype, device, num_layer = 6, gamma = 0.9):
        super(GMN, self).__init__()
        self.gamma = gamma
        self.n = A.shape[0]
        self.num_layer = num_layer
        self.dtype = dtype
        self.device = device
        
        self.A = torch.tensor(A, dtype = self.dtype, device = self.device)
        self.A = torch.clamp(self.A, min=0, max=1)
        self.A1 = self.A
        
        self.A_list = torch.empty(self.num_layer, self.n, self.n, dtype = self.dtype, device = self.device)
        for i in range(self.num_layer):
            if i == 0:
                self.A_list[i] = self.A
            else:
                self.A_list[i] = torch.clamp(torch.matmul(self.A_list[i-1], self.A), min=0, max=1)
        W_list = torch.empty(self.num_layer, self.n, self.n, dtype = self.dtype, device = self.device)
        nn.init.uniform_(W_list)
        self.W_list = torch.nn.Parameter(W_list)
        
        
    def forward(self, input):
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        X = torch.squeeze(input[:,0,:,:])
        M = torch.squeeze(input[:,1,:,:])
        
        Y_hat = None
        for i in range(0, self.num_layer):
            if i == 0:
                Y_hat = self.gamma * (torch.mm(X[:,-1,:], self.A_list[0] * self.W_list[0]))
            elif i == 1:
                Y_hat += self.gamma**2 * (torch.mm(X[:,-2,:] * (1-M[:,-1,:]), self.A_list[1] * self.W_list[1]))
            else:
                # i >= 2
                NonMissing = (1-M[:,-1,:])
                for j in range(1, i):
                    NonMissing = NonMissing * (1-M[:,-(j+1),:])
                Y_hat += self.gamma**(i+1) * (torch.mm(X[:,-(i+1),:] * NonMissing, self.A_list[i] * self.W_list[i]))
        return Y_hat
    

class GNN(nn.Module):
    def __init__(self, A, layer = 5, gamma = 0.9):
        
        super(GNN, self).__init__()
        
        self.gamma = gamma
        
        self.n = A.shape[0]
        
        self.layer = layer
        
        self.A = torch.tensor(A, dtype = dtype, device = device)
        self.A = torch.clamp(self.A, min=0, max=1)
        self.A1 = self.A
        self.W1 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A2 = torch.matmul(self.A1, self.A)
        self.A2 = torch.clamp(self.A2, min=0, max=1)
        self.W2 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A3 = torch.matmul(self.A2, self.A)
        self.A3 = torch.clamp(self.A3, min=0, max=1)
        self.W3 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A4 = torch.matmul(self.A3, self.A)
        self.A4 = torch.clamp(self.A4, min=0, max=1)
        self.W4 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A5 = torch.matmul(self.A4, self.A)
        self.A5 = torch.clamp(self.A5, min=0, max=1)
        self.W5 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A6 = torch.matmul(self.A5, self.A)
        self.A6 = torch.clamp(self.A6, min=0, max=1)
        self.W6 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A7 = torch.matmul(self.A6, self.A)
        self.A7 = torch.clamp(self.A7, min=0, max=1)
        self.W7 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A8 = torch.matmul(self.A7, self.A)
        self.A8 = torch.clamp(self.A8, min=0, max=1)
        self.W8 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A9 = torch.matmul(self.A8, self.A)
        self.A9 = torch.clamp(self.A9, min=0, max=1)
        self.W9 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        self.A10 = torch.matmul(self.A9, self.A)
        self.A10 = torch.clamp(self.A10, min=0, max=1)
        self.W10 = torch.nn.Parameter(torch.randn([self.n, self.n],dtype = dtype, device = device))
        
        
    def forward(self, input):
        
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        X = torch.squeeze(input[:,0,:,:])
        M = torch.squeeze(input[:,1,:,:])
        
        Y_hat = self.gamma * (torch.mm(X[:,-1,:], self.A1 * self.W1))
        Y_hat += self.gamma**2 * (torch.mm(X[:,-2,:] * (1-M[:,-1,:]), self.A2 * self.W2))
        if self.layer >=3:
            Y_hat += self.gamma**3 * (torch.mm(X[:,-3,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]), self.A3 * self.W3))
        if self.layer >=4:
            Y_hat += self.gamma**4 * (torch.mm(X[:,-4,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]), self.A4 * self.W4))
        if self.layer >=5:
            Y_hat += self.gamma**5 * (torch.mm(X[:,-5,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]), self.A5 * self.W5))
        if self.layer >=6:
            Y_hat += self.gamma**6 * (torch.mm(X[:,-6,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]), self.A6 * self.W6))
        if self.layer >=7:
            Y_hat += self.gamma**7 * (torch.mm(X[:,-7,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]), self.A7 * self.W7))
        if self.layer >=8:
            Y_hat += self.gamma**8 * (torch.mm(X[:,-8,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]) * (1-M[:,-7,:]), self.A8 * self.W8))
        if self.layer >=9:
            Y_hat += self.gamma**9 * (torch.mm(X[:,-9,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]) * (1-M[:,-7,:]) * (1-M[:,-8,:]), self.A9 * self.W9))
        if self.layer >=10:
            Y_hat += self.gamma**10 * (torch.mm(X[:,-10,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]) * (1-M[:,-7,:]) * (1-M[:,-8,:])* (1-M[:,-9,:]), self.A10 * self.W10))
        
        return Y_hat
    
    

class SGNN(nn.Module):
    def __init__(self, A, layer = 5, gamma = 0.9):
        
        super(SGNN, self).__init__()
        
        self.gamma = gamma
        
        self.layer = layer
        
        self.n = A.shape[0]
        A = torch.tensor(A, dtype = dtype, device = device)
        D_rsqrt = torch.tensor(torch.diag(torch.sum(A, dim=0).pow(-0.5)), dtype = dtype, device = device)
        I = torch.tensor(torch.eye(self.n, self.n), dtype = dtype, device = device)
        self.L = I - D_rsqrt.mm(A).mm(D_rsqrt)
        self.e, self.v = torch.symeig(self.L, eigenvectors=True)
        
        self.F_x_param = Parameter(torch.Tensor(self.n))
        self.F_h_param = Parameter(torch.Tensor(self.n))
        
        self.A1 = Parameter(torch.Tensor(self.n))
        self.A2 = Parameter(torch.Tensor(self.n))
        self.A3 = Parameter(torch.Tensor(self.n))
        self.A4 = Parameter(torch.Tensor(self.n))
        self.A5 = Parameter(torch.Tensor(self.n))
        self.A6 = Parameter(torch.Tensor(self.n))
        self.A7 = Parameter(torch.Tensor(self.n))
        self.A8 = Parameter(torch.Tensor(self.n))
        self.A9 = Parameter(torch.Tensor(self.n))
        self.A10 = Parameter(torch.Tensor(self.n))

        self.reset_parameters()
        
    def forward(self, input):
        
        batch_size = input.size(0)
        type_size = input.size(1)
        step_size = input.size(2)
        spatial_size = input.size(3)

        X = torch.squeeze(input[:,0,:,:])
        M = torch.squeeze(input[:,1,:,:])
        
        A1_diag = torch.diag(self.A1)
        A2_diag = torch.diag(self.A2)
        A3_diag = torch.diag(self.A3)
        A4_diag = torch.diag(self.A4)
        A5_diag = torch.diag(self.A5)
        A6_diag = torch.diag(self.A6)
        A7_diag = torch.diag(self.A7)
        A8_diag = torch.diag(self.A8)
        A9_diag = torch.diag(self.A9)
        A10_diag = torch.diag(self.A10)
        
        
        Y_hat = self.gamma * (torch.mm(X[:,-1,:], self.v.matmul(A1_diag).matmul(torch.t(self.v))))
        Y_hat += self.gamma**2 * (torch.mm(X[:,-2,:] * (1-M[:,-1,:]), self.v.matmul(A2_diag).matmul(torch.t(self.v))))
        Y_hat += self.gamma**3 * (torch.mm(X[:,-3,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]), self.v.matmul(A3_diag).matmul(torch.t(self.v))))
        Y_hat += self.gamma**4 * (torch.mm(X[:,-4,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]), self.v.matmul(A4_diag).matmul(torch.t(self.v))))
        if self.layer >=5:
            Y_hat += self.gamma**5 * (torch.mm(X[:,-5,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]), self.v.matmul(A5_diag).matmul(torch.t(self.v))))
        if self.layer >=6:
            Y_hat += self.gamma**6 * (torch.mm(X[:,-6,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]), self.v.matmul(A6_diag).matmul(torch.t(self.v))))
        if self.layer >=7:
            Y_hat += self.gamma**7 * (torch.mm(X[:,-7,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]), self.v.matmul(A7_diag).matmul(torch.t(self.v))))
        if self.layer >=8:
            Y_hat += self.gamma**8 * (torch.mm(X[:,-8,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]) * (1-M[:,-7,:]), self.v.matmul(A8_diag).matmul(torch.t(self.v))))
        if self.layer >=9:
            Y_hat += self.gamma**9 * (torch.mm(X[:,-9,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]) * (1-M[:,-7,:]) * (1-M[:,-8,:]), self.v.matmul(A9_diag).matmul(torch.t(self.v))))
        if self.layer >=10:
            Y_hat += self.gamma**10 * (torch.mm(X[:,-10,:] * (1-M[:,-1,:]) * (1-M[:,-2,:]) * (1-M[:,-3,:]) * (1-M[:,-4,:]) * (1-M[:,-5,:]) * (1-M[:,-6,:]) * (1-M[:,-7,:]) * (1-M[:,-8,:])* (1-M[:,-9,:]), self.v.matmul(A10_diag).matmul(torch.t(self.v))))
        
        return Y_hat
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.n)
        self.A1.data.uniform_(-stdv, stdv)
        self.A2.data.uniform_(-stdv, stdv)
        self.A3.data.uniform_(-stdv, stdv)
        self.A4.data.uniform_(-stdv, stdv)
        self.A5.data.uniform_(-stdv, stdv)
        self.A6.data.uniform_(-stdv, stdv)
        self.A7.data.uniform_(-stdv, stdv)
        self.A8.data.uniform_(-stdv, stdv)
        self.A9.data.uniform_(-stdv, stdv)
        self.A10.data.uniform_(-stdv, stdv)
        