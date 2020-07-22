#!/usr/bin/env python
# coding: utf-8


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
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# %matplotlib inline  
# plt.ioff()
# plt.ion()

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

sys.path.append('../')


import models
import utils
import importlib

import argparse

dtype = torch.float32
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

torch.cuda.set_device(0)



# Load data

def loadDataset(dataset = None):
    if dataset == 'PEMS-BAY':
        speed_matrix = pd.read_hdf('../Data/PEMS-BAY/pems-bay.h5')
        A = pd.read_pickle('../Data/PEMS-BAY/adj_mx_bay.pkl')
        A = A[2]
        A[np.where(A != 0)] = 1
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[0]):
                if A[i,j] == 1:
                    A[j,i] = 1
    elif dataset == 'METR-LA':
        speed_matrix = pd.read_hdf('../Data/METR-LA/metr-la.h5')
        A = pd.read_pickle('../Data/METR-LA/adj_mx.pkl')
        A = A[2]
        A[np.where(A != 0)] = 1
        for i in range(0, A.shape[0]):
            for j in range(i, A.shape[0]):
                if A[i,j] == 1:
                    A[j,i] = 1
    elif dataset == 'LOOP-SEA':
        speed_matrix = pd.read_pickle('../Data/LOOP-SEA/speed_matrix_2015_1mph')
        A = np.load('../Data/LOOP-SEA/Loop_Seattle_2015_A.npy')
    elif dataset == 'INRIX-SEA':
        speed_matrix = pd.read_pickle('../Data/INRIX-SEA/INRIX_Seattle_Speed_Matrix__2012_v2.pkl')
        A = np.load('../Data/INRIX-SEA/INRIX_Seattle_Adjacency_matrix_2012_v2.npy')
    else:
        print('Dataset not found.')
        return None, None
    print('Dataset loaded.')
    return speed_matrix, A


# def StoreData(result_dict, model_name, train_result, test_result, directory, model, random_seed = 1024, save_model=True):
#     result_dict[model_name] = {}
#     result_dict[model_name]['train_loss'] = train_result[2]
#     result_dict[model_name]['valid_loss'] = train_result[3] 
#     result_dict[model_name]['MAE'] = test_result[3] 
#     result_dict[model_name]['RMSE'] = test_result[4] 
#     result_dict[model_name]['MAPE'] = test_result[5] 
#     f = open(directory + '/gmn_log_rs_' + str(random_seed) + '.pkl', "wb")
#     pickle.dump(result_dict,f)
#     f.close()
#     if save_model:
#         torch.save(model.state_dict(), directory + '/' + model_name)

def StoreData(model_name, train_result, test_result, directory, model, random_seed = 1024, save_model=True):
    
    if os.path.isfile(directory + '/gmn_log_rs_' + str(random_seed) + '.pkl'):
        f = open(directory + '/gmn_log_rs_' + str(random_seed) + '.pkl', "rb")
        result_dict = pickle.load(f)
        f.close()
    else:
        result_dict = {}
        
    result_dict[model_name] = {}
    result_dict[model_name]['train_loss'] = train_result[2]
    result_dict[model_name]['valid_loss'] = train_result[3] 
    result_dict[model_name]['train_time'] = train_result[5]
    result_dict[model_name]['valid_time'] = train_result[6] 
    result_dict[model_name]['MAE'] = test_result[3] 
    result_dict[model_name]['RMSE'] = test_result[4] 
    result_dict[model_name]['MAPE'] = test_result[5] 
    f = open(directory + '/gmn_log_rs_' + str(random_seed) + '.pkl', "wb")
    pickle.dump(result_dict,f)
    f.close()
    if save_model:
        torch.save(model.state_dict(), directory + '/' + model_name)


        
if __name__ == "__main__":
    
#     dataset == 'PEMS-BAY'
#     missing_rate = 0.2
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '-dataset', default='PEMS-BAY', type=str, required=True, help="specify dataset")
    parser.add_argument('-m', '-missingrate', default=0.2, type=float, required=True, help="specify missing rate")
    parser.add_argument('-o', '-optm', default='Adam', type=str, required=False, help="specify training optimizer")
    parser.add_argument('-l', '-learningrate', default=0.001, type=float, required=False, help="specify initial learning rate")
    parser.add_argument('-g', '-gamma', default=0.9, type=float, required=False, help="specify gamma for GMN")
    parser.add_argument('-t', '-maskingtype', default='random', type=str, required=False, help="specify masking type")
    parser.add_argument('-r', '-randomseed', default=1024, type=int, required=False, help="specify random seed")
    parser.add_argument('-s', '-savemodel', default=1, type=int, required=False, help="specify whether save model")
    args = parser.parse_args()
    
    dataset = args.d
    missing_rate = args.m
    optm = args.o
    learning_rate = args.l
    gamma = args.g
    random_seed = args.r
    masking_type = args.t
    save_model = args.s
    
    np.random.seed(random_seed)
    
    print('Exp: baseline models')
    print('\tDataset:', dataset)
    print('\tMissing rate:', missing_rate)
    print('\tOptimizer:', optm) 
    print('\tLearnig ate:', learning_rate)
    print('\tGamma:', gamma)
    print('\tMasking type:', masking_type)
    print('\tRandom seed:', random_seed)
    if save_model==1:
        print('\tSave model:', 'True')
    else:
        print('\tSave model:', 'False')


    directory = './Masking_' + masking_type + '/GMN_' + str(dataset) + '_MR=' + str(missing_rate) + '_gamma=' + str(gamma)

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    speed_matrix, A = loadDataset(dataset = dataset)


    importlib.reload(utils)
    from utils import PrepareDataset, TrainModel, TestModel


    importlib.reload(utils)
    from utils import PrepareDataset
    mask_ones_proportion = 1 - missing_rate
    train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean = PrepareDataset(speed_matrix, BATCH_SIZE = 64, seq_len = 10, pred_len = 1, \
                   train_propotion = 0.6, valid_propotion = 0.2, \
                   mask_ones_proportion = mask_ones_proportion, \
                   masking = True, masking_type = masking_type, delta_last_obsv = False, \
                   shuffle = True, random_seed = random_seed)


    inputs, labels = next(iter(train_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()


    importlib.reload(utils)
    from utils import TrainModel, TestModel

    # SGNN-6
    importlib.reload(models)
    from models import SGNN
    importlib.reload(utils)
    from utils import TrainModel, TestModel
    model_name = 'SGMN6'
    print(model_name)
    model = SGNN(A, layer = 6, gamma = gamma)
    model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
    test_result = TestModel(model, test_dataloader, max_speed)
    StoreData(model_name, train_result, test_result, directory, model, random_seed, save_model)

    # SGNN-8
    importlib.reload(models)
    from models import SGNN
    importlib.reload(utils)
    from utils import TrainModel, TestModel
    model_name = 'SGMN8'
    print(model_name)
    model = SGNN(A, layer = 8, gamma = gamma)
    model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
    test_result = TestModel(model, test_dataloader, max_speed)
    StoreData(model_name, train_result, test_result, directory, model, random_seed, save_model)
    
    # SGNN-10
    importlib.reload(models)
    from models import SGNN
    importlib.reload(utils)
    from utils import TrainModel, TestModel
    model_name = 'SGMN10'
    print(model_name)
    model = SGNN(A, layer = 10, gamma = gamma)
    model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
    test_result = TestModel(model, test_dataloader, max_speed)
    StoreData(model_name, train_result, test_result, directory, model, random_seed, save_model)

    # GNN-6
    importlib.reload(models)
    from models import GNN
    importlib.reload(utils)
    from utils import TrainModel, TestModel
    model_name = 'GMN6'
    print(model_name)
    model = GNN(A, layer = 6, gamma = gamma)
    model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
    test_result = TestModel(model, test_dataloader, max_speed)
    StoreData(model_name, train_result, test_result, directory, model, random_seed, save_model)

    # GNN-8
    importlib.reload(models)
    from models import GNN
    importlib.reload(utils)
    from utils import TrainModel, TestModel
    model_name = 'GMN8'
    print(model_name)
    model = GNN(A, layer = 8, gamma = gamma)
    model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
    test_result = TestModel(model, test_dataloader, max_speed)
    StoreData(model_name, train_result, test_result, directory, model, random_seed, save_model)
    
    # GNN-10
    importlib.reload(models)
    from models import GNN
    importlib.reload(utils)
    from utils import TrainModel, TestModel
    model_name = 'GMN10'
    print(model_name)
    model = GNN(A, layer = 10, gamma = gamma)
    model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
    test_result = TestModel(model, test_dataloader, max_speed)
    StoreData(model_name, train_result, test_result, directory, model, random_seed, save_model)
    
    

#     # ANN
#     importlib.reload(models)
#     from models import ANN
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'FCMN'
#     print(model_name)
#     model = ANN(A, layer = 10, gamma = 0.9)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, optm = optm, learning_rate = learning_rate, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # LSTM
#     importlib.reload(models)
#     from models import LSTM
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'LSTM'
#     model = LSTM(A.shape[0])
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # LSTM-I
#     importlib.reload(models)
#     from models import LSTM
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'LSTMI'
#     model = LSTM(A.shape[0], imputation = True)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # LSTM-D
#     import LSTMD
#     importlib.reload(LSTMD)
#     from LSTMD import LSTMD
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'LSTMD'
#     print(model_name)
#     model = LSTMD(fea_size, fea_size, fea_size, X_mean)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # GRU
#     importlib.reload(models)
#     from models import GRU
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'GRU'
#     print(model_name)
#     gru = GRU(A.shape[0])
#     gru, train_result = TrainModel(gru, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(gru, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # GRU-I
#     importlib.reload(models)
#     from models import GRU
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'GRUI'
#     print(model_name)
#     model = GRU(A.shape[0], imputation = True)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # GRU-D
#     import GRUD
#     importlib.reload(GRUD)
#     from GRUD import GRUD
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'GRUD'
#     print(model_name)
#     model = GRUD(fea_size, fea_size, fea_size, X_mean)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # Graph GRU-D
#     import GGRUD
#     importlib.reload(GGRUD)
#     from GGRUD import GGRUD
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'GGRUD'
#     print(model_name)
#     model = GGRUD(A, fea_size, fea_size, fea_size, X_mean)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-3, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)

#     # GCLSTM
#     importlib.reload(models)
#     from models import GCLSTM
#     importlib.reload(utils)
#     from utils import TrainModel, TestModel
#     model_name = 'GCLSTM'
#     print(model_name)
#     model = GCLSTM(A)
#     model, train_result = TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-2, patience = 5)
#     test_result = TestModel(model, test_dataloader, max_speed)
#     StoreData(result_dict, model_name, train_result, test_result, directory, model)




