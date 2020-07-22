import torch
from torch.autograd import Variable
import torch.utils.data as utils

# import torch.utils.data as utils
# import torch.nn.functional as F
# import torch
# import torch.nn as nn

# from torch.nn.parameter import Parameter

import math
import numpy as np
import copy
import time

def PrepareDataset(speed_matrix, \
                   BATCH_SIZE = 64, \
                   seq_len = 10, \
                   pred_len = 1, \
                   train_propotion = 0.7, \
                   valid_propotion = 0.2, \
                   mask_ones_proportion = 0.8, \
                   masking = True, \
                   masking_type = 'random', \
                   delta_last_obsv = False, \
                   shuffle = True, \
                   random_seed = 1024, \
                   ):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert speed/volume/occupancy matrix to training and testing dataset. 
    The vertical axis of speed_matrix is the time axis and the horizontal axis 
    is the spatial axis.
    
    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """
    
    print('Start Generate Data')
    print('\t batch_size:', BATCH_SIZE, '\t\t input_len:', seq_len, '\t\t\t pred_len:', pred_len)
    print('\t train_set:', train_propotion, '\t\t valid_set:', valid_propotion, '\t\t test_set:', np.around(1 - train_propotion - valid_propotion, decimals=2))
    print('\t masking:', masking, '\t\t\t mask_ones_rate:', mask_ones_proportion, '\t\t delta & last_obsv:', delta_last_obsv)
    print('\t shuffle dataset:', shuffle, '\t\t random_seed:', random_seed)
    
    np.random.seed(random_seed)
    
    time_len = speed_matrix.shape[0]
    speed_matrix = speed_matrix.clip(0, 100)
    
    max_speed = speed_matrix.max().max()
    speed_matrix =  speed_matrix / max_speed
    
    data_zero_values = np.where(speed_matrix == 0)[0].shape[0]
    data_zero_rate = data_zero_values / (speed_matrix.shape[0] * speed_matrix.shape[1])
    print('Orignal dataset missing rate:', data_zero_rate)
    
    print('Generating input and labels...')
    if not masking:
        speed_sequences, speed_labels = [], []
        for i in range(time_len - seq_len - pred_len):
            speed_sequences.append(speed_matrix.iloc[i:i+seq_len].values)
            speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
        speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
        print('Input sequences and labels are generated.')
        
    else:
        # using zero-one mask to randomly set elements to zeros
        # genertate mask
        
        if masking_type == 'random':
            Mask = np.random.choice([0,1], size=(speed_matrix.shape), p = [1 - mask_ones_proportion, mask_ones_proportion])
            masked_speed_matrix = np.multiply(speed_matrix, Mask)
            Mask[np.where(masked_speed_matrix == 0)] = 0
            mask_zero_values = np.where(Mask == 0)[0].shape[0] / (Mask.shape[0] * Mask.shape[1])
            print('\t Masked dataset missing rate:', np.around(mask_zero_values, decimals=4),'(mask zero rate:', np.around(1 - mask_ones_proportion, decimals=4), ')')
        
        speed_sequences, speed_labels = [], []
        for i in range(time_len - seq_len - pred_len):
            speed_sequences.append(masked_speed_matrix.iloc[i:i+seq_len].values)
            speed_labels.append(speed_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
        speed_sequences, speed_labels = np.asarray(speed_sequences), np.asarray(speed_labels)
        print('Input sequences, labels, and masks are generated.')
        
        # Mask sequences
        Mask = np.ones_like(speed_sequences)
        Mask[np.where(speed_sequences == 0)] = 0
        
        if delta_last_obsv:
            # temporal information
            interval = 5 # 5 minutes
            S = np.zeros_like(speed_sequences) # time stamps
            for i in range(S.shape[1]):
                S[:,i,:] = interval * i

            Delta = np.zeros_like(speed_sequences) # time intervals
            for i in range(1, S.shape[1]):
                Delta[:,i,:] = S[:,i,:] - S[:,i-1,:]
                
            Delta = Delta / Delta.max() # normalize

            missing_index = np.where(speed_sequences == 0)

            X_last_obsv = np.copy(speed_sequences)
            for idx in range(missing_index[0].shape[0]):
                i = missing_index[0][idx] 
                j = missing_index[1][idx]
                k = missing_index[2][idx]
                if j != 0 and j != seq_len-1:
                    Delta[i,j+1,k] = Delta[i,j+1,k] + Delta[i,j,k]
                if j != 0:
                    X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
        
            print('Time intervals of missing values and last_observations are generated.')
    
    # shuffle and split the dataset to training and testing datasets
    print('Start to shuffle dataset ...')
    if shuffle:
        sample_size = speed_sequences.shape[0]
        index = np.arange(sample_size, dtype = int)
        np.random.shuffle(index)
        speed_sequences = speed_sequences[index]
        speed_labels = speed_labels[index]
        if masking:
            Mask = Mask[index]
            if delta_last_obsv:
                Delta = Delta[index]
                X_last_obsv = X_last_obsv[index]
    print('Dataset Shuffled. Start to split dataset ...')
    
    if not masking:
        dataset_agger = speed_sequences
    else:
        speed_sequences = np.expand_dims(speed_sequences, axis=1)
        Mask = np.expand_dims(Mask, axis=1)
        if delta_last_obsv:
            Delta = np.expand_dims(Delta, axis=1)
            X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
            dataset_agger = np.concatenate((speed_sequences, Mask, Delta, X_last_obsv), axis = 1)
        else:
            dataset_agger = np.concatenate((speed_sequences, Mask), axis = 1)
    
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
    if masking:
        train_data, train_label = dataset_agger[:train_index], speed_labels[:train_index]
        valid_data, valid_label = dataset_agger[train_index:valid_index], speed_labels[train_index:valid_index]
        test_data, test_label = dataset_agger[valid_index:], speed_labels[valid_index:]
    else:
        train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
        valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
        test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = False)
    
    X_mean = np.mean(speed_sequences, axis = 0)
    
    print('Finished')
    
    return train_dataloader, valid_dataloader, test_dataloader, max_speed, X_mean



def TrainModel(model, train_dataloader, valid_dataloader, learning_rate = 1e-5, optm = 'Adam', num_epochs = 300, patience = 10, min_delta = 0.00001):
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    model.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = learning_rate
    if optm == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    elif optm == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr = learning_rate)
    elif optm == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
        
    use_gpu = torch.cuda.is_available()
    
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    train_time_list = []
    valid_time_list = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    
    losses_epochs_valid_steps_l1 = [] 
    
    sub_epoch = 1
    
    for epoch in range(0, num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
        
        losses_epoch_train = []
        losses_epoch_valid = []

        train_start = time.time()
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
                
            model.zero_grad()

            outputs = model(inputs)
            
            if len(labels[labels==0]): 
                label_mask = torch.ones_like(labels).cuda()
                label_mask = label_mask * labels
                label_mask[label_mask!=0] = 1
                loss_train = loss_MSE(outputs * label_mask, torch.squeeze(labels)) 
            else:
                loss_train = loss_MSE(outputs, torch.squeeze(labels)) 
#             print(loss_train.item())
#             print([loss_train.data[0]])
#             print(loss_train.data.cpu().numpy()[0])
#             print(loss_train.item())
            losses_epoch_train.append(loss_train.item())
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
        train_end = time.time()
        
        # validation   
        valid_start = time.time()
        
        losses_l1_allsteps_val = []
        
        for data in valid_dataloader:
            
            inputs_val, labels_val = data
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            outputs_val= model(inputs_val)
            
            if len(labels_val[labels_val==0]): 
                labels_val_mask = torch.ones_like(labels_val).cuda()
                labels_val_mask = labels_val_mask * labels_val
                labels_val_mask[labels_val_mask!=0] = 1

                loss_valid = loss_MSE(outputs_val * labels_val_mask, torch.squeeze(labels_val))
            else:
                loss_valid = loss_MSE(outputs_val , torch.squeeze(labels_val))
            
                
#                 print(inputs_val[:,1:,:].shape)
                
#                 print(labels_val.shape)
                
#                 full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                
#                 delta = torch.abs(full_labels_val - outputs_val)
                
#                 delta_batch_mean = torch.mean(delta, dim = 0)
                
#                 delta_batch_spatial_mean = torch.mean(delta_batch_mean, dim = 1)

#                 losses_l1_allsteps_val.append(delta_batch_spatial_mean.data.cpu().numpy())

#                 if losses_l1_allsteps_val is None:
#                     losses_l1_allsteps_val = delta_batch_spatial_mean
#                 else:
#                     losses_l1_allsteps_val += delta_batch_spatial_mean
                
#                 # mean l1 loss for each step of the sequences of a batched sample 
#                 mean_loss_l1_allsteps = torch.mean(torch.mean(torch.abs(full_labels_val - outputs_val), dim = 0), dim = 1)
                
# #                 print(mean_loss_l1_allsteps)
#                 losses_l1_allsteps_val.append(delta_batch_spatial_mean.data.cpu().numpy(), )
    
            losses_epoch_valid.append(loss_valid.item())
        
        valid_end = time.time()
        
        avg_losses_epoch_train = sum(losses_epoch_train) / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid) / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
#         print(losses_l1_allsteps_val)
        
        if losses_l1_allsteps_val:
            losses_l1_allsteps_val = np.asarray(losses_l1_allsteps_val)
            losses_l1_allsteps_mean = np.mean(losses_l1_allsteps_val, 0)
            losses_epochs_valid_steps_l1.append(losses_l1_allsteps_mean)
#             print(losses_l1_allsteps_val)
# #             losses_l1_allsteps_mean = (losses_l1_allsteps_mean / valid_batch_number).data.cpu().numpy()
# #             losses_l1_allsteps_mean = torch.mean(losses_l1_allsteps_val, dim = 0).data.cpu().numpy()
# # #             print(losses_l1_allsteps_mean.shape)
# # #             losses_epochs_valid_steps_l1.append(losses_l1_allsteps_mean)
            
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.plot(losses_l1_allsteps_mean)
            plt.ylim((0, 0.1))
            plt.title('Epoch: ' + str(epoch))
            plt.ylabel('MSE')
            plt.xlabel('Step')
            fig.savefig('./Figures/' + type(model).__name__ + '_' + str(epoch) + '.png')   # save the figure to file
            plt.close(fig)
            
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = copy.deepcopy(model)
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
                
#             sub_epoch += 1
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
                    
        if (epoch >= 5 and (patient_epoch == 4 or sub_epoch % 50 == 0)) and learning_rate > 1e-5:
            learning_rate = learning_rate / 10
            if optm == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
            elif optm == 'Adadelta':
                optimizer = torch.optim.Adadelta(model.parameters(), lr = learning_rate)
            elif optm == 'RMSprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
            sub_epoch = 1
        else:
            sub_epoch += 1
                        
        
        # Print training parameters
        cur_time = time.time()
        train_time = np.around([train_end - train_start] , decimals=2)
        train_time_list.append(train_time)
        valid_time = np.around([valid_end - valid_start] , decimals=2)
        valid_time_list.append(valid_time)
        
        print('Epoch: {}, train_loss: {}, valid_loss: {}, lr: {}, train_time: {}, valid_time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    learning_rate,\
                    np.around([train_end - train_start] , decimals=2),\
                    np.around([valid_end - valid_start] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
    
    train_time_avg = np.mean(np.array(train_time_list))
    valid_time_avg = np.mean(np.array(valid_time_list))
    
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid, losses_epochs_valid_steps_l1, train_time_avg, valid_time_avg]



def TestModel(model, test_dataloader, max_speed):
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 

    output_list = []
    label_list = []
    
    losses_l1_allsteps = None
    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        
        loss_mse = loss_MSE(outputs, torch.squeeze(labels))
        loss_l1 = loss_L1(outputs, torch.squeeze(labels))
        
        losses_mse.append(loss_mse.cpu().data.numpy())
        losses_l1.append(loss_l1.cpu().data.numpy())
        
        output_list.append(torch.squeeze(outputs).cpu().data.numpy())
        label_list.append(torch.squeeze(labels).cpu().data.numpy())
        
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
#     print(losses_l1)
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    output_list = np.array(output_list)
    label_list = np.array(label_list)
    
    non_zero_index = np.nonzero(label_list)
    MAE = np.mean(np.absolute(output_list[non_zero_index] - label_list[non_zero_index])) * max_speed
    RMSE = np.sqrt(np.mean(np.square(output_list[non_zero_index]* max_speed - label_list[non_zero_index]* max_speed)))
    MAPE = np.mean(np.absolute(output_list[non_zero_index] - label_list[non_zero_index])/label_list[non_zero_index]) * 100         
    MAE = np.around(MAE, decimals=3)
    RMSE = np.around(RMSE, decimals=3)
    MAPE = np.around(MAPE, decimals=3)
    print('Tested: MAE: {}, RMSE : {}, MAPE : {} %, '.format( MAE, RMSE, MAPE))
    return [losses_l1, losses_mse, None, MAE, RMSE, MAPE, losses_l1_allsteps]