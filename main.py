import sys
import os.path
import time
import pandas as pd
import numpy as np
import mlflow
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utilities import data, evaluations
from utilities.cifar10 import cifar10
from utilities.dataset import CIFAR10
from utilities.utils import *

id = sys.argv[1] # experiment id

# paths
create_dir(id)
train_dir = data_dir(id,'train.csv')
val_dir = data_dir(id,'val.csv')
test_dir = data_dir(id,'test.csv')
save_model = model_dir(id)
writer_train_acc = SummaryWriter(log_dir(id,'train_acc'))
writer_val_acc = SummaryWriter(log_dir(id,'val_acc'))
writer_train_loss = SummaryWriter(log_dir(id,'train_loss'))
logger = Logger(id).logging

# configs
config_pool = readConfig('config.yml')
if not any(param['id'] == id for param in config_pool):
    sys.exit("Configuration {} does not exist!".format(id))
else:
    for param in config_pool:
        if param['id'] == id:
            config = param
            logger(config)

global_step = 0

def main(seed):
    set_seed(seed)

    # 1. create master
    cifar10(id, config.get('root'), config.get('val_ratio'), config.get('unlabel_ratio')).master()
    # 2. check GPU
    device = torch.device('cuda')
    logger('device: {}'.format(torch.cuda.device_count()))
    
    # 4. data loader: train & val
    if config.get('label_batch_size') is None:
        label_batch_size = config.get('batch_size')/2 # mixmatch paper suggestion
    else: 
        label_batch_size = config.get('label_batch_size')
    unlabel_batch_size = config.get('batch_size') - label_batch_size

    train_dataset = CIFAR10('train', train_dir, remove_unlabel=False)
    unlabel_idx, label_idx = train_dataset.get_index()
    batch_sampler = data.TwoStreamBatchSampler(unlabel_idx, label_idx, config.get('batch_size'), label_batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)
    
    val_dataset = CIFAR10('val', val_dir, remove_unlabel=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.get('batch_size'), shuffle=True)

    logger('training labels: {}'.format(len(label_idx)))
    logger('training dataset: {}'.format(len(train_dataset)))
    logger('training dataloader: {}'.format(len(train_loader)))
    logger('validation dataset: {}'.format(len(val_dataset)))
    logger('validation dataloader: {}'.format(len(val_loader)))
    
    # 6. import torch.renet50 & change output layer
    model = nn.DataParallel(WideResnet50(config.get('num_class'))).to(device)
    ema_model = nn.DataParallel(WideResnet50(config.get('num_class'), ema=True)).to(device)

    consis_criterion = nn.MSELoss().to(device)
    # 7. optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate'), amsgrad=True)

    # 9. train & val
    train_start = time.time()
    best_acc = 0
    best_epoch = 0

    for epoch in range(config.get('num_epochs')):
        train_loss, train_acc = train(model, ema_model, optimizer, train_loader, epoch, unlabel_batch_size, label_batch_size, consis_criterion, device)
        print('global_step:', global_step)
        val_loss, val_acc = validation(val_loader, ema_model, epoch, device)
        
        is_best = val_acc >= best_acc
        best_acc = max (val_acc, best_acc)
        if is_best:
            torch.save(ema_model.state_dict(), save_model)
            best_epoch = epoch+1

        logger ('Epoch: {} | Second: {:.4f} | Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}'
                .format(epoch+1, time.time() - train_start, train_loss/len(train_loader), train_acc/len(train_loader), val_loss/len(val_loader), val_acc/len(val_loader)))
        
        writer_train_acc.add_scalar('acc',train_acc/len(train_loader), epoch) # same tag id, plot two lines in same graph
        writer_val_acc.add_scalar('acc', val_acc/len(val_loader), epoch)
    
    logger('global step: {}'.format(global_step))
    logger('best model epoch: {}'.format(best_epoch))

    # 11. close tensorboard writer
    writer_train_acc.close()
    writer_val_acc.close()

def train(model, ema_model, optimizer, train_loader, epoch, unlabel_batch_size, label_batch_size, consis_criterion, device):
    
    global global_step
    softmax = nn.Softmax(dim=1)
    train_loss = 0
    train_acc = 0

    for i, (input_1, input_2, target) in enumerate(train_loader):
        input_u1, input_l = torch.split(input_1, [unlabel_batch_size, label_batch_size])
        input_u2 = torch.split(input_2, [unlabel_batch_size, label_batch_size])[0]
        target_l = torch.split(target, [unlabel_batch_size, label_batch_size])[1]
    
        input_u1 = input_u1.to(device)
        input_u2 = input_u2.to(device)
        input_l = input_l.to(device)
        target_l = target_l.to(device)

        # guess label for unlabel data
        with torch.no_grad():
            outputs_u1 = softmax(model(input_u1))
            outputs_u2 = softmax(model(input_u2))
            guess_u = sum([outputs_u1, outputs_u2]) / 2 
            guess_u = sharpen(guess_u, config.get('T'))
            guess_u = guess_u.repeat(2, 1)

        # mixup
        target_l_encode = torch.cuda.FloatTensor(label_batch_size, config.get('num_class')).zero_().scatter_(1, target_l.view(-1,1), 1) # one-hot to concate with outputs_u
        all_inputs = torch.cat([input_l, input_u1,input_u2], 0)
        all_targets = torch.cat([target_l_encode, guess_u], 0)
        
        index = torch.randperm(all_inputs.shape[0])
        shuffled_all_inputs, shuffled_all_targets = all_inputs[index], all_targets[index]

        beta_distirb = torch.distributions.beta.Beta(config.get('mm_alpha'), config.get('mm_alpha'))
        lam = beta_distirb.sample().item()
        lam = max(lam, 1-lam) # lam always > 0.5

        mixed_inputs = lam * all_inputs + (1-lam) * shuffled_all_inputs # mix two unlabeld image into one
        mixed_targets = lam * all_targets + (1-lam) * shuffled_all_targets

        mixed_input_l, mixed_target_l = mixed_inputs[:label_batch_size], mixed_targets[:label_batch_size]
        mixed_input_u, mixed_target_u = mixed_inputs[label_batch_size:], mixed_targets[label_batch_size:]

        # loss
        class_loss = class_criterion(softmax(model(mixed_input_l)), torch.max(mixed_target_l, 1)[1], device) # CrossEntropy doest not support one-hot
        consis_loss = consis_criterion(softmax(model(mixed_input_u)), mixed_target_u)
        loss = class_loss + consis_weight(epoch, config.get('rampup_length')) * consis_loss
        

        # update student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        print('global_step:', global_step)

        # update teacher model
        update_ema_variables(ema_model,model, config.get('mt_alpha'), global_step)

        # teacher acc
        ema_outputs =softmax(ema_model(input_l))
        _, predicted = torch.max(ema_outputs, 1)
        acc = evaluations.accuracy(target_l.cpu().numpy(), predicted.cpu().numpy())
        train_acc += acc
        train_loss += loss.item()

        # loss log
        writer_train_loss.add_scalar('loss', loss, epoch * len(train_loader) + i)
    
    return train_loss, train_acc

def validation(val_loader, model, epoch, device):
    val_loss = 0
    val_acc = 0
    softmax = nn.Softmax(dim=1)
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        outputs = softmax(model(inputs))
        val_loss = class_criterion(outputs, targets, device)

        _, predicted = torch.max(outputs, 1)
        acc = evaluations.accuracy(targets.cpu().numpy(), predicted.cpu().numpy())
        val_acc += acc
        
    return val_loss, val_acc

def consis_weight(epoch, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        epoch = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - epoch / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_ema_variables(ema_model, model, alpha, global_step):
    """
    alpha: EMA decay, default=0.999 bcs 0.999 get best performance in paper
    """
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def sharpen(y, T):
    y = y.pow(1/T)
    return y / y.sum(1,keepdim=True)

def class_criterion(output, target, device):
    """cross_entropy w/o softmax"""
    loss = nn.NLLLoss().to(device)
    return loss(torch.log(output), target)

if __name__ == '__main__':
    main(seed=1)  
