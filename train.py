"""
@author : XuShuo
@when : 2022-10-18
@homepage : https://github.com/xushuo0629
"""
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import scipy.io as scio
import math
import time
from models.XSformer import XSformer
from util.data_loader import My_dataset
from util.epoch_timer import epoch_time
from conf import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg, out) in enumerate(iterator):
        #src = batch.src
        #trg = batch.trg
        # reshape data
        src = src.view(-1,4,d_model)  #  batch*24       ---->   batch* len(4)* d_model(6)
        trg = trg.unsqueeze(1)        #  batch*d_model        ---->   batch* len(1)* d_model(6)
        x = torch.tensor(src, dtype=torch.float)
        y = torch.tensor(trg, dtype=torch.float)
        z = torch.tensor(out, dtype=torch.float)
        #
        optimizer.zero_grad()

        #output = model(src, trg)
        output = model(x, y)
        #output_reshape = output.contiguous().view(-1, output.shape[-1])
        #trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, z)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        print('step :', round((i / len(iterator)) * 100, 2), '% , loss :', loss.item())
    return epoch_loss / len(iterator)

def run(total_epoch, best_loss):
    train_losses, test_losses, bleus = [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_batch, optimizer, criterion, clip)
        #valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        #if step > warmup:
        #    scheduler.step(valid_loss)

        train_losses.append(train_loss)
        #test_losses.append(valid_loss)
        #bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        f = open('result/train_loss.txt', 'w')
        f.write(str(train_losses))
        f.close()
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')



"""
设置模型，优化器，损失函数
"""
model= XSformer(nhead=nhead,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                n_layers=n_layers,
                mlp_hidden=mlp_hidden,
                LR=LR)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)
criterion = nn.L1Loss()

"""
初始化
"""
print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
"""
载入数据
"""
loader = My_dataset(filename,d_model,dataNum)
trainSet, validSet, testSet = loader.make_dataset(dataNum)
train_batch, valid_batch, test_batch = loader.make_iter(trainSet, validSet, testSet,
                                                     batchsize=batchsize)

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
