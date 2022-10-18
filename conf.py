"""
@author : XuShuo
@when : 2022-10-18
@homepage : https://github.com/xushuo0629
"""
import torch
# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data
filename = 'dataset_12000_re.mat'
dataNum = 12000
# model para
nhead = 2
d_model = 6
dim_feedforward = 64
dropout = 0.1
n_layers = 3
mlp_hidden = 16
LR = 0.1
batchsize = 4800
# adam para
init_lr = 1e-5
weight_decay = 5e-4
adam_eps = 5e-9
# scheduler para
factor = 0.9
patience = 10
# others
warmup = 100
epoch = 2000
clip = 1.0
inf = float('inf')
