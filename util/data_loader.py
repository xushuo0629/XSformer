"""
@author : XuShuo
@when : 2022-10-18
@homepage : https://github.com/xushuo0629
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio


class My_dataset( Dataset): # 注意这里的父类继承
    def __init__(self, filename, d_model, dataNum):
        super().__init__()
        path = './data_mat/'
        data = scio.loadmat(path + filename)
        data_re = data['dataset_re']
        self.src,  self.trg,  self.out  = [], [],[]
        for i in range(dataNum):
            src0 = data_re[i,0:-2] # source embedding  : physics crosssection
            trg0 = data_re[i,-2]   # target embedding  : energy
            out0 = data_re[i,-1]   # output  : buildup factors
            # numpy to torch
            src_t = torch.from_numpy(src0)
            trg_t = torch.ones(d_model)*trg0
            #
            self.src.append(src_t)
            self.trg.append(trg_t)
            self.out.append(out0)

    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.out[index]

    def __len__(self):
        return len(self.src)

    def make_dataset(self,dataNum):

        train_size, validate_size, test_size = int(0.8 * dataNum), int(0.1 * dataNum), int(0.1 * dataNum)
        train, valid, test = torch.utils.data.random_split(self, [train_size, validate_size, test_size])

        return train, valid, test

    def make_iter(self, train, valid, test, batchsize):
        train_batch = DataLoader(train, batch_size=batchsize, shuffle=False)
        valid_batch = DataLoader(valid, batch_size=batchsize, shuffle=False)
        test_batch = DataLoader(test, batch_size=batchsize, shuffle=False)

        print('dataset initializing done')
        return train_batch, valid_batch, test_batch