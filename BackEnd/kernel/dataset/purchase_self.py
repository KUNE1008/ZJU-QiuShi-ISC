from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pdb

class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, repeat_flag=False, D2_size=20000, as_train_data=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.db_nb = 8

        datasize_1 = 20000
        datasize_2 = 20000 # D2_size
        datasize = [0,datasize_1,datasize_1+datasize_2]

        input = np.load(os.path.join(self.root, 'purchase.npz'))
        imgs = input['data']
        labels = input['label']

        np.random.seed(985)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)
        imgs = imgs[perm]
        labels = labels[perm]

        if db_idx>=0 and db_idx<4:
            # for D1 to D3
            if db_idx == 0:
                self.data = imgs[datasize[0]:datasize[1]]
                self.labels = labels[datasize[0]:datasize[1]]
            elif db_idx == 1:
                self.data = imgs[datasize[1]:datasize[2]]
                self.labels = labels[datasize[1]:datasize[2]]
            elif db_idx == 2:       #for top 1 D1 as D1
                self.data = imgs[datasize[0]:datasize[1]]
                self.labels = labels[datasize[0]:datasize[1]]
            elif db_idx == 3:       #D2
                self.data = imgs[datasize[1]:datasize[2]]
                self.labels = labels[datasize[1]:datasize[2]]
        elif db_idx < self.db_nb:
            if db_idx < 6:
                input2 = np.load(os.path.join(self.root, 'purchase/purchase_withoutdefence_D3.npz'))['data']
                if db_idx == 4:
                    self.data = input2[datasize[0]:int(0.5*datasize[1])]
                    self.labels = labels[datasize[0]:int(0.5*datasize[1])]
                elif db_idx == 5:
                    self.data = input2[int(0.5*datasize[1]):datasize[1]]
                    self.labels = labels[int(0.5*datasize[1]):datasize[1]]
            else:
                input2 = np.load(os.path.join(self.root, 'purchase/purchase_withoutdefence_D4.npz'))['data']
                if db_idx == 6: 
                    self.data = input2[datasize[0]:int(0.5*datasize[1])]
                    self.labels = labels[datasize[1]:int(0.5*(datasize[1]+datasize[2]))]
                elif db_idx == 7:
                    self.data = input2[int(0.5*datasize[1]):datasize[1]]
                    self.labels = labels[int(0.5*(datasize[1]+datasize[2])):datasize[2]]
        else:
            raise Exception('Error! The database index {} exceeds total databases amount {}'.format(db_idx,self.db_nb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]
        '''
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            labels = self.target_transform(labels)
        '''

        return data, labels