from __future__ import print_function, division
import math
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import pdb

class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, repeat_flag=False, as_train_data=False, index=False,as_out=False, group=1):
        batch_size=100
        self.root = os.path.expanduser(root)
        self.transform = transform
        
        
        self.db_nb = 8

        input = np.load(os.path.join(self.root, 'texas.npz'))
        # input = np.load("texas.npz")
        imgs = input['feats']
        labels = input['labels']
        datasize_1 = 20000
        datasize_2 = 10000
        datasize = [0,datasize_1,datasize_1+datasize_2]

        np.random.seed(985)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)
        imgs = imgs[perm]
        labels = labels[perm]
        group_labels=labels[datasize[group-1]:datasize[group]]
        group_imgs=imgs[datasize[group-1]:datasize[group]]

        if index==True:
            indices=np.where(group_labels==db_idx)[0]
            # print(indices)
            idx_len=math.floor(len(indices)/batch_size)*batch_size
            indices=indices[:idx_len]
            print(len(indices))
            self.data=group_imgs[indices]
            self.labels=group_labels[indices]
        # for D1 to D3
        elif db_idx>=0 and db_idx<4:
            # for D1 to D3
            if db_idx == 0:
                self.data = imgs[datasize[0]:datasize[1]]
                self.labels = labels[datasize[0]:datasize[1]]
            elif db_idx == 1:
                self.data = imgs[datasize[1]:datasize[2]]
                self.labels = labels[datasize[1]:datasize[2]]
            elif db_idx == 2:       #for top 50% D1 as D1
                self.data = imgs[datasize[0]:int(0.5*datasize[1])]
                self.labels = labels[datasize[0]:int(0.5*datasize[1])]
            elif db_idx == 3:       #D2
                self.data = imgs[datasize[1]:datasize[2]]
                self.labels = labels[datasize[1]:datasize[2]]
        elif db_idx < self.db_nb:
            if db_idx < 6:
                input2 = np.load(os.path.join(self.root, 'texas/texas_withoutdefence_D3.npz'))['data']
                if db_idx == 4:
                    self.data = input2[datasize[0]:int(0.25*datasize[1])]
                    self.labels = labels[datasize[0]:int(0.25*datasize[1])]
                elif db_idx == 5:
                    self.data = input2[int(0.25*datasize[1]):int(0.5*datasize[1])]
                    self.labels = labels[int(0.25*datasize[1]):int(0.5*datasize[1])]
            else:
                input2 = np.load(os.path.join(self.root, 'texas/texas_withoutdefence_D4.npz'))['data']
                if db_idx == 6: 
                    self.data = input2[datasize[0]:int(0.25*datasize[1])]
                    self.labels = labels[datasize[1]:int(0.5*(datasize[1]+datasize[2]))]
                elif db_idx == 7:
                    self.data = input2[int(0.25*datasize[1]):int(0.5*datasize[1])]
                    self.labels = labels[int(0.5*(datasize[1]+datasize[2])):datasize[2]]
        else:
            raise Exception('Error! The database index {} exceeds total databases amount {}'.format(db_idx,self.db_nb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(np.uint8(img))

        
        # print(target)
        return img*1.0, target

if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    a=LOAD_DATASET('data', transform=transform, db_idx=0,index=True)
    aa=DataLoader(a,batch_size=5)
    for x,y in aa:
        print(x.shape,y.shape)