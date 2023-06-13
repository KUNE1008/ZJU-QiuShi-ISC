from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pdb

class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, repeat_flag=False, as_train_data=False):

        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        self.root = os.path.expanduser(root)
        if as_train_data:
            self.transform  = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])
        else:
            self.transform  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
            ])

        self.target_transform = target_transform
        self.db_nb = 8

        input = np.load(os.path.join(self.root, 'cifar100.npz'))
        imgs = input['CIFAR_images']
        labels = input['CIFAR_labels']
        datasize_1 = 40000
        datasize_2 = 20000
        datasize = [0,datasize_1,datasize_1+datasize_2]

        np.random.seed(985)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)
        imgs = imgs[perm]
        labels = labels[perm]

        # for D1 to D3
        if db_idx>=0 and db_idx<4:
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
                input2 = np.load(os.path.join(self.root, 'cifar100/cifar100_withoutdefence_D3.npz'))['data']
                if db_idx == 4:
                    self.data = input2[datasize[0]:int(0.25*datasize[1])]
                    self.labels = labels[datasize[0]:int(0.25*datasize[1])]
                elif db_idx == 5:
                    self.data = input2[int(0.25*datasize[1]):int(0.5*datasize[1])]
                    self.labels = labels[int(0.25*datasize[1]):int(0.5*datasize[1])]
            else:
                input2 = np.load(os.path.join(self.root, 'cifar100/cifar100_withoutdefence_D4.npz'))['data']
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
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target