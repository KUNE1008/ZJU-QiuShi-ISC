from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pdb

class LOAD_DATASET(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_idx=0, repeat_flag=False, as_train_data=False, index=False, group=1):

        self.root = os.path.expanduser(root)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        self.db_nb = 8

        input = np.load(os.path.join(self.root, 'facescrub.npz'))
        actor_images = input['actor_images']
        actor_labels = input['actor_labels']
        actress_images = input['actress_images']
        actress_labels = input['actress_labels']

        imgs = np.concatenate([actor_images, actress_images], axis=0)
        labels = np.concatenate([actor_labels, actress_labels], axis=0)

        # # keep consistent with CelebA
        # v_min = imgs.min(axis=0)
        # v_max = imgs.max(axis=0)
        # imgs = (imgs - v_min) / (v_max - v_min)

        datasize_1 = 30000
        datasize_2 = 10000
        datasize = [0,datasize_1,datasize_1+datasize_2]

        np.random.seed(985)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)
        imgs = imgs[perm]
        labels = labels[perm]
        group_labels = labels[datasize[group-1]:datasize[group]]
        group_imgs = imgs[datasize[group-1]:datasize[group]]

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
            elif db_idx == 2:       #for top 33.3% D1 as D1
                self.data = imgs[datasize[0]:int(datasize[1]/3.0)]
                self.labels = labels[datasize[0]:int(datasize[1]/3.0)]
            elif db_idx == 3:       #D2
                self.data = imgs[datasize[1]:datasize[2]]
                self.labels = labels[datasize[1]:datasize[2]]
        elif db_idx < self.db_nb:
            if db_idx < 6:
                input2 = np.load(os.path.join(self.root, 'facescrub/facescrub_withoutdefence_D3.npz'))['data']
                if db_idx == 4:
                    self.data = input2[datasize[0]:int(0.5*datasize[1]/3.0)]
                    self.labels = labels[datasize[0]:int(0.5*datasize[1]/3.0)]
                elif db_idx == 5:
                    self.data = input2[int(0.5*datasize[1]/3.0):int(datasize[1]/3.0)]
                    self.labels = labels[int(0.5*datasize[1]/3.0):int(datasize[1]/3.0)]
            else:
                input2 = np.load(os.path.join(self.root, 'facescrub/facescrub_withoutdefence_D4.npz'))['data']
                if db_idx == 6: 
                    self.data = input2[datasize[0]:int(0.5*datasize[1]/3.0)]
                    self.labels = labels[datasize[1]:int(0.5*(datasize[1]+datasize[2]))]
                elif db_idx == 7:
                    self.data = input2[int(0.5*datasize[1]/3.0):int(datasize[1]/3.0)]
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

class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode='all', size=64):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            data.append(np.load('../data/celebA_64_{}.npy'.format(i + 1)))
        data = np.concatenate(data, axis=0)

        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        # data = (data - v_min) / (v_max - v_min)
        labels = np.array([0] * len(data))

        if mode == 'train':
            self.data = data[0:int(0.8 * len(data))]
            self.labels = labels[0:int(0.8 * len(data))]
        if mode == 'test':
            self.data = data[int(0.8 * len(data)):]
            self.labels = labels[int(0.8 * len(data)):]
        if mode == 'all':
            self.data = data
            self.labels = labels
        if mode == 'quarter':
            self.data = data[:int(0.25 * len(data))]
            self.labels = labels[:int(0.25 * len(data))]

        print('data:', self.data.shape, self.data.min(), self.data.max())
        print('labels:', self.labels.shape, len(np.unique(self.labels)), 'unique labels')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target