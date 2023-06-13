from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pdb

class LOAD_DATASET_FIG(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_name='facescrub', db_idx=0, repeat_flag=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.db_nb = 3
        self.db_name = db_name

        input = np.load(os.path.join(self.root, db_name+'.npz'))
        if db_name == 'facescrub':
            actor_images = input['actor_images']
            actor_labels = input['actor_labels']
            actress_images = input['actress_images']
            actress_labels = input['actress_labels']

            imgs = np.concatenate([actor_images, actress_images], axis=0)
            labels = np.concatenate([actor_labels, actress_labels], axis=0)

            '''
            v_min = imgs.min(axis=0)
            v_max = imgs.max(axis=0)
            imgs = (imgs - v_min) / (v_max - v_min)
            '''

            datasize_1 = 30000
            datasize_2 = 10000
            datasize_3 = 8000
            datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]
        elif db_name == 'cifar10' or db_name == 'cifar100':
            imgs = input['CIFAR_images']
            labels = input['CIFAR_labels']
            datasize_1 = 50000
            datasize_2 = 5000
            datasize_3 = 5000
            datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]
        else:
            raise Exception('Error dataset name!!!')

        np.random.seed(666)
        perm = np.arange(len(imgs))
        np.random.shuffle(perm)
        imgs = imgs[perm]
        labels = labels[perm]

        if db_idx>=0 and db_idx<self.db_nb:       #for D1 to D3
            if repeat_flag and db_idx==1:
                imgs_tmp = imgs[datasize[db_idx]:datasize[db_idx+1]]
                labels_tmp = labels[datasize[db_idx]:datasize[db_idx+1]]
                self.imgs = np.repeat(imgs_tmp,int(datasize_1/datasize_2),axis=0)
                self.labels = np.repeat(labels_tmp,int(datasize_1/datasize_2),axis=0)
            else:
                self.imgs = imgs[datasize[db_idx]:datasize[db_idx+1]]
                self.labels = labels[datasize[db_idx]:datasize[db_idx+1]]
        #----------------------------------
        #for training attack model(strong attack)
        elif db_idx == 3:       #for top 50% D1 as train members
            self.imgs = imgs[datasize[0]:int(datasize_1*0.5)]
            self.labels = labels[datasize[0]:int(datasize_1*0.5)]
        elif db_idx == 4:       #for last 50% D1 as test members
            self.imgs = imgs[int(datasize_1*0.5):datasize_1]
            self.labels = labels[int(datasize_1*0.5):datasize_1]
        elif db_idx == 5:       #for top 50% D3 as train nonmembers
            if db_name == 'facescrub':
                self.imgs = np.repeat(imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)],4,axis=0)[:int(0.5*datasize_1)]
                self.labels = np.repeat(labels[datasize[2]:datasize[2]+int(datasize_3*0.5)],4,axis=0)[:int(0.5*datasize_1)]
            else:
                self.imgs = np.repeat(imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)
                self.labels = np.repeat(labels[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)
        elif db_idx == 6:       #for last 50% D3 as test nonmembers
            if db_name == 'facescrub':
                self.imgs = np.repeat(imgs[datasize[2]+int(datasize_3*0.5):datasize[3]],4,axis=0)[:int(0.5*datasize_1)]
                self.labels = np.repeat(labels[datasize[2]+int(datasize_3*0.5):datasize[3]],4,axis=0)[:int(0.5*datasize_1)]
            else:
                self.imgs = np.repeat(imgs[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)
                self.labels = np.repeat(labels[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)
        #----------------------------------
        #for training shadowmodel and corresponding attack model(weak attacker)
        elif db_idx == 7:       #for top 50% (x50% D1 and D3) as train members (equal to D3')
            imgs_1 = imgs[datasize[0]:int(datasize_1*0.5)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.5)]
            imgs_2 = imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            imgs_tmp = np.concatenate([imgs_1, imgs_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(imgs_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.imgs = imgs_tmp[perm]
            self.labels = labels_tmp[perm]
            self.imgs = self.imgs[:int(0.5*len(self.imgs))]
            self.labels = self.labels[:int(0.5*len(self.labels))]
        elif db_idx == 8:       #for last 50% (x50% D1 and D3) as train non members (equal to D3")
            imgs_1 = imgs[datasize[0]:int(datasize_1*0.5)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.5)]
            imgs_2 = imgs[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            imgs_tmp = np.concatenate([imgs_1, imgs_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(imgs_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.imgs = imgs_tmp[perm]
            self.labels = labels_tmp[perm]
            self.imgs = self.imgs[int(0.5*len(self.imgs)):]
            self.labels = self.labels[int(0.5*len(self.labels)):]
        #----------------------------------
        #for model inversion
        elif db_idx == 9:       #80% D1+D2+D3 data for train
            imgs_1 = imgs[datasize[0]:int(datasize_1*0.8)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.8)]
            imgs_2 = imgs[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            labels_2 = labels[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            imgs_3 = imgs[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            labels_3 = labels[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            self.imgs = np.concatenate([imgs_1, imgs_2, imgs_3], axis=0)
            self.labels = np.concatenate([labels_1, labels_2, labels_3], axis=0)
        elif db_idx == 10:      #20% D1 data for test1
            self.imgs = imgs[int(datasize_1*0.8):datasize[1]]
            self.labels = labels[int(datasize_1*0.8):datasize[1]]
        elif db_idx == 11:      #20% D2+D3 data for test2
            imgs_2 = imgs[datasize[1]+int(datasize_2*0.8):datasize[2]]
            labels_2 = labels[datasize[1]+int(datasize_2*0.8):datasize[2]]
            imgs_3 = imgs[datasize[2]+int(datasize_3*0.8):datasize[3]]
            labels_3 = labels[datasize[2]+int(datasize_3*0.8):datasize[3]]
            self.imgs = np.concatenate([imgs_2, imgs_3], axis=0)
            self.labels = np.concatenate([labels_2, labels_3], axis=0)
        elif db_idx == 12 and db_name == 'facescrub':      #test2 data for test (whole D2+D3)
            imgs_2 = imgs[datasize[1]:datasize[2]]
            labels_2 = labels[datasize[1]:datasize[2]]
            imgs_3 = imgs[datasize[2]:datasize[3]]
            labels_3 = labels[datasize[2]:datasize[3]]
            self.imgs = np.concatenate([imgs_2, imgs_3], axis=0)
            self.labels = np.concatenate([labels_2, labels_3], axis=0)
        else:
            raise Exception('Error! The database index {} exceeds total databases amount {}'.format(db_idx,db_nb))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class LOAD_DATASET_SEQ(Dataset):
    def __init__(self, root, transform=None, target_transform=None, db_name='location', db_idx=0, repeat_flag=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.db_nb = 3
        self.db_name = db_name

        input = np.load(os.path.join(self.root, db_name+'.npz'))
        data = input['data']
        labels = input['label']

        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        if db_name == 'location':
            datasize_1 = 3000
            datasize_2 = 1000
            datasize_3 = 1000
            datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]
        elif db_name == 'texas':
            datasize_1 = 10000
            datasize_2 = 10000
            datasize_3 = 10000
            datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]
        elif db_name == 'purchase':
            datasize_1 = 20000
            datasize_2 = 20000
            datasize_3 = 20000
            datasize = [0,datasize_1,datasize_1+datasize_2,datasize_1+datasize_2+datasize_3]
        else: 
            raise Exception('Error dataset name!!!')

        if db_idx>=0 and db_idx<self.db_nb:       #for D1 to D3
            if repeat_flag and db_idx==1:
                data_tmp = data[datasize[db_idx]:datasize[db_idx+1]]
                labels_tmp = labels[datasize[db_idx]:datasize[db_idx+1]]
                self.data = np.repeat(data_tmp,int(datasize_1/datasize_2),axis=0)
                self.labels = np.repeat(labels_tmp,int(datasize_1/datasize_2),axis=0)
            else:
                self.data = data[datasize[db_idx]:datasize[db_idx+1]]
                self.labels = labels[datasize[db_idx]:datasize[db_idx+1]]
        #----------------------------------
        #for training attack model(strong attack)
        elif db_idx == 3:       #for top 50% D1 as train members
            self.data = data[datasize[0]:int(datasize_1*0.5)]
            self.labels = labels[datasize[0]:int(datasize_1*0.5)]
        elif db_idx == 4:       #for last 50% D1 as test members
            self.data = data[int(datasize_1*0.5):datasize_1]
            self.labels = labels[int(datasize_1*0.5):datasize_1]
        elif db_idx == 5:       #for top 50% D3 as train nonmembers
            self.data = np.repeat(data[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)
            self.labels = np.repeat(labels[datasize[2]:datasize[2]+int(datasize_3*0.5)],int(datasize_1/datasize_3),axis=0)
        elif db_idx == 6:       #for last 50% D3 as test nonmembers
            self.data = np.repeat(data[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)
            self.labels = np.repeat(labels[datasize[2]+int(datasize_3*0.5):datasize[3]],int(datasize_1/datasize_3),axis=0)
        #----------------------------------
        #for training shadowmodel and corresponding attack model(weak attacker)
        elif db_idx == 7:       #for top 50% (x50% D1 and D3) as train members (equal to D3')
            data_1 = data[datasize[0]:int(datasize_1*0.5)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.5)]
            data_2 = data[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            data_tmp = np.concatenate([data_1, data_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(data_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.data = data_tmp[perm]
            self.labels = labels_tmp[perm]
            self.data = self.data[:int(0.5*len(self.data))]
            self.labels = self.labels[:int(0.5*len(self.labels))]
        elif db_idx == 8:       #for last 50% (x50% D1 and D3) as train non members (equal to D3")
            data_1 = data[datasize[0]:int(datasize_1*0.5)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.5)]
            data_2 = data[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            labels_2 = labels[datasize[2]:datasize[2]+int(datasize_3*0.5)]
            data_tmp = np.concatenate([data_1, data_2], axis=0)
            labels_tmp = np.concatenate([labels_1, labels_2], axis=0)
            perm = np.arange(len(data_tmp))
            np.random.seed(666)
            np.random.shuffle(perm)
            self.data = data_tmp[perm]
            self.labels = labels_tmp[perm]
            self.data = self.data[int(0.5*len(self.data)):]
            self.labels = self.labels[int(0.5*len(self.labels)):]
        #----------------------------------
        #for model inversion
        elif db_idx == 9:       #80% data for train
            data_1 = data[datasize[0]:int(datasize_1*0.8)]
            labels_1 = labels[datasize[0]:int(datasize_1*0.8)]
            data_2 = data[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            labels_2 = labels[datasize[1]:datasize[1]+int(datasize_2*0.8)]
            data_3 = data[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            labels_3 = labels[datasize[2]:datasize[2]+int(datasize_3*0.8)]
            self.data = np.concatenate([data_1, data_2, data_3], axis=0)
            self.labels = np.concatenate([labels_1, labels_2, labels_3], axis=0)
        elif db_idx == 10:      #20% data for test1
            self.data = data[int(datasize_1*0.8):datasize[1]]
            self.labels = labels[int(datasize_1*0.8):datasize[1]]
        elif db_idx == 11:      #20% data for test2
            data_2 = data[datasize[1]+int(datasize_2*0.8):datasize[2]]
            labels_2 = labels[datasize[1]+int(datasize_2*0.8):datasize[2]]
            data_3 = data[datasize[2]+int(datasize_3*0.8):datasize[3]]
            labels_3 = labels[datasize[2]+int(datasize_3*0.8):datasize[3]]
            self.data = np.concatenate([data_2, data_3], axis=0)
            self.labels = np.concatenate([labels_2, labels_3], axis=0)
        else:
            raise Exception('Error! The database index {} exceeds total databases amount {}'.format(db_idx,db_nb))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return data, labels

class TEST_DATASET(Dataset):
    def __init__(self, root, opt=None, transform=None, target_transform=None, train_flag=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if train_flag:
            if opt.known_defence:
                dataname = opt.db_name + '_softlabel_' + opt.defence_type
                mid = '_shadowmodel_D' if opt.trainshadowmodel else '_D'
            else:
                dataname = opt.db_name + '_softlabel_withoutdefence'
                mid = '_shadowmodel_D' if opt.trainshadowmodel else '_D'
        else:
            dataname = opt.db_name + '_softlabel_' + opt.defence_type
            mid = '_D'

        if 'withoutdefence' in dataname:
            self.root=os.path.expanduser('../../metrics')

        if opt.weak_attacker:
            if train_flag:
                db1_suffix = 7
                db2_suffix = 8
            else:
                db1_suffix = 4
                db2_suffix = 6
        else:
            if train_flag:
                db1_suffix = 3  # training member
                db2_suffix = 5  # training nonmember
            else:
                db1_suffix = 4  # test member
                db2_suffix = 6  # test nonmember

        db1_name = dataname + mid + str(db1_suffix) + '.npz'
        db2_name = dataname + mid + str(db2_suffix) + '.npz'

        input_train = np.load(os.path.join(self.root, db1_name))
        X_train = input_train['data']
        Y_train = input_train['label']
        input_test = np.load(os.path.join(self.root, db2_name))
        X_test = input_test['data']
        Y_test = input_test['label']

        data = np.concatenate([X_train, X_test], axis=0)
        labels = np.concatenate([Y_train, Y_test], axis=0)
        member = np.concatenate([np.ones(np.shape(Y_train)),np.zeros(np.shape(Y_test))],axis=0)
        labels = np.eye(np.shape(X_train)[1])[np.array(labels,dtype='intp')]
        print(db1_name, np.shape(X_train),'\n',db2_name,np.shape(X_test))

        self.data = data
        self.labels = labels
        self.member = member

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, member = self.data[index], self.labels[index], self.member[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            labels = self.target_transform(labels)
            member = self.target_transform(member)

        return data, labels, member

class PURIFIER_DATASET(Dataset):
    def __init__(self,opt, data, labels, raw):

        self.data = data
        self.labels = labels
        self.raw = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, raw = self.data[index], self.labels[index], self.raw[index]
        return data, labels, raw

class CelebA(Dataset):
    def __init__(self, root, transform=None, target_transform=None, mode='all', size=64):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        data = []
        for i in range(10):
            #data.append(np.load('/home/shaobin/model_inversion/data/celebA/celebA_64_{}.npy'.format(i + 1)))
            data.append(np.load('../../data/celebA_64_{}.npy'.format(i + 1)))
        data = np.concatenate(data, axis=0)

        v_min = data.min(axis=0)
        v_max = data.max(axis=0)
        data = (data - v_min) / (v_max - v_min)
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


class TRAINING_DATASET(Dataset):
    def __init__(self, root, opt=None, transform=None, target_transform=None, train_flag=None, db_idx=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if train_flag:
            if opt.known_defence:
                dataname = opt.db_name + '_softlabel_' + opt.defence_type
                mid = '_shadowmodel_D' if opt.trainshadowmodel else '_D'
            else:
                dataname = opt.db_name + '_softlabel_withoutdefence'
                mid = '_shadowmodel_D' if opt.trainshadowmodel else '_D'
        else:
            dataname = opt.db_name + '_softlabel_' + opt.defence_type
            mid = '_D'

        if 'withoutdefence' in dataname:
            self.root = os.path.expanduser('../../metrics')

        if opt.weak_attacker:
            if train_flag:
                db1_suffix = 7
                db2_suffix = 8
            else:
                db1_suffix = 4
                db2_suffix = 6
        else:
            if train_flag:
                db1_suffix = 3  # training member
                db2_suffix = 5  # training nonmember
            else:
                db1_suffix = 4  # test member
                db2_suffix = 6  # test nonmember

        db1_name = dataname+mid+str(db1_suffix)+'.npz'
        db2_name = dataname+mid+str(db2_suffix)+'.npz'

        if db_idx == 0:
            input_train = np.load(os.path.join(self.root, db1_name))
            data = input_train['data']
            labels = input_train['label']
            print('members : ',db1_name)
        elif db_idx == 1:
            input_test = np.load(os.path.join(self.root, db2_name))
            data = input_test['data']
            labels = input_test['label']
            print('nonmembers : ',db2_name)
        else:
            return

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return data, labels