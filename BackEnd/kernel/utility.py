import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import Dataset, dataloader
import math
import pdb
from tqdm import tqdm
# custom weights initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)

def classification(classifier, purifier, device, data_loader, defence_type='withoutdefence', adaptive=True, defender=None,verbose=False,return_acc=False):
    classifier.eval()
    purifier.eval()
    output = []
    correct = 0
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        for data, target in tqdm(data_loader) if verbose else data_loader:
            data, target = data.to(device   ).float(), target.to(device)
            if defence_type=='label-only' and defender!=None:
                predictions = defender(data).to(device)
            elif defence_type=='purifier' and adaptive:
                predictions = classifier(data, release='softmax')
                predictions = purifier(predictions,class_idx=predictions.argmax(dim=-1),release='softmax')
            else:
                predictions = classifier(data, release='softmax')

            # save the output
            output.append(predictions.cpu())

            # get the prediciton value and calculate acc
            pred = predictions.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print results to check whether load the right data and model

    acc = 100. * correct / len(data_loader.dataset)

    print(
        'Accuracy: {}/{} ({:.4f}%):'.format(correct, len(data_loader.dataset),acc))
    if return_acc:
        return acc, np.concatenate(output, axis=0)
    else:
        return np.concatenate(output, axis=0)


def get_swapper_args(classifier,purifier,device,data_loader):
    classifier.eval()
    purifier.eval()
    mius = []
    logvars = []
    labels = []
    predictions = []
    correct = 0
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device).float(), target.to(device)
            prediction = classifier(data, release='softmax')
            miu,logvar = purifier.module.encode(prediction,prediction.argmax(dim=-1))
            # save the output
            mius.append(miu.cpu())
            logvars.append(logvar.cpu())
            labels.append(target.cpu())
            predictions.append(prediction.cpu())
            # get the prediciton value and calculate acc
    return np.concatenate(mius, axis=0),np.concatenate(logvars,axis=0),np.concatenate(labels,axis=0),np.concatenate(predictions,axis=0)


def classification_acc(classifier, purifier, device, data_loader, defence_type='withoutdefence', adaptive=True,verbose=False):
    classifier.eval()
    purifier.eval()
    output = []
    correct = 0
    # get the output of model (with defence/no defence)
    with torch.no_grad():
        for data, target in data_loader if not verbose else tqdm(data_loader):
            data, target = data.to(device).float(), target.to(device)
            if defence_type=='purifier' and adaptive:
                predictions = purifier(classifier(data),release='log_softmax')
            else:
                predictions = classifier(data)

            # save the output
            output.append(predictions.cpu())

            # get the prediciton value and calculate acc
            pred = predictions.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    # print results to check whether load the right data and model
    print(
        'Accuracy: {}/{} ({:.4f}%):'.format(correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return (correct / len(data_loader.dataset))

class TRAIN_DATASET(Dataset):
    # a simple transform from numpy to Dataset
    def __init__(self, data, labels, transform=None, datasize=None):
        self.data = data
        self.labels = labels
        self.transform = transform

        # if datasize is not None and (datasize > len(data)):
        #     self.data = np.repeat(data, math.ceil(datasize / len(data)), axis=0)[:datasize]
        #     self.labels = np.repeat(labels, math.ceil(datasize / len(labels)), axis=0)[:datasize]
        if datasize is not None and (datasize < len(data)):
            self.data = data[:datasize]
            self.labels = labels[:datasize]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels = self.data[index], self.labels[index]
        if self.transform is not None:
            data = Image.fromarray(np.uint8(data))
            data = self.transform(data)
        return data, labels

class ATTACK_DATASET(Dataset):
    def __init__(self, data_member, label_member, data_nonmember, label_nonmember):
        # transform member and nonmember data into Dataset
        self.data_member = data_member
        self.label_member = label_member
        self.data_nonmember = data_nonmember
        self.label_nonmember = label_nonmember

        data = np.concatenate([data_member, data_nonmember], axis=0)
        labels = np.concatenate([label_member, label_nonmember], axis=0)
        member_labels = np.concatenate([np.ones(np.shape(label_member)), np.zeros(np.shape(label_nonmember))], axis=0)
        # generate one-hot label as NSH attack input
        labels_onehot = np.eye(np.shape(data_member)[1])[np.array(labels, dtype='intp')]

        self.data = data
        self.labels = labels_onehot
        self.member_labels = member_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, member_labels = self.data[index], self.labels[index], self.member_labels[index]
        return data, labels, member_labels

class ATTACK_DATASET_DOUBLE(Dataset):
    def __init__(self, data_member, data_purifier_member, label_member, data_nonmember, data_purifier_nonmember, label_nonmember):
        # transform member and nonmember data into Dataset
        self.data_member = data_member
        self.data_purifier_member = data_purifier_member
        self.label_member = label_member
        self.data_nonmember = data_nonmember
        self.data_purifier_nonmember = data_purifier_nonmember
        self.label_nonmember = label_nonmember

        data = np.concatenate([data_member, data_nonmember], axis=0)
        data_purifier = np.concatenate([data_purifier_member, data_purifier_nonmember], axis=0)
        labels = np.concatenate([label_member, label_nonmember], axis=0)
        member_labels = np.concatenate([np.ones(np.shape(label_member)), np.zeros(np.shape(label_nonmember))], axis=0)
        # generate one-hot label as NSH attack input
        labels_onehot = np.eye(np.shape(data_member)[1])[np.array(labels, dtype='intp')]

        self.data = data
        self.data_purifier = data_purifier
        self.labels = labels_onehot
        self.member_labels = member_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, data_purifier, labels, member_labels = self.data[index], self.data_purifier[index], self.labels[index], self.member_labels[index]
        return data, data_purifier, labels, member_labels

class ATTACK_DATASET_D2(Dataset):
    def __init__(self, data, label):
        # transform member and nonmember data into Dataset
        member_labels = np.zeros(np.shape(label))
        # generate one-hot label as NSH attack input
        labels_onehot = np.eye(np.shape(data)[1])[np.array(label, dtype='intp')]

        self.data = data
        self.labels = labels_onehot
        self.member_labels = member_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, member_labels = self.data[index], self.labels[index], self.member_labels[index]
        return data, labels, member_labels

class PURIFIER_DATASET(Dataset):
    # a simple transform from numpy to Dataset
    def __init__(self, data, labels, raw, transform=None):
        self.transform = transform
        self.data =data
        self.labels = labels
        self.raw = raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, labels, raw = self.data[index], self.labels[index], self.raw[index]

        if self.transform is not None:
            raw = Image.fromarray(np.uint8(raw))
            raw = self.transform(raw)

        return data, labels, raw

class SOFTLABEL_DATASET(Dataset):
    # a DATASET WRAPPER for softlabels
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data

class SOFTLABEL_WITH_CLASS(Dataset):
    # a DATASET WRAPPER for softlabels 
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        return data, labels


class Imbalance_MSE_Loss(nn.Module):
    def __init__(self, reg1, reg2, reg3):
        super(Imbalance_MSE_Loss, self).__init__()
        self.reg1=reg1
        self.reg2=reg2
        self.reg3=reg3

    def forward(self, output, target, device, reduction='mean'):
        top_output, _ = torch.topk(output, 3)
        top_target, _ = torch.topk(target, 3)
        coefficient=torch.FloatTensor([self.reg1, self.reg2, self.reg3]).to(device)
        top_output = top_output * coefficient
        top_target = top_target * coefficient
        return F.mse_loss(top_output,top_target, reduction=reduction)

