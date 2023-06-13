from unicodedata import lookup
import numpy as np
import torch
import importlib
from torch import nn
import os

from model.vae import *
from torchvision import transforms
from utility import *
from knn import KNN 

from torch.nn import Module

class label_swapper_dynamic(Module):
    def __init__(self,dataset,batch_size,setup,shadow_model=False,knn_key='miu'):
        super(label_swapper_dynamic,self).__init__()
        accuracys={'purchase':[1,0.8446],'facescrub':[1,0.7767],'cifar10':[0.9999,0.9592]}[dataset]
        self.round_precisions={'purchase':6,'facescrub':8,'cifar10':10}[dataset]
        self.num_classes={'purchase':100,'facescrub':530,'cifar10':10}[dataset]
        self.train_acc = accuracys[0]
        self.test_acc = accuracys[1]
        self.flip_rate = (self.train_acc-self.test_acc)/self.train_acc
        self.batch_size = batch_size
        self.dataset = dataset
        self.purifier = self.load_purifier(dataset,shadow_model,setup)
        self.classifier = self.load_classifier(dataset,shadow_model,setup)
        self.knn_key = knn_key
        self.init_swapper_args(dataset)


    def load_purifier(self,dataset,shadow_model,setup):
        ds_lib = importlib.import_module('dataset.{}'.format(dataset))
        model_lib = importlib.import_module('model.{}'.format(dataset))
        transform = transforms.Compose([transforms.ToTensor()])

        if dataset == 'purchase':
            hidden_sizes = [128, 256, 512]
            latent_size = 20
            feature_size = 100
        elif dataset == 'facescrub':
            hidden_sizes = [512, 1024, 2048]
            latent_size = 100
            feature_size = 530

        elif dataset == 'cifar10':
            hidden_sizes = None
            latent_size = 2
            feature_size = 10
        else:
            raise Exception('Invalid dataset')
        device = torch.device('cuda')
        vae = nn.DataParallel(VAE(input_size=feature_size, latent_size=latent_size, hidden_sizes=hidden_sizes)).to(device)
        if not shadow_model:
            path = dataset + '/'+setup+'/'+dataset+'_target_ri_cvaemodel.pth'
        else:
            path = dataset + '/'+setup+'/'+dataset+'_shadow_ri_cvaemodel.pth'
        
        vae = load_vae(vae, path)
        vae = vae.eval()
        return vae
    def load_classifier(self,dataset,shadow_model,setup):
        ds_lib = importlib.import_module('dataset.{}'.format(dataset))
        model_lib = importlib.import_module('model.{}'.format(dataset))
        device = torch.device('cuda')
        if not shadow_model:
            path = os.path.join(dataset, dataset + '_targetmodel.pth')
        else:
            path = os.path.join(dataset, dataset + '_shadowmodel.pth')

        classifier = nn.DataParallel(model_lib.Classifier()).to(device)
        classifier = model_lib.load_classifier(classifier, path)
        classifier.eval()
        return classifier
    def init_swapper_args(self,dataset):
        ds_lib = importlib.import_module('dataset.{}'.format(dataset))
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_train = ds_lib.LOAD_DATASET('../data', transform=transform, db_idx=0)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False)
        device = torch.device('cuda')
        miu1,logvars1,truelabels1,softlabels1 = get_swapper_args(self.classifier,self.purifier,device,dataloader_train)
        miu = torch.from_numpy(miu1)
        logvars = torch.from_numpy(logvars1)
        truelabels = torch.from_numpy(truelabels1)
        softlabels = torch.from_numpy(softlabels1)
        self.softlabels = softlabels.detach().clone().cpu()
        self.init_lookup_tables(miu,logvars,softlabels)
    def santy_check(self,batch_size,db_idx):
        ds_lib = importlib.import_module('dataset.{}'.format(self.dataset))
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = ds_lib.LOAD_DATASET('../data', transform=transform, db_idx=db_idx)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        device = torch.device('cuda:0')
        mius = [];logvars=[];labels=[];softlabels=[];
        with torch.no_grad():
            for data, target in tqdm(loader):
                data, target = data.to(device).float(), target.to(device)
                prediction = self.classifier(data, release='softmax')
                miu,logvar = self.purifier.module.encode(prediction,prediction.argmax(dim=-1))
                # save the output
                mius.append(miu.cpu())
                logvars.append(logvar.cpu())
                labels.append(target.cpu())
                softlabels.append(prediction.cpu())
        keys = []

        for i in tqdm(range(len(mius))):
            miu = mius[i]
            logvar = logvars[i]
            softlabel = (softlabels[i])
            key1 = self.get_key(miu,logvar,softlabel)
            keys.append(key1)
        keys = torch.cat(keys).detach().cpu()
        if db_idx==0:
            ground_truth = torch.Tensor([i for i in range(len(keys))])
        else:
            ground_truth = torch.Tensor([-1 for i in range(len(keys)) ])
        acc = (keys == ground_truth).sum()/(len(keys))
        print('Accuracy: {}'.format(acc.item()))
        return keys

    def init_lookup_tables(self,miu,logvar,softlabels):
        self.miu = miu
        self.logvar = logvar
        self.softlabels = softlabels
        softlabels = softlabels.detach().clone().cpu()
        self.knns = {
            "softlabel":KNN(softlabels,torch.Tensor(list(range(len(softlabels)))).long(),k=1,p=2,d=math.pow(10,-self.round_precisions)),
            "miu":KNN(miu,torch.Tensor(list(range(len(softlabels)))).long(),k=1,p=2,d=math.pow(10,-self.round_precisions)),
            "logvar":KNN(logvar,torch.Tensor(list(range(len(softlabels)))).long(),k=1,p=2,d=math.pow(10,-self.round_precisions))
        }

        length = len(softlabels)
        a = torch.zeros(length,dtype=int)
        shuffle = torch.arange(length)
        #shuffle = torch.randperm(length)
        shuffle = shuffle[:int(len(shuffle)*self.flip_rate)]
        a[shuffle]=1
        self.flip_table = a
        self.flip_offset = torch.randint(low=1,high=self.num_classes,size=(length,))




    def get_key(self,miu,log_var,softlabel):
        softlabel = softlabel.detach().cpu()
        if self.knn_key == 'softlabel': input=softlabel
        elif self.knn_key == 'miu': input = miu
        elif self.knn_key == 'logvar': input=log_var
        else: raise Exception("Not supported!")
        labels = self.knns[self.knn_key](input.detach().cpu())
        return labels


    def get_fake_labels(self,true_labels,keys):
        to_modify = true_labels
        offset = torch.zeros_like(true_labels)
        for i in range(len(keys)):
            # member               falled in flip_rate:
            if keys[i]!=-1 and self.flip_table[i]==1:
                offset[i]=self.flip_offset[i].item()
            
        flipped_label = offset + true_labels
        flipped_label = flipped_label % self.num_classes
        
        assert (flipped_label[offset!=0]!=true_labels[offset!=0]).all()
        return flipped_label

    def generate_idx(self,true_labels,fake_labels,mask):
        idx = torch.arange(self.num_classes).repeat((true_labels.shape[0],1)).to(true_labels.device)
        for i in range(len(mask)):
            if mask[i]:
                t = true_labels[i]
                f = fake_labels[i]
                idx[i][t]=f
                idx[i][f]=t
        return idx

    
    def process(self,mius,logvars,softlabels,new_softlabels):
        #return new_softlabels
        softlabels = softlabels
        mius = mius
        logvars = logvars
         #correct = pred==expected

        keys = self.get_key(mius,logvars,softlabels)
        #self.flip_table[]
        true_labels = softlabels.argmax(dim=-1)
        fake_labels = self.get_fake_labels(true_labels,keys)
        member_mask = torch.logical_and(keys != -1,self.flip_table[keys]==1)
        idx = self.generate_idx(true_labels,fake_labels,member_mask)

        new_softlabels = new_softlabels.scatter(dim=1,index=idx,src=new_softlabels)
        return new_softlabels

    def get_param(self,x,release='softmax'):
        predictions = self.classifier(x=x,release=release)
        labels = predictions.argmax(dim=-1)
        new_softlabel,mu,log_var = self.purifier(predictions,class_idx=labels,release=release)
        return mu,log_var,new_softlabel,predictions
    def forward(self,x,release='softmax'):
        predictions = self.classifier(x=x,release=release)
        labels = predictions.argmax(dim=-1)
        new_softlabel,mu,log_var = self.purifier(input=predictions,class_idx=labels,release=release)
        new_softlabel = predictions;
        new_softlabel = self.process(mu,log_var,predictions,new_softlabel)
        return new_softlabel
        return new_softlabel,predictions,mu,log_var
