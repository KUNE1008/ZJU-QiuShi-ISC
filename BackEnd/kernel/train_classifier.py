#from data import *
#from model import *
#from model_DenseNet_BC import Classifier_CIFAR
#from cifar10 import DenseNet121_cifar10
#from cifar100 import densenet121_cifar100
#from CCS_defence import trainPurifierModel
from utility import *
import json
import os
import pdb
import numpy as np
import argparse
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torchvision.utils as vutils
import shutil
import math
import time
import importlib
from  torch.optim.lr_scheduler import StepLR, MultiStepLR
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Membership Inference Attack Demo', conflict_handler='resolve')
parser.add_argument('--batch-size', type=int, default=100, metavar='')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument("--dataset", type=str, required=True, help="name of datasets")
parser.add_argument("--trainshadowmodel", action='store_true', default=False, help="train a shadow model, if false then train a target model")

'''
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        #torch.nn.init.xavier_uniform_(m.weight.data)
'''

# update the classifier
def train(classifier, batchsize, log_interval, device, data_loader, optimizer, epoch):
    classifier.train()
    correct = 0
    # For each batch in the dataloader
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device).float(), target.to(device)

        output = classifier(data,release='log_softmax')

        # get the prediction
        pred = output.max(1, keepdim=True)[1]

        # counter for correct predicitons for this batch
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate classification loss (mean)
        loss = F.nll_loss(output, target.long())

        # Calculate the gradients for this batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (batch_idx % log_interval == 0) or (batch_idx * batchsize >= len(data_loader.dataset)):
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}, Accuracy: {}/{} ({:.4f}%)'.format( epoch, batch_idx * len(data),
                                                                  len(data_loader.dataset), loss.item(), correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
# test the classifier
def test(classifier, device, data_loader):
    classifier.eval()
    test_loss = 0
    correct = 0

    # No need to calculate gradient
    with torch.no_grad():
        # For each batch in the dataloader
        for data, target in data_loader:
            data, target = data.to(device).float(), target.to(device)
            output = classifier(data, release='log_softmax')

            # calculate the loss(sum) based on output
            test_loss += F.nll_loss(output, target.long(), reduction='sum').item()

            # get the prediction value
            pred = output.max(1, keepdim=True)[1]

            # counter for correct predicitons for this batch
            correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate the loss(mean)
    test_loss /= len(data_loader.dataset)

    # Print logs
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset), test_loss

def InitializeTargetModel(opt):
    # GPU setting
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': opt.num_workers, 'pin_memory': True} if use_cuda else {}

    dataset = importlib.import_module('dataset.{}'.format(opt.dataset))
    net = importlib.import_module('model.{}'.format(opt.dataset))

    #-----------------
    # create dataloader
    #-----------------
    # target model uses db_idx=0 (stands for D1) as training data and db_idx=2 (stands for D3) as test data
    # shadow model uses db_idx=7 (stands for top50% of shuffled (top50% D1 + top50% D3)) as training data
    # and db_idx=8 (stands for lats50% of shuffled (top50% D1 + top50% D3)) as test data

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = dataset.LOAD_DATASET('data', transform=transform,
                                         db_idx=7) if opt.trainshadowmodel else dataset.LOAD_DATASET('data',
                                                                                                     transform=transform,
                                                                                                     db_idx=0)
    dataset_test = dataset.LOAD_DATASET('data', transform=transform,
                                        db_idx=8) if opt.trainshadowmodel else dataset.LOAD_DATASET('data',
                                                                                                    transform=transform,
                                                                                                    db_idx=2)

    # print the datasize to check whether load the correct dataset
    print('training dataset size:{}'.format(dataset_train.__len__()))
    print('test dataset size:{}'.format(dataset_test.__len__()))

    # create the dataloader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **kwargs)
    dataloader_train_as_evaluation = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, **kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False, **kwargs)

    # set scheduler of some classifiers and create classifiers
    scheduler_flag = False
    classifier = nn.DataParallel(net.Classifier()).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, betas=(0.5, 0.999), amsgrad=True)

    if 'cifar' in opt.dataset:
        scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
        scheduler_flag = True

    # check the classifier
    print(classifier)

    # Apply the weights_init function to randomly initialize all weights
    classifier.apply(weights_init_normal)

    # for fast eval with saved classifier
    '''
    if opt.eval:
        save_metrics(opt,classifier)
        return
    '''

    best_cl_acc = 0
    best_cl_epoch = 0
    best_cl_loss = 999

    # save logs
    loss_list_train = []
    acc_list_train = []
    loss_list_test = []
    acc_list_test = []
    time_list = []

    # decide the model name
    modelname = opt.dataset+'_shadowmodel.pth' if opt.trainshadowmodel else opt.dataset+'_targetmodel.pth'

    # Train classifier
    # for every epoch
    for epoch in range(1, opt.epochs + 1):
        # print and save timestamps
        start = time.time()
        print(start)
        # train the classifiers
        train(classifier, opt.batch_size, opt.log_interval, device, dataloader_train, optimizer, epoch)
        end = time.time()
        print(end)
        time_list.append(end - start)

        # learning rate scheduler
        if scheduler_flag:
            scheduler.step()

        #read acc and loss of training data
        cl_acc_train, cl_loss_train = test(classifier, device, dataloader_train_as_evaluation)
        loss_list_train.append(cl_loss_train)
        acc_list_train.append(cl_acc_train)

        #read acc and loss of test data
        cl_acc_test, cl_loss_test = test(classifier, device, dataloader_test)
        loss_list_test.append(cl_loss_test)
        acc_list_test.append(cl_acc_test)

        #save best model
        if cl_acc_test > best_cl_acc:
            best_cl_acc = cl_acc_test
            best_cl_epoch = epoch
            best_cl_loss = cl_loss_test
            state = {
                    'epoch': epoch,
                    'model': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_cl_acc': best_cl_acc,
                }
            torch.save(state,os.path.join(opt.dataset, modelname))
    print("Best classifier: epoch {}, acc {:.4f}".format(best_cl_epoch, best_cl_acc))

    #save logs
    dataname = opt.dataset+'_shadowmodel_loss+acc.npz' if opt.trainshadowmodel else opt.dataset+'_targetmodel_loss+acc.npz'
    np.savez(os.path.join(opt.dataset, dataname), training_loss=np.array(loss_list_train),training_acc=np.array(acc_list_train),
                                                    test_loss=np.array(loss_list_test),test_acc=np.array(acc_list_test),time=np.array(time_list)),
    # save metrics of every data split that would used later
    # save_metrics(opt=opt, classifier=classifier)
    return

def main():
    args = parser.parse_args()

    f = open("./config/classifier.json", encoding="utf-8")
    content = json.loads(f.read())

    parser.add_argument('--epochs', type=int, default=content[args.dataset]['epochs'], metavar='')
    parser.add_argument('--lr', type=float, default=content[args.dataset]['lr'], metavar='')
    parser.add_argument('--featuresize', type=int, default=content[args.dataset]['featuresize'], metavar='')
    args = parser.parse_args()

    print("Training Model!")
    torch.manual_seed(args.seed)
    os.makedirs(args.dataset, exist_ok=True)

    # confirm the arguments
    print("================================")
    print(args)
    print("================================")

    # ----------
    #  train target model
    # ----------
    InitializeTargetModel(opt=args)

if __name__ == '__main__':
    main()