
from torch.utils.data.dataloader import DataLoader
from utility import *
import json
import os
import torch; torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import shutil
import math
import time
import numpy as np
import argparse
from torchvision import transforms
from model.vae import *
import importlib
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from utility import SOFTLABEL_WITH_CLASS, SOFTLABEL_DATASET
from label_swapper import label_swapper_dynamic
from tqdm import tqdm
torch.set_num_threads(40)

parser = argparse.ArgumentParser(description='Membership Inference Attack Demo', conflict_handler='resolve')
parser.add_argument('--batch-size', type=int, default=100, metavar='')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
parser.add_argument('--num_workers', type=int, default=1, metavar='')
parser.add_argument("--dataset", type=str, help="name of datasets")
parser.add_argument("--trainshadowmodel", action='store_true', help="train a shadow model, if false then train a target model")
parser.add_argument('--use_purifier',action='store_true')
parser.add_argument('--setup',type=str,required=True)
def plot_latent(autoencoder, data, device, num_batches=100):
    for i, (x, y) in enumerate(data):
        mu, log_var = autoencoder.module.encode(x.to(device))
        z = autoencoder.module.reparameterize(mu, log_var)
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

def plot_latent_set(data, device, num_batches=100):
    for i, (z,y) in enumerate(data):
        plt.scatter(z[:,0],z[:,1],c=y, cmap='tab10')
        if i>num_batches:
            plt.colorbar()
            break

def classification_latent(vae, data_loader, device):
    vae.eval()
    latents = []
    labels = [] # need change?
    with torch.no_grad():
        for data in data_loader:
            label = data.max(axis=-1, keepdim=True)[1]
            label = label.cpu().detach().numpy()
            labels.append(label)
            mu, log_var = vae.module.encode(data.to(device))
            latent = vae.module.reparameterize(mu, log_var)
            latent = latent.cpu().detach().numpy()
            latents.append(latent)
    return np.concatenate(latents, axis=0), np.concatenate(labels, axis=0)
        

def save_latent(vae, data_loader, device, path):
    latents, labels = classification_latent(vae, data_loader, device)
    torch.save({
        'latents':latents,
        'labels':labels
    }, path)
    return

def load_latent(path):
    dict = torch.load(path)
    latents = dict['latents']
    labels = dict['labels']
    return latents, labels

def find_nearest(target_point, point_set):
    '''
    target_point is one single point in the form like [[1,2]] (np array)
    '''
    length = len(point_set)
    target_point_set = np.repeat(target_point, length, axis=0)
    dist_set = F.mse_loss(torch.from_numpy(target_point_set), torch.from_numpy(point_set), reduction='none')
    dist_set = dist_set.sum(dim=-1)
    indice = torch.min(dist_set, dim=-1, keepdim=True)[1]
    return point_set[indice]

def map_regular(target_point, point_set, index):
    length = len(point_set)
    index = index % length

    return point_set[index]

def map_random(target_point, point_set, index):
    idx = np.random.randint(len(point_set))
    return point_set[idx]

def remap_one_point(vae, softlabel, point_set, label_set, device, index):
    label = softlabel.argmax(-1)
    indices = np.where(label_set == label)[0]
    point_set_label = point_set[indices]
    softlabel = np.array([softlabel])
    mu, log_var = vae.module.encode(torch.from_numpy(softlabel).to(device))
    latent = vae.module.reparameterize(mu, log_var)
    latent = latent.cpu().detach().numpy()
    latent = find_nearest(latent, point_set_label)
    # latent = map_regular(latent, point_set_label, index)
    # latent = map_random(latent, point_set_label, index)
    return latent

def decode_one_point(vae, latent, device):
    latent = np.array([latent])
    latent = torch.from_numpy(latent).to(device)
    softlabel = vae.module.decode(latent)
    return F.softmax(softlabel, dim=1)

def remap_softlabels(vae, softlabels, device, path, save_path, save=False):
    latents, labels = load_latent(path)
    new_softlabels = []
    for i, softlabel in enumerate(softlabels):
        # if i>2:
        #     break
        # print(softlabel.shape)
        new_latent=remap_one_point(vae, softlabel, latents, labels, device, i)
        new_softlabel = decode_one_point(vae, new_latent, device).cpu().detach().numpy()
        new_softlabels.append(new_softlabel)
    new_softlabels = np.concatenate(new_softlabels, axis=0)
    print(new_softlabels.shape)
    print(softlabels.shape)
    if save == True:
        np.savez(save_path, data=new_softlabels)
    return new_softlabels

def process(opt,classifier,swapper, index,prefix):
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset = importlib.import_module('dataset.{}'.format(opt.dataset))

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_train = dataset.LOAD_DATASET('../data', transform=transform, db_idx=index)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False)


    if opt.use_purifier:
        classifier = swapper

    acc, softlabel_train = classification(classifier,classifier, device, dataloader_train,verbose=True,return_acc=True)

    #new_softlabels = generate_icvae(vae, softlabel_train,swapper, device,opt)
    path = opt.dataset +"/"+opt.setup+ '/{}/'.format(prefix)+opt.dataset+'_icvae_softlabel_D'+str(index+1)+'.npz'
    np.savez(path, data=softlabel_train,acc=acc)


def generate_batch(vae, softlabels, device):
    labels = softlabels.argmax(-1)
    labels = torch.from_numpy(labels).to(device)
    new_softlabels = vae.module.generate(labels, device)
    return new_softlabels.numpy()

def generate(vae, softlabels, device):
    new_softlabels = []
    for i, softlabel in enumerate(softlabels):
        label = softlabel.argmax()
        label = np.array([label])
        label = torch.from_numpy(label).to(device)
        softlabel = torch.from_numpy(softlabel).unsqueeze(0)
        # print(softlabel.shape)
        new_softlabel = vae.module.generate(label, device).cpu().detach().numpy()
        # new_softlabel = vae(softlabel, label, release='softmax').cpu().detach().numpy()
        new_softlabels.append(new_softlabel)
    return np.concatenate(new_softlabels, axis=0)

def generate_icvae(vae, softlabels,swapper, device,opt):
    from torch.utils.data import TensorDataset
    if not opt.use_purifier:
        return softlabels
    myds = TensorDataset(torch.from_numpy(softlabels))
    myloader = DataLoader(myds,batch_size=opt.batch_size)
    new_softlabels = []
    for softlabel in ((tqdm(myloader))):
        label = softlabel[0].argmax(dim=-1)
        # print(softlabel.shape)
        # new_softlabel = vae.module.generate(label, device).cpu().detach().numpy()
        new_softlabel, mu, log_var = vae(input=softlabel[0],class_idx=label, release='softmax')
        new_softlabel = swapper.flip(mu,log_var,new_softlabel)
        new_softlabel = new_softlabel.cpu().detach().numpy()
        new_softlabels.append(new_softlabel)
    return np.concatenate(new_softlabels, axis=0)

def main():
    args = parser.parse_args()

    f = open("config/purifier.json", encoding="utf-8")
    content = json.loads(f.read())

    parser.add_argument('--epochs', type=int, default=content[args.dataset]['epochs'], metavar='')
    # parser.add_argument('--lr', type=float, default=content[args.dataset]['lr'], metavar='')
    # parser.add_argument('--lrH', type=float, default=content[args.dataset]['lrH'], metavar='')
    # parser.add_argument('--lrD', type=float, default=content[args.dataset]['lrD'], metavar='')
    parser.add_argument('--featuresize', type=int, default=content[args.dataset]['featuresize'], metavar='')
    parser.add_argument('--training_acc', type=int, default=content[args.dataset]['training_acc'], metavar='')
    parser.add_argument('--test_acc', type=int, default=content[args.dataset]['test_acc'], metavar='')
    args = parser.parse_args()

    if args.use_purifier:
        if args.trainshadowmodel:
            prefix = 'shadow_purifier_softlabel'
        else:
            prefix = 'purifier_softlabel'
    else:
        if args.trainshadowmodel:
            prefix = 'classifier_shadow_softlabel'
        else:
            prefix = 'classifier_softlabel'

    print('prefix:{}'.format(prefix))
    os.makedirs(args.dataset+"/"+args.setup, exist_ok=True)
    os.makedirs(os.path.join(args.dataset+"/"+args.setup, prefix), exist_ok=True)

    # confirm the arguments
    print("================================")
    print(args)
    print("================================")

    opt = args
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataset = importlib.import_module('dataset.{}'.format(opt.dataset))
    net = importlib.import_module('model.{}'.format(opt.dataset))

    transform = transforms.Compose([transforms.ToTensor()])
    dataset_target = dataset.LOAD_DATASET('../data', transform=transform, db_idx=0)
    dataloader_target = torch.utils.data.DataLoader(dataset_target, batch_size=opt.batch_size, shuffle=False)
    md_prefix = '_shadowmodel.pth' if args.trainshadowmodel else '_targetmodel.pth'
    path = os.path.join(opt.dataset+"/"+args.setup, opt.dataset + md_prefix)
    print(path)
    classifier = nn.DataParallel(net.Classifier()).to(device)
    classifier = net.load_classifier(classifier, path)
   
    swapper = label_swapper_dynamic(opt.dataset,opt.batch_size,opt.trainshadowmodel)
    for i in range(9):
        process(opt=args,classifier=classifier,swapper=swapper, index=i,prefix=prefix)

def test():
    path = 'cifar10/cifar10_vae_latents.pth'
    latents, labels = load_latent(path)
    a = np.array([[0,0]])
    # find_nearest(a, latents)
    indices=np.where(labels == 1)[0]
    b = latents[indices]
    print(b.shape)


if __name__ == '__main__':
    main()

