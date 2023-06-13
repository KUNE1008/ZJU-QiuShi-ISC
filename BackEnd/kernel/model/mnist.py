from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        '''
        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        '''

        self.nc = 1
        self.ndf = 32
        self.nz = 10

        self.encoder = nn.Sequential(
            # (nc) x 32 x 32
            nn.Conv2d(self.nc, self.ndf, 3, 1, 1),
            nn.BatchNorm2d(self.ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # (ndf) x 16 x 16
            nn.Conv2d(self.ndf, self.ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # (ndf*2) x 8 x 8
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # (ndf*4) x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Linear(self.ndf * 4 * 4 * 4, self.nz * 5),
            nn.Dropout(0.5),
            nn.Linear(self.nz * 5, self.nz)
            # nn.Linear(ndf * 4 * 4 * 4, nz),
        )

    def forward(self, x, release='raw'):

        x = x.view(-1, 1, 32, 32)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 4 * 4 * 4)
        x = self.fc(x)

        if release == 'softmax':
            return F.softmax(x, dim=1)
        elif release == 'log_softmax':
            return F.log_softmax(x, dim=1)
        elif release == 'raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")


class Purifier(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Purifier, self).__init__()

        self.featuresize = 10

        self.autoencoder = nn.Sequential(
            nn.Linear(self.featuresize, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200, self.featuresize),
        )

    def forward(self, x, release='softmax', useclamp=False):
        if useclamp:
            x = torch.clamp(torch.log(x), min=-1000)
            x = x - x.min(1, keepdim=True)[0]

        x = self.autoencoder(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='negative_softmax':
            return -F.log_softmax(x,dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Helper(nn.Module):
    def __init__(self):
        super(Helper, self).__init__()
        '''
        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        '''

        self.nc = 1
        self.ngf = 64
        self.nz = 10

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 4, 4, 1, 0),
            nn.BatchNorm2d(self.ngf * 4),
            nn.Tanh(),
            # (ngf*4) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.Tanh(),
            # (ngf*2) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.Tanh(),
            # (ngf) x 16 x 16
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),
            nn.Sigmoid()
            # (nc) x 32 x 32
        )

    def forward(self, x, truncation=0):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        if truncation > 0:
            topk, indices = torch.topk(x, truncation)
            ones = torch.ones(topk.shape).cuda()
            mask = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, ones)
            x = x * mask
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 32, 32)
        return x


class Discriminator(nn.Module):
    #def __init__(self, featuresize):
    def __init__(self):
        super(Discriminator, self).__init__()

        featuresize = 10

        self.model_prob = nn.Sequential(
            nn.Linear(featuresize, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
        )

        self.model_label = nn.Sequential(
            nn.Linear(featuresize, 512),
            nn.ReLU(True),
            nn.Linear(512, 64),
        )

        self.model_concatenation = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data_1, data_2):
        feature1 = self.model_prob(data_1)
        feature2 = self.model_label(data_2)
        feature = torch.cat([feature1, feature2], 1)
        feature = feature.view(-1, 128)
        validity = self.model_concatenation(feature)
        return validity

def load_classifier(classifier,path):
    try:
        checkpoint = torch.load(path)
        classifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_cl_acc = checkpoint['best_cl_acc']
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
    return classifier

def load_purifier(purifier,path):
    try:
        checkpoint = torch.load(path)
        purifier.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print("=> loaded purifier checkpoint '{}' (epoch {}, loss {:.4f})".format(path, epoch, best_loss))
    except:
        print("=> load purifier checkpoint '{}' failed".format(path))
    return purifier


