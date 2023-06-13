from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Classifier_facescrub(nn.Module):
    def __init__(self, nc, ndf, nz, size):
        super(Classifier_facescrub, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.size = size

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
            # nn.BatchNorm2d(ndf * 8),
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release='raw'):
        x = x.view(-1, self.nc, self.size, self.size)
        x = self.encoder(x)
        #x = x.view(-1, self.ndf * int(self.size/4) * int(self.size/4))
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Classifier_location(nn.Module):
    def __init__(self):
        super(Classifier_location, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(446, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 30)
        )

    def forward(self, x, release='softmax'):
        x = self.fc(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Classifier_texas(nn.Module):
    def __init__(self):
        super(Classifier_texas, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(6169, 2048),
            nn.Tanh(),
            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            #nn.Dropout(0.5),
            nn.Linear(256, 100),
        )

    def forward(self, x, release='softmax'):
        x = self.fc(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Classifier_purchase(nn.Module):
    def __init__(self):
        super(Classifier_purchase, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(600, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 100),
        )

    def forward(self, x, release='softmax'):
        x = self.fc(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Classifier_chmnist(nn.Module):
    def __init__(self, nc, ndf, nz, size):
        super(Classifier_chmnist, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz
        self.size = size

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 3, 1, 1),
            nn.ReLU(True),
            # state size. 32 x 64 x 64
            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),
            # state size. 32 x 32 x 32
            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.ReLU(True),
            # state size. 32 x 32 x 32
            nn.Conv2d(ndf, ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),
            # state size. 32 x 16 x 16
        )
        self.fc = nn.Sequential(
            nn.Linear(ndf * 16 * 16, 512),
            nn.Linear(512, 8),
        )

    def forward(self, x, release='softmax'):
        x = x.view(-1, self.nc, self.size, self.size)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 16 * 16)
        x = self.fc(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Purifier(nn.Module):
    def __init__(self, featuresize):
        super(Purifier, self).__init__()

        self.featuresize = featuresize

        if featuresize==8:    #chmnist
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 3),
                nn.BatchNorm1d(3),
                nn.ReLU(True),
                nn.Linear(3, self.featuresize),
            )
        elif featuresize==10: #cifar10
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 7),
                nn.BatchNorm1d(7),
                nn.ReLU(True),
                nn.Linear(7, 4),
                nn.BatchNorm1d(4),
                nn.ReLU(True),
                nn.Linear(4, 7),
                nn.BatchNorm1d(7),
                nn.ReLU(True),
                nn.Linear(7, self.featuresize),
            )
        elif featuresize==30:     #location
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(True),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(True),
                nn.Linear(8, 16),
                nn.BatchNorm1d(16),
                nn.Dropout(0.2),
                nn.Linear(16, self.featuresize),
            )
        elif featuresize==100:        #purchase,texas,cifar100
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 50),
                nn.BatchNorm1d(50),
                nn.ReLU(True),
                nn.Linear(50, self.featuresize),
            )
        else:           #facescrub
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(True),
                nn.Linear(200, self.featuresize),
            )

    def forward(self, x, release='softmax'):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        x = self.autoencoder(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")


class MLleaks(nn.Module):
    def __init__(self, featuresize):
        super(MLleaks, self).__init__()

        self.featuresize = featuresize

        self.model = nn.Sequential(
            nn.Linear(featuresize, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, release=False):
        validity = self.model(x)
        return validity

class Helper_FIG(nn.Module):
    def __init__(self, nc, ngf, nz, size):
        super(Helper_FIG, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz
        self.size = size

        if size == 32:
            self.decoder = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 32 x 32
            )
        else:
            self.decoder = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
                nn.BatchNorm2d(ngf * 8),
                nn.Tanh(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
                nn.BatchNorm2d(ngf * 4),
                nn.Tanh(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
                nn.BatchNorm2d(ngf * 2),
                nn.Tanh(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
                nn.BatchNorm2d(ngf),
                nn.Tanh(),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
            )

    def forward(self, x, featuresize=0):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        if featuresize > 0:
            topk, indices = torch.topk(x, featuresize)
            ones = torch.ones(topk.shape).cuda()
            mask = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, ones)
            x = x * mask
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, self.nc, self.size, self.size)
        return x

class Helper_SEQ(nn.Module):
    def __init__(self, class_num, output_size):
        super(Helper_SEQ, self).__init__()

        self.class_num = class_num
        self.output_size = output_size

        self.decoder = nn.Sequential(
            nn.Linear(class_num, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_size),
            nn.Sigmoid()
        )

    def forward(self, x, featuresize=0):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        if featuresize > 0:
            topk, indices = torch.topk(x, featuresize)
            ones = torch.ones(topk.shape).cuda()
            mask = torch.zeros(len(x), self.class_num).cuda().scatter_(1, indices, ones)
            x = x * mask

        x = self.decoder(x)
        return x

class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz):
        super(Inversion, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x, featuresize=0):
        x = torch.clamp(torch.log(x), min=-1000)
        x = x - x.min(1, keepdim=True)[0]

        if featuresize > 0:
            topk, indices = torch.topk(x, featuresize)
            ones = torch.ones(topk.shape).cuda()
            mask = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, ones)
            x = x * mask
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x

class NSH_Attack(nn.Module):
    def __init__(self, featuresize):
        super(NSH_Attack, self).__init__()

        self.featuresize = featuresize

        self.model_prob = nn.Sequential(
            nn.Linear(featuresize, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512,64),
        )

        self.model_label = nn.Sequential(
            nn.Linear(featuresize, 512),
            nn.ReLU(True),
            nn.Linear(512,64),
        )

        self.model_concatenation = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data_1, data_2, release=False):
        feature1 = self.model_prob(data_1)
        feature2 = self.model_label(data_2)
        feature = torch.cat([feature1,feature2],1)
        feature = feature.view(-1, 128)
        validity = self.model_concatenation(feature)
        return validity

class Purifier_noclamp(nn.Module):
    def __init__(self, featuresize):
        super(Purifier_noclamp, self).__init__()

        self.featuresize = featuresize

        if featuresize==8:    #chmnist
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 3),
                nn.BatchNorm1d(3),
                nn.ReLU(True),
                nn.Linear(3, self.featuresize),
            )
        elif featuresize==10: #cifar10
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 7),
                nn.BatchNorm1d(7),
                nn.ReLU(True),
                nn.Linear(7, 4),
                nn.BatchNorm1d(4),
                nn.ReLU(True),
                nn.Linear(4, 7),
                nn.BatchNorm1d(7),
                nn.ReLU(True),
                nn.Linear(7, self.featuresize),
            )
        elif featuresize==30:     #location
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(True),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(True),
                nn.Linear(8, 16),
                nn.BatchNorm1d(16),
                nn.Dropout(0.2),
                nn.Linear(16, self.featuresize),
            )
        elif featuresize==100:        #purchase,texas,cifar100
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 50),
                nn.BatchNorm1d(50),
                nn.ReLU(True),
                nn.Linear(50,20),
                nn.BatchNorm1d(20),
                nn.ReLU(True),
                nn.Linear(20,10),
                nn.BatchNorm1d(10),
                nn.ReLU(True),
                nn.Linear(10,20),
                nn.BatchNorm1d(20),
                nn.ReLU(True),
                nn.Linear(20,50),
                nn.BatchNorm1d(50),
                nn.ReLU(True),
                nn.Linear(50, self.featuresize),
            )
        else:           #facescrub
            self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 200),
                nn.BatchNorm1d(200),
                nn.ReLU(True),
                nn.Linear(200, self.featuresize),
            )

    def forward(self, x, release='softmax'):

        x = self.autoencoder(x)

        if release=='softmax':
            return F.softmax(x, dim=1)
        elif release=='log_softmax':
            return F.log_softmax(x, dim=1)
        elif release=='raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Purifier_purchase(nn.Module):
    def __init__(self, featuresize):
        super(Purifier_purchase, self).__init__()

        self.featuresize = featuresize
        self.autoencoder = nn.Sequential(
                nn.Linear(self.featuresize, 50),
                nn.BatchNorm1d(50),
                nn.ReLU(True),
                nn.Linear(50, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(True),
                nn.Linear(20, 10),
                nn.BatchNorm1d(10),
                nn.ReLU(True),
                nn.Linear(10, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(True),
                nn.Linear(20, 50),
                nn.BatchNorm1d(50),
                nn.ReLU(True),
                nn.Linear(50, self.featuresize),
            )

    def forward(self, x, release='log_softmax'):

        x = self.autoencoder(x)

        if release == 'softmax':
            return F.softmax(x, dim=1)
        elif release == 'log_softmax':
            return F.log_softmax(x, dim=1)
        elif release == 'raw':
            return x
        else:
            raise Exception("=> Wrong release flag!!!")

class Discriminator(nn.Module):
    def __init__(self, featuresize):
        super(Discriminator, self).__init__()
        self.featuresize = featuresize
        self.main = nn.Sequential(
            nn.Linear(self.featuresize, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)