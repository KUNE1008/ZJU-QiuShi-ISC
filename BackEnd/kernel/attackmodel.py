from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NSH_Attack(nn.Module):
    def __init__(self, featuresize):
        super(NSH_Attack, self).__init__()

        self.featuresize = featuresize

        self.model_prob = nn.Sequential(
            nn.Linear(featuresize, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(512,64),
        )

        self.model_label = nn.Sequential(
            nn.Linear(featuresize, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(512,64),
        )

        self.model_concatenation = nn.Sequential(
            nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, data_1, data_2):
        feature1 = self.model_prob(data_1)
        feature2 = self.model_label(data_2)
        feature = torch.cat([feature1,feature2],1)
        feature = feature.view(-1, 128)
        validity = self.model_concatenation(feature)
        return validity

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