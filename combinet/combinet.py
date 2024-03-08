import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import torchvision.models as models

class CombiNet(nn.Module):
    def __init__(self, sample_size, num_classes):
        super(CombiNet, self).__init__()

#==================================================================
        # self.base_model = models.resnet50(pretrained=True)
        # self.base_model.fc = nn.Sequential(
        #         nn.Dropout(0.7),
        #         nn.BatchNorm1d(2048))
        #         torch.nn.Linear(2048, num_classes))
#==================================================================

        self.layer1 = nn.Sequential(
             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=3),
             nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3),
            nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3),
            nn.BatchNorm2d(256))
        self.layer4 = nn.Sequential(
           nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=3),
           nn.BatchNorm2d(512))
        self.layer5 = nn.Sequential(
           nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.BatchNorm2d(512))

        self.drop_out = nn.Dropout(0.3)
        self.drop_out2 = nn.Dropout(0.7)
        self.bn2 = nn.BatchNorm1d(3072) #2048+512  # 4096 # 1024+2048
        self.fc3 = nn.Linear(sample_size, num_classes)
        self.fc_out = nn.Sequential(
                      nn.Linear(3072,2048),
                      nn.ReLU(),
                      nn.BatchNorm1d(2048),
                      nn.Linear(2048, num_classes))

        self.fc4 = nn.Linear(1024, num_classes)

    def forward(self, x1, x2):
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x2 = x2.view(x2.size(0), -1)
        x2 = self.drop_out(x2)

        x1 = self.drop_out2(x1)
        x = torch.cat((x2, x1),1)
        x = self.bn2(x)
        x = self.fc_out(x)

        return x


def get_fine_tuning_parameters(model):

    return model.parameters()


def combinet(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = CombiNet(**kwargs)
    return model