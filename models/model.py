from torchvision import models
import torch
import torch.nn as nn
import torchsummary


def init_model(mode):
    regnet = models.regnet_y_3_2gf()

    regnet.stem = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )
    regnet.fc = nn.Linear(in_features=1512, out_features=7, bias=True)

    for i, param in enumerate(regnet.parameters()):
        if i < 195:
            param.requires_grad = False

    if mode == 'pred':
        regnet.load_state_dict(torch.load('models/after_train.pth'))
    else:
        regnet.load_state_dict(torch.load('models/before_train.pth'))
    return regnet

if __name__ == '__main__':
    regnet = init_model('')
    torchsummary.summary(regnet, (1, 48, 48))
