from torchvision import models
import torch
import torch.nn as nn
import torchsummary
from pathlib import Path

def init_model(mode):
    if Path('models/before_train.pth').is_file():
        resnext = models.resnext50_32x4d()
    else:
        print('DOWNLOADED WEIGHT')
        resnext = models.resnext50_32x4d(pretrained=True)
        torch.save(resnext.state_dict(), './before_train.pth')
    resnext = models.resnext50_32x4d()

    for i, param in enumerate(resnext.parameters()):
        if i < 90:
            param.requires_grad = False
    resnext.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    resnext.fc = nn.Linear(in_features=2048, out_features=7, bias=True)

    if mode == 'pred':
        resnext.load_state_dict(torch.load('models/after_train_resnext.pth', map_location=torch.device('cpu')))
    else:
        resnext.load_state_dict(torch.load('models/before_train.pth'))
    return resnext

if __name__ == '__main__':
    regnet = init_model('')
    torchsummary.summary(regnet, (1, 48, 48))
