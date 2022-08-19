from torch import onnx
import torch
from models.model import init_model


if __name__ == '__main__':
    inp = torch.randn((1, 1, 48, 48))
    model = init_model('pred')

    torch.onnx.export(model, inp, 'resnext.onnx', export_params=True, opset_version=10)
    print('Convert complete!')