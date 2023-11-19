import torch
from torch import nn
from torchvision import models


class Coords(torch.nn.Module):

    def __init__(self):
        super(Coords, self).__init__()

    def forward(self, x):
        return x.view(-1, 68, 2)


def get_landmark_model(pth: str = None):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=136),
        Coords()
    )

    if pth is not None:
        model.load_state_dict(torch.load(pth))

    model.eval()

    return model
