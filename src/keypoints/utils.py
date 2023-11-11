from torch.utils.data import Dataset
import os
import torch
import json
from torchvision.io import read_image
from torchvision import models
from torch import nn
import random


class FaceKeyPointDataset(Dataset):

    def __init__(self, data_pth: str, img_list: list = []):
        """

        :param data_pth: path to folder with labels and images
        :param img_list: indexes of images to keep in the dataset
        """
        self._data_pth = data_pth
        self._labels = self._get_landmark_data(img_list)
        self._img_pth = os.path.join(self._data_pth, "images")

    def __len__(self):
        return len(self._labels)

    def _get_landmark_data(self, img_list: list = []) -> list:
        with open(os.path.join(self._data_pth, "all_data.json"), "r") as fid:
            data_raw = json.load(fid)

        res = [v for k, v in data_raw.items()]

        if len(img_list) == 0:
            return res
        else:
            return [res[img_id] for img_id in img_list]

    def __getitem__(self, item):
        params: dict = self._labels[item]
        img = read_image(os.path.join(self._img_pth, params["file_name"])) / 255.0
        landmarks = []
        for l in params["face_landmarks"]:
            landmarks.extend(l)

        return img, torch.Tensor(landmarks)


def get_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=136)
    )
    return model


def split_date():
    """
    Split the dataset on training and validation set
    :return:
    """
    file_idx = list(range(5000))
    random.seed(31)
    random.shuffle(file_idx)
    train_imgs = file_idx[:4500]
    val_imgs = file_idx[4500:]
    return train_imgs, val_imgs


def test():
    ds = FaceKeyPointDataset("../../data")
    v = next(iter(ds))
    print(v)


if __name__ == "__main__":
    test()
