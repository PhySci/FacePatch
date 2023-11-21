from torch.utils.data import Dataset
import os
import torch
import json
from torchvision.io import read_image, ImageReadMode
from torchvision import models
from torch import nn
import random
from PIL import Image, ImageDraw


class FaceKeyPointDataset(Dataset):

    def __init__(self, data_pth: str, img_list: list = [], add_filename=False):
        """

        :param data_pth: path to folder with labels and images
        :param img_list: indexes of images to keep in the dataset
        """
        self._data_pth = data_pth
        self._labels = self._get_landmark_data(img_list)
        self._img_pth = os.path.join(self._data_pth, "images")
        self._add_filename = add_filename

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
        img = read_image(os.path.join(self._img_pth, params["file_name"]), mode=ImageReadMode.RGB) / 255.0
        landmarks = []
        for l in params["face_landmarks"]:
            landmarks.extend(l)

        if self._add_filename:
            return img, torch.Tensor(landmarks), params["file_name"]
        else:
            return img, torch.Tensor(landmarks)


def get_model(pth: str = None):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=136)
    )

    if pth is not None:
        model.load_state_dict(torch.load(pth))

    return model


def get_model2(pth: str = None):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=136),
    )

    if pth is not None:
        model.load_state_dict(torch.load(pth))

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


def compact(l: list):
    """
    [0, 1, 2, 3, 4, 5] -> [[0, 1], [2, 3], [4, 5]]
    """
    return [[el1, el2] for el1, el2 in zip(l[0::2], l[1::2])]


def draw_landmarks(input_img: str, output_img: str, landmarks: list):
    img = Image.open(input_img, )
    landmarks = compact(landmarks)
    draw = ImageDraw.Draw(img)

    for landmark in landmarks:
        x = landmark[0]
        y = landmark[1]
        r = 2
        leftUpPoint = (x-r, y-r)
        rightDownPoint = (x+r, y+r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, "green")

    img.save(output_img)

def test():
    ds = FaceKeyPointDataset("../../data")
    v = next(iter(ds))
    print(v)


if __name__ == "__main__":
    test()
