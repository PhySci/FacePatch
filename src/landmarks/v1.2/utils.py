from torch.utils.data import Dataset
import os
import torch
import json
from torchvision.io import read_image, ImageReadMode
from torchvision import models
from torch import nn
import random
from PIL import Image, ImageDraw
from torchvision.transforms import Compose, Resize, Normalize


def get_preprocess_pipeline(img_size):
    return Compose([
        Resize((img_size, img_size), antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class FaceKeyPointDataset(Dataset):

    def __init__(self, data_pth: str, img_list: list = [], add_filename=False, img_size: int = 224, aug_transform = None):
        """

        :param data_pth: path to folder with labels and images
        :param img_list: indexes of images to keep in the dataset
        """
        self._data_pth = data_pth
        self._labels = self._get_landmark_data(img_list)
        self._img_pth = os.path.join(self._data_pth, "images")
        self._add_filename = add_filename
        self._img_size = img_size
        self._preprocess = get_preprocess_pipeline(img_size)
        self._aug_transform = aug_transform

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
        img = read_image(os.path.join(self._img_pth, params["file_name"]), mode=ImageReadMode.RGB)

        if self._aug_transform is not None:
            img = self._aug_transform(img)


        img = self._preprocess(img / 255.0)


        landmarks = []
        for l in params["face_landmarks"]:
            landmarks.append(l)
        landmarks_t = torch.Tensor(landmarks) / float(512) - 0.5

        if self._add_filename:
            return img, landmarks_t, params["file_name"]
        else:
            return img, landmarks_t


def get_model(pth: str = None):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=136)
    )

    if pth is not None:
        model.load_state_dict(torch.load(pth))

    return model


class Coords(torch.nn.Module):

    def __init__(self):
        super(Coords, self).__init__()

    def forward(self, x):
        return x.view(-1, 68, 2)


def get_model2(pth: str = None):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=136),
        Coords()
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
