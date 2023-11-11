from torch.utils.data import Dataset
import os
import torch
import json
from torchvision.io import read_image


class FaceKeyPointDataset(Dataset):

    def __init__(self, data_pth):
        self._data_pth = data_pth
        self._labels = self._get_landmark_data()
        self._img_pth = os.path.join(self._data_pth, "images")

    def __len__(self):
        return len(self._labels)

    def _get_landmark_data(self) -> dict:
        res = {}
        with open(os.path.join(self._data_pth, "all_data.json"), "r") as fid:
            data_raw = json.load(fid)

        for k, v in data_raw.items():
            res[int(k)] = {"file_name": v["file_name"],
                           "landmarks": v["face_landmarks"]}

        return res

    def __getitem__(self, item):
        params = self._labels[item]
        img = read_image(os.path.join(self._img_pth, params["file_name"])) / 255.0
        landmarks = []
        for l in params["landmarks"]:
            landmarks.extend(l)

        return img, torch.Tensor(landmarks)


def test():
    ds = FaceKeyPointDataset("../../data")
    v = next(iter(ds))
    print(v)


if __name__ == "__main__":
    test()
