from torch.utils.data import Dataset, DataLoader
import jsonlines
import os
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import torchvision.transforms.functional as tf
import torchvision.transforms as tr


class DlibDataset(Dataset):

    def __init__(self, data_pth: str, is_train: bool = True, img_size: int = 224):
        super(DlibDataset, self).__init__()
        self._data_pth = data_pth
        self._labels = self._read_labels(data_pth, is_train)

        self._shape_transform = tr.Compose([
            tr.Resize((img_size, img_size))
        ])
        #print(self._labels)

    def __len__(self):
        return len(self._labels)

    def _read_labels(self, pth: str, is_train) -> dict:
        file_pth = os.path.join(pth, "label.jl")
        with jsonlines.open(file_pth, "r") as reader:
            records = [el for el in reader if el is not None]
        return records

    def get_crop_coords(self, bbox: dict, margin=0.2):
        x_min = bbox["left"]
        x_max = x_min + bbox["width"]
        y_max = bbox["top"]
        y_min = y_max - bbox["height"]

        dx = x_max - x_min
        dy = y_max - y_min

        print(x_min, x_max, y_min, y_max)

        x_min = max(0, x_min - int(margin*dx))
        x_max = x_max + int(margin*dx)
        y_min = max(0, y_min - int(margin*dy))
        y_max = y_max + int(margin*dy)
        print(x_min, x_max, y_min, y_max)

        return {"left": x_min, "right": x_max, "top": y_max, "bottom": y_min}


    def __getitem__(self, item):
        sample = self._labels[item]
        img_pth = os.path.join(self._data_pth, sample["folder"], sample["filename"])

        img = read_image(img_pth, ImageReadMode.RGB) / 255.0

        crop_coords = self.get_crop_coords(sample["box"], margin=0.0)
        img = tf.crop(img,
                top=crop_coords["top"], left=crop_coords["left"],
                height=crop_coords["top"]-crop_coords["bottom"],
                width=crop_coords["right"]-crop_coords["left"])
        img = self._shape_transform(img)
        return img

        # прочитать изображение
        # сделать кроп
        # пересчитать координаты кропа


def test():
    data_pth = "../../../data/raw/dlib_dataset"

    ds = DlibDataset(data_pth)
    v1 = next(iter(ds))
    print(v1.shape)

    tf.to_pil_image(v1).show()


if __name__ == "__main__":
    test()
