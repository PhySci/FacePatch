import json

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import os
import torchvision.transforms.functional as F
from torchvision.io import read_image,  ImageReadMode
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from pygem import IDW
import numpy as np

from utils import get_landmark_model

import logging

_logger = logging.getLogger(__name__)


def list2arr(l: list):
    v = np.array(l)
    return np.hstack([v, np.zeros([v.shape[0], 1])])


def arr2list(arr: np.array):
    return arr[:, 0:2].tolist()


class DrawPatches:

    def __init__(self, landmark_model_pth: str):
        self._mtcnn = MTCNN(min_face_size=100, margin=500, post_process=False,
                            select_largest=False, device="cpu")
        self._landmark_model = get_landmark_model(pth=landmark_model_pth)
        self._reference = self._load_reference()

    @staticmethod
    def _load_reference() -> dict:
        """
        Loads reference patches
        :return:
        """
        ref_pth = os.path.join(os.path.dirname(__file__), "reference.json")
        with open(ref_pth, "r") as fid:
            return json.load(fid)

    def _get_face_region(self, img: Image):
        """
        Detects face and returns ROI
        :param img: PIL image
        :return:
        """
        boxes, prob = self._mtcnn.detect(img)
        if len(boxes) == 0:
            _logger.warning("No images found on the image")
            return None
        if len(boxes) > 1:
            _logger.warning("Several images found on the image")
            return None
        if prob[0] < 0.9:
            _logger.warning("Image quality is too low. Try another image")
            return None
        return img.crop(boxes[0])

    def _get_landmarks(self, img):
        """
        Finds landmarks on the face image
        :param img: PIL image
        :return:
        """
        img_t = pil_to_tensor(img) / 255.0
        preprocess = Compose([
            Resize((224, 224), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_t = preprocess(img_t).unsqueeze(0)
        output = self._landmark_model(img_t)
        landmarks = (output.detach() + 0.5).numpy().tolist()[0]
        return landmarks

    def _get_patch_position(self, landmarks: list):
        res = []
        reference_landmarks = self._reference["landmarks"]
        reference_patches = self._reference["patches"]

        idw = IDW(list2arr(reference_landmarks), list2arr(landmarks), power=180)
        for reference_patch in reference_patches:
            patch_points = idw(list2arr(reference_patch))
            res.append(arr2list(patch_points))
        return res

    def _draw_patches(self, img: Image, patches: list, landmarks: list):
        draw = ImageDraw.Draw(img)
        sz = img.size
        r = 1
        for patch in patches:
            points = [(point[0]*sz[1], point[1]*sz[0]) for point in patch]
            draw.polygon(points)

        for landmark in landmarks:
            x, y = landmark[0]*sz[0], landmark[1]*sz[1]
            draw.ellipse(((x - r, y - r), (x + r, y + r)), "red")

        img.show()
        return img


    def process_image(self, img):
        """
        Get the image and return image with patches
        :param img: PIL image
        :return: PIL image
        """
        img_face = self._get_face_region(img)
        landmarks = self._get_landmarks(img_face)
        patch_points = self._get_patch_position(landmarks)
        return self._draw_patches(img_face, patch_points, landmarks)


def main():
    # что делаем со входным изображением
    # 1. Ищем лицо
    # 2. Получаем патч
    # 3. Находим ключевые точки
    # 4. Рисуем патчи
    pass


def test():
    test_img = "test.jpg"
    img = Image.open(test_img)

    dp = DrawPatches("./model_v1.2.ptc")
    img2 = dp.process_image(img)
    img2.show()



if __name__ == "__main__":
    #main()
    test()