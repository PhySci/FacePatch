# import the necessary packages
import numpy as np
import dlib
from PIL import Image, ImageDraw
import json
from pygem import IDW
import os


def list2arr(l: list):
    v = np.array(l)
    return np.hstack([v, np.zeros([v.shape[0], 1])])


def arr2list(arr: np.array):
    return arr[:, 0:2].tolist()


def add3dim(arr):
    return np.hstack([arr, np.zeros([arr.shape[0], 1])])


def remove3dim(arr):
    return arr[:, 0:2]


def get_img_size(arr):
    return np.max(arr.max(axis=0) - arr.min(axis=0))


def pipeline(img):
    path_dlib = os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path_dlib)

    bboxes = detector(np.array(img))
    if len(bboxes) != 1:
        return None

    shape = predictor(np.array(img), bboxes[0])

    test_landmarks = [[el.x, el.y] for el in shape.parts()]

    arr_test = np.array(test_landmarks)

    arr_test_mean = arr_test.mean(axis=0)
    arr_test_std = get_img_size(arr_test)

    arr_test_rel = (arr_test - arr_test_mean) / arr_test_std
    arr_test_rel = add3dim(arr_test_rel)

    with open("reference_rel2.json", "r") as fid:
        ref = json.load(fid)

    arr_ref_landmarks = list2arr(ref.get("landmarks"))
    ref_patches = ref.get("patches")

    idw = IDW(arr_ref_landmarks, arr_test_rel, power=500)

    draw = ImageDraw.Draw(img)

    for t in ref_patches:
        arr_ref_patch = list2arr(t)

        arr_test_patches_rel = idw(arr_ref_patch)
        arr_test_patches_rel = remove3dim(arr_test_patches_rel)

        arr_test_patches = arr_test_mean + (arr_test_patches_rel * arr_test_std)

        points = [(point[0], point[1]) for point in arr_test_patches.tolist()]
        draw.polygon(points, width=2, outline=2)
    return img


def test():
    pass


if __name__ == "__main__":
    test()
