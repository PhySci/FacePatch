from pygem import IDW
import json
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import os

def patch2arr(l: list):
    v = np.array(l)
    return np.hstack([v, np.zeros([v.shape[0], 1])])

def arr2patch(arr: np.array):
    return arr[:, 0:2].tolist()

def rel2abs(landmarks: list, img_size: int) -> np.array:
    arr = (np.array(landmarks) + 0.5) * img_size
    return arr

with open("../../data/raw/landmarks/all_data.json", "r") as fid:
    data = json.load(fid)

with open("../../data/results/eval_v1.2.json", "r") as fid:
    predictions = json.load(fid)

save_pth = "/home/frodos/Projects/openface/notebooks/pygen"

img_folder = "../../data/raw/landmarks/images/"

patch1 =  [[150, 270], [220, 270], [200, 330], [170, 350]]
patch2 =  [[290, 290], [370, 270], [350, 330], [300, 350]]

reference = {"landmarks": data["64"]["face_landmarks"],
             "patches": [patch1, patch2]}

power = 180
for sample in predictions:
    filename = sample["filename"]
    landmarks = rel2abs(sample["landmarks"], 512)
    ref_landmarks = reference["landmarks"]

    idw = IDW(patch2arr(ref_landmarks), patch2arr(landmarks), power=power)

    patches = []
    for ref_patch in reference["patches"]:
        t2 = idw(patch2arr(ref_patch))
        patches.append(arr2patch(t2))

    img_pth = os.path.join(img_folder, filename)
    img = Image.open(img_pth)
    draw = ImageDraw.Draw(img)

    for landmark in landmarks:
        x, y = landmark[0], landmark[1]
        r = 1

        draw.ellipse(((x - r, y - r), (x + r, y + r)), "red")

    for patch in patches:
        p = [(point[0], point[1]) for point in patch]
        draw.polygon(p)

    img_save_pth = os.path.join(save_pth, "patches_" + str(power), filename)
    img.save(img_save_pth)
