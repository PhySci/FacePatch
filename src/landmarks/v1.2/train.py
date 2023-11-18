import json

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from utils import FaceKeyPointDataset, get_model, get_model2, split_date, draw_landmarks, compact
from tqdm import tqdm
import argparse
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
import numpy as np

DATA_PTH = "../../../data/raw/landmarks"
DEVICE = "cuda"


def train():
    batch_size = 64
    lr = 1e-3
    gamma = 0.8
    n_epoch = 50

    aug_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, hue=0.0, contrast=0.3, saturation=0.0)
    ])

    train_ids, test_ids = split_date()

    ds_train = FaceKeyPointDataset(DATA_PTH, img_list=train_ids, aug_transform=None)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=20, shuffle=True)

    ds_val = FaceKeyPointDataset(DATA_PTH, img_list=test_ids)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=4)

    model = get_model2().to(device=DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter()

    loss_val_best = np.inf

    for epoch0 in range(n_epoch):
        epoch = epoch0+1
        loss_train = 0
        loss_val = 0
        model.train()
        for batch_id, batch in tqdm(enumerate(dl_train)):
            img = batch[0].to(DEVICE)
            landmarks = batch[1].to(DEVICE)
            output = model(img)
            loss = loss_fn(output, landmarks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_train += loss.item()
        loss_train /= (batch_id+1)

        model.eval()
        for batch_id, batch in enumerate(dl_val):
            with torch.no_grad():
                img = batch[0].to(DEVICE)
                landmarks = batch[1].to(DEVICE)
                output = model(img)
                loss = loss_fn(output, landmarks)
                loss_val += loss.item()
        loss_val /= (batch_id + 1)

        print("Epoch {:d}, train loss {:4.6f}, eval loss {:4.6f}".format(epoch, loss_train, loss_val))
        lr = scheduler.get_last_lr()[0]
        print("Learning rate is {:4.6f}".format(lr))

        writer.add_scalar("Loss/train", loss_train, epoch)
        writer.add_scalar("Loss/val", loss_val, epoch)
        writer.add_scalar("LR", lr, epoch)

        scheduler.step()

        if loss_val < loss_val_best:
            #torch.save(model.state_dict(), "../../../models/model_v1.2.ptc")
            loss_val_best = loss_val
            print("Save the model")

    writer.close()


def eval():
    _, test_ids = split_date()

    batch_size = 16

    ds_val = FaceKeyPointDataset(DATA_PTH, img_list=test_ids, add_filename=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=4)

    model = get_model2().to(device=DEVICE)
    model.load_state_dict(torch.load("../../../models/model_v1.2.ptc"))
    model.to(DEVICE)
    model.eval()

    loss_fn = nn.MSELoss()
    loss_val = 0

    res = []
    for batch_id, batch in tqdm(enumerate(dl_val)):
        img = batch[0].to(DEVICE)
        landmarks = batch[1].to(DEVICE)
        filenames = batch[2]
        output: torch.Tensor = model(img)
        loss = loss_fn(output, landmarks)
        loss_val += loss.item() / dl_val.batch_size
        arr = output.cpu().detach().numpy().tolist()
        r = [{"filename": filename, "landmarks": keypoints} for filename, keypoints in zip(filenames, arr)]
        res.extend(r)

    loss_val /= (batch_id+1)
    print(loss_val)

    with open("../../../data/results/eval_v1.2.json", "w") as fid:
        json.dump(res, fid)


def inference(img_path: str):
    # img_path = "../../data/raw/images/00017.png"
    img = read_image(img_path, mode=ImageReadMode.RGB) / 255.0

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])

    img = preprocess(img).unsqueeze(0)

    model = get_model("../../models/model_v2.ptc")
    model.eval()
    output = model(img)
    landmarks = output.cpu().detach().numpy().tolist()[0]

    draw_img = to_pil_image(img[0, :, :, :])
    landmarks = compact(landmarks)
    draw = ImageDraw.Draw(draw_img)

    for landmark in landmarks:
        x = landmark[0]
        y = landmark[1]
        r = 2
        leftUpPoint = (x-r, y-r)
        rightDownPoint = (x+r, y+r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, "green")

    draw_img.save("test.jpg")


def parse_args():
    parser = argparse.ArgumentParser(description="Train face region model")
    parser.add_argument("-a", "--action", type=str, default="train")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    if args["action"] == "train":
        train()
    elif args["action"] == "eval":
        eval()
    elif args["action"] == "inference":
        inference("me.jpg")
