import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from utils import FaceKeyPointDataset, get_model, split_date
from tqdm import tqdm
import random


def train():
    data_pth = "../../data"
    device = "cuda"
    batch_size = 12

    train_ids, test_ids = split_date()

    ds_train = FaceKeyPointDataset(data_pth, img_list=train_ids)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=4, shuffle=True)

    ds_val = FaceKeyPointDataset(data_pth, img_list=test_ids)
    dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=4)

    model = get_model().to(device=device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        loss_train = 0
        loss_eval = 0
        model.train()
        for batch_id, batch in tqdm(enumerate(dl_train)):
            img = batch[0].to(device)
            landmarks = batch[1].to(device)
            output = model(img)
            loss = loss_fn(output, landmarks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_train += loss.item() / dl_train.batch_size
        loss_train /= (batch_id+1)

        model.eval()
        for batch_id, batch in enumerate(dl_val):
            with torch.no_grad():
                img = batch[0].to(device)
                landmarks = batch[1].to(device)
                output = model(img)
                loss = loss_fn(output, landmarks)
                loss_eval += loss.item() / dl_val.batch_size
        loss_eval /= (batch_id + 1)

        print("Epoch {:d}, train loss {:4.3f}, eval loss {:4.3f}".format(epoch, loss_train, loss_eval))

    torch.save(model.state_dict(), "../../models/model_v1.ptc")






if __name__ == "__main__":
    train()