import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from utils import FaceKeyPointDataset
from torchvision import models
from tqdm import tqdm


def get_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=136)
    )
    return model


def train():
    data_pth = "../../data"
    device="cuda"

    ds = FaceKeyPointDataset(data_pth)
    dl = DataLoader(ds, batch_size=12, num_workers=4)
    model = get_model().to(device=device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(10):

        total_loss = 0
        for batch in tqdm(dl):
            img = batch[0].to(device)
            landmarks = batch[1].to(device)
            output = model(img)
            loss = loss_fn(output, landmarks)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print("Epoch {:d}, loss {:4.3f}".format(epoch, total_loss))

    torch.save(model.state_dict(), "model_v1")


if __name__ == "__main__":
    train()