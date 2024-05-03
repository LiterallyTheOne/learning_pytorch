import torch
from torch import nn

from torch.utils.data import DataLoader, random_split

import torchvision

from pathlib import Path


def find_device():
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    return device


def load_data(data_path: Path) -> [DataLoader, DataLoader, DataLoader]:
    tr = torchvision.transforms.Compose(
        [torchvision.transforms.Resize([90, 160]), torchvision.transforms.ToTensor(), ]
    )

    image_folder = torchvision.datasets.ImageFolder(data_path, transform=tr)

    g1 = torch.Generator().manual_seed(20)
    train_data, val_data, test_data = random_split(
        image_folder, [0.6, 0.2, 0.2], g1)

    train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader


def load_model(device: str) -> nn.Module:
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(1280, 4)
    model = model.to(device)

    return model


def train(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
          device: str = 'cuda', tensorboard_writer=None, epoch: int = 0):
    model = model.to(device)
    model.train()

    running_loss = 0
    i_c = 0
    number_of_batches = len(data_loader)

    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        prediction = model(images)
        loss = loss_fn(prediction, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        i_c += 1
        if (i % 10 == 9) or i == number_of_batches - 1:
            print(f'\r{i + 1}/{number_of_batches}, loss = {loss.item():>4f}', end='')
            if tensorboard_writer:
                tensorboard_writer.add_scalar('training loss', running_loss / i_c, epoch * number_of_batches + i)
            running_loss = 0
            i_c = 0
    print()


def evaluate(model: nn.Module,
             data_loader: torch.utils.data.DataLoader,
             loss_fn: nn.Module,
             device: str = 'cuda') -> tuple[torch.float, torch.float]:
    data_size = len(data_loader.dataset)
    number_of_batches = len(data_loader)

    model.eval()

    loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            prediction = model(images)
            loss += loss_fn(prediction, labels).item()
            correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()

        loss /= number_of_batches
        correct /= data_size

    return correct, loss
