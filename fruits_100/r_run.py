import torch
from torch import nn

from r_utils import prepare_and_get_data_path, find_device, load_model, load_train_data, load_valid_data, train, \
    evaluate, predict

from matplotlib import pyplot as plt

import torchvision

from pathlib import Path
from PIL import Image


def main():
    device = find_device()

    print(f'device: {device}')

    data_path = prepare_and_get_data_path('pc')
    #
    model = load_model(device)
    model = model.to(device)
    #
    # train_data_loader, val_data_loader = load_data(data_path)
    train_data_loader = load_train_data(data_path / 'train')
    valid_data_loader = load_valid_data(data_path / 'val')
    #
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    #
    losses = []
    for epoch in range(1):
        print(f'in epoch: {epoch}')
        losses = train(model, train_data_loader, loss_fn, optimizer, device)
        accuracy, loss = evaluate(model, valid_data_loader, loss_fn, device)
        print(f'validation -> accuracy: {accuracy:.2f}, loss: {loss:.2f}')

    plt.plot(losses)
    plt.show()

    # haha = Path('/home/ramin/ramin_programs/files/datasets/fruits-100/test/0/0.jpg')
    #
    # tr = torchvision.transforms.Compose(
    #     [torchvision.transforms.Resize([90, 160]), torchvision.transforms.ToTensor(), ]
    # )
    #
    # image = Image.open(haha)
    #
    # kekw = tr(image).to(device)
    # kekw = kekw.unsqueeze(0)
    #
    # model = load_model(device)
    #
    # result = model(kekw)
    #
    # print(torch.argmax(result))

    # result = predict(model, data_path / 'test')

    # print(result)


if __name__ == '__main__':
    main()
