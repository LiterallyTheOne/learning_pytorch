from r_utils import train, evaluate, find_device, load_data, load_model

import torch
from torch import nn

from pathlib import Path

from matplotlib import pyplot as plt


def main():
    data_path = Path(
        '/home/ramin/ramin_programs/files/datasets/tom-and-jerry-image-classification/tom_and_jerry/tom_and_jerry')

    device = find_device()

    print(f'device: {device}')

    train_data_loader, val_data_loader, test_data_loader = load_data(data_path)

    # for tensor_image, label in train_data_loader:
    #     figure, axes = plt.subplots(1, 1)
    #
    #     axes.imshow(torchvision.transforms.ToPILImage()(tensor_image[0]))
    #     axes.set_axis_off()
    #     print(label[0])
    #     plt.show()
    #     break

    model = load_model(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(5):
        print(f'in epoch: {epoch}')
        train(model, train_data_loader, loss_fn,
              optimizer, device)
        accuracy, loss = evaluate(model, val_data_loader, loss_fn, device)
        print(f'validation -> accuracy: {accuracy:.2f}, loss: {loss:.2f}')

    accuracy, loss = evaluate(model, test_data_loader, loss_fn, device)
    print('test results:')
    print(f'accuracy: {accuracy}, loss: {loss}')

    torch.save(model.state_dict(), "mobile_net_v2_transfer_learning.pth")


if __name__ == '__main__':
    main()
