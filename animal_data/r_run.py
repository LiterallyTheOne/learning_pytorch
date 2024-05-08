import torch
from torch import nn

from r_utils import prepare_and_get_data_path, load_data, find_device, load_model, train, evaluate, plot_standard

from matplotlib import pyplot as plt


def main():
    # data_path = prepare_and_get_data_path('pc')
    #
    # device = find_device()
    # print(device)
    #
    # train_data_loader, valid_data_loader, test_data_loader = load_data(data_path)
    #
    # model = load_model(device)
    #
    # loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    #
    # train(model, train_data_loader, loss_fn, optimizer, device)
    #
    # print(train_data_loader)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    a = [1, 2, 3, 4]
    plot_standard(axes[0, 0], a, 'accuracy')

    plt.show()


if __name__ == '__main__':
    main()
