from torch import nn
import torch
from typing_extensions import Self
import config


class NeuralNetwork(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(3, 128, config.conv_kernel_size)
        self.act1: nn.ReLU = nn.ReLU()
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(
            config.pool_kernel_size, config.stride_size
        )
        self.drop1: nn.Dropout2d = nn.Dropout2d(config.dropout_prob)

        """
        self.conv2: nn.Conv2d = nn.Conv2d(128, 224, config.conv_kernel_size)
        self.act2: nn.ReLU = nn.ReLU()
        self.pool2: nn.MaxPool2d = nn.MaxPool2d(
            config.pool_kernel_size, config.stride_size
        )
        self.drop2: nn.Dropout2d = nn.Dropout2d(config.dropout_prob)

        self.fc3: nn.Linear = nn.Linear(224 * 6 * 6, 128)
        self.act3: nn.ReLU = nn.ReLU()
        self.drop3 = nn.Dropout(config.dropout_prob)

        self.fc4 = nn.Linear(128, 128)
        self.act4: nn.ReLU = nn.ReLU()
        self.drop4 = nn.Dropout(config.dropout_prob)
        """

        self.fc5 = nn.Linear(128 * 15 * 15, 10)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        #x = self.conv2(x)
        #x = self.act2(x)
        #x = self.pool2(x)
        #x = self.drop2(x)


        x = x.view(-1, 128 * 15 * 15)

        #x = self.drop3(self.act3(self.fc3(x)))
        #x = self.drop4(self.act4(self.fc4(x)))

        x = self.fc5(x)

        return x
