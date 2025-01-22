import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(10, 32, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x
