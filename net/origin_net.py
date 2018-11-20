import torch.nn as nn
import torch.nn.functional as F


class OriginNet(nn.Module):
    def __init__(self):
        super(OriginNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        conv1 = self.layer1(x)
        maxpool1 = self.maxpool(conv1)
        conv2 = self.layer2(maxpool1)
        maxpool2 = self.maxpool(conv2)
        out = maxpool2.reshape(maxpool2.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return conv1, maxpool1, conv2, maxpool2, out
