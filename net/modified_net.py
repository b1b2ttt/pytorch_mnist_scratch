import torch.nn as nn


class ModifiedNet(nn.Module):
    def __init__(self):
        super(ModifiedNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(28 * 28 * 32, 10)

    def forward(self, x):
        conv1 = self.layer1(x)
        out = conv1.reshape(conv1.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        return None, None, None, None, out

