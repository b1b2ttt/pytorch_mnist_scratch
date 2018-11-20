import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from datasets.my_mnist_dataset import MyMnistDataset
from net.origin_net import OriginNet
from net.modified_net import ModifiedNet
from visualization.visual_feature_map import plot_mid_layer_output

class IndicatorHistory():
    def __init__(self):
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []
    def reset(self):
        self.__init__()
    def update(self, tr_loss, te_loss, tr_acc, te_acc):
        self.train_loss.append(tr_loss)
        self.test_loss.append(te_loss)
        self.train_acc.append(tr_acc)
        self.test_acc.append(te_acc)
    

# construct dataloader
data_root = 'datasets/'
train_data = MyMnistDataset(txtfile_path=data_root+'trainsets.txt',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

test_data = MyMnistDataset(txtfile_path=data_root+'testsets.txt',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)


# trainng function
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, _, _, _, output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# test function
def test(args, model, device, test_loader, train_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, _, _, _, output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    train_loss = 0
    train_correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            _, _, _, _, output = model(data)
            train_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        train_loss, train_correct, len(train_loader.dataset),
        100. * train_correct / len(train_loader.dataset)))
    
    return train_loss, test_loss, train_correct/len(train_loader.dataset), correct/len(test_loader.dataset)


# tranining settings
parser = argparse.ArgumentParser(description='PyTorch MNIST From Scratch')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--show-mid-output', type=bool, default=False,
                        help='show middle layer outputs')
parser.add_argument('--use-ori-net', type=bool, default=False,
                        help='using original network')


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

if args.use_ori_net:
    plotname = 'original'
    model = OriginNet().to(device)
    # using Adam optimizer (original implemention)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    plotname = 'modified'
    model = ModifiedNet().to(device) 
    # using SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)



# record history indecator information
mnist_history = IndicatorHistory()
mnist_history.reset()

# trainning 
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    train_loss, test_loss, train_correct, test_correct = test(args, model, device, test_loader, train_loader)
    mnist_history.update(train_loss, test_loss, train_correct, test_correct)    


if args.show_mid_output:
    model.eval()
    with torch.no_grad():
        img_example = np.array(Image.open('datasets/imgs/0.jpg'), dtype = 'float32')
        img_example = torch.from_numpy(img_example).unsqueeze(0).unsqueeze(0).to(device)
        conv1, maxpool1, conv2, maxpool2, _ = model(img_example)
    #plot_mid_layer_output(img_example.cpu().numpy(), name='origin_image') 
    plot_mid_layer_output(conv1.cpu().numpy(), name='conv1')
    plot_mid_layer_output(maxpool1.cpu().numpy(), name='maxpool1')
    plot_mid_layer_output(conv2.cpu().numpy(), name='conv2') 
    plot_mid_layer_output(maxpool2.cpu().numpy(), name='maxpool2')


# plot figure
plt.figure(figsize=(8, 6))
plt.title('loss w.r.t. epochs', size=14)
plt.xlabel('epochs', size=14)
plt.ylabel('loss', size=14)
plt.plot(np.array(range(1, args.epochs + 1)), np.array(mnist_history.train_loss), color='b', linestyle='--', marker='o', label='train loss')
plt.plot(np.array(range(1, args.epochs + 1)), np.array(mnist_history.test_loss), color='r', linestyle='-', label='test loss')
plt.legend(loc='upper left')
plt.savefig('visualization/loss-epochs_' + plotname + '.png', format='png')


plt.figure(figsize=(8, 6))
plt.title('error rate w.r.t. epochs', size=14)
plt.xlabel('epochs', size=14)
plt.ylabel('error', size=14)
plt.plot(np.array(range(1, args.epochs + 1)), 1. - np.array(mnist_history.train_acc), color='b', linestyle='--', marker='o', label='train error')
plt.plot(np.array(range(1, args.epochs + 1)), 1. - np.array(mnist_history.test_acc), color='r', linestyle='-', label='test error')
plt.legend(loc='upper left')
plt.savefig('visualization/acc-epochs_' + plotname + '.png', format='png')
