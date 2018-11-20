from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class MyMnistDataset():
    def __init__(self, txtfile_path, transform=None, target_transform=None):
        fh = open(txtfile_path, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = np.array(Image.open(img), dtype = 'float32')
        img = np.expand_dims(img,2)
        #print(img.shape)
        if self.transform is not None:
            img = self.transform(img)
            #print(img.shape)
        return img,label

    def __len__(self):
        return len(self.imgs)
