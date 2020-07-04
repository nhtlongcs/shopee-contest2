import csv
import os

import numpy as np
import torch
from torch import Tensor
from PIL import Image
from torch.utils import data
from torchvision import transforms as tvtf


class shopee_raw(data.Dataset):
    def __init__(self, data_root_dir, train_dir, test_dir, csv_train_dir, csv_val_dir, csv_test_dir, is_train=True, infer_mode=False, nclasses=[-1]):
        super().__init__()
        self.is_train = is_train
        self.root_dir = data_root_dir
        self.train_dir = os.path.join(self.root_dir, train_dir)
        self.val_dir = os.path.join(self.root_dir, train_dir)
        self.test_dir = os.path.join(self.root_dir, test_dir)

        csv_train_dir = self.root_dir + csv_train_dir
        csv_val_dir = self.root_dir + csv_val_dir

        self.data = None
        self.labels = None

        if self.is_train:
            self.data = list(csv.reader(open(csv_train_dir)))
        elif self.is_train == False:
            self.data = list(csv.reader(open(csv_val_dir)))
            # self.data = self.data[:min(12000, len(self.data))]

        self.data.pop(0)  # clear header
        self.data, self.labels = zip(*self.data)
        self.labels = list(map(int, self.labels))

        self.data = [os.path.join("{0:0=2d}".format(self.labels[i]), self.data[i])
                     for i in range(len(self.data))]

    def __getitem__(self, index):
        if self.is_train:
            item_path = self.train_dir + self.data[index]
        else:
            item_path = self.val_dir + self.data[index]
        # return (item_path, self.labels[index])  # debug line
        image = Image.open(item_path).convert('RGB')

        tf = tvtf.Compose([tvtf.Resize(224),
                           tvtf.CenterCrop(224),
                           tvtf.ToTensor(),
                           tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                           ])
        image = tf(image)
        return (image, self.labels[index])

    def __len__(self):
        return len(self.data)


# test
if __name__ == "__main__":
    dataset = shopee_raw('/home/ken/shopee_ws/data/', 'train/train/',
                         'test/test/', 'train.csv', 'val.csv', 'test.csv')
    print(dataset[0])
