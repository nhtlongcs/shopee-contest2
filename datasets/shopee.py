from torch.utils import data
import torch
import os
import csv


class shopee_raw(data.Dataset):
    def __init__(self, data_root_dir, train_dir, test_dir, csv_train_dir, csv_test_dir, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.root_dir = data_root_dir

        self.train_dir = os.path.join(self.root_dir, train_dir)
        self.test_dir = os.path.join(self.root_dir, test_dir)

        csv_train_dir = self.root_dir + csv_train_dir 
        csv_test_dir = self.root_dir + csv_test_dir 

        self.data = None
        self.labels = None

        # if self.is_train:
        #     self.data = list(csv.reader(open(csv_train_dir)))
        # elif self.is_train == False:
        #     self.data = list(csv.reader(open(csv_test_dir)))

        # _, self.labels = zip(*self.data)
        # self.labels = list(map(int, self.labels))

    def __getitem__(self, index):
        if self.is_train:
            item_path = self.train_dir + self.data[index]
        else:
            item_path = self.test_dir + self.data[index]
        return (torch.load(item_path), self.labels[index])

    def __len__(self):
        return len(data)


# test
if __name__ == "__main__":
    dataset = shopee_raw('/content/data/', 'train/',
                         'test/', 'train.csv', 'test.csv')
