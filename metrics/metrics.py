import torch
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os


class Accuracy():
    def __init__(self):
        self.reset()

    def calculate(self, output, target):
        # [B, N]
        pred = torch.argmax(output, dim=1)
        # [B]
        return (pred == target).float().sum().item(), pred.size(0)

    def update(self, value):
        self.nsamples += value[1]
        self.ncorrects += value[0]

    def reset(self):
        self.nsamples = 0.0
        self.ncorrects = 0.0

    def value(self):
        return self.ncorrects / self.nsamples

    def summary(self):
        return self.ncorrects / self.nsamples

    def display(self):
        print(self.ncorrects / self.nsamples)


class ConfusionMatrix():
    def __init__(self, nclasses, output_dir, verbose=False):
        self.nclasses = nclasses
        self.reset()
        self.output_dir = output_dir
        self.count = 1
        self.verbose = verbose

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        return confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(), labels=range(self.nclasses))

    def update(self, value):
        self.cm += value

    def reset(self):
        self.cm = np.zeros(shape=(self.nclasses, self.nclasses))

    def value(self):
        return 0

    def summary(self):
        # self.display(self.output_dir, str(self.count))
        self.count += 1
        return self.cm

    def display(self, output_dir, name):
        df_cm = pd.DataFrame(self.cm, index=range(
            self.nclasses), columns=range(self.nclasses))
        if self.verbose:
            print(df_cm)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, cmap='YlGnBu')
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(output_dir, name))
        plt.close()
