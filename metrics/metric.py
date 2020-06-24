import torch 
import numpy as np

class Accuracy():
    def __init__(self):
        self.reset()

    def calculate(self, output, target):
        # [B, N]
        pred = torch.argmax(output, dim=1)
        #print(pred, target)
        return ((pred == target).float().sum() / output.size(0)).item()

    def update(self, value):
        self.acc.append(value)

    def reset(self):
        self.acc = []

    def value(self):
        return np.mean(self.acc)

    def summary(self):
        return np.mean(self.acc)

from sklearn.metrics import confusion_matrix

class ConfusionMatrix():
    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1)
        return confusion_matrix(target, pred, labels=range(self.nclasses))

    def update(self, value):
        self.cm += value

    def reset(self):
        self.cm = np.zeros(self.nclasses, self.nclasses)

    def value(self):
        return 0

    def summary(self):
        return self.cm

