import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

class _EfficientNet(nn.Module):
    def __init__(self, version, is_frozen=False):
        super().__init__()
        assert version in [f'b{id}' for id in range(7)]
        self.net = EfficientNet.from_pretrained(f'efficientnet-{version}')
        self.is_frozen = is_frozen
        if self.is_frozen: self.freeze()
        self.num_features = self.net._fc.in_features
    
    def freeze(self):
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.net.extract_features(x)
        x = self.net._avg_pooling(x)
        x = x.view(x.size(0), -1)
        #x = self.net._dropout(x)
        return x


class SimpleEfficientNet(nn.Module):
    def __init__(self, nclasses, backbone = 'effnetb0', freeze_backbone = True,):
        super().__init__()
        if (backbone[:6] == 'effnet'):
            self.cnn = _EfficientNet(backbone.replace('effnet', ''), is_frozen=freeze_backbone)
            self.num_features = self.cnn.num_features
        #self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(self.num_features, nclasses)
        self.nclasses = nclasses

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1) # Flatten
        out = self.cls(x)
        return out

if __name__ == "__main__":
    print("Hello")