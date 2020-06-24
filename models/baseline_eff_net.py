import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class baseline_eff_net(nn.Module):
    """Baseline model"""

    def __init__(self, version, nclasses):
        super().__init__()
        assert version in range(9)
        self.extractor = EfficientNet.from_pretrained(
            f'efficientnet-b{version}')
        self.feature_dim = self.extractor._fc.in_features

        self.cls = nn.Linear(self.feature_dim, nclasses)

    def forward(self, x):
        x = self.extractor.extract_features(x)
        x = self.extractor._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.cls(x)
        return x
