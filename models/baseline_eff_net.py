import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractors.efficient_net import EfficientNetExtractor


# class baseline_eff_net(nn.Module):
#     """Baseline model"""

#     def __init__(self, version, nclasses, depth=3):
#         super().__init__()
#         assert version in range(9)
#         self.extractor = EfficientNet.from_pretrained(
#             f'efficientnet-b{version}')
#         self.feature_dim = self.extractor._fc.in_features
#         self.cls_1 = nn.Linear(self.feature_dim, 640)
#         self.cls_2 = nn.Linear(640, 320)
#         self.cls_3 = nn.Linear(320, 160)
#         self.cls_4 = nn.Linear(160, 80)
#         self.cls_5 = nn.Linear(80, nclasses)

#     def forward(self, x):
#         x = self.extractor.extract_features(x)
#         x = self.extractor._avg_pooling(F.relu(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.cls_1(x))
#         x = F.relu(self.cls_2(x))
#         x = F.relu(self.cls_3(x))
#         x = F.relu(self.cls_4(x))
#         x = self.cls_5(x)
#         return x


class baseline_eff_net(nn.Module):
    """Baseline model"""

    def __init__(self, version, nclasses, freeze=False):
        super().__init__()
        assert version in range(8)
        self.extractor = EfficientNetExtractor(version)
        if freeze:
            self.extractor.freeze()

        self.feature_dim = self.extractor.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, nclasses)
        )

    def forward(self, x):
        x = self.extractor.get_embedding(x)
        x = self.classifier(x)
        return x
