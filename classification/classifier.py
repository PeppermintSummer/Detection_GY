import torch.nn as nn
import torch
import torchvision
import cv2

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torchvision.models.resnet50(pretrained=True)

    def forward(self,x):
        feature = self.backbone(x)
        # print('feature::',feature.size())
        prob = torch.softmax(feature,dim=1) # feature size() ===> [1, 1000]
        return prob

