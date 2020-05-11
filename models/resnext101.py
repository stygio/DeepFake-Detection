from torchvision.models.resnet import ResNet, model_urls, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
import torch.nn as nn

from models.classifier import BinaryClassifier


class Binary_ResNeXt101(ResNet):
    """
    My version of ResNet152 with a modified classifier and functions for unfreezing certain groups of layers.
    """
    def __init__(self, pretrained = False):
        # Constructor
        super(Binary_ResNeXt101, self).__init__(block = Bottleneck, layers = [3, 4, 23, 3], groups = 32, width_per_group = 8)
        self.fc = BinaryClassifier(in_channels = self.fc.in_features)

        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress = False)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.load_state_dict(model_dict)
    
    def unfreeze_classifier(self):
        for param in self.fc.parameters():
            param.requires_grad = True