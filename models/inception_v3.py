from torchvision.models.inception import Inception3, model_urls
from torchvision.models.utils import load_state_dict_from_url
import torch.nn as nn


class Binary_Inception(Inception3):
    """
    My version of Inception with functions for unfreezing certain groups of layers.
    """
    def __init__(self, pretrained = False):
        # Constructor
        super(Binary_Inception, self).__init__(num_classes = 1, aux_logits = False, init_weights = False)

        if pretrained:
            model_dict = self.state_dict()
            pretrained_dict = load_state_dict_from_url(model_urls['inception_v3_google'], progress = False)
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict) 
            # 3. load the new state dict
            self.load_state_dict(model_dict)

        # Initialize new fc weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization
                nn.init.xavier_normal_(m.weight, gain = 1)
    
    def unfreeze_classifier(self):
        for param in self.fc.parameters():
            param.requires_grad = True