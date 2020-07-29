from efficientnet_pytorch import EfficientNet

class Binary_EfficientNet(EfficientNet):
    """
    My extension of EfficientNet with functions for unfreezing certain groups of layers.
    """
    def __init__(self, blocks_args = None, global_params = None):
        # Constructor
        super(Binary_EfficientNet, self).__init__(blocks_args = blocks_args, global_params = global_params)
    
    def classifier_parameters(self):
        return self._fc.parameters()

    def higher_level_parameters(self):
        hl_parameters = []
        hl_parameters += list(self._bn1.parameters())
        hl_parameters += list(self._conv_head.parameters())
        for block in self._blocks[37:]:
            hl_parameters += list(block.parameters())
        return hl_parameters

    def lower_level_parameters(self):
        ll_parameters = []
        for block in self._blocks[10:37]:
            ll_parameters += list(block.parameters())        
        return ll_parameters
    
    def unfreeze_classifier(self):
        for param in self.classifier_parameters():
            param.requires_grad = True

    def unfreeze_higher_level(self):
        for param in self.higher_level_parameters():
            param.requires_grad = True

    def unfreeze_lower_level(self):
        for param in self.lower_level_parameters():
            param.requires_grad = True