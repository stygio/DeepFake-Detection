import torch.nn as nn

class BinaryClassifier(nn.Module):
    def __init__(self, in_channels):
        super(BinaryClassifier, self).__init__()
        
        self.fc = nn.Linear(in_channels, 1)

    def init_weights():
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization
                nn.init.xavier_normal_(m.weight, gain = 1)
    
    def forward(self, x):
        x = nn.functional.dropout(x, training = self.training)
        x = self.fc(x)

        return x