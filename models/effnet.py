from torch import nn
import timm


class EffNet(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(EffNet, self).__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features = 1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_outputs)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output