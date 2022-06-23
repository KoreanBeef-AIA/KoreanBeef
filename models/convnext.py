from torch import nn
import timm


class ConvNext(nn.Module):
    def __init__(self, n_outputs:int, **kwargs):
        super(ConvNext, self).__init__()
        self.model = timm.create_model('convnext_xlarge_in22ft1k', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features = 1792, out_features=625),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=625),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.Linear(in_features=256, out_features=n_outputs)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output