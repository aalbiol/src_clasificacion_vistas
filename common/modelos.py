import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class ResNet(nn.Module):
    def __init__(self, model, num_channels_in, num_classes, tune_fc_only,p_dropout=0.5):
        super(ResNet, self).__init__()
        
        self.num_channels_in=num_channels_in
        self.conditioner=nn.Conv2d(num_channels_in, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        capas=list(model.children())
        cap=capas[:-1]
        self.features=nn.Sequential(*cap)
        
        self.dropout=nn.Dropout(p=p_dropout)
        num_features =capas[-1].in_features
        self.classifier =  nn.Conv2d(num_features,num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            
        
    def forward(self,x):
        if self.num_channels_in !=3:
            y= self.conditioner(x)
        else:
            y=x
        f=self.features(y)
        f=self.dropout(f)
        return self.classifier(f)
    



def resnet50(num_channels_in, num_classes, pretrained=True,tune_fc_only=False,p_dropout=0.5):
    model = models.resnet50(pretrained)
    return ResNet(model, num_channels_in,num_classes,tune_fc_only,p_dropout=p_dropout)

