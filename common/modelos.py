import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class ResNet(nn.Module):
    def __init__(self, model, num_channels_in, num_classes,p_dropout=0.5):
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
    

class ResNetMIL(nn.Module):
    def __init__(self, model, num_channels_in, num_classes, p_dropout=0.5):
        super(ResNetMIL, self).__init__()
        
        print('Resnet Num channels_in:',num_channels_in)
        self.num_channels_in=num_channels_in
        self.conditioner=None
        if self.num_channels_in !=3:
            self.conditioner=nn.Conv2d(num_channels_in, 3, kernel_size=1, stride=1, padding=0, bias=False)
                
        self.features = nn.Sequential( OrderedDict([
            ('conv1', model.conv1),
            ('bn1',model.bn1),
            ('relu', model.relu),
            ('maxpool',model.maxpool),
            ('layer1',model.layer1),
            ('layer2',model.layer2),
            ('layer3',model.layer3),
            ('layer4',model.layer4),
            ]))
        
        # classification layer
        self.dropout=nn.Dropout(p=p_dropout)
        

        #num_features = self.features.layer4[2].conv3.out_channels
        num_features = model.fc.in_features
        #print('num_features:',num_features)
        
        self.averagepool= nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.classifier =  nn.Linear(num_features,num_classes, bias=True)
            
        

    def forward(self,x):
        if self.num_channels_in !=3:
            y= self.conditioner(x)
        else:
            y=x
        f=self.features(y)
        f=self.dropout(f)
        f=self.averagepool(f)
        f=f[:,:,0,0]
        #f=f.mean(axis=(2,3))
        #print(f.shape)
        return self.classifier(f)



def resnet50(num_channels_in, num_classes, pretrained=True,tune_fc_only=False,p_dropout=0.5):
    model = models.resnet50(pretrained)
    return ResNet(model, num_channels_in,num_classes,p_dropout=p_dropout)

def resnet50MIL(num_channels_in, num_classes, pretrained=True,p_dropout=0.5):
    model = models.resnet50(pretrained)
    return ResNetMIL(model, num_channels_in,num_classes,p_dropout=p_dropout)

