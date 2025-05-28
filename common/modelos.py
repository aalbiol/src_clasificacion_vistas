import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import timm
import torch
import modelos
import torch.nn.functional as F
import math

class ResNet(nn.Module):
    def __init__(self, model, num_channels_in, num_classes,p_dropout=0.5):
        super(ResNet, self).__init__()
        
        self.num_channels_in=num_channels_in
        self.conditioner=nn.Conv2d(num_channels_in, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        capas=list(model.children())
        cap=capas[:-1]
        self.features=nn.Sequential(*cap)
        #features incluye hasta el avgpool inclusive
        
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
        return self.classifier(f)[:,:,0,0]

class VitPatch(nn.Module):
    def __init__(self,tipo, num_channels_in, num_classes):
        super(VitPatch, self).__init__()
        
        self.num_channels_in=num_channels_in
        self.conditioner=nn.Conv2d(num_channels_in, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.num_classes=num_classes
        
        if tipo=='vit_16':
            self.model= timm.create_model('vit_base_patch16_224', pretrained=True)
        elif tipo=='vit_32':
            self.model= timm.create_model('vit_base_patch32_224', pretrained=True)
        else:
            print('Error: tipo de modelo no reconocido Falling back to vit_base_patch16_224')
            self.model= timm.create_model('vit_base_patch16_224', pretrained=True)

        self.model.head=nn.Linear(self.model.head.in_features, num_classes)
        capas=list(self.model.children())
        cap=capas[:-1]
        self.features=nn.Sequential(*cap)
        self.classifier=self.model.head

        self.model.global_pool=None

        self.prediction_pool=nn.AdaptiveAvgPool2d(output_size=(6, 6))


            
        
    def forward(self,x):
        if self.num_channels_in !=3:
            y= self.conditioner(x)
        else:
            y=x
        z = self.model.forward_features(y)
        z = self.model.forward_head(z)
        tam_espacial=z.shape[1]
        lado=int(math.sqrt(tam_espacial-1))
        #print("lado:",lado)
        z=z[:,1:,:].permute(0,2,1).reshape(-1,self.num_classes,lado,lado)
        z=self.prediction_pool(z)

        #print("Forward vit output size:",z.shape)
        return z  

# class ResNetMIL(nn.Module):
#     ''' Otra forma de tener la misma estructura de red  
#     '''
#     def __init__(self, model, num_channels_in, num_classes, p_dropout=0.5):
#         super(ResNetMIL, self).__init__()
        
#         print('Resnet Num channels_in:',num_channels_in)
#         self.num_channels_in=num_channels_in
#         self.conditioner=None
#         if self.num_channels_in !=3:
#             self.conditioner=nn.Conv2d(num_channels_in, 3, kernel_size=1, stride=1, padding=0, bias=False)
                
#         self.features = nn.Sequential( OrderedDict([
#             ('conv1', model.conv1),
#             ('bn1',model.bn1),
#             ('relu', model.relu),
#             ('maxpool',model.maxpool),
#             ('layer1',model.layer1),
#             ('layer2',model.layer2),
#             ('layer3',model.layer3),
#             ('layer4',model.layer4),
#             ]))
        
#         # classification layer
#         self.dropout=nn.Dropout(p=p_dropout)
        

#         #num_features = self.features.layer4[2].conv3.out_channels
#         num_features = model.fc.in_features
#         #print('num_features:',num_features)
        
#         self.averagepool= nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         # nhidden_classifier=256
#         # self.classifier =  nn.Sequential(nn.Dropout(p=p_dropout),
#         #                         nn.Linear(num_features,nhidden_classifier, bias=True),
#         #                         nn.ReLU(inplace=True),
#         #                         nn.Linear(nhidden_classifier,num_classes, bias=True)
#         #                         )
#         self.classifier =  nn.Sequential(nn.Dropout(p=p_dropout),
#                                 nn.Linear(num_features,num_classes, bias=True),
#                                 )
                                         
            
        

#     def forward(self,x):
#         if self.num_channels_in !=3:
#             y= self.conditioner(x)
#         else:
#             y=x

#         f=self.features(y) 
#         # print("x:",x.shape)       
#         # print('y:',f.shape) 
#         f=self.averagepool(f)

#         f=f[:,:,0,0]
#         #f=f.mean(axis=(2,3))
#         #print(f.shape)
#         return self.classifier(f)


class ResNetPatchMIL(nn.Module):
    '''
    Devuelve una imagencita de baja resolución
    Cambia el avgpool + linear por un una convolución 1x1 como clasificador
    '''
    def __init__(self, model, num_channels_in,num_classes, p_dropout=0.5):
        super(ResNetPatchMIL, self).__init__()

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
        num_features = self.features.layer4[2].conv3.out_channels

        
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features,num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            )

        
    def forward(self,x):
        if self.num_channels_in !=3:
            y= self.conditioner(x)
        else:
            y=x
        f=self.features(y)
        f=self.dropout(f)
        return self.classifier(f)
    @torch.jit.export
    def forward_fruit(self,x):
        y = self.forward(x)
        maximo, pos_maximo = torch.max(y,dim=-1)
        maximo, pos_maximo = torch.max(maximo,dim=-1)
        return maximo
    


def resnet50(num_channels_in, num_classes, pretrained=True,tune_fc_only=False,p_dropout=0.5):
    model = models.resnet50(pretrained)
    return ResNet(model, num_channels_in,num_classes,p_dropout=p_dropout)

# def resnet50MIL(num_channels_in, num_classes, pretrained=True,p_dropout=0.5):
#     model = models.resnet50(pretrained)
#     return ResNetMIL(model, num_channels_in,num_classes,p_dropout=p_dropout)


def resnet18PatchMIL(num_channels_in,num_classes, p_dropout=0.5):
    model = models.resnet18(weights=models.ResNet18_Weights)
    return ResNetPatchMIL(model, num_channels_in,num_classes,p_dropout=p_dropout)

def resnet34PatchMIL(num_channels_in,num_classes, p_dropout=0.5):
    model = models.resnet34(weights=models.ResNet34_Weights)
    return ResNetPatchMIL(model, num_channels_in,num_classes,p_dropout=p_dropout)

def resnet50PatchMIL(num_channels_in,num_classes, p_dropout=0.5):
    model = models.resnet50(weights=models.ResNet50_Weights)
    return ResNetPatchMIL(model, num_channels_in,num_classes,p_dropout=p_dropout)

def resnet101PatchMIL(num_channels_in,num_classes, p_dropout=0.5):
    model = models.resnet101(weights=models.ResNet101_Weights)
    return ResNetPatchMIL(model, num_channels_in,num_classes,p_dropout=p_dropout)

def focal_loss_binary_with_logits(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal loss with logits, pytorch version, uses binary_cross_entropy_with_logits
    """
    #print("Gamma focal:",gamma)
    # print('logits_device:',logits.device) 
    # print('labels_device:',labels.device) 
    probs = torch.sigmoid(logits)
    labels = labels.float()

    pt = torch.where(labels > 0.5 , probs, 1 - probs)
    at = torch.where(labels > 0.5 , torch.tensor(alpha).to(logits.device), torch.tensor(1 - alpha).to(logits.device))

    ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    focal_loss = at * (1 - pt).pow(gamma) * ce_loss
    #print(at)
    #print(ce_loss)
    return 10.0*torch.mean(focal_loss)