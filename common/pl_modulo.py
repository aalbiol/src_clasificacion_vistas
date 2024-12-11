import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)
# print("PATHS:",sys.path)


import warnings
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from torchmetrics import ConfusionMatrix
from torchmetrics import F1Score
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from m_dataLoad_json import lee_vista

import datetime

import pickle
from tqdm import tqdm

import numpy as np

import modelos
import wandb

torch.set_printoptions(precision=3)

    
def write_names(fichero,nombres):
    with open(fichero, 'w') as fp:
        for item in nombres:
            fp.write("%s\n" % item)
    #print(f' Finished writing {fichero}')

def mixup_data(x, y, alpha):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
   
    return mixed_x, mixed_y

class ViewClassifier(pl.LightningModule):
    def __init__(self, num_channels_in,
                lr=1e-3,
                class_names=None,
                weight_decay=1e-3,
                mixup_alpha=0.4,
                label_smoothing=0.01,
                warmup_iter=5,
                p_dropout=0.5,
                normalization_dict=None,
                multilabel=True,
                training_size=[120,120]):
        super().__init__()

        self.class_names=class_names
        self.num_classes=len(class_names)

        self.normalization_dict=normalization_dict
        self.multilabel=multilabel

        self.num_channels_in=num_channels_in
        self.weight_decay=weight_decay
        self.mixup_alpha=mixup_alpha

        self.label_smoothing = label_smoothing
        self.p_dropout=p_dropout
        self.warmup_iter=warmup_iter
        self.lr=lr
        self.training_size=training_size

        self.modelo=modelos.resnet50(num_channels_in=self.num_channels_in,
                                         num_classes=self.num_classes,
                                         p_dropout=self.p_dropout)               


# Monitorizacion
        self.estadisticas_labels=None
        self.pos_weights = None     
        if self.num_classes>1:
            self.F1Score_macro=F1Score('multilabel',num_labels=self.num_classes,average='macro').to(self.device)
            self.F1Score_micro=F1Score('multilabel',num_labels=self.num_classes,average='micro').to(self.device)
        else:
            self.F1Score_macro=F1Score('binary').to(self.device)
            self.F1Score_micro=F1Score('binary').to(self.device)
        self.valpreds=None
        self.valtargets=None
        self.valfilenames=None

                    
    def forward(self, X):# Para training
        Y=self.modelo.forward(X)
        return Y[:,:,0,0]



    def criterion(self, logits_batch, labels):
        '''labels lista con batch_size elementos
        Cada elemento tiene tantos elementos como etiquetas tenga la vista
        logits (batch_size,num_classes) '''

        
           
        # if self.pos_weights is None:
        #     pos_weight = torch.ones([self.num_classes],dtype=torch.float32,device=self.device)
        #     pos_weight *=10
        # else:
        #     pos_weight=self.pos_weights
                     
        #Lossfnc = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weight) 
        Lossfnc = nn.BCEWithLogitsLoss(reduction='mean') 
        smoothed_labels=(1-self.label_smoothing)*labels + self.label_smoothing/2 # Para multilabel cada label es binaria
        
        loss = Lossfnc(logits_batch,smoothed_labels)
        return loss 



    def configure_optimizers(self):
        #Activar inicialmente todos los gradientes En fintuning ya se ordenara el freezing
        parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        
        optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)   
        scheduler = ExponentialLR(optimizer, gamma=0.99) 
        
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=self.warmup_iter)
        
        return [optimizer],[warmup_lr_scheduler, scheduler]


    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']

        transform = transforms.Compose([
        transforms.Resize(self.training_size),
        transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        ])

        images = torch.cat([transform(v).unsqueeze(0) for v in images], dim=0)
        
        if torch.isnan(labels).any():
            print("Etiquetas con nans en train",labels)

        if self.estadisticas_labels is None:
            self.estadisticas_labels = labels   
        else:
            self.estadisticas_labels=torch.concat((self.estadisticas_labels,labels),dim=0)

        images,labels = mixup_data(images,labels,self.mixup_alpha)

        logits = self.forward(images)
        loss = self.criterion(logits, labels)
                
        # perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss



    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        fruit_ids=batch['fruit_ids']
        view_ids=batch['view_ids']
        transform = transforms.Compose([
        transforms.Resize(self.training_size),
        transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        ])

        images = torch.cat([transform(v).unsqueeze(0) for v in images], dim=0)
        if torch.isnan(labels).any():
            print("Etiquetas con nans en val",labels)

        logits = self.forward(images) # mapas de logits [b,c,11,11]

        loss = self.criterion(logits, labels)
        
        preds=F.sigmoid(logits)
        self.F1Score_macro(preds,labels)
        self.F1Score_micro(preds,labels)
            
        if self.valpreds is None:
            self.valpreds = preds
            self.valtargets = labels   
            self.valfilenames = view_ids 
        else:
            self.valpreds=torch.cat((self.valpreds,preds))
            self.valtargets = torch.cat((self.valtargets,labels))
            self.valfilenames += view_ids
        
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        

        


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['images']
        casos=batch['casos']
        normalizador = transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        
        images = [ normalizador(v) for v in images ]
        logits = self.forward(images)
        preds=F.sigmoid(logits)   
        return preds,casos


    def on_validation_epoch_end(self, ) -> None:
        if self.estadisticas_labels is not None:
            medias=torch.mean(self.estadisticas_labels,dim=0)
            self.pos_weights=(1-medias)/medias
            self.estadisticas_labels=None


        F1macro=self.F1Score_macro.compute()
        F1micro=self.F1Score_micro.compute()
        self.log("Val F1-macro",F1macro,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val F1-micro",F1micro,on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
        if self.num_classes>1:
            self.F1Score_macro=F1Score('multilabel',num_labels=self.num_classes,average='macro').to(self.device)
            self.F1Score_micro=F1Score('multilabel',num_labels=self.num_classes,average='micro').to(self.device)
        else:
            self.F1Score_macro=F1Score('binary').to(self.device)
            self.F1Score_micro=F1Score('binary').to(self.device)

        if self.valpreds is not None:
            fname1 = 'valpreds_multilabel.pt'
            fname2 = 'valtargets_multilabel.pt'
            print('valpreds.shape',self.valpreds.shape)
            torch.save(self.valpreds,fname1)
            torch.save(self.valtargets,fname2)
            self.valpreds=None
            self.valtargets=None
        
            
        if self.valfilenames is not None:
            fname3 = 'valfilenames_multilabel.txt'            
            write_names(fname3,self.valfilenames)
            self.valfilenames = None
        return    

    def save(self, path,config=None):
        salida={'state_dict':self.state_dict(),
        'normalization_dict':self.normalization,
        'image_size':self.image_size,
        'class_names':self.class_names,
        'training_date': datetime.datetime.now()}
        if config is not None:
            salida['config']=config

        with open(path, 'wb') as f:
                pickle.dump(salida, f)
        print(f' Finished writing {path}')

    def load(self, path):
        with open(path, 'rb') as f:
            leido = pickle.load(f)
        print(f' Finished Loading {path}')            
        self.load_state_dict(leido['state_dict'])
        self.normalization=leido['normalization_dict']
        self.image_size=leido['image_size']
        self.class_names=leido['class_names']
    def predict(self,vista):
        '''
        Recibe una lista de PILS.
        Devuelve lista de mapas de probabilidad de cada clase
        Cada mapa es un tensor con tantos canales como clases
        '''

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(self.image_size),
        ])

        normalizador = transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        
        # Normalizar las imágenes y crear batch
        vistas_transformadas = [ transform(v) for v in vistas ]
        vistas_normalizadas = [ normalizador(v) for v in vistas_transformadas ]

        batch = torch.stack(vistas_transformadas)
        batchn = torch.stack(vistas_normalizadas)
       
        logits =self.modelo.forward(batchn)
   

        if self.multilabel==False:
            probs= F.softmax(logits, dim = 1)
        else:
            probs=F.sigmoid(logits)
        
        
        return probs,batch
    

    def evaluate(self, X,device):
        if not isinstance(X ,list):
            X=[X]
        self.eval()
        print(device)
        self.to(device)
        resultados=[]
        for x in tqdm(X):
            #print(x)
            im=x.to(device)

            redimensionada=transforms.Resize(self.training_size)(im)
            medias=self.normalization_dict['medias_norm']
            stds=self.normalization_dict['stds_norm']
            normalizada=transforms.Normalize(medias,stds)(redimensionada)  


            with torch.no_grad():
                logits=self.forward(normalizada.unsqueeze(0))
                
            if self.multilabel==False:
                probs= F.softmax(logits, dim = 1)
            else:
                probs=F.sigmoid(logits)
                
            im=im.cpu().numpy().transpose(1,2,0)
            
            resultado={'imgname':x,'probs':probs}
            resultados.append(resultado)
        return resultados

    def on_training_epoch_end(self) -> None:
        pass



class ViewClassifier_Old(pl.LightningModule):
    def __init__(self, num_channels_in,
                lr=1e-3,
                class_names=None,
                weight_decay=1e-3,
                mixup_alpha=0.4,
                label_smoothing=0.01,
                warmup_iter=5,
                p_dropout=0.5):
        super().__init__()

        self.class_names=class_names
        self.num_classes=len(class_names)

        self.num_channels_in=num_channels_in
        self.weight_decay=weight_decay
        self.mixup_alpha=mixup_alpha

        self.label_smoothing = label_smoothing
        self.p_dropout=p_dropout
        self.warmup_iter=warmup_iter
        self.lr=lr

        self.modelo=modelos.resnet50(num_channels_in=self.num_channels_in,
                                         num_classes=self.num_classes,
                                         p_dropout=self.p_dropout)               


# Monitorizacion
        self.estadisticas_labels=None
        self.pos_weights = None     
        if self.num_classes>1:
            self.F1Score_macro=F1Score('multilabel',num_labels=self.num_classes,average='macro').to(self.device)
            self.F1Score_micro=F1Score('multilabel',num_labels=self.num_classes,average='micro').to(self.device)
        else:
            self.F1Score_macro=F1Score('binary').to(self.device)
            self.F1Score_micro=F1Score('binary').to(self.device)
        self.valpreds=None
        self.valtargets=None
        self.valfilenames=None

                    
    def forward(self, X):# Para training
        Y=self.modelo.forward(X)
        return Y[:,:,0,0]



    def criterion(self, logits_batch, labels):
        '''labels lista con batch_size elementos
        Cada elemento tiene tantos elementos como etiquetas tenga la vista
        logits (batch_size,num_classes) '''

        
           
        # if self.pos_weights is None:
        #     pos_weight = torch.ones([self.num_classes],dtype=torch.float32,device=self.device)
        #     pos_weight *=10
        # else:
        #     pos_weight=self.pos_weights
                     
        #Lossfnc = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weight) 
        Lossfnc = nn.BCEWithLogitsLoss(reduction='mean') 
        smoothed_labels=(1-self.label_smoothing)*labels + self.label_smoothing/2 # Para multilabel cada label es binaria
        
        loss = Lossfnc(logits_batch,smoothed_labels)
        return loss 



    def configure_optimizers(self):
        #Activar inicialmente todos los gradientes En fintuning ya se ordenara el freezing
        parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        
        optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)   
        scheduler = ExponentialLR(optimizer, gamma=0.99) 
        
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=self.warmup_iter)
        
        return [optimizer],[warmup_lr_scheduler, scheduler]


    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']

        if torch.isnan(labels).any():
            print("Etiquetas con nans en train",labels)

        if self.estadisticas_labels is None:
            self.estadisticas_labels = labels   
        else:
            self.estadisticas_labels=torch.concat((self.estadisticas_labels,labels),dim=0)

        images,labels = mixup_data(images,labels,self.mixup_alpha)

        logits = self.forward(images)
        loss = self.criterion(logits, labels)
                
        # perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss



    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['labels']
        fruit_ids=batch['fruit_ids']
        view_ids=batch['view_ids']

        if torch.isnan(labels).any():
            print("Etiquetas con nans en val",labels)

        logits = self.forward(images) # mapas de logits [b,c,11,11]

        loss = self.criterion(logits, labels)
        
        preds=F.sigmoid(logits)
        self.F1Score_macro(preds,labels)
        self.F1Score_micro(preds,labels)
            
        if self.valpreds is None:
            self.valpreds = preds
            self.valtargets = labels   
            self.valfilenames = view_ids 
        else:
            self.valpreds=torch.cat((self.valpreds,preds))
            self.valtargets = torch.cat((self.valtargets,labels))
            self.valfilenames += view_ids
        
            
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        

        


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch['images']
        casos=batch['casos']

        logits = self.forward(images)
        preds=F.sigmoid(logits)   
        return preds,casos


    def on_validation_epoch_end(self, ) -> None:
        if self.estadisticas_labels is not None:
            medias=torch.mean(self.estadisticas_labels,dim=0)
            self.pos_weights=(1-medias)/medias
            self.estadisticas_labels=None


        F1macro=self.F1Score_macro.compute()
        F1micro=self.F1Score_micro.compute()
        self.log("Val F1-macro",F1macro,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val F1-micro",F1micro,on_step=False, on_epoch=True, prog_bar=True, logger=True)
                
        if self.num_classes>1:
            self.F1Score_macro=F1Score('multilabel',num_labels=self.num_classes,average='macro').to(self.device)
            self.F1Score_micro=F1Score('multilabel',num_labels=self.num_classes,average='micro').to(self.device)
        else:
            self.F1Score_macro=F1Score('binary').to(self.device)
            self.F1Score_micro=F1Score('binary').to(self.device)

        # if self.valpreds is not None:
        #     fname1 = 'valpreds_multilabel.pt'
        #     fname2 = 'valtargets_multilabel.pt'
        #     print('valpreds.shape',self.valpreds.shape)
        #     torch.save(self.valpreds,fname1)
        #     torch.save(self.valtargets,fname2)
        #     self.valpreds=None
        #     self.valtargets=None
        
            
        # if self.valfilenames is not None:
        #     fname3 = 'valfilenames_multilabel.txt'            
        #     write_names(fname3,self.valfilenames)
        #     self.valfilenames = None
        return    

    def predice_fruto(self,vista):
        '''
        Recibe una lista de PILS.
        Devuelve lista de mapas de probabilidad de cada clase
        Cada mapa es un tensor con tantos canales como clases
        '''
        
        medias_norm = [0.7726, 0.4272, 0.2231]
        stds_norm = [0.4191, 0.4947, 0.4163]

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(self.image_size),
        ])

        normalizador = transforms.Normalize(medias_norm, stds_norm)
        
        # Normalizar las imágenes y crear batch
        vistas_transformadas = [ transform(v) for v in vistas ]
        vistas_normalizadas = [ normalizador(v) for v in vistas_transformadas ]

        batch = torch.stack(vistas_transformadas)
        batchn = torch.stack(vistas_normalizadas)

        features = self.modelo.features(batchn)        
        logits =self.modelo.classifier(features)
   

        if self.multilabel==False:
            probs= F.softmax(logits, dim = 1)
        else:
            probs=F.sigmoid(logits)
        
        return probs,batch

    def on_training_epoch_end(self) -> None:
        pass

