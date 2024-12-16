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
from torchmetrics import AUROC
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
import matplotlib.pyplot as plt

import datetime

import pickle
from tqdm import tqdm

import numpy as np

import modelos
import pl_datamodule
import dataset
import json
import pycimg
#import wandb

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
    def __init__(self, num_channels_in=3,
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
        if class_names is not None:
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
        
        if self.class_names is not None:
            self.modelo=modelos.resnet50(num_channels_in=self.num_channels_in,
                                         num_classes=self.num_classes,
                                         p_dropout=self.p_dropout)               
        else:
            self.modelo=None


# Monitorizacion
        self.estadisticas_labels=None
        self.pos_weights = None     

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

        assert self.class_names is not None

        images = batch['images']
        labels = batch['labels']


        # Durante el entrenamiento las imágenes las normaliza el dataloader

        # transform = transforms.Compose([
        # transforms.Resize(self.training_size),
        # transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        # ])

        # images = torch.cat([transform(v).unsqueeze(0) for v in images], dim=0)
        
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

        # Durante el entrenamiento las imágenes las normaliza el dataloader
        # transform = transforms.Compose([
        # transforms.Resize(self.training_size),
        # transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        # ])

        # images = torch.cat([transform(v).unsqueeze(0) for v in images], dim=0)


        if torch.isnan(labels).any():
            print("Etiquetas con nans en val",labels)

        logits = self.forward(images) # mapas de logits [b,c,11,11]

        loss = self.criterion(logits, labels)
        
        preds=F.sigmoid(logits)

            
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
        pass
        # images = batch['images']
        # casos=batch['casos']
        # normalizador = transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        
        # images = [ normalizador(v) for v in images ]
        # logits = self.forward(images)
        # preds=F.sigmoid(logits)   
        # return preds,casos


    def on_validation_epoch_end(self, ) -> None:
        # Actualizar pos_weights
        if self.estadisticas_labels is not None:
            medias=torch.mean(self.estadisticas_labels,dim=0)
            self.pos_weights=(1-medias)/medias
            self.estadisticas_labels=None

        #Calcular AUC

        aucfunc=AUROC(task='multilabel',average='none',num_labels=self.num_classes)
        auc=aucfunc(self.valpreds,(self.valtargets>0.5).int())

        self.aucs={}
        for i in range(self.num_classes):
            self.aucs[f'Val AUC - {self.class_names[i]}']=auc[i].item()
        self.log_dict(self.aucs)#,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.valpreds is not None:
            self.valpreds=None
            self.valtargets=None
            self.valfilenames = None
        
        return    

    def on_training_epoch_end(self) -> None:
        pass


    def save(self, path,config=None):
        salida={'state_dict':self.state_dict(),
        'normalization_dict':self.normalization_dict,
        'image_size':self.training_size,
        'class_names':self.class_names,
        'training_date': datetime.datetime.now(),
        'final_val_aucs':self.aucs,
        'num_channels_in':self.num_channels_in,
        'p_dropout':self.p_dropout,
        'model_type':'Classifier',
    }

        if config is not None:
            salida['config']=config

        with open(path, 'wb') as f:
                pickle.dump(salida, f)
        print(f' Finished writing {path}')

    def load(self, path):
        with open(path, 'rb') as f:
            leido = pickle.load(f)
        print(f' Finished Loading {path}')
        self.num_channels_in=leido['num_channels_in']
        self.class_names=leido['class_names']
        self.num_classes=len(self.class_names)
        self.p_dropout=leido['p_dropout']
        self.modelo=modelos.resnet50(num_channels_in=self.num_channels_in,
                                         num_classes=self.num_classes,
                                         p_dropout=self.p_dropout)

        self.load_state_dict(leido['state_dict'])
        self.normalization_dict=leido['normalization_dict']
        self.training_size=leido['image_size']
        self.class_names=leido['class_names']
        self.training_date=leido['training_date']
        self.crop_size=leido['config']['data']['crop_size']
    

    def predict(self, nombres,device,delimiter="_",max_value=None,terminaciones=None,include_images=False,remove_suffix=True):
        '''
        lista de nombres de imágenes

        Se le pueden pasar nombres _RGB.png, _NIR.png, _UV.png,...

        Para generar la lista de casos,
        se les quita el sufijo , por ejemplo _RGB.png usando el delimiter "_" para generar el viewid y luego se lee con 
        '''
        if not isinstance(nombres ,list):
            nombres=[nombres]
        self.eval()
        #print(device)
        self.to(device)
        resultados=[]
        

        if remove_suffix:
            sin_prefijos=[ pl_datamodule.remove_sufix(nombre,delimiter) for nombre in nombres]
        else:
            sin_prefijos=nombres

        sin_prefijos=list(set(sin_prefijos))
        if self.crop_size is None:
            transformacion=transforms.Compose([
            transforms.Resize(self.training_size),
            transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        ])
        else:
            transformacion=transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize(self.training_size),
                transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
            ])
            
        if len(sin_prefijos) >1:
            for nombre in tqdm(sin_prefijos):
                #print("Processing ",nombre)
                x=dataset.lee_vista(images_folder='.',view_id=nombre,terminaciones=terminaciones,max_value=max_value,carga_mask=False)            

                im=x.to(device)

#                redimensionada=transforms.Resize(self.training_size)(im)
#                medias=self.normalization_dict['medias_norm']
#                stds=self.normalization_dict['stds_norm']
#                normalizada=transforms.Normalize(medias,stds)(redimensionada)  
                normalizada=transformacion(im)
                
                with torch.no_grad():
                    logits=self.forward(normalizada.unsqueeze(0))
                    
                if self.multilabel==False:
                    probs= F.softmax(logits, dim = 1)
                else:
                    probs=F.sigmoid(logits)
                    
                im=im.cpu().numpy().transpose(1,2,0)
                
                probsdict={}
                for i in range(self.num_classes):
                    probsdict[self.class_names[i]]=probs[0,i].item()
                resultado={'imgname':nombre,'probs':probsdict,'tensor_probs':probs.cpu()}
                if include_images:
                    resultado['img']=im

                resultados.append(resultado)
        else:
            for nombre in sin_prefijos:
                #print("Processing ",nombre)
                x=dataset.lee_vista(images_folder='.',view_id=nombre,terminaciones=terminaciones,max_value=max_value,carga_mask=False)            
#                print("Entrada ",x.shape, x.max(), x.min())

                im=x.to(device)

                # _=plt.imshow(im[:3,:,:].cpu().numpy().transpose(1,2,0),clim=[-2,2])
                # _=plt.show()

                #redimensionada=transforms.Resize(self.training_size)(im)
                #medias=self.normalization_dict['medias_norm']
                #stds=self.normalization_dict['stds_norm']
                #normalizada=transforms.Normalize(medias,stds)(redimensionada)  
                normalizada=transformacion(im)
                

#                print("Normalizada ",  normalizada.shape, normalizada.mean(dim=(-1,-2)), normalizada.std(dim=(-1,-2)))
                with torch.no_grad():
                    logits=self.forward(normalizada.unsqueeze(0))
                    
                if self.multilabel==False:
                    probs= F.softmax(logits, dim = 1)
                else:
                    probs=F.sigmoid(logits)
                    
                im=im.cpu().numpy().transpose(1,2,0)
                
                
                probsdict={}
                for i in range(self.num_classes):
                    probsdict[self.class_names[i]]=probs[0,i].item()
                resultado={'imgname':nombre,'probs':probsdict,'tensor_probs':probs.cpu()}
                if include_images:
                    resultado['img']=im

                resultados.append(resultado)
            
        return resultados

    # def evaluate(self, dataloader,device):
    #     self.eval()
    #     self.to(device)
    #     self.valpreds=None
    #     self.valtargets=None
    #     self.valfilenames=None
    #     for batch in tqdm(dataloader):
    #         images = batch['images']
    #         labels = batch['labels']
    #         fruit_ids=batch['fruit_ids']
    #         view_ids=batch['view_ids']
    #         logits = self.forward(images)
    


def load_model_info(path):
    with open(path, 'rb') as f:
        model_info = pickle.load(f)
    print(f' Finished Loading {path}')            
        
    return model_info

def print_model_info(path):
    with open(path, 'rb') as f:
        leido = pickle.load(f)
    print(f' Finished Loading {path}')            
        
    normalization=leido['normalization_dict']
    image_size=leido['image_size']
    class_names=leido['class_names']
    config=None
    if 'config' in leido:
        config=leido['config']
    training_date=leido['training_date']
    final_val_aucs=leido['final_val_aucs']
    print("normalization: ",normalization)
    print("image_size: ",image_size)
    print("class_names: ",class_names)
    print("training_date: ",training_date)
    print("final_val_aucs: ",final_val_aucs)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if config is not None:
        if 'evaluate' in config:
            del config['evaluate']
        if 'predict' in config:
            del config['predict']   
        config_string=json.dumps(config,indent=3)
        print("config", config_string)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

       
