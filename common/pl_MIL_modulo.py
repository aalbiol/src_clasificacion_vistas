
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

from torchmetrics.classification import AUROC

import numpy as np
import modelos
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR

import pickle
import datetime
import pl_datamodule
import dataset
from torchvision import transforms

from tqdm import tqdm
import pycimg


# Para clasificar vistas entrenando con MIL cuando anotaciones son por fruto y no por vista    
class MILClassifier(pl.LightningModule):
    def __init__(self, num_channels_in=None,
                optimizer='sgd', 
                lr=1e-3,
                weight_decay=1e-3,
                model_version = None,
                class_names=None,
                warmup_iter=0,
                label_smoothing=0.01,
                p_dropout=0.5,
                normalization_dict=None,
                training_size=None):
        super().__init__()


        
        self.class_names=class_names
        
        self.__dict__.update(locals())
        
        if class_names is not None:
            self.num_classes=len(class_names)
        self.num_channels_in=num_channels_in
        self.weight_decay=weight_decay
        self.normalization_dict=normalization_dict # Para normalizar en inferencia. Para entrenar no se emplea
        self.training_size=training_size # Para normalizar en inferencia. Para entrenar no se emplea
        self.lr=lr
        self.warmup_iter=warmup_iter
        self.label_smoothing = label_smoothing
        self.p_dropout=p_dropout
                                
        self.optimizer_name = optimizer
        
        if self.class_names is not None:
            print('FruitMILClassifier num clases out=',self.num_classes)
        # Using a pretrained ResNet backbone
        
        if model_version is not None:
            if model_version == "50":
                self.modelo=modelos.resnet50MIL(num_channels_in=num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout) 
                print('Using resnet50')  
            # elif model_version == "68":
            #     self.modelo=modelos.dualpathnet68(num_channels_in=num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout)  
            #     print('Using dualpathnet68')
            # elif model_version == "201":
            #     self.modelo=modelos.densenet201(num_channels_in=num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout)  
            #     print('Using densenet201')
            # elif model_version == "mobilesmall":
            #     self.modelo=modelos.mobilenetV3small(num_channels_in=num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout)  
            #     print('Using MobileNetV3_Small')
            # elif model_version == "mobilelarge":
            #     self.modelo=modelos.mobilenetV3large(num_channels_in=num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout)  
            #     print('Using MobileNetV3_Large')      
            else:
                print(f"\n***** Warning. Version resnet solicitada {model_version} no contemplada. Usando resnet50")
                self.modelo=modelos.resnet50MIL(num_channels_in=num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout)    
        else:
            self.modelo=None

        
        self.epoch_counter=1
        
        self.pos_weights=None
        self.valpreds=None
        self.valtargets=None
        self.valfilenames=None
        self.estadisticas_labels = None

    def vistas2fruit(self, logits_vistas, nviews,labels):
        logits_fruits = torch.split(logits_vistas, nviews) # Esto es una lista 
        logits_fruit=[] # Lista de los logits de la vista critica de cada fruto. Al final, tantos elementos como frutos
        for fruit_idx,logits in enumerate(logits_fruits):
            # logits matriz de cada uno de los frutos de nvistas x nclases
            nvistas,nclasses=logits.shape
            logits_clases=[]
            for c in range(nclasses):
                if labels[fruit_idx,c]>0.5:
                    logits_clases.append(logits[:,c].max(axis=0,keepdim=False)[0])
                else:
                    logits_clases.append(logits[:,c].mean(axis=0,keepdim=False))
            logits_clases=torch.stack(logits_clases)
            logits_fruit.append(logits_clases)
        logits_fruit = torch.stack(logits_fruit)
        return logits_fruit

            

    def forward(self, X, nviews):
        '''
        X: tensor con vistas de diferentes frutos
        nviews: tantos elementos como frutos El elemento k-ésimo es el número de vistas del fruto k-ésimo
        Recibe tantos elementos como vistas
        Devuelve tantos elementos como frutos
        Los logits de cada fruto son el máximo de los de sus vistas para cada tipo de defecto
        
        El número de columnas igual al número de tipos de defecto
        '''
        
        assert self.class_names is not None
        
        logits_all_views = self.modelo(X)
        
        
        # Agrupar lo que es de cada fruto
        logits_fruits = torch.split(logits_all_views, nviews) # Esto es una lista 
        

        logits_fruit=[] # Lista de los logits de la vista critica de cada fruto. Al final, tantos elementos como frutos
        for logits in logits_fruits:         
            logits_fruit.append(logits.max(axis=0,keepdim=True)[0])

         
        logits_fruit = torch.concat(logits_fruit,axis = 0)
        
        return logits_fruit,logits_all_views 
    
    def criterion(self, logits, labels):       
        if self.pos_weights is None:
            pos_weight = torch.ones([self.num_classes],dtype=torch.float32,device=self.device)
            #scale tensor by 10 
            pos_weight *=10
            
            
        else:  
            pos_weight = self.pos_weights     
        
        smoothed_labels=(1-self.label_smoothing)*labels + self.label_smoothing/2
                    
        binaryLoss = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=pos_weight)
        #binaryLoss = nn.BCEWithLogitsLoss(reduction='mean')
        # print('logit.shape:',logits.shape)
        # print('smoothed_labels.shape:',smoothed_labels.shape)
        #   print(">>>>>>>>>>>>smoothed labels:", smoothed_labels)
        loss=binaryLoss(logits,smoothed_labels)
        
        if torch.isnan(loss):
            print ('NAN Loss Logits:',logits, " labels:", labels, 'epoch:', self.epoch_counter)
        return loss
            
        
    def configure_optimizers(self):
        print("weight decay = ",self.weight_decay)
        parameters = list(filter(lambda x: x.requires_grad, self.parameters()))
        if self.optimizer_name.lower() == 'sgd':
            optimizer = SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name.lower() == 'adam':
            optimizer = Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        else:
            print(f'**** WARNING : Optimizer configured to {self.optimizer_name}. Falling back to SGD')
            optimizer = SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)
                       
        if self.warmup_iter > 0:           
            warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_iter)
            schedulers = [warmup_lr_scheduler, ExponentialLR(optimizer, gamma=0.99) ]
        else:
           schedulers = [ ExponentialLR(optimizer, gamma=0.99) ] 
        return [optimizer],schedulers
    
    
    def training_step(self, batch, batch_idx):
        assert self.modelo is not None

        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        
        if self.estadisticas_labels is None:
            self.estadisticas_labels = labels   
        else:
            self.estadisticas_labels=torch.concat((self.estadisticas_labels,labels),dim=0)

        
        logits_fruto,logits_vistas = self(images, nviews) #logits de fruto

        #logits_fruto=self.vistas2fruit(logits_vistas,nviews,labels)

        loss = self.criterion(logits_fruto, labels)
        # perform logging
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    
    def validation_step(self, batch, batch_idx):
        assert self.modelo is not None
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        casos = batch['paths']
        logits_fruto,logits_vistas = self(images, nviews)
        
        loss = self.criterion(logits_fruto, labels)
        
        preds=F.sigmoid(logits_fruto)

        if self.valpreds is None:
            self.valpreds = preds
            self.valtargets = (labels>0.5).to(torch.int32)   
            self.valfilenames = casos 
        else:
            self.valpreds=torch.cat((self.valpreds,preds))
            self.valtargets = torch.cat((self.valtargets,labels))
            self.valfilenames += casos    
        # perform logging
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    
    def predict_step(self, batch, batch_idx, dataloader_idx=0): 
        ''' Con este metodo se predice la clase del fruto
        Se toma la vista con mas probabilidad de algun tipo de defecto
        Se mira la clase de dicha vista
        
        TODO: Se puede pensar en otras estrategias tale como contar el número de vistas clasificadas como defecto.
        O también consider la vista con el defecto mas serio. Por ejemplo si 1 vista es de 3ª y otra de 2ª y el resto buenas,
        se clasificaria como de 3ª
        '''
        images = batch['images']
        nviews = batch['nviews']
        paths = batch['paths'] 
        
        logits = self(images, nviews)
        preds_class = logits.argmax(axis = 1)
        return preds_class,paths    
    
    
    def on_validation_epoch_end(self, ) -> None:     
        
        if self.valpreds is not None:
        #Calcular AUC
            aucfunc=AUROC(task='multilabel',average='none',num_labels=self.num_classes)
            auc=aucfunc(self.valpreds,(self.valtargets>0.5).int())

            self.aucs={}
            for i in range(self.num_classes):
                self.aucs[f'Val AUC - {self.class_names[i]}']=auc[i].item()
            self.log_dict(self.aucs)#,on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # Borrar para recomenzar en epoch siguiente    
            self.valpreds=None
            self.valtargets=None
        
            
        if self.valfilenames is not None:
            # fname3 = 'valfilenames_multiclass.txt' if self.multilabel == False else 'valfilenames_multilabel.txt'            
            # write_names(fname3,self.valfilenames)
            self.valfilenames = None
            if self.estadisticas_labels is not None:
                medias=torch.nanmean(self.estadisticas_labels,dim=0)
                self.pos_weights=(1-medias)/medias
                self.estadisticas_labels=None
        
        self.epoch_counter += 1
            


    def save(self, path,config=None):
        salida={'state_dict':self.state_dict(),
        'normalization_dict':self.normalization_dict,
        'image_size':self.training_size,
        'class_names':self.class_names,
        'training_date': datetime.datetime.now(),
        'final_val_aucs':self.aucs,
        'num_channels_in':self.num_channels_in,
        'p_dropout':self.p_dropout,
        'model_type':'MILClassifier',
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
        self.num_classes=len(self.class_names)
        self.modelo=modelos.resnet50MIL(num_channels_in=self.num_channels_in,
                                         num_classes=self.num_classes,
                                         p_dropout=self.p_dropout)

        self.load_state_dict(leido['state_dict'])
        self.normalization_dict=leido['normalization_dict']
        self.training_size=leido['image_size']
        self.class_names=leido['class_names']
        self.training_date=leido['training_date']
        self.crop_size=leido['config']['data']['crop_size']
        self.maxvalues=leido['config']['data']['maxvalues']
        #print("Loading maxvalues:",self.maxvalues)
        self.channel_list=leido['config']['data']['channel_list']
    
    
    
    def predict(self, nombres,device,include_images=False,json_file=None):
        '''
        lista de nombres de imágenes de cimgs

        Cada archivo incluye una lista de las vistas de un fruto.
        Las vistas pueden tener distintos tamaños

        
        '''
        if not isinstance(nombres ,list):
            nombres=[nombres]
        self.eval()
        #print(device)
        self.to(device)
        resultados=[]
        
        #print(">>>>>>>>>> Nombres:",nombres)

        sin_prefijos=nombres

        sin_prefijos=list(set(sin_prefijos))
        
        #print(">>>>>>>>>> sin prefijos:",sin_prefijos)
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

        for nombre in sin_prefijos:
            #print("Processing ",nombre)
            extension=sin_prefijos[0].split('.')[-1]
            if extension == 'cimg':
                x = pycimg.cimglistread_torch(nombre,self.maxvalues,channel_list=self.channel_list)
            elif extension == 'npz':
                assert json_file is not None
                x = pycimg.npzread_torch(nombre,json_file,channel_list=self.channel_list)
            else:
                print(f"   >>>>>>>>>>>>> {nombre}: Extension no contemplada")
                continue

            normalizada=[transformacion(j) for j in x] # Las vistas pueden tener distintos tamaños
            normalizada=torch.stack(normalizada)
            normalizada=normalizada.to(device)                            
            nviews=len(x)
            with torch.no_grad():
                logits_fruto,logits_vistas=self.forward(normalizada,nviews=[nviews])
                

            probs_fruto=F.sigmoid(logits_fruto).round(decimals=3)
            probs_vistas=F.sigmoid(logits_vistas).round(decimals=3)
            
                             
            #print(">>>>>>>>>>>",probs_fruto[0,:])    
            ims_vistas= [ v.cpu().numpy().transpose(1,2,0) for v in x]
            
            probsdictfruto={}
            probsdictvistas={}
            for i in range(self.num_classes):
                probsdictfruto[self.class_names[i]]=int(1000*probs_fruto[0,i].item())/1000
                kk=probs_vistas[:,i].cpu().numpy().tolist()
                kk=[int(1000*k)/1000 for k in kk]
                probsdictvistas[self.class_names[i]]=kk
            resultado={'imgname':nombre,'probs_fruto':probsdictfruto,'probs_vistas' : probsdictvistas,'probs_fruto_tensor':probs_fruto[0,:],'probs_vistas_tensor':probs_vistas}
            if include_images:
                resultado['img']=ims_vistas #Lista de vistas normalizadas entre 0 y 1

            resultados.append(resultado)
            
        return resultados
