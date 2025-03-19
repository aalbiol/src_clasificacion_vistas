
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
from torchmetrics.classification import AUROC
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

from torch.optim.lr_scheduler import LinearLR

import numpy as np

import modelos
#import metricas
import os
import datetime
import pickle
import sys
import pycimg



torch.set_printoptions(precision=3)


class PatchMILClassifier(pl.LightningModule):
    def __init__(self, num_channels_in=3,
                optimizer='sgd', 
                lr=1e-3,
                weight_decay=1e-3,
                model_version = None,
                class_names=None,
                warmup_iter=0,
                label_smoothing=0.05,
                p_dropout=0.5,
                normalization_dict=None,
                training_size=None,
                config=None):
        super().__init__()


        #assert class_names is not None
        self.class_names=class_names
        
        #assert len(class_names)

        self.__dict__.update(locals())
        if not self.class_names is  None:
            self.num_classes=len(class_names)
        self.num_channels_in=num_channels_in
        self.weight_decay=weight_decay
        self.lr=lr
        self.warmup_iter=warmup_iter
        self.label_smoothing = label_smoothing
        self.p_dropout=p_dropout
        self.normalization_dict=normalization_dict # Para normalizar en inferencia. Para entrenar no se emplea
        self.training_size=training_size # Para normalizar en inferencia. Para entrenar no se emplea


        self.optimizer_name =optimizer
        self.config=config

        self.epoch_counter=1
        self.estadisticas_labels=None
        self.pos_weights = None
#metricas

        self.valpreds=None
        self.valtargets=None
        self.valfilenames=None
        resnet_version = model_version

        if config is not None:
            if 'focal_loss_gamma' in config['train']:
                print("Using focal loss gamma=",config['train']['focal_loss_gamma'])
                self.focal_loss_gamma=config['train']['focal_loss_gamma']
            else:
                self.focal_loss_gamma=None
            
            if 'focal_loss_alpha' in config['train']:
                self.focal_loss_alpha=config['train']['focal_loss_alpha']
            else:
                self.focal_loss_alpha=None
        
        self.resnet_version = resnet_version



        print("Resnet Version: ",resnet_version,type(resnet_version))

        if resnet_version is None or self.class_names is None:
            return
        if resnet_version == 50 or resnet_version =="50":
            self.modelo=modelos.resnet50PatchMIL(num_channels_in,num_classes=self.num_classes,  p_dropout=self.p_dropout)               
        elif resnet_version == 18 or resnet_version =="18":
            self.modelo=modelos.resnet18PatchMIL(num_channels_in,num_classes=self.num_classes,p_dropout=self.p_dropout)
        elif resnet_version == 34 or resnet_version =="34":
            self.modelo=modelos.resnet34PatchMIL(num_channels_in,num_classes=self.num_classes,p_dropout=self.p_dropout)
        elif resnet_version == 101 or resnet_version =="101":
            self.modelo=modelos.resnet101PatchMIL(num_channels_in,num_classes=self.num_classes,p_dropout=self.p_dropout)
        elif model_version=="vit_16":
                self.modelo=modelos.VitPatch(tipo=model_version,
                                              num_channels_in=num_channels_in, num_classes=self.num_classes)              
                                
        else:
            print(f"\n***** Warning. Version resnet solicitada {resnet_version} no contemplada. Usando resnet18")
            self.modelo=modelos.resnet18PatchMIL(num_channels_in,num_classes=self.num_classes,p_dropout=self.p_dropout)


            
    
    def aggregate_multilabel(self,logits_patches):
        ''' Selecciona el maximo logit de cada clase de defecto '''
        
        tam = logits_patches.shape
        #print("Tam antes agregación ",tam) tam=[20,10,7,7]
        
        logits_patches_reshaped = torch.reshape(logits_patches , (tam[0],tam[1], tam[2]*tam[3] ) )
        #print("logits_patches:", logits_patches_reshaped)
        logits_views= torch.max(logits_patches_reshaped,dim=2)
        
        return logits_views[0]



    def forward(self, X,nviews):# Para training
        # features = self.modelo.features(X)
        # features_dropped=self.modelo.dropout(features)        
        # logits_patches =self.modelo.classifier(features_dropped)
        logits_patches=self.modelo(X)
        

        logits_view = self.aggregate_multilabel(logits_patches)

         # Agrupar lo que es de cada fruto
        logits_fruits = torch.split(logits_view, nviews) # Esto es una lista 
        logits_fruit=[] # Lista de los logits de la vista critica de cada fruto. Al final, tantos elementos como frutos
        
        for logits in logits_fruits:
            #logits tiene dimensiones [nviews,num_classes]           
            logits_fruit.append(logits.max(axis=0,keepdim=True)[0])
        #logits_fruit tiene dimensiones nfrutos,num_classes    
        logits_fruit = torch.concat(logits_fruit,axis = 0)            
        return logits_patches,logits_view,logits_fruit

    
    def criterion(self, logits, labels): 
        
        # Gamma del focal loss va disminuyendo a medida que avanza el entrenamiento gamma annealing
        num_epochs=self.config['train']['epochs'] 
        num_epochs23=num_epochs*2/3
        
        gamma=self.focal_loss_gamma*(1-(self.epoch_counter-1)/num_epochs23)
        gamma=max(gamma,0)
        gamma=self.focal_loss_gamma

        if self.focal_loss_alpha is not None:
            loss=modelos.focal_loss_binary_with_logits(logits,labels,alpha=self.focal_loss_alpha,gamma=gamma)
            # print("Logits:",logits)
            # print("Labels:",labels)
            #print("Focal Loss:",loss)
            return loss,loss,loss                         
        binaryLoss = nn.BCEWithLogitsLoss(reduction='none')
        #binaryLoss = nn.BCEWithLogitsLoss(reduction='mean')
        # print('logit.shape:',logits.shape)
        # print('smoothed_labels.shape:',smoothed_labels.shape)
        #   print(">>>>>>>>>>>>smoothed labels:", smoothed_labels)
        losses=binaryLoss(logits,labels)
        
        losses_cols=[]
        pos_losses_col=[]
        neg_losses_col=[]
        for col in range(labels.shape[1]):
            labels_col=labels[:,col]
            losses_col=losses[:,col]
            pos_loss_col=losses_col[labels_col>0.5].mean()
            neg_loss_col=losses_col[labels_col<0.5].mean()
            pos_losses_col.append(pos_loss_col)
            neg_losses_col.append(neg_loss_col)
            loss_col=(pos_loss_col+neg_loss_col)/2
            losses_cols.append(loss_col)
        loss=torch.stack(losses_cols).mean()
        loss_pos=torch.stack(pos_losses_col).mean()
        loss_neg=torch.stack(neg_losses_col).mean()
        
        
        
        if torch.isnan(loss):            
            print ('\nNAN Loss Logits:',logits, " labels:", labels, 'epoch:', self.epoch_counter)
            print("NAN Losses:",loss,losses_cols)
        return loss,loss_pos,loss_neg
    
    def criterionval(self, logits, labels):  
        if self.focal_loss_alpha is not None:
            loss=modelos.focal_loss_binary_with_logits(logits,labels,alpha=self.focal_loss_alpha,gamma=self.focal_loss_gamma)
            return loss                                   
        binaryLoss = nn.BCEWithLogitsLoss(reduction='mean')
        loss=binaryLoss(logits,labels)
        return loss
    



    def criterion_old(self, logits_fruits, labels):
        '''labels lista con batch_size elementos
        Cada elemento tiene tantos elementos como etiquetas tenga la vista
        logits (batch_size,num_classes) '''


        if self.pos_weights is None:
            pos_weight = torch.ones([self.num_classes],dtype=torch.float32,device=self.device)
            pos_weight *=10
        else:
            pos_weight=self.pos_weights           

            
        #pesos=pos_weight.unsqueeze(0).repeat(logits_fruits.shape[0],1).to(self.device)         
        smoothed_labels=(1-self.label_smoothing)*labels + self.label_smoothing/2          
        #   mascara=~smoothed_labels.isnan()
        binaryLoss = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=self.pos_weights)

        loss=binaryLoss(logits_fruits,smoothed_labels)
            # loss=metricas.myNAN_BCElogitsLoss(logits_fruits[mascara], smoothed_labels[mascara],pesos[mascara])
            # if loss.isnan():
            #     print(" *** WARNING Loss=Nan. Replacing by 0")
            #     print ("Labels Nan Loss: ",labels)
            #     print ("Smoothed Labels Nan Loss: ",smoothed_labels)
            #     print ("LogitsBatch: ",logits_batch)
            #     print ("Pos Weights NAN: ",pesos[mascara])
            #     #print("Pesos Clasificador:",self.modelo.classifier[0].weight)
                
            #     loss=0 
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
        num_epochs=self.config['train']['epochs']           
        gamma=0.2**(1/num_epochs)    
        if self.warmup_iter > 0:           
            warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=self.warmup_iter)
            schedulers = [warmup_lr_scheduler, ExponentialLR(optimizer, gamma=gamma) ]
        else:
           schedulers = [ ExponentialLR(optimizer, gamma=gamma) ] 
        return [optimizer],schedulers

    def criterion_patches(self, logits_patches, labels,nviews):
        nbatch,nclases,nfilas,ncolumnas=logits_patches.shape
        ngroups=len(nviews)
        ngroups2,nclases2=labels.shape
        
        assert ngroups==ngroups2
        assert nclases==nclases2
        
        split_logits=torch.split(logits_patches,nviews,dim=0)
        

        pos_losses=[]
        neg_losses=[]
        #print("labels.device:",labels.device)
        #print("logits_patches.device:",logits_patches.device) 
        
        print("\n\n***************************************************")
        print("Labels:",labels.shape)
        print("Logits:",logits_patches.shape)
        print("nviews",nviews)
        for g in range(ngroups):
            logits_group=split_logits[g]
            labels_group=labels[g]
            print(g,"Labels Group:",labels_group.shape)
            print(g,"Logits Group:",logits_group.shape)
            print("================")
            for c in range(nclases):
                labels_group_c=labels_group[c]
                logits_group_c=logits_group[:,c,:,:]
                print(c,"Labels Group C:",labels_group_c.shape)
                print(c,"Logits Group:C",logits_group_c.shape)                
                if labels_group_c < 0.5:
                    loss_groupc=modelos.focal_loss_binary_with_logits(logits_group_c,torch.full(logits_group_c.shape,labels_group_c).to(logits_group_c.device),
                                                                      alpha=self.focal_loss_alpha,gamma=self.focal_loss_gamma)
                    #print("Loss Group C:",loss_groupc)
                    neg_losses.append(loss_groupc)
                else:
                    max_logit=torch.max(logits_group_c) 
                    loss_groupc=modelos.focal_loss_binary_with_logits(max_logit,torch.full(max_logit.shape,labels_group_c).to(max_logit.device),
                                                                      alpha=self.focal_loss_alpha,gamma=self.focal_loss_gamma)
                    pos_losses.append(loss_groupc)
        pos_loss=torch.stack(pos_losses).mean()
        neg_loss=torch.stack(neg_losses).mean()
        loss=(pos_loss+neg_loss)/2
        return loss,pos_loss,neg_loss
                    

    def training_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        if self.resnet_version is None:
            print(">>>>>>>>>>>>>>>>> ERROR. Resnet version not defined in training step")
            sys.exit(1)
        
        if self.estadisticas_labels is None:
            self.estadisticas_labels = labels   
        else:
            self.estadisticas_labels=torch.concat((self.estadisticas_labels,labels),dim=0)

        logits_patches,logits_view,logits_fruit = self.forward(images,nviews)
        
        #loss,pos_loss,neg_loss = self.criterion_patches(logits_patches, labels,nviews) 
        loss,pos_loss,neg_loss = self.criterion(logits_fruit, labels)
        if loss==0:
            print("*** WARNNG Training Zero Loss. Labels= ",labels)
            if images.isnan().any():
                print("Nans en images")
            else:
                print("No hay Nans en imagen")
        
        # perform logging
        log_dict = {'train_loss': loss, 'pos_loss':pos_loss, 'neg_loss':neg_loss}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss



    def validation_step(self, batch, batch_idx):
        images = batch['images']
        labels = batch['label']
        nviews = batch['nviews']
        
        logits_patches,logits_view,logits_fruit = self.forward(images,nviews=nviews) # mapas de logits [b,c,11,11]
        loss = self.criterionval(logits_fruit, labels)
        #loss,pos_loss,neg_loss = self.criterion_patches(logits_patches, labels,nviews)
        
        preds=F.sigmoid(logits_fruit)
       
        if self.valpreds is None:
            self.valpreds = preds
            self.valtargets = labels   
        else:
            self.valpreds=torch.cat((self.valpreds,preds))
            self.valtargets = torch.cat((self.valtargets,labels))
                    
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    # def predict_step(self, batch, batch_idx, dataloader_idx=0):    
    #     if 'casos' in batch:
    #         casos=batch['casos']
    #     else:
    #         casos=batch['view_ids']
    #     labels = batch['labels']
    #     batchn = batch['images'] # Vienen normalizadas del dataloader
    #     features = self.modelo.features(batchn)        
    #     logits =self.modelo.classifier(features)   
    #     probs_parches=F.sigmoid(logits)
        
    #     probs_parches=probs_parches.detach().cpu().numpy()
    #     probs_vista=probs_parches.max(axis=(2,3)) 
        
    #     out=dict()
    #     for m in zip(casos,probs_vista,labels):
    #         zz=np.round(m[1]*100).astype('int')
    #         lista_probs = zz.tolist() 
    #         lista_labels=m[2].tolist()
    #         dict_predictions={ ww[0]:ww[1] for ww in zip(self.class_names,lista_probs)}
    #         dict_annotations={ ww[0]:ww[1] for ww in zip(self.class_names,lista_labels)}
    #         out[m[0]]= {'predictions':dict_predictions,'annotations':dict_annotations}
    #     return out

    def on_validation_epoch_end(self, ) -> None:

        if self.estadisticas_labels is not None:
            medias=torch.nanmean(self.estadisticas_labels,dim=0)
            self.pos_weights=(1-medias)/medias
            self.estadisticas_labels=None
        #print("self.pos_weights: ",self.pos_weights )
        self.epoch_counter += 1

        aucfunc=AUROC(task='multilabel',average='none',num_labels=self.num_classes)
        auc=aucfunc(self.valpreds,(self.valtargets>0.5).int()) 


            
        self.aucs={}
        for i in range(self.num_classes):
            self.aucs[f'Val AUC - {self.class_names[i]}']=auc[i].item()
        self.log_dict(self.aucs)#,on_step=False, on_epoch=True, prog_bar=True, logger=True)         
        self.valpreds=None
        self.valtargets=None
            
        return    

    def on_training_epoch_end(self) -> None:
        pass
###################################################################################
    def save(self, path,config=None):
        salida={'state_dict':self.state_dict(),
        'resnet_version':self.resnet_version,                
        'normalization_dict':self.normalization_dict,
        'image_size':self.training_size,
        'class_names':self.class_names,
        'training_date': datetime.datetime.now(),
        'final_val_aucs':self.aucs,
        'num_channels_in':self.num_channels_in,
        'p_dropout':self.p_dropout,
        'model_type':'PatchMILClassifier',
    }

        if config is not None:
            salida['config']=config

        with open(path, 'wb') as f:
            pickle.dump(salida, f)
        print(f' Finished writing {path}')

###################################################################################
    def load(self, path):
        with open(path, 'rb') as f:
            leido = pickle.load(f)
        print(f' Finished Loading {path}')
        self.num_channels_in=leido['num_channels_in']
        self.class_names=leido['class_names']
        self.num_classes=len(self.class_names)
        self.p_dropout=leido['p_dropout']
        self.num_classes=len(self.class_names)
        self.resnet_version=leido['resnet_version']
        if self.resnet_version == 50:
            self.modelo=modelos.resnet50PatchMIL(num_channels_in=self.num_channels_in, num_classes=self.num_classes,  p_dropout=self.p_dropout)               
        elif self.resnet_version == 18:
            self.modelo=modelos.resnet18PatchMIL(num_channels_in=self.num_channels_in,um_classes=self.num_classes,p_dropout=self.p_dropout)
        elif self.resnet_version == 34:
            self.modelo=modelos.resnet34PatchMIL(num_channels_in=self.num_channels_in,num_classes=self.num_classes,p_dropout=self.p_dropout)
        elif self.resnet_version == 101:
            self.modelo=modelos.resnet101PatchMIL(num_channels_in=self.num_channels_in,num_classes=self.num_classes,p_dropout=self.p_dropout)
                                
        else:
            print(f"\n***** Warning. Version resnet solicitada {self.resnet_version} no contemplada. Usando resnet18")
            self.modelo=modelos.resnet18MIL(num_classes=self.num_classes,p_dropout=self.p_dropout)


        self.load_state_dict(leido['state_dict'])
        self.normalization_dict=leido['normalization_dict']
        self.training_size=leido['image_size']
        self.class_names=leido['class_names']
        self.training_date=leido['training_date']
        self.maxvalues=leido['config']['data']['maxvalues']
        self.channel_list=leido['config']['data']['channel_list']
    
    def predict(self, nombres,device,include_images=False,json_file=None):
        '''
        lista imagenes. Cada imagen es una tupla (fich.npz,fich.json)

        Cada archivo incluye una lista de las vistas de un fruto.
        Las vistas pueden tener distintos tamaños
        El ficher json contine información de maxValChannels y canales

        
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

        transformacion=transforms.Compose([
            transforms.Resize(self.training_size),
            transforms.Normalize(self.normalization_dict['medias_norm'],self.normalization_dict['stds_norm'])
        ])


        for caso in sin_prefijos:
            #print("Processing ",nombre)
            
            extension=caso.split('.')[-1]

            if extension == 'npz':
                #print("nombre_json:",json_file)
                x = pycimg.npzread_torch(caso,nombre_json=json_file,channel_list=self.channel_list)
            else:
                print(f"   >>>>>>>>>>>>> {caso}: Extension no contemplada")
                continue

            normalizada=[transformacion(j) for j in x] # Las vistas pueden tener distintos tamaños
            normalizada=torch.stack(normalizada)
            normalizada=normalizada.to(device)                            
            nviews=len(x)
            with torch.no_grad():
                logits_patches,logits_view,logits_fruit=self.forward(normalizada,nviews=[nviews])
                

            probs_fruto=F.sigmoid(logits_fruit).round(decimals=3)
            probs_vistas=F.sigmoid(logits_view).round(decimals=3)
            probs_patches=F.sigmoid(logits_patches).round(decimals=3)
            
                             
            #print(">>>>>>>>>>>",probs_fruto[0,:])    
            ims_vistas= [ v.cpu().numpy().transpose(1,2,0) for v in x]
            
            probsdictfruto={}
            probsdictvistas={}
            probsdictpatches={}
            for i in range(self.num_classes):
                probsdictfruto[self.class_names[i]]=int(1000*probs_fruto[0,i].item())/1000
                kk=probs_vistas[:,i].cpu().numpy().tolist()
                kk=[int(1000*k)/1000 for k in kk]
                probsdictvistas[self.class_names[i]]=kk

                kk=probs_patches[:,i,:,:].cpu().numpy().tolist()
                kk=[np.round(k,decimals=3) for k in kk]
                probsdictpatches[self.class_names[i]]=kk

            resultado={'imgname':caso,'jsonname':json_file,
                       'probs_fruto':probsdictfruto,
                       'probs_vistas' : probsdictvistas,
                       'probs_patches':probsdictpatches,
                       'probs_fruto_tensor':probs_fruto[0,:],
                       'probs_vistas_tensor':probs_vistas,
                       'probs_patches_tensor':probs_patches,}
            if include_images:
                resultado['img']=ims_vistas #Lista de vistas normalizadas entre 0 y 1

            resultados.append(resultado)
            
        return resultados
   
