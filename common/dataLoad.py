
import warnings
warnings.filterwarnings('ignore')
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)

# torch and lightning imports
import torch
import pytorch_lightning as pl

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from typing import Tuple,Any

import os
from PIL import Image
import sampler
from transformaciones import Aumentador_Imagenes

import m_dataLoad_json

class FileNamesDataSet(Dataset):
    '''
    Clase para suministrar archivos de una lista cuando no hay anotacion
    Solo se devuelve la imagen y el fruitid y el viewid.
    Por compatibilidad se devuelve un target = None
    '''
    def __init__(self,root_folder=None, filenames_list=None ,transform=None, 
                 field_delimiter='-',max_value = 255, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.root_folder=root_folder
        self.filenames_list=list(filenames_list)
        self.transform = transform
        self.field_delimiter=field_delimiter
        self.max_value = max_value
        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:              
        view_id=m_dataLoad_json.view_id(self.filenames_list[index])
        fruit_id=m_dataLoad_json.fruit_id(self.filenames_list[index],delimiter=self.field_delimiter)
        
        img_fullname=(self.filenames_list[index] if self.root_folder is None else os.path.join(self.root_folder,self.filenames_list[index]))
        imagen= m_dataLoad_json.lee_png16(img_fullname,self.max_value)
           
        if self.transform is not None:                
            imagen2 = self.transform(imagen)
        else:
            imagen2=imagen
        target=None     # Este data set no está etiquetado               
        return imagen2, target, view_id,fruit_id 
              
    def __len__(self) -> int:
        return len(self.filenames_list)
 
 


def matriz_etiquetas(dataset):
    lista=[]
    for caso in dataset:
        labels=caso['labels']
        lista.append(labels)
    matriz=torch.stack(lista)

    print(f"** Info. Núm de instancias= {matriz.shape[0]}")
    print(" ** Numero de casos por categoria:" , torch.sum(matriz,axis=0))
    return matriz
        
    
    
# Esto es el data set    
class OliveViewsDataSet(Dataset):
    def __init__(self,dataset=None ,transform=None, carga_mask=False, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.dataset=dataset   
        self.transform = transform
        self.carga_mask = carga_mask
        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        caso=self.dataset[index]
        
        target =caso['labels'].float()
        view_id=caso['view_id']
        fruit_id=caso['fruit_id']
        
        
        if caso['image'] is None: #Cuando no está en memoria
            imags_folder=caso['imag_folder']
            sufijos=caso['sufijos']
            max_value=caso['max_value']
            crop_size=caso['crop_size']
            #print("Reading ", view_id)
            imagen=m_dataLoad_json.lee_vista(imags_folder,view_id,sufijos,max_value=max_value,carga_mask=self.carga_mask)
        else:
            #print("Imagen ya disponible:" , view_id)
            imagen=caso['image']
                       
        if self.transform is not None:                
            imagen2 = self.transform(imagen)
        else:
            imagen2=imagen
                           
        return imagen2,target,view_id,fruit_id # imagen, lista_de_etiquetas, (ruta_al_archivo, vista_id) 
       
    def __len__(self) -> int:
        return len(self.dataset)
    

def matriz_etiquetas(dataset):
    lista=[]
    for caso in dataset:
        labels=caso['labels']
        lista.append(labels)
    matriz=torch.stack(lista)

    return matriz

def view_ids(dataset):
    lista=[]
    for caso in dataset:
        v_id=caso['view_id']
        lista.append(v_id)

    return lista
        
    
 
def my_collate_fn(data): # Crear batch a partir de lista de casos
    '''
    images: tensor de batch_size x num_channels_in x height x width
    labels: tensor de batch_size x num_classes
    view_ids: lista de batch_size elementos 
    fruit_ids: lista de batch_size elementos 
    '''
    images = [d[0] for d in data]
    images = torch.stack(images, dim = 0) # tendra dimensiones numvistastotalbatch, 3,250,250
    
    labels = [d[1] for d in data]
    if labels[0] is not None:
        labels = torch.stack(labels,dim=0)
    #labels = torch.tensor(labels).long()
    #labels es una lista con tantos elementos como batch_size
    # Cada elemento
    
    view_ids = [d[2] for d in data]
    fruit_ids = [d[3] for d in data]  
    return { 
        'images': images, 
        'labels': labels,
        'view_ids': view_ids,
        'fruit_ids': fruit_ids
    }
 


class ViewDataModule(pl.LightningDataModule):
    def __init__(self, 
                 training_path = None, 
                 train_dataplaces=None,
                 val_dataplaces=None,
                 suffixes=None,
                 normalization_means=None,
                 normalization_stds=None,
                 defect_types=None,              
                 predict_path = None , 
                 predictions_file_list=None,
                 batch_size =30, 
                 num_workers=-1,
                 max_value=1024,
                 crop_size=(120,120),
                 delimiter='_',
                 carga_mask=True,
                 multilabel=True,
                 in_memory=True,

                 **kwargs):
        '''
        self.train_set_categories es una lista de nombres de ficheros
        Cada fichero corresponde a una categoria
        Los ficheros contienen lineas del tipo

        fich.cimg, 2,3,7

        Indicando el nombre del fichero y qué vistas tomar
        
        base_path: Directorio base para entrenamiento y validacion
        annotations: Subdirectorio del anterior donde estan los json
          Las diferentes etiquetas serán la unión de los sets de defectos de todos los JSON
        images: Subdirectorio de base_path que contine las imagenes. Las imagenes son pngs de 16 bits uno por canal. 
          El nombre del json es el prefijo para las imágenes. 
        suffixes: diferente para cada canal, _R.png, _G.png, _B.png

        Si los parámetros de normalización son None, se estiman dichos parámetros a partir del dataset de training

        predict_set_folder: carpeta conteniendo Ficheros json vacíos. 

        defect_types: O bien se le pasan o si no por defecto los determina él a partir de los JSONS
        '''
        super().__init__()

  

        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers >= 0 else multiprocessing.cpu_count()-1

        print('DataLoader Num Workers: ',self.num_workers)
        
        self.set_defect_types=defect_types,
        self.base_path=training_path
        self.medias_norm = normalization_means
        self.stds_norm = normalization_stds
        
        self.multilabel=multilabel
        
        self.predict_path=predict_path
        self.predictions_filelist=predictions_file_list
        self.delimiter=delimiter
        self.crop_size=crop_size
        self.carga_mask=carga_mask
        
        if train_dataplaces is not None:
            self.trainset,_,self.tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(training_path,
                dataplaces=train_dataplaces, 
                sufijos=suffixes,
                max_value=max_value, 
                prob_train=1.0,
                crop_size=crop_size,
                defect_types=defect_types,
                multilabel=multilabel,
                splitname_delimiter=delimiter,
                in_memory=in_memory,
                carga_mask=self.carga_mask)
        if val_dataplaces is not None:
            self.valset,_,self.tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(training_path,
                dataplaces=val_dataplaces, 
                sufijos=suffixes,
                max_value=max_value, 
                prob_train=1.0,
                crop_size=crop_size,
                defect_types=defect_types,
                multilabel=multilabel,
                splitname_delimiter=delimiter,
                in_memory=in_memory,
                carga_mask=self.carga_mask)
            
        if train_dataplaces is None or val_dataplaces is None:
            print("No train dataplace or val dataplace")
            self.trainset=None
            self.valset=None
            self.train_dataset = None
            self.val_dataset = None
        
        if self.medias_norm is None or self.stds_norm is None:
            self.medias_norm, self.stds_norm=m_dataLoad_json.calcula_media_y_stds(self.trainset,self.crop_size)
            print('He calculado parametros de normalizacion')
            print(f'Medias: {self.medias_norm}')
            print(f'Stds: {self.stds_norm}')

 

        transform_geometry= transforms.Compose([   
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, shear=15, scale=(0.7, 1.1),translate=(0.15,0.15)),
        ])
        transform_intensity_rgb= transforms.Compose([
            transforms.ColorJitter(brightness=(0.8,1.4), hue=0.01,contrast=(0.8,1.55),saturation=0.15)            
            ])
        transform_intensity= transforms.Compose([
            transforms.ColorJitter(brightness=(0.8,1.4),contrast=(0.8,1.55))            
            ])
        transform_normalize=transforms.Compose([transforms.Normalize(self.medias_norm, self.stds_norm),
                                                ])
    
        
        transform_train=Aumentador_Imagenes(transform_geometry,
                                                       transform_intensity_rgb,transform_intensity,transform_normalize)

        self.transform_val = Aumentador_Imagenes(transforms.CenterCrop(self.crop_size),
                                                       None,None,transform_normalize)


        if self.trainset is not None:
            self.train_dataset = OliveViewsDataSet(dataset=self.trainset, transform = transform_train,carga_mask=self.carga_mask)           
            self.val_dataset = OliveViewsDataSet(dataset=self.valset, transform = self.transform_val,carga_mask=self.carga_mask) 
            
        
        if self.predict_path is not None:
            self.predict_dataset = FileNamesDataSet( root_folder= self.predictions_base_path, filenames_list = self.predictions_filelist, 
                                                 transform = self.transform_val, field_delimiter=self.delimiter)
        else:
            self.pred_dataset=None
                 
    
        if self.trainset is not None: # Si estamos en prediccion no lo hago
            self.matriz_casos_train=matriz_etiquetas(self.trainset)
            self.matriz_casos_val=matriz_etiquetas(self.valset)
            self.view_ids_train=view_ids(self.trainset)
            self.view_ids_val=view_ids(self.valset)

        if self.train_dataset is not None:
            print(f"len total trainset =   {len(self.train_dataset )}")

        if self.val_dataset is not None:
            print(f"len total valset =   {len(self.val_dataset )}")
        if self.predictions_filelist is not None:
            print(f"len total predset =   {len(self.predictions_filelist)}")
       

        print("batch_size in ViewDataModule", self.batch_size)
        
    def get_len_trainset(self):
        return len(self.train_dataset)
    
    def get_num_classes(self):
        return len(self.set_defect_types)  

    def get_viewids_train(self):
        return self.view_ids_train

    def get_viewids_val(self):
        return self.view_ids_val


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        return None

    def train_dataloader(self):
        print("batch_size in Dataloader train", self.batch_size)
        misampler=sampler.Balanced_BatchSamplerMultiLabel(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.batch_size,  sampler = misampler,  num_workers=self.num_workers, collate_fn=my_collate_fn)
        
    def val_dataloader(self):
        print("batch_size in Dataloader val", self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn)

    
    def predict_dataloader(self):
        print("batch_size in predict data loader", self.batch_size)
        return DataLoader(self.pred_dataset , batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate_fn)



    
