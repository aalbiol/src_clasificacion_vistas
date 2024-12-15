
import warnings
warnings.filterwarnings('ignore')
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)

# torch and lightning imports
import torch
import pytorch_lightning as pl
import numpy as np
import math

from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from typing import Tuple,Any

import os
from PIL import Image
import sampler
from transformaciones import Aumentador_Imagenes


from dataset import ViewsDataSet,CImgListDataSet
import pandas as pd
import json
import glob

from tqdm import tqdm

import pycimg

from dataset import lee_vista

def remove_sufix(filename,delimiter):
    '''
    Dado un nombre de fichero, 
    obtiene el basename
    elimina el _xx.png

    '''
    
    id=filename.split(delimiter)

    id=id[:-1]
    id=delimiter.join(id)

    return id

def parse_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def extract_tipos_defecto(d):
    anot =d['annotations']
    tipos_defecto=list(anot.keys())
    tipos_defecto= [e.lower() for e in tipos_defecto]
    return tipos_defecto

def extract_one_hot(d,tipos_defecto):
    anot =d['annotations']
    anot2={}
    for k,v in anot.items():
        anot2[k.lower()]=v
    anot.update(anot2)
    v=[]
    for defecto in tipos_defecto:
        if defecto in anot:
            tmp=anot[defecto]
            if isinstance(tmp,str):
                tmp=float(tmp)
            if tmp <0 :
                tmp=math.nan
            v.append(tmp)
        else:
            v.append(math.nan)
    return torch.tensor(v)

# def split_train_val(lista,p):
#     n1=int(len(lista)*p)
#     train=lista[:n1]
#     val=lista[n1:]
#     return train,val

def fruit_id(filename,delimiter):
    '''
    Dado un nombre de fichero, 
    obtiene el basename
    elimina el _xx.json

    '''
    basename=os.path.basename(filename)
    id=basename.split(delimiter)

    id=id[:-1]
    id=delimiter.join(id)

    return id


def fruit_id_MIL(filename):
    '''
    Dado un nombre de fichero, 
    obtiene el basename
    elimina el _xx.json

    '''
    basename=os.path.basename(filename)
    id=os.path.splitext(basename)[0]
 

    return id

def view_id(filename):
    '''
    Dado un nombre de fichero, 
    obtiene el basename
    elimina el _xx.json

    '''
    basename=os.path.basename(filename)
    id=basename.split('.')
    return ''.join(id[:-1])

def add_good_category(onehot):
     v= (1 if onehot.sum()==0 else 0)
     return torch.concat((torch.tensor(v).unsqueeze(0),onehot))


def genera_ds_jsons_multilabel(root,  dataplaces, sufijos=None,max_value=255, prob_train=0.7,crop_size=(120,120),defect_types=None,splitname_delimiter='-',
                               multilabel=True, in_memory=True, carga_mask=True):
    '''
    dataplaces: lista de tuplas 
      Si la tupla tiene dos elementos, el primero es el directorio de jsons y el segundo el de imágenes La lista de jsons se obtiene con glob
      Si la tupla tiene 3 elementos, el primero es la lista de jsons, el segundo el directorio de jsons, y el tercero el directorio de imágenes
    '''
    assert sufijos is not None

    assert sufijos is not None
    json_files=[]
    imags_directorio=[]
    for place in dataplaces: # Si el dataplace tiene dos elementos (jsons)
        if len(place)==2:
            anotaciones = place[0]
            imagenes=place[1]
            anot_folder=os.path.join(root,anotaciones)
            imags_folder=os.path.join(root,imagenes)
            fichs=glob.glob('*.json',root_dir=anot_folder)
            ficheros=[os.path.join(anot_folder,f) for f in fichs]
            

        if len(place)==3:
            lista_filename=place[0]
            anotaciones = place[1]
            imagenes=place[2]
            anot_folder=os.path.join(root,anotaciones)
            imags_folder=os.path.join(root,imagenes)
            lista_filename=os.path.join(anot_folder,lista_filename)
            #Leer lista_filename
            with open(lista_filename) as f:
                fichs = f.readlines()
            fichs=[f.strip() for f in fichs]
            ficheros=[os.path.join(anot_folder,f) for f in fichs]
 
        json_files +=ficheros
        imagenes= [imags_folder]*len(ficheros)
        imags_directorio +=imagenes


    if defect_types is None:
        tipos_defecto=set()
        for json_file in json_files:
            d=parse_json(json_file)
            defects=extract_tipos_defecto(d)
            defects=set(defects)   
            tipos_defecto = tipos_defecto.union(defects)
       
        tipos_defecto=list(tipos_defecto)
        tipos_defecto.sort()
        print('Tipos Defecto de JSONS:', tipos_defecto)
    else:
        tipos_defecto=defect_types
        print('Tipos de Defecto por configuracion:', tipos_defecto)
            
    df=pd.DataFrame()
    df['json_files']=json_files
    df['fruit_ids']=[fruit_id(a,splitname_delimiter) for a in df['json_files']]
    df['view_ids']=[view_id(a) for a in df['json_files']]
    df['imags_folder']=imags_directorio


    frutos =pd.Series(list(set(df['fruit_ids'])))
    train=frutos.sample(frac=prob_train)
    val=frutos.drop(train.index)
    dftrain=df[df['fruit_ids'].isin( list(train) )]
    dfval=df[df['fruit_ids'].isin( list(val) )]


    trainset=[]

    for k in tqdm(range(len(dftrain))):
        json_file=dftrain.iloc[k]['json_files']
        f_id=dftrain.iloc[k]['fruit_ids']
        d=parse_json(json_file)
        v_id=dftrain.iloc[k]['view_ids']
        imags_folder= dftrain.iloc[k]['imags_folder']
        if in_memory:
            channels=lee_vista(imags_folder,v_id,sufijos,max_value=max_value,carga_mask=carga_mask)
        else:
            channels=None
        onehot=extract_one_hot(d,tipos_defecto)
        if multilabel==False and onehot.sum()> 1: #skip instances with multiple labels if multiclass
            continue
        if multilabel==False:
            onehot = add_good_category(onehot)
        dict_vista={'fruit_id':f_id, 'view_id':v_id, 'image': channels, 'labels': onehot, 
                     'imag_folder': imags_folder, 'sufijos': sufijos, 'max_value':max_value, 'crop_size':crop_size} 
        trainset.append(dict_vista)

    valset=[]


    for k in tqdm(range(len(dfval))):
        json_file=dfval.iloc[k]['json_files']
        f_id=dfval.iloc[k]['fruit_ids']
        d=parse_json(json_file)
        v_id=dfval.iloc[k]['view_ids']
        imags_folder= dfval.iloc[k]['imags_folder']
        if in_memory:
            channels=lee_vista(imags_folder,v_id,sufijos,max_value=max_value)
        else:
            channels=None
        onehot=extract_one_hot(d,tipos_defecto)
        if multilabel==False and onehot.sum()> 1: #skip instances with multiple labels if multiclass
            continue
        if multilabel==False:
            onehot = add_good_category(onehot)
        dict_vista={'fruit_id':f_id, 'view_id':v_id, 'image': channels, 'labels': onehot ,
                'imag_folder': imags_folder, 'sufijos': sufijos, 'max_value':max_value, 'crop_size':crop_size}
        valset.append(dict_vista)
    
    if multilabel ==False:
        tipos_defecto.insert(0,'bueno')        
    return trainset,valset,tipos_defecto
        



def genera_ds_jsons_multilabelMIL(root,  dataplaces, sufijos=None,maxvalue=255, defect_types=None, in_memory=True,channel_list=None):

    json_files=[]
    imags_directorio=[]

    for place in dataplaces:
        
        lista_filename=place[0]
        anotaciones = place[1]
        imagenes=place[2]
        anot_folder=os.path.join(root,anotaciones)
        imags_folder=os.path.join(root,imagenes)
        lista_filename=os.path.join(anot_folder,lista_filename)        
        with open(lista_filename) as f:
            fichs = f.readlines()
        fichs=[f.strip() for f in fichs]
        
        ficheros=[os.path.join(anot_folder,f) for f in fichs]
        
        json_files +=ficheros#FullPath
        imagenes= [imags_folder]*len(fichs)
        imags_directorio += imagenes


    if defect_types is None:
        tipos_defecto=set()
        for json_file in json_files:
            d=parse_json(json_file)
            defects=extract_tipos_defecto(d)
            defects=set(defects)   
            tipos_defecto = tipos_defecto.union(defects)
       
        tipos_defecto=list(tipos_defecto)
        tipos_defecto.sort()
        print('Tipos Defecto de JSONS:', tipos_defecto)
    else:
        tipos_defecto=defect_types
        print('Tipos de Defecto por configuracion:', tipos_defecto)
    
    out=[]
    for fruto in zip(json_files,imags_directorio):
        jsonfile=fruto[0]
        imags_folder=fruto[1]
        d=parse_json(jsonfile)
        onehot=extract_one_hot(d,tipos_defecto)
        fruitid=fruit_id_MIL(jsonfile)
        # json_sin_ext=os.path.splitext(jsonfile)[0]
        # json_sin_ext_sin_dir=os.path.basename(json_sin_ext)
        # fruit_id=json_sin_ext_sin_dir
        if in_memory:
            nombre_cimg=os.path.join(imags_folder,fruitid)
            nombre_cimg += ".cimg"        
            vistas = pycimg.cimglistread_torch(nombre_cimg,maxvalue,channel_list=channel_list) # lista de tensores normalizados en intensidad 
        else:
            vistas=None
        dict_fruto={'fruit_id':fruitid, 'image': vistas, 'labels': onehot, 
                     'imag_folder': imags_folder,  'max_value':maxvalue}
        out.append(dict_fruto)     
            
    
    return out,tipos_defecto
        



# ===================================== NORMALIZACION ==================
# 
def calcula_media_y_stds(trainset,crop_size=None):
    
    print("Estimando medias y stds...\n")
    medias=[]
    medias2=[]
    
    pix_dimensions=(1,2)
    area_total=0
    for caso in tqdm(trainset):
        img=caso['image']
        view_id=caso['view_id']
        if img is None: #Cuando no está en memoria
            imags_folder=caso['imag_folder']
            sufijos=caso['sufijos']
            max_value=caso['max_value']
            
            #print("Reading ", view_id)
            img=lee_vista(imags_folder,view_id,sufijos,max_value=max_value,carga_mask=True)
        
        if crop_size is not None:
            image=transforms.functional.center_crop(img,crop_size)
        else:
            image=img
        
        
        mask=(image[0]>0)
        area=torch.sum(mask)
        
        suma=torch.sum(image,axis=pix_dimensions)
        image2=image*image
        suma2=torch.sum(image2,axis=pix_dimensions)
        area_total +=area
        #print('Area util :', area)


        medias.append(suma)
        medias2.append(suma2)

    #print('medias:',medias) 
    medias=torch.stack(medias).sum(axis=0)/area_total
    medias2=torch.stack(medias2).sum(axis=0)/area_total
    stds=np.sqrt(medias2 - medias*medias)

    return medias,stds



def calcula_media_y_stds_MIL(trainset):
    suma=0
    suma2=0
    npixels=0
    
    pix_dimensions=(1,2) # El eje 0 es el color
    area_total=0
    for caso in trainset:
        vistas=caso['image']
        fruit_id=caso['fruit_id']
        maxvalue=caso['max_value']
        if vistas is None: #Cuando no está en memoria
            imags_folder=caso['imag_folder']
            nombre_cimg=os.path.join(imags_folder,fruit_id)
            nombre_cimg += ".cimg"        
            vistas = pycimg.cimglistread_torch(nombre_cimg,maxvalue) # lista de tensores normalizados en intensidad 
            
        for v in vistas:
            suma +=torch.sum(v,axis=pix_dimensions)
            suma2 +=torch.sum(v*v,axis=pix_dimensions)
            npixels += v.shape[pix_dimensions[0]]* v.shape[pix_dimensions[1]]
            

    #print('medias:',medias) 
    medias= suma / npixels
    medias2=suma2 / npixels
    stds=np.sqrt(medias2 - medias*medias)

    return medias.tolist(),stds.tolist()




def matriz_etiquetas(dataset):
    lista=[]
    for caso in dataset:
        labels=caso['labels']
        lista.append(labels)
    matriz=torch.stack(lista)

    print(f"** Info. Núm de instancias= {matriz.shape[0]}")
    print(" ** Numero de casos por categoria:" , torch.sum(matriz,axis=0))
    return matriz
        
    

# def matriz_etiquetas(dataset):
#     lista=[]
#     for caso in dataset:
#         labels=caso['labels']
#         lista.append(labels)
#     matriz=torch.stack(lista)
#     return matriz

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
 
def my_collate_fn_MIL(data): # Genera un batch a partir de una lista de frutos
    
    images = [d[0] for d in data]
    images = torch.concat(images, axis = 0) # tendra dimensiones numvistastotalbatch, 3,250,250
    
    nviews = [d[0].shape[0] for d in data] # contiene (nviews0, nviews1,,... ) con tantos elementos como frutos tenga el batch
    # Sirve para poder trocear luego por frutos
    
    labels = [d[1] for d in data]
    labels = torch.stack(labels,axis=0) #(5)

    paths = [d[2] for d in data]
    return { #(6)
        'images': images, 
        'label': labels,
        'nviews': nviews,
        'paths': paths
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
                 training_size=(120,120),
                 delimiter='_',
                 carga_mask=True,
                 multilabel=True,
                 in_memory=True,
                 augmentation=None,
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

  
        assert augmentation is not None
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
        self.augmentation=augmentation
        self.training_size=training_size
        
        if train_dataplaces is not None:
            if in_memory:
                print("Reading  Training jsons labels and images ...")
            else:
                print("Reading  Training jsons labels ...")
            self.trainset,_,self.tipos_defecto=genera_ds_jsons_multilabel(training_path,
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
            if in_memory:
                print("Reading  Val jsons labels and images ...")
            else:
                print("Reading  Val jsons labels ...")            
            self.valset,_,self.tipos_defecto=genera_ds_jsons_multilabel(training_path,
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
            print("Calculando parametros de normalizacion...")
            self.medias_norm, self.stds_norm=calcula_media_y_stds(self.trainset,self.crop_size)
            print('He calculado parametros de normalizacion')
            print(f'Medias: {self.medias_norm}')
            print(f'Stds: {self.stds_norm}')

        transform_normalize=transforms.Compose([transforms.Normalize(self.medias_norm, self.stds_norm),
                                                ])
        augmentation=self.augmentation
        if self.crop_size is not None:
            transform_geometry= transforms.Compose([   
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(augmentation['random_rotation']),
            transforms.RandomAffine(degrees=augmentation['affine']['degrees'], shear=augmentation['affine']['shear'], 
                                    scale=augmentation['affine']['scale'],translate=augmentation['affine']['translate']
                                    ),
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.training_size)
            ])
        else:
            transform_geometry= transforms.Compose([   
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(augmentation['random_rotation']),
            transforms.RandomAffine(degrees=augmentation['affine']['degrees'], shear=augmentation['affine']['shear'], 
                                    scale=augmentation['affine']['scale'],translate=augmentation['affine']['translate']
                                    ),
            transforms.Resize(self.training_size)
            ])



        transform_intensity_rgb= transforms.Compose([
            transforms.ColorJitter(brightness=augmentation['brightness'], hue=augmentation['hue'],contrast=augmentation['contrast'],saturation=augmentation['contrast'])            
            ])
        transform_intensity= transforms.Compose([
            transforms.ColorJitter(brightness=augmentation['brightness'],contrast=augmentation['contrast'])            
            ])
    

        transform_train=Aumentador_Imagenes(transform_geometry,
                                                    transform_intensity_rgb,transform_intensity,transform_normalize)
        # transform_val = Aumentador_Imagenes(transforms.CenterCrop(self.crop_size),
        #                                             None,None,transform_normalize)            
        if self.crop_size is not None:             
            tamanyo=transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.training_size)])

            transform_val = Aumentador_Imagenes(tamanyo,                                                
                                                    None,None,transform_normalize)            
        else:
            transform_val = Aumentador_Imagenes(transforms.Resize(self.training_size),
                                                    None,None,transform_normalize)

        if self.trainset is not None:
            self.train_dataset = ViewsDataSet(dataset=self.trainset, transform = transform_train,carga_mask=self.carga_mask)           
            self.val_dataset = ViewsDataSet(dataset=self.valset, transform = transform_val,carga_mask=self.carga_mask) 
            
        
        if self.predict_path is not None:
            self.predict_dataset = FileNamesDataSet( root_folder= self.predictions_base_path, filenames_list = self.predictions_filelist, 
                                                 transform = transform_val, field_delimiter=self.delimiter)
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



## Cuando las vistas de un fruto están almacenadas en CIMGLists y las anotaciones en JSONS arandanos
class JSONSCImgDataModule(pl.LightningDataModule):
    def __init__(self, 
                 root_path=None,
                train_dataplaces=None,
                val_dataplaces=None,
                pred_dataplaces=None,                
                batch_size=5,
                num_workers = -1,
                defect_types = None,
                maxvalue=255,
                normalization_means=None, #if none calculate from training data
                normalization_stds=None,
                in_memory=True,
                imagesize=(112,112),
                crop_size=None,
                channel_list=[0,1,2],
                augmentation=None,
                  **kwargs):
        super().__init__()

        assert augmentation is not None    
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()-1
        
        self.tipos_defecto=defect_types,
        self.root_path=root_path
        self.medias_norm = normalization_means
        self.stds_norm = normalization_stds
        self.target_image_size =imagesize
        self.train_dataplaces=train_dataplaces
        self.val_dataplaces=val_dataplaces
        self.pred_dataplaces=pred_dataplaces
        self.root_path=root_path
        self.imagesize=imagesize
        self.training_size=imagesize
        self.channel_list=channel_list
        self.augmentation = augmentation
        self.crop_size=crop_size
        
        self.trainset =None
        self.valset = None
        self.maxvalue=maxvalue
        if self.train_dataplaces is not None:
            self.trainset,self.tipos_defecto=genera_ds_jsons_multilabelMIL(self.root_path, 
                                                                                    dataplaces=self.train_dataplaces, 
                                                                                    maxvalue=maxvalue,
                                                                                    defect_types=defect_types,
                                                                                    in_memory=in_memory,
                                                                                    channel_list=self.channel_list)
 
        if self.val_dataplaces is not None:
            self.valset,self.tipos_defecto=genera_ds_jsons_multilabelMIL(self.root_path, 
                                                                                    dataplaces=self.val_dataplaces, 
                                                                                    maxvalue=maxvalue,
                                                                                    defect_types=defect_types,in_memory=in_memory,
                                                                                    channel_list=self.channel_list)
        
        if self.medias_norm is None or self.stds_norm is None:
            print('\n *** INFO: Estimando medias y varianzas normalizacion')
            self.medias_norm,self.stds_norm=calcula_media_y_stds_MIL(self.trainset)
            print('     ** Estimated medias_norm:', self.medias_norm)
            print('     ** Estimated stds_norm:', self.stds_norm)
            

        transform_normalize=transforms.Compose([transforms.Normalize(self.medias_norm, self.stds_norm) ])
            
        augmentation=self.augmentation
        if self.crop_size is not None:
            transform_geometry= transforms.Compose([   
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(augmentation['random_rotation']),
            transforms.RandomAffine(degrees=augmentation['affine']['degrees'], shear=augmentation['affine']['shear'], 
                                    scale=augmentation['affine']['scale'],translate=augmentation['affine']['translate']
                                    ),
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.training_size)
            ])
        else:
            transform_geometry= transforms.Compose([   
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(augmentation['random_rotation']),
            transforms.RandomAffine(degrees=augmentation['affine']['degrees'], shear=augmentation['affine']['shear'], 
                                    scale=augmentation['affine']['scale'],translate=augmentation['affine']['translate']
                                    ),
            transforms.Resize(self.training_size)
            ])



        transform_intensity_rgb= transforms.Compose([
            transforms.ColorJitter(brightness=augmentation['brightness'], hue=augmentation['hue'],contrast=augmentation['contrast'],saturation=augmentation['contrast'])            
            ])
        transform_intensity= transforms.Compose([
            transforms.ColorJitter(brightness=augmentation['brightness'],contrast=augmentation['contrast'])            
            ])
    

        transform_train=Aumentador_Imagenes(transform_geometry,
                                                    transform_intensity_rgb,transform_intensity,transform_normalize)
        
        if self.crop_size is not None:             
            tamanyo=transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.training_size)])

            transform_val = Aumentador_Imagenes(tamanyo,                                                
                                                    None,None,transform_normalize)            
        else:
            transform_val = Aumentador_Imagenes(transforms.Resize(self.training_size),
                                                    None,None,transform_normalize)
        
        if defect_types is None:
            print(f"JSONSCImgDataModule tipos defecto desde JSONS: {self.tipos_defecto}")
        else:
            print(f"JSONSCImgDataModule tipos defecto desde configuración: {self.tipos_defecto}")
            

        
            
        self.numlabels=len(self.tipos_defecto)           

        if self.train_dataplaces is not None:
                self.train_dataset=CImgListDataSet(dataset=self.trainset,transform=transform_train,channel_list=self.channel_list)  
                
        if self.val_dataplaces is not None:            
            self.val_dataset=CImgListDataSet(dataset=self.valset,transform=transform_val,channel_list=self.channel_list)

            
        print(f"JSONSCImgDataModule num labels = {self.numlabels}")
        if self.train_dataplaces is not None:
            print(f"JSONSCImgDataModule len total trainset =   {len(self.trainset )}")
        if self.val_dataplaces is not None:
            print(f"JSONSCImgDataModule len total valset =   {len(self.valset )}")
            
            

       

        print("batch_size in JSONSCImgDataModule ", self.batch_size)
        

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        return None
    
     
    def train_dataloader(self):
        print("batch_size in Dataloader train", self.batch_size)
        misampler=sampler.Balanced_BatchSamplerMultiLabel(self.train_dataset)   
        #return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=my_collate_fn)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=misampler, num_workers=self.num_workers, collate_fn=my_collate_fn_MIL)

    def val_dataloader(self):
        print("batch_size in Dataloader val", self.batch_size)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False,shuffle=False, collate_fn=my_collate_fn_MIL)
    # def predict_dataloader(self):
    #     print("batch_size in predict data loader", self.batch_size)
    #     return DataLoader(self.val_dataset , batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=my_collate_fn)





    @staticmethod
    def add_model_specific_args(parser):
        #parser = parent_parser.add_argument_group("model")
        #parser.add_argument("--data.train_set_csv", type=str, default='text_files/train_set_articulo.csv')
        #parser.add_argument("--data.test_set_csv", type=str, default='text_files/test_set_articulo.csv')
        return parser

      
