import os
import sys
import json
import glob
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import torch.nn.functional as F
import torchvision

from tqdm import tqdm


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

def lee_png16(filename,max_value):
    #print(f'Reading {filename}...')
    im=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if im.ndim ==3 :
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im=im.astype('float32')/max_value
    im[im>1]=1

    im=torch.tensor(im)
    if im.ndim ==2:# convertir de hxw --> 1 x h x w
        im=im.unsqueeze(0)
    else:
        im=im.permute((2,0,1))
    return im


def lee_vista(images_folder,view_id,terminaciones,max_value,carga_mask=True):
    #print("Reading ", view_id)
    nombre_base=os.path.join(images_folder,view_id)
    canales=[]

    assert isinstance(max_value,list), 'maxvalue tiene que ser una lista de tantos elementos como canales o una lista con un unico elemento que se emplea para todos los canales'
   
    if len(max_value) == len(terminaciones):
            max_values=max_value
    else:
        assert len(max_value)==1, "Es una lista que no tiene ni un solo elemento ni un numero de elementos igual al numero de canales"
        max_values= max_value*len(terminaciones)

    for k,t in enumerate(terminaciones):
        nombre=nombre_base+t
        canal=lee_png16(nombre,max_values[k])
        canales.append(canal)
    canales=torch.concat(canales,0)
    if carga_mask:
        term_mascara="_auxb1.png"
        nombre=nombre_base+term_mascara
        mascara=lee_png16(nombre,255)
        color_centro=mascara[0,mascara.shape[1]//2, mascara.shape[2]//2]
        mascara =((mascara==color_centro)*(mascara < 0.5)).float()

        #print(canales.shape,mascara.shape)
        canales *= mascara
        # plt.imshow(canales[:3,:,:].numpy().transpose((1,2,0)),clim=(0,0.25))
        # plt.show()
    
    return canales



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
                               multilabel=True, in_memory=True, carga_mask=False):
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
    if in_memory:
        print("Reading  Training jsons labels and images ...")
    else:
        print("Reading  Training jsons labels ...")
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
    if in_memory:
        print("Reading  Validation jsons labels and images ...")
    else:
        print("Reading  Validation jsons labels ...")

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
        


# =================== Salvar listas en jsons ============            
    
def write_list(a_list,filename):
    print("Started writing list data into a json file")
    with open(filename, "w") as fp:
        json.dump(a_list, fp)
        print(f"Done writing JSON data into {filename}")

# Read list to memory
def read_list(filename):
    # for reading also binary mode is important
    with open(filename, 'rb') as fp:
        n_list = json.load(fp)
        return n_list
    

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
            img=lee_vista(imags_folder,view_id,sufijos,max_value=max_value,carga_mask=False)
        
        if crop_size is not None:
            image=torchvision.transforms.functional.center_crop(img,crop_size)
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

