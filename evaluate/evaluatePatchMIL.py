
import warnings
from pathlib import Path
from argparse import ArgumentParser
import argparse
import os

import yaml
import json
import pickle

warnings.filterwarnings('ignore')


import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)
sys.path.append(current_file_dir)
# print("PATHS:",sys.path)

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import  ModelCheckpoint
import multiprocessing
import base64
from PIL import Image
import io

from tqdm import tqdm
from pl_datamodule import ViewDataModule, my_collate_fn

import pl_datamodule
import torch.nn.functional as F
from dataset import ViewsDataSet
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from metrics import calculate_auc_multilabel
from pl_patch_MIL_modulo import PatchMILClassifier

from datetime import datetime


#from m_dataLoad_json import write_list

# Función para serializar tensores
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):  # Si es un tensor de PyTorch
        return obj.tolist()  # Convierte a lista
    raise TypeError(f"Objeto de tipo {type(obj).__name__} no es serializable.")


if __name__ == "__main__":
    # parser = ArgumentParser()

    # parser.add_argument("--config", default = "configs/train.yaml", help="""YAML config file""")

    # args = parser.parse_args()

    config_file = sys.argv[1]
    with open(config_file,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    
    
       # Comprobar si hay GPU
    cuda_available=torch.cuda.is_available()
    if cuda_available:
        cuda_count=torch.cuda.device_count()
        cuda_device=torch.cuda.current_device()
        tarjeta_cuda=torch.cuda.device(cuda_device)
        tarjeta_cuda_device_name=torch.cuda.get_device_name(cuda_device)
        print(f'Cuda Disponible. Num Tarjetas: {cuda_count}. Tarjeta activa:{tarjeta_cuda_device_name}\n\n')
        device='cuda'
        gpu=1
    else:
        device='cpu'
        gpu=0

    train_dataplaces = config['data']['train']
    val_dataplaces = config['data']['val']
    terminaciones=config['data']['terminaciones']
    print("Terminaciones:",terminaciones)
    root_folder=config['data']['root_folder']
    
    crop_size=config['data']['crop_size']
    
    batch_size=config['train']['batch_size']
    in_memory=config['train']['in_memory']
    maxvalues=config['data']['maxvalues']
    crop_size=config['data']['crop_size']
    tipos_defecto=config['model']['defect_types']
    
    
    model_dir=config['evaluate']['model_dir']
    model_file=config['evaluate']['model_file']
  
    
    #multilabel=config['model']['multilabel']

    out_dir=config['evaluate']['report_dir']

    
    if not os.path.exists(out_dir):
        print(f'No existe el directorio para report : {out_dir}. Se crea')
        Path( out_dir ).mkdir( parents=True, exist_ok=True )

   
    train_predictions=config['evaluate']['train_predictions']
    val_predictions=config['evaluate']['val_predictions']
    aucs_jsonfile=config['evaluate']['aucs']

    model_path=os.path.join(model_dir,model_file)
    
    model = PatchMILClassifier()
    model.load(model_path)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n** Device:",device)
    print("\n")
    
    training_date=str(model.training_date)
    
    print("\n\n====================================================================")
    print("  ******************** TRAINING SET *************************")
    print("====================================================================")        
  
    preds_train=[]
    targets_train=[]
    #ids_train=[]
    json_data=[]

    print('root_folder:',root_folder)
    
    dataset,tipos_defecto=pl_datamodule.genera_ds_jsons_multilabelMIL(root_folder,  train_dataplaces, maxvalue=maxvalues, defect_types=tipos_defecto, 
                                                                      terminacion=terminaciones,in_memory=False,channel_list=model.channel_list)
    print("Train dataset:",len(dataset))
    class_results=[]


    for caso in tqdm(dataset):
        #image=caso['image']
        targets=caso['labels']
        fruit_id=caso['fruit_id']
        #view_id=caso['view_id']
        imags_folder=caso['imag_folder']
        json_file=caso['json_file_full_path']
        #   print("Main Json file:",json_file)
        #view_id=os.path.join(imags_folder,view_id)
        nombre_cimg=caso['image_file_full_path']
        results=model.predict(nombre_cimg,device,include_images=False,json_file=json_file)

        result= results[0] #introducimos los frutos de 1 en 1
        #print("Result:",result) 
        #results=model.predict(image,device)
        tensor_probs=torch.tensor(result['probs_fruto_tensor'])
        preds_train.append(tensor_probs)



        targets_views=None
        if targets.ndim==1:
            targets_fruit=targets
        else:
            targets_fruit=targets.max(dim=0).values
            with open(json_file,'r') as f:
                data=json.load(f)
                targets_views=data['views_annotations']
        
        targets_train.append(targets_fruit.unsqueeze(dim=0))
        targets_fruit_dict={}
        for nclase in range(len(tipos_defecto)):
            targets_fruit_dict[tipos_defecto[nclase]]=int(targets_fruit[nclase].item())
        a_guardar={'filename': result['imgname'],
                            'scores':result['probs_fruto'],
                            'ground_truth': targets_fruit_dict,
                            'probs_vistas':result['probs_vistas']}
        if targets_views is not None:
            a_guardar['ground_truth_views']=targets_views
        
        json_data.append(a_guardar)

                        

    data_train={'train_results':json_data,
                'evaluation_date': str(datetime.now()),
                'training_date':training_date,
                'model_file':model_path}
    out_json_trainfile=os.path.join(out_dir,train_predictions)

    #print("Dataval:",data_val)
    print("Writing ",out_json_trainfile)
    with open(out_json_trainfile, 'w') as f:
        json.dump(data_train, f, indent=3)
    

    preds_train = torch.stack(preds_train)
    targets_train = torch.concat(targets_train)
    aucs_train=calculate_auc_multilabel(preds_train.cpu(),targets_train.cpu(),tipos_defecto)   

    preds_val=[]
    targets_val=[]
    #ids_train=[]
    json_data=[]
    print("\n\n====================================================================")
    print("  ******************** VALIDATION SET *************************")
    print("====================================================================")
    print('root_folder:',root_folder)
    #print('val_dataplaces:',train_dataplaces) 
    dataset,tipos_defecto=pl_datamodule.genera_ds_jsons_multilabelMIL(root_folder,  val_dataplaces, maxvalue=maxvalues, defect_types=tipos_defecto, 
                                                                      terminacion=terminaciones,in_memory=False,channel_list=model.channel_list)
    print("Val dataset:",len(dataset))
    class_results=[]
    for caso in tqdm(dataset):
        #image=caso['image']
        targets=caso['labels']
        fruit_id=caso['fruit_id']
        #view_id=caso['view_id']
        imags_folder=caso['imag_folder']
        json_file=caso['json_file_full_path']
        #view_id=os.path.join(imags_folder,view_id)
        nombre_cimg=caso['image_file_full_path']
        
        results=model.predict(nombre_cimg,device,include_images=False,json_file=json_file)
        result= results[0]
        #print("Result:",result) 
        #results=model.predict(image,device)
        preds_val.append(result['probs_fruto_tensor'])
        
        


        targets_views=None
        if targets.ndim==1:
            targets_fruit=targets
        else:
            targets_fruit=targets.max(dim=0).values
            with open(json_file,'r') as f:
                data=json.load(f)
                targets_views=data['views_annotations']
        targets_val.append(targets_fruit.unsqueeze(dim=0))
        
        targets_fruit_dict={}
        for nclase in range(len(tipos_defecto)):
            targets_fruit_dict[tipos_defecto[nclase]]=int(targets_fruit[nclase].item())
        a_guardar={'filename': result['imgname'],
                            'scores':result['probs_fruto'],
                            'ground_truth': targets_fruit_dict,
                            'probs_vistas':result['probs_vistas']}
        if targets_views is not None:
            a_guardar['ground_truth_views']=targets_views
        
        json_data.append(a_guardar)
                        


    data_val={'val_results':json_data,
              'evaluation_date': str(datetime.now()),
              'training_date':training_date,
              'model_file':model_path}
    
    out_json_valfile=os.path.join(out_dir,val_predictions)

    #print("Dataval:",data_val)
    print("Writing ",out_json_valfile)
    with open(out_json_valfile, 'w') as f:
        json.dump(data_val, f, indent=3)
         
    

    # preds_train = torch.concat(preds_train)
    # targets_train = torch.concat(targets_train)

    print("\n====================================================================")
    print("  ******************** AUCs *************************")
    print("====================================================================")
    
    preds_val = torch.stack(preds_val)
    targets_val = torch.concat(targets_val)    
    aucs_val=calculate_auc_multilabel(preds_val.cpu(),targets_val.cpu(),tipos_defecto)

    
    aucdata={'Train AUCs':aucs_train,
             'Val AUCs': aucs_val,
             'evaluation_date': str(datetime.now()),
             'training_date':training_date,
             'model_file':model_path}

    aucs_file=os.path.join(out_dir,aucs_jsonfile)
    print("Writing ",aucs_file)
    with open(aucs_file, 'w') as f:
        json.dump(aucdata, f, indent=3)
    
     


    
    