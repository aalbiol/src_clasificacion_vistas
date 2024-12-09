
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

common_folder = os.path.join(current_file_dir, "..\common")
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

import m_dataLoad_json
import torch.nn.functional as F
from dataset import ViewsDataSet
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from metrics import calculate_auc_multilabel
from pl_modulo import ViewClassifier


#from m_dataLoad_json import write_list

# Función para serializar tensores
def tensor_to_serializable(obj):
    if isinstance(obj, torch.Tensor):  # Si es un tensor de PyTorch
        return obj.tolist()  # Convierte a lista
    raise TypeError(f"Objeto de tipo {type(obj).__name__} no es serializable.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", default = "configs/train.yaml", help="""YAML config file""")

    args = parser.parse_args()


    with open(args.config,'r') as f:
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
    root_folder=config['data']['root_folder']
    
    crop_size=config['data']['crop_size']
    delimiter=config['data']['delimiter']
    batch_size=config['train']['batch_size']
    in_memory=config['train']['in_memory']
    maxvalues=config['data']['maxvalues']
    crop_size=config['data']['crop_size']
    tipos_defecto=config['model']['defect_types']
    normalization=config['train']['output']['normalization_file']
    model_path=os.path.join(config['train']['output']['path'],config['train']['output']['model_file'])
    normalization_path=os.path.join(config['train']['output']['path'],normalization)
    multilabel=config['model']['multilabel']

    out_dir=Path(config['evaluate']['output']['root_folder'])
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    out_file_dir=os.path.join(config['evaluate']['output']['root_folder'],config['evaluate']['output']['out_file'])


    with open(normalization_path, 'r') as infile:
        dict_norm=json.load(infile)


    num_channels_in=config['model']['num_channels_input']
    model=ViewClassifier(num_channels_in,
                class_names=tipos_defecto,
                p_dropout=config['train']['p_dropout'],
                normalization_dict=dict_norm)
    
            
    # Cargar modelo entrenado
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
  
    print("Tipos defecto:",tipos_defecto)
  

    preds_train=[]
    targets_train=[]
    ids_train=[]
    json_data=[]
    for dataplace in train_dataplaces:
            _,dataset,tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(root_folder,  train_dataplaces, sufijos=terminaciones,max_value=maxvalues, prob_train=0.0,crop_size=crop_size,
                                                                               defect_types=tipos_defecto,splitname_delimiter=delimiter,multilabel=multilabel, in_memory=in_memory, carga_mask=True)
            class_results=[]
            for caso in dataset:
                image=caso['image']
                targets=caso['labels']
                fruit_id=caso['fruit_id']
                results=model.evaluate(image,device)
                preds_train.append(results[0]['probs'])
                targets_train.append(targets.unsqueeze(dim=0))
                for nclase in range(results[0]['probs'].shape[1]):
                    class_results.append({'name': tipos_defecto[nclase],
                                        'score':results[0]['probs'][0][nclase],
                                        'ground_truth': targets[nclase]})
                json_data.append({'filename': fruit_id,
                                  'class_results':class_results})
                                #   'image': image})


    data={'results':json_data}
    

    preds_train = torch.concat(preds_train)
    targets_train = torch.concat(targets_train)

    aucs=calculate_auc_multilabel(preds_train,targets_train,tipos_defecto)

    data.update({'AUC': aucs})

    with open(out_file_dir, 'wb') as f:
        pickle.dump(data, f)
    out_file_dir_json=out_file_dir.replace('pkl','json')
    with open(out_file_dir_json, 'wb') as f:
        json.dump(data, f, indent=4)


    system.exit()
    