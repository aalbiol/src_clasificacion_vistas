
import warnings
from pathlib import Path
from argparse import ArgumentParser
import argparse
import os

import yaml
import json

warnings.filterwarnings('ignore')


import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)
print("PATHS:",sys.path)

# torch and lightning imports
import torch
torch.set_float32_matmul_precision('medium')
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import  ModelCheckpoint

#from m_finetuning import FeatureExtractorFreezeUnfreeze

from dataLoad import ViewDataModule

#from pl_modulo import OliveClassifier


from m_dataLoad_json import write_list

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.



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


    # training_path = config['root_folder']
    # print('Config Root Folder:',training_path)
    # user = os.getenv('USER') or os.getenv('USERNAME')


    # print('Training Root Folder:',training_path)

    # suffixes=['_r.png','_g.png','_b.png','_a.png','_nir.png','_nir1.png','_nir2.png','_nir4.png']
    # num_channels_in=len(suffixes)
    # max_values=config['max_value']
    # crop_size=(config['crop_size'],config['crop_size'])
    
    
  
    # tipos_defecto=config['defect_types']

    # tipos_defecto = [str(d) for d in tipos_defecto]

    # train_dataplaces=config['train_dataplaces']
    # val_dataplaces=config['val_dataplaces']

    # print('In memory: ', config['in_memory'])

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

    print('Loading images...')
    datamodule = ViewDataModule(
        training_path = root_folder, 
		 train_dataplaces=train_dataplaces,
         val_dataplaces=val_dataplaces,
                 suffixes=terminaciones,
                 defect_types=tipos_defecto,              
                 predict_path = None , 
                 batch_size =batch_size,    
                 #num_workers=config['num_workers'],
                 normalization_means=None, # Para entrenar estimamos valores de normalizacion
                 normalization_stds=None,             
                 max_value=maxvalues,              
                 crop_size=crop_size,
                 delimiter=delimiter,
                 carga_mask=False, # En clasificación no se cargan las máscaras
                 in_memory=in_memory)
    print('... done!')

   

    sys.exit()

    medias_norm=datamodule.medias_norm.tolist()
    stds_norm=datamodule.stds_norm.tolist()
    print('medias_norm:',medias_norm)
    print('stds_norm:',stds_norm)
    




    model=OliveClassifier(num_channels_in,
                lr = config['learning_rate'],
                class_names=tipos_defecto,
                weight_decay=config['weights_decay'],
                mixup_alpha=config['mixup_alpha'],
                label_smoothing=config['label_smoothing'],
                warmup_iter=config['warmup'],
                p_dropout=config['p_dropout'])
        
    # Continuar entrenamiento a partir de un punto
    if config['initial_model'] is not None:
        checkpoint = torch.load(config['initial_model'])
        model.load_state_dict(checkpoint['state_dict'])



    # Instantiate lightning trainer and train model
    miwandb= WandbLogger(name=config['log_name'], project=config['wandb_project'],config=config)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
   
    trainer_args = {'max_epochs': config['num_epochs'], 'logger' : miwandb}
    
    
    print('num_epochs:',config['num_epochs'])
    

    callbacks3=[FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['unfreeze_epoch'],initial_denom_lr=2),
                #ModelCheckpoint(monitor='val_loss',dirpath='.',filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',save_top_k=1),
                lr_monitor]
    
    trainer = pl.Trainer(callbacks=callbacks3,**trainer_args)
    
      
    trainer.fit(model, datamodule=datamodule)
    
    os.system('wandb sync --clean')
    
    # Save trained model


    model_name=config['model_name']
    save_path = os.path.join(config['save_path'],model_name) if config['save_path'] is not None else  model_name
    trainer.save_checkpoint(save_path)
    
    dict_norm={'medias_norm': medias_norm,
    'stds_norm': stds_norm }
    print('Saving normalization_last_train.json')
    with open('normalizacion_last_train.json', 'w') as outfile:
        json.dump(dict_norm,outfile,indent=4)
