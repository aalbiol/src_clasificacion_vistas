
import warnings
from pathlib import Path
from argparse import ArgumentParser
import argparse
import os
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch

import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger


from dataLoad import ViewDataModule

from pl_modulo import OliveClassifier
from pytorch_lightning.callbacks import LearningRateMonitor

#from finetuning_scheduler import FinetuningScheduler

from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import  ModelCheckpoint

from m_finetuning import FeatureExtractorFreezeUnfreeze

from m_dataLoad_json import write_list

import yaml
import json

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.

torch.set_float32_matmul_precision('medium')

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


    
    suffixes=config['terminaciones']
    num_channels_in=5 # RGB + NIR + UV
    max_values=config['maxValues']

    tipos_defecto=config['defect_types']

    tipos_defecto = [str(d) for d in tipos_defecto]

    train_dataplaces=config['train_dataplaces']
    val_dataplaces=config['val_dataplaces']

    print('In memory: ', config['in_memory'])



    print('Loading images...')
    datamodule = ViewDataModule(
        training_path = config['root_folder'], 
		 train_dataplaces=train_dataplaces,
         val_dataplaces=val_dataplaces,
                 suffixes=suffixes,
                 defect_types=tipos_defecto,              
                 predict_path = None , 
                 batch_size =config['batch_size'],
                 normalization_means=None,
                 normalization_stds=None,             
                 max_value=max_values,              
                 crop_size=None,
                 delimiter='_',
                 in_memory=config['in_memory'],
                 carga_mask=False)
    print('... done!')
    tipos_defecto=datamodule.tipos_defecto
    print('Tipos de defecto de JSONS: ',tipos_defecto)
   
# Saving info
    print('Saving clases_last_train.json')
    with open('clases_last_train.json', 'w') as outfile:
        json.dump({'clases':list(datamodule.tipos_defecto) },outfile,indent=4)
    


    medias_norm=datamodule.medias_norm.tolist()
    stds_norm=datamodule.stds_norm.tolist()
    print('medias_norm:',medias_norm)
    print('stds_norm:',stds_norm)
    print('Categorias:',datamodule.tipos_defecto)

    dict_norm={'medias_norm': medias_norm,
    'stds_norm': stds_norm }
    print('Saving normalization_last_train.json')
    with open('normalizacion_last_train.json', 'w') as outfile:
        json.dump(dict_norm,outfile,indent=4)


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
