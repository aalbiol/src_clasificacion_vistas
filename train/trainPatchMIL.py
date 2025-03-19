
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

from m_finetuning import FeatureExtractorFreezeUnfreeze

from pl_datamodule import JSONSCImgDataModule

from pl_patch_MIL_modulo import PatchMILClassifier




#from m_dataLoad_json import write_list


if __name__ == "__main__":
    parser = ArgumentParser()

    # parser.add_argument("--config", default = "configs/train.yaml", help="""YAML config file""")

    # args = parser.parse_args()


    # with open(args.config,'r') as f:
    #     config=yaml.load(f,Loader=yaml.FullLoader)

    if len(sys.argv) < 2:
        print("Usage: trainPatchMIL.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1],'r') as f:
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
    terminacion=config['data']['terminaciones']
    root_folder=config['data']['root_folder']
    
    crop_size=config['data']['crop_size']
    
    batch_size=config['train']['batch_size']
    in_memory=config['train']['in_memory']
    maxvalues=config['data']['maxvalues']
    crop_size=config['data']['crop_size']
    channel_list=config['data']['channel_list']
    training_size=config['data']['training_size']
    tipos_defecto=config['model']['defect_types']
    aumentacion=config['train']['augmentation']
    
    
    
    output=config['train']['output']
    save_path=output['path']
    model_file=output['model_file']
    


    num_channels_in=config['model']['num_channels_input']
    model_version= config['model']['model_version']
   
    
    print('Creating DataModule...')
    datamodule =  JSONSCImgDataModule( root_path = root_folder, 
                 train_dataplaces=train_dataplaces,
                 val_dataplaces=val_dataplaces,
                 defect_types=tipos_defecto,                                                 
                 predict_dataplaces = None , 
                 batch_size=batch_size, #Num frutos
                 imagesize=training_size,
                 normalization_means=None,
                 normalization_stds= None,
                 max_value=maxvalues,
                 in_memory=in_memory,
                 num_workers=config['train']['num_workers'],
                 channel_list=channel_list,
                 augmentation=aumentacion,
                 terminacion=terminacion)
    print('... done!')

  # Output
    model_name=config['train']['output']['model_file']
    model_path=config['train']['output']['path'] 


    medias_norm=datamodule.medias_norm
    stds_norm=datamodule.stds_norm
    print('medias_norm:',medias_norm)
    print('stds_norm:',stds_norm)
    
    dict_norm={'medias_norm': medias_norm,
        'stds_norm': stds_norm }

    model = PatchMILClassifier(
                            optimizer =config['train']['optimizer'], 
                            lr = config['train']['learning_rate'],
                            num_channels_in=len(channel_list),
                            model_version=model_version,
                            warmup_iter=config['train']['warmup'],
                            class_names=tipos_defecto,
                            p_dropout=config['train']['p_dropout'],
                            label_smoothing=config['train']['label_smoothing'],
                            weight_decay=config['train']['weights_decay'],
                            normalization_dict=dict_norm,
                            training_size=training_size,
                            config=config)
    
    
            
    # Continuar entrenamiento a partir de un punto
    if config['train']['initial_model'] is not None:
        model.load(config['train']['initial_model'])


    # Instantiate lightning trainer and train model
    logname=config['train']['logname']
    miwandb= WandbLogger(name=logname, project='WANDB_DVC',config=config, entity='multiscan')

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    num_epochs=config['train']['epochs']
    
    trainer_args = {'max_epochs': num_epochs, 'logger' : miwandb}
    print('num_epochs:',num_epochs)
    

    callbacks3=[FeatureExtractorFreezeUnfreeze(unfreeze_at_epoch=config['train']['unfreeze_epoch'],initial_denom_lr=2),]
    
    trainer = pl.Trainer(callbacks=callbacks3,**trainer_args)
    
    print("\n***************************************\n")
    print("Starting training...")  
    trainer.fit(model, datamodule=datamodule)

    
    # Create model_path if it does not exist
    Path(model_path).mkdir(parents=True, exist_ok=True)
    output_model_filename = os.path.join(model_path,model_name)
    model.save(output_model_filename,config=config)

