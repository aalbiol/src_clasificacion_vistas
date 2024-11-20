
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

    # parser.add_argument("--num_epochs", default = 3000, help="""Number of Epochs to Run.""", type=int)

    # parser.add_argument("-o", "--optimizer", help="""PyTorch optimizer to use. Defaults to sgd.""", default='adam')
    # parser.add_argument("-lr", "--learning_rate", help="Adjust learning rate of optimizer.", type=float, default=1e-3)
    # parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
    #                      type=int, default=40)
    
    # parser.add_argument("-s", "--save_path", default='./out_models/', help="""Path to save model trained model checkpoint.""")
    # parser.add_argument("-i", "--initial_model", default=None, help="""Initial Model . If None Resnet is used""")
  
    # parser.add_argument("--log_name",help="""WandB log name""", default='PatchMIL')
    # parser.add_argument("--root_folder",help="""Folder containing the training data""",
    #                    # default='/home/aalbiol/owc/mscanData/pngs_anotados/aceitunas')
    #                    default='/home/aalbiol/owc/mscanData/pngs_anotados/Smart4Olives/season_2022/manzanilla/Season_oct_2022_set01_rev1')
    # parser.add_argument("-wd", "--weights_decay", help="Regulatization param.", type=float, default=1e-3)
    # parser.add_argument("-mu_a", "--mixup_alpha", help="MixUp Alpha. Zero: no regularization", type=float, default=0.4)
    # parser.add_argument("--multilabel", action=argparse.BooleanOptionalAction)
    # parser.add_argument("--in_memory", action=argparse.BooleanOptionalAction)
    # parser.add_argument( "--crop_size", help="Image target size.", type=int, default=120)
    # parser.add_argument( "--resnet_version", help="Resnet Version. Valores posibles 18 y 50", type=int, default=50)
    # parser.add_argument("--label_smoothing", help="Label Smoothing to prevent overfitting.", type=float, default=0.1)
    # parser.add_argument("--unfreeze_epoch", help="Layers unfrozen each unfreeze_epoch.", type=int, default=30)
    # parser.add_argument("--p_dropout", help="Dropout Probability", type=float, default=0.5)
    # parser.add_argument( "--warmup", help="""Warmup number of batches.""", type=int, default=0)
    # parser.set_defaults(multilabel=True)
    # parser.set_defaults(in_memory=True)

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


    training_path = config['root_folder']
    print('Config Root Folder:',training_path)
    user = os.getenv('USER') or os.getenv('USERNAME')
    if user=='csanchis':
        training_path=os.path.join(r"C:\\Users\\csanchis",training_path)
    elif user=="aalbiol":
        training_path=os.path.join("/home/aalbiol/owc",training_path)

    print('Training Root Folder:',training_path)

    suffixes=['_r.png','_g.png','_b.png','_a.png','_nir.png','_nir1.png','_nir2.png','_nir4.png']
    num_channels_in=len(suffixes)
    max_values=config['max_value']
    crop_size=(config['crop_size'],config['crop_size'])
    
    
  
    tipos_defecto=config['defect_types']

    tipos_defecto = [str(d) for d in tipos_defecto]

    train_dataplaces=config['train_dataplaces']
    val_dataplaces=config['val_dataplaces']

    print('In memory: ', config['in_memory'])



    print('Loading images...')
    datamodule = ViewDataModule(
        training_path = training_path, 
		 train_dataplaces=train_dataplaces,
         val_dataplaces=val_dataplaces,
                 suffixes=suffixes,
                 defect_types=tipos_defecto,              
                 predict_path = None , 
                 batch_size =config['batch_size'],
                 num_workers=config['num_workers'],
                 normalization_means=None,
                 normalization_stds=None,             
                 max_value=max_values,              
                 crop_size=crop_size,
                 delimiter='_',
                 carga_mask=config['carga_mask'],
                 in_memory=config['in_memory'])
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
