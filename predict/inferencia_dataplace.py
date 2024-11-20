
import warnings
from pathlib import Path
from argparse import ArgumentParser

import yaml
import json
from tqdm import tqdm
import os

import multiprocessing
warnings.filterwarnings('ignore')


import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


from pl_modulo import OliveClassifier
import m_dataLoad_json
import dataLoad

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

    #device='cpu'
    suffixes=['_r.png','_g.png','_b.png','_a.png','_nir.png','_nir1.png','_nir2.png','_nir4.png']
    
    num_channels_in=len(suffixes)
    max_value=config['max_value']
    crop_size=(config['crop_size'],config['crop_size'])

    clases=config['defect_types']
    pred_path = config['root_folder']
    print('Config Root Folder:',pred_path)
    user = os.getenv('USER') or os.getenv('USERNAME')
    if user=='csanchis':
        pred_path=os.path.join(r"C:\\Users\\csanchis",pred_path)
    elif user=="aalbiol":
        pred_path=os.path.join("/home/aalbiol/owc",pred_path)

    print('Training Root Folder:',pred_path)
 

    pred_ds,_,tipos_defecto=m_dataLoad_json.genera_ds_jsons_multilabel(pred_path,
                dataplaces= config['pred_dataplaces'], 
                sufijos=suffixes,
                max_value=max_value, 
                prob_train=1.0, # Si no quiero split poner 1.0
                crop_size=crop_size,
                defect_types=clases,
                multilabel=True,
                splitname_delimiter='_',
                in_memory=False,
                ) 
   
    print("Tipos defecto en jsons:",tipos_defecto)
    print("Tipos defecto en config:",clases)
    
    print('Loading normalization_last_train.json')
    with open('normalizacion_last_train.json', 'r') as infile:
        dict_norm=json.load(infile)

    transform_val = transforms.Compose([
        transforms.CenterCrop(config['crop_size']),    
        transforms.Normalize(dict_norm['medias_norm'],dict_norm['stds_norm'])
        ])      
    
    print("Length Dataset:",len(pred_ds))
    pred_dataset=dataLoad.OliveViewsDataSet(pred_ds,transform=transform_val)


  

    model=OliveClassifier(num_channels_in,
                class_names=tipos_defecto,
                p_dropout=config['p_dropout'])
        
    # Continuar entrenamiento a partir de un punto
    assert config['initial_model'] is not None
    initial_model_path = config['initial_model'] if config['model_path'] is None else os.path.join(config['model_path'],config['initial_model'])
    checkpoint = torch.load(initial_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.modelo.eval()
    modelo = model.modelo.to(device)

    num_workers = multiprocessing.cpu_count()-1 if config['num_workers']<0 else config['num_workers']
    dl= DataLoader(pred_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, collate_fn=dataLoad.my_collate_fn)

    preds=[]
    targets=[]
    ids=[]
    with torch.no_grad():
        for batch in tqdm(dl):
            images = batch['images']
            labels = batch['labels']
            view_ids=batch['view_ids']
            
            Y=modelo.forward(images.to(device))
            logits = Y[:,:,0,0] # como es un clasif de vista devuelve nbatch x ndefectos x 1 x 1
            probs =F.sigmoid(logits)
            probs=probs.to('cpu')
            preds.append(probs)
            targets.append(labels)
            ids+= view_ids
            #print(probs.shape)

    preds = torch.concat(preds)
    targets = torch.concat(targets)

    outpred_path = config['preds_file'] 
    outtargets_path = config['targets_file'] 
    names_path=config['filenames_file']

    torch.save(preds,outpred_path)
    torch.save(targets,outtargets_path)

    with open(names_path, mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(ids))
    

    print ("Saved ", outpred_path , ' and ', outtargets_path)