from glob import glob
import os
import sys

import random
import yaml

import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)

from pl_datamodule import view_id, fruit_id


def split_train_val(directorio,prob_train=0.8,delimiter='_'):
    patron=os.path.join(directorio,'*.json')
    files = glob(patron)
    
    files=[os.path.basename(f) for f in files]
    
    print(f'Num json files: {len(files)}')
    fruit_ids = [fruit_id(f,delimiter) for f in files]
    fruit_ids=list(set(fruit_ids))
    random.shuffle(fruit_ids)
    
    n = len(fruit_ids)
    print(f'Num Fruid Ids:{n}')
    train_ids = fruit_ids[:int(n*prob_train)]
    
    
    train_files = []
    val_files = []
    for f in files :
        if fruit_id(f,delimiter) in train_ids:
            train_files.append(f)
        else:
            val_files.append(f)
    
    
    train_files.sort()
    val_files.sort()
    return train_files,val_files

def create_train_val_lists(directorio,prob_train=0.8,delimiter='_', train_list_name=None,val_list_name=None):
    '''
    recibe un directorio o lista de directorios con jsons y crea dos listas de ficheros de entrenamiento y validación
    separando por fruit_id
    '''
    
    assert train_list_name is not None, "Train list name is None"
    assert val_list_name is not None, "Val list name is None"
    
    if not isinstance(directorio,list):
        t,v=split_train_val(directorio,prob_train,delimiter)
        nombre_t=os.path.join(directorio,train_list_name)
        nombre_v=os.path.join(directorio,val_list_name)
        with open(nombre_t, 'w') as fp:
            for item in t:
                fp.write("%s\n" % item)
        print(f"Train list saved in {nombre_t}")
        with open(nombre_v, 'w') as fp:
            for item in v:
                fp.write("%s\n" % item)            
        print(f"Val list saved in {nombre_v}")
    else:
        for d in directorio:
            create_train_val_lists(d,prob_train,delimiter,train_list_name,val_list_name)


if __name__ == '__main__':
    config_file=sys.argv[1]
    
    directorios=[]
    
    with open(config_file,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    
    
    root_folder=config['data']['root_folder']
    train=config['data']['train']
    for dp in train:
        directorios.append(os.path.join(root_folder,dp[1]))
    
    val=config['data']['val']
    for dp in val:
        directorios.append(os.path.join(root_folder,dp[1]))
    directorios=list(set(directorios)) 
    prob_train=config['data']['prob_train']   
    create_train_val_lists(directorios,prob_train=prob_train,delimiter='_',train_list_name='train_list.txt',val_list_name='val_list.txt')
            
    
