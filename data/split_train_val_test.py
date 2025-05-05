
# Para cuando las vistas de un fruto están en un mismo cimg y las anotaciones son a nivvel de fruto
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




def split_train_val(directorio,prob_train=0.8,delimiter='_'):
    patron=os.path.join(directorio,'*.json')
    files = glob(patron)
    
    files=[os.path.basename(f) for f in files]
    

    random.shuffle(files)
    
    n = len(files)
    print(f'Num Fruid Ids:{n}')
    train_files = files[:int(n*prob_train)]
    val_files = files[int(n*prob_train):]
    
    
    return train_files,val_files

def create_train_val__tests_lists(directorios,prob_train=0.8,prob_test=0.1,train_list_name=None,val_list_name=None,test_list_name=None):
    '''
    recibe un directorio o lista de directorios con jsons y crea dos listas de ficheros de entrenamiento y validación
    separando por fruit_id
    '''
    
    assert train_list_name is not None, "Train list name is None"
    assert val_list_name is not None, "Val list name is None"
    assert test_list_name is not None, "Test list name is None"
    
    
    allfiles=[]
    for d in directorios:
        patron=os.path.join(d,'*.json')
        filesdir = glob(patron)
        allfiles+=filesdir
    allfiles=list(set(allfiles))
    random.shuffle(allfiles)
    print(f'Num jsons: {len(allfiles)}')
    
    ntrain=int(len(allfiles)*prob_train)
    ntest=int(len(allfiles)*prob_test)
    print("ntest:",ntest)
    print("TestListName:",test_list_name)
    if os.path.exists(test_list_name): # Leerlo y coger la lista de jsons de test
        print(f'\n *** Test List File {test_list_name} exists, reading it')
        with open(test_list_name,'r') as f:
            lineas=f.readlines()
        testjsons=[l.strip() for l in lineas]
    else:
        testjsons=allfiles[:ntest]
    
    trainvaljsons=[]
    
    print("Lentestjsons=",len(testjsons))
    basenamestestest=[os.path.basename(f) for f in testjsons]
    for f in allfiles :
        bn=os.path.basename(f)
        if bn  in basenamestestest:
            continue
        else:
            trainvaljsons.append(f)
            
    print(f'Num Train Val jsons: {len(trainvaljsons)}')
    
    random.shuffle(trainvaljsons)
    trainjsons=trainvaljsons[:ntrain]
    valjsons=trainvaljsons[ntrain:]
    with open(train_list_name, 'w') as fp:
        for item in trainjsons:
            fp.write("%s\n" % item)
    with open(val_list_name, 'w') as fp:
        for item in valjsons:
            fp.write("%s\n" % item)
            
    if not os.path.exists(test_list_name):
        with open(test_list_name, 'w') as fp:
            for item in testjsons:
                fp.write("%s\n" % item)   
    else:
        print(f'\n *** Test List File {test_list_name} exists, Not Updating it')
        


if __name__ == '__main__':
    config_file=sys.argv[1]
    
    directorios=[]
    
    with open(config_file,'r') as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    
    
    root_folder=config['data']['root_folder']
    train=config['data']['dirs']
    for dp in train:
        directorios.append(os.path.join(root_folder,dp))

    directorios=list(set(directorios)) 
    prob_train=config['data']['prob_train']
    prob_test=config['data']['prob_test']   
    
    train_list_name=config['data']['train_list_name']
    val_list_name=config['data']['val_list_name']
    test_list_name=config['data']['test_list_name']
    
    train_list_name=os.path.join(root_folder,train_list_name)
    val_list_name=os.path.join(root_folder,val_list_name)
    test_list_name=os.path.join(root_folder,test_list_name)
    create_train_val__tests_lists(directorios,prob_train=prob_train,prob_test=prob_test,
                           train_list_name=train_list_name,val_list_name=val_list_name,test_list_name=test_list_name)
            
    
