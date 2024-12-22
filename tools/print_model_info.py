import sys
import os
import pickle
import json

current_file_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the folder you want to add

common_folder = os.path.join(current_file_dir, "../common")
sys.path.append(common_folder)


def print_model_info(path):
    with open(path, 'rb') as f:
        leido = pickle.load(f)
    print(f' Finished Loading {path}')
    config=None
    if 'config' in leido:
        config=leido['config']
    print(">>>>>>>>>>>>>>> CONFIG >>>>>>>>>>>>>>>>>>")
    if config is not None:
        if 'evaluate' in config:
            del config['evaluate']
        if 'predict' in config:
            del config['predict']   
        config_string=json.dumps(config,indent=3)
        print("config", config_string)
    print(">>>>>>>>>>>>>>>END CONFIG >>>>>>>>>>>>>>>>>>")   
    
    if 'normalization_dict' in leido:  
        normalization=leido['normalization_dict']
    else:
        normalization=None
    if 'image_size' in leido:
        training_size=leido['image_size']
    else:
        training_size=None
    if 'class_names' in leido:        
        class_names=leido['class_names']
    else:
        class_names=None

    training_date=leido['training_date']
    if 'final_val_aucs' in leido:
        final_val_aucs=leido['final_val_aucs']
    else:
        final_val_aucs=None
    model_type=leido['model_type']    
        
    if config is not None:
        data =config['data']
        model=config['model']
        if 'dict_norm' in data:
            normalization=data['dict_norm']
        class_names=model['defect_types']
        training_size=data['training_size']
        
        
       
    print("normalization: ",normalization)
    print("training_size: ",training_size)
    print("class_names: ",class_names)
    print("training_date: ",training_date)
    print("final_val_aucs: ",final_val_aucs)
    print("model_type: ",model_type)




if __name__ == '__main__':
    modelofilaname = sys.argv[1]

    print_model_info(modelofilaname)