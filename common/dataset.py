
import warnings
warnings.filterwarnings('ignore')
import os
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)

# torch and lightning imports
import torch    
from typing import Tuple,Any

from torch.utils.data import Dataset
import m_dataLoad_json
    
# Esto es el data set para entrenar validar   
class ViewsDataSet(Dataset):
    def __init__(self,dataset=None ,transform=None, carga_mask=False, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.dataset=dataset   
        self.transform = transform
        self.carga_mask = carga_mask
        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        caso=self.dataset[index]
        
        target =caso['labels'].float()
        view_id=caso['view_id']
        fruit_id=caso['fruit_id']
        
        
        if caso['image'] is None: #Cuando no está en memoria
            imags_folder=caso['imag_folder']
            sufijos=caso['sufijos']
            max_value=caso['max_value']
            crop_size=caso['crop_size']
            #print("Reading ", view_id)
            imagen=m_dataLoad_json.lee_vista(imags_folder,view_id,sufijos,max_value=max_value,carga_mask=self.carga_mask)
        else:
            #print("Imagen ya disponible:" , view_id)
            imagen=caso['image']
                       
        if self.transform is not None:                
            imagen2 = self.transform(imagen)
        else:
            imagen2=imagen
                           
        return imagen2,target,view_id,fruit_id # imagen, lista_de_etiquetas, (ruta_al_archivo, vista_id) 
    
    def __get_target__(self, index: int) -> Any:
        return self.dataset[index]['labels']
       
    def __len__(self) -> int:
        return len(self.dataset)


# Este se emplea para inferencia
class FileNamesDataSet(Dataset):
    '''
    Clase para suministrar archivos de una lista cuando no hay anotacion
    Solo se devuelve la imagen y el fruitid y el viewid.
    Por compatibilidad se devuelve un target = None
    '''
    def __init__(self,root_folder=None, filenames_list=None ,transform=None, 
                 field_delimiter='-',max_value = 255, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.root_folder=root_folder
        self.filenames_list=list(filenames_list)
        self.transform = transform
        self.field_delimiter=field_delimiter
        self.max_value = max_value
        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:              
        view_id=m_dataLoad_json.view_id(self.filenames_list[index])
        fruit_id=m_dataLoad_json.fruit_id(self.filenames_list[index],delimiter=self.field_delimiter)
        
        
        img_fullname=(self.filenames_list[index] if self.root_folder is None else os.path.join(self.root_folder,self.filenames_list[index]))
        print("Getting ", img_fullname)
        imagen= m_dataLoad_json.lee_png16(img_fullname,self.max_value)
           
        if self.transform is not None:                
            imagen2 = self.transform(imagen)
        else:
            imagen2=imagen
        target=None     # Este data set no está etiquetado               
        return imagen2, target, view_id,fruit_id 
              
    def __len__(self) -> int:
        return len(self.filenames_list)
 
