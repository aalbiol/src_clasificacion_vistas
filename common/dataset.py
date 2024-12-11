
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
#import m_dataLoad_json
import cv2



def lee_png16(filename,max_value):
    #print(f'Reading {filename}...')
    assert os.path.exists(filename), f'No existe el archivo {filename}'
    im=cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if im.ndim ==3 :
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im=im.astype('float32')/max_value
    im[im>1]=1

    im=torch.tensor(im)
    if im.ndim ==2:# convertir de hxw --> 1 x h x w
        im=im.unsqueeze(0)
    else:
        im=im.permute((2,0,1))
    return im


def lee_vista(images_folder,view_id,terminaciones,max_value,carga_mask=True):
    #print("Reading ", view_id)
    nombre_base=os.path.join(images_folder,view_id)
    canales=[]

    assert isinstance(max_value,list), 'maxvalue tiene que ser una lista de tantos elementos como canales o una lista con un unico elemento que se emplea para todos los canales'
   
    if len(max_value) == len(terminaciones):
            max_values=max_value
    else:
        assert len(max_value)==1, "Es una lista que no tiene ni un solo elemento ni un numero de elementos igual al numero de canales"
        max_values= max_value*len(terminaciones)

    for k,t in enumerate(terminaciones):
        nombre=nombre_base+t
        canal=lee_png16(nombre,max_values[k])
        canales.append(canal)
    canales=torch.concat(canales,0)
    if carga_mask:
        term_mascara="_auxb1.png"
        nombre=nombre_base+term_mascara
        mascara=lee_png16(nombre,255)
        color_centro=mascara[0,mascara.shape[1]//2, mascara.shape[2]//2]
        mascara =((mascara==color_centro)*(mascara < 0.5)).float()

        #print(canales.shape,mascara.shape)
        canales *= mascara
        # plt.imshow(canales[:3,:,:].numpy().transpose((1,2,0)),clim=(0,0.25))
        # plt.show()

    return canales



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
            imagen=lee_vista(imags_folder,view_id,sufijos,max_value=max_value,carga_mask=self.carga_mask)
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

