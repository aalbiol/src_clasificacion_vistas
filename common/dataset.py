
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
import pycimg

from PIL import Image



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


def lee_vista(images_folder,view_idd,terminaciones,max_value,carga_mask=True):
    print("Reading ", view_idd,images_folder)
    print("Maxvalues:",max_value)
    nombre_base=os.path.join(images_folder,view_idd)
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

 
class CImgListDataSet(Dataset):
    def __init__(self,dataset=None , transform=None, channel_list=None, terminacion=".cimg", *args, **kwargs):
        super().__init__(*args,  **kwargs)
        
        self.dataset=dataset   
        self.transform = transform
        self.channel_list=channel_list
        self.terminacion=terminacion
        

        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        caso=self.dataset[index]
        
        target =caso['labels'].float()
        fruit_id=caso['fruit_id']
        vistas=caso['image']
        imags_folder=caso['imag_folder']
        maxvalue=caso['max_value']
        
        if target.isnan().sum() >0:
            print("\n\n******Imagen ", fruit_id, "labels:",target )
        if vistas is None: #Cuando no está en memoria
            nombre_img=os.path.join(imags_folder,fruit_id)
            nombre_img += self.terminacion
            jsonfile=nombre_img.replace(self.terminacion,".json")
            assert os.path.exists(nombre_img), f'No existe el archivo {nombre_img}'       
            if "cimg" in self.terminacion:
                    #Aqui channel_list es una lista de enteros
                    vistas = pycimg.cimglistread_torch(nombre_img,maxvalue,channel_list=self.channel_list) # lista de tensores normalizados en intensidad 
            elif "npz" in self.terminacion:
                    #Aqui channel_list es una lista de strings
                vistas = pycimg.npzread_torch(nombre_img,jsonfile,channel_list=self.channel_list)
            else:
                vistas=None
                print(f"ERROR en genera_ds_jsons_multilabelMIL: terminacion '{terminacion}' no reconocida")
                sys.exit(1)
            #Tamaños diferentes
        
        assert vistas is not None, f'No se ha podido cargar la imagen {fruit_id} Type Vistas: {type(vistas)}'
        vistas_transformed = [] # cada vista sufre una aumentacion diferente
        if self.transform is not None:        
            for vista in vistas:
                    assert vista is not None, f'No se ha podido cargar la imagen {fruit_id} Type Vista: {type(vista)}'
                    vista = self.transform(vista)                                
                    vistas_transformed.append(vista)               
        else:
            vistas_transformed=vistas
        vistas_transformed=torch.stack(vistas_transformed,axis=0)                    
        return vistas_transformed,target,fruit_id,imags_folder
    def __get_target__(self, index: int) -> Any:
        return self.dataset[index]['labels_fruit'] 
    def __len__(self) -> int:
        return len(self.dataset)
    

def get_clase(filename):
    """
    Get the class of the image from the filename
    """
    # Get the class from the filename
    # The class is the first part of the filename
    # before the first underscore
    directorio=os.path.dirname(filename)
    clase=directorio.split("/")[-1]
    return clase

class ListasDataSet(Dataset):
    def __init__(self,lista_ficheros=None , clases=None, transform=None):
        
        
        assert transform is not None, "No se ha definido la transformacion"
        self.lista_ficheros=lista_ficheros   
        self.transform = transform
        self.clases=clases
        self.clases_dict={}
        #print("Clases:",clases)
        for i, clase in enumerate(clases):
            self.clases_dict[clase]=i

        self.onehots={}
        for i, clase in enumerate(clases):
            self.onehots[clase]=torch.zeros(len(clases))
            self.onehots[clase][i]=1
        
        
        
        
              
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        #print("Getting item",index)
        assert index >=0 and index < len(self.lista_ficheros), f"Index {index} out of range {len(self.lista_ficheros)}"
        caso=self.lista_ficheros[index]
        
        clase=get_clase(caso)
        target =self.clases_dict[clase]
        onehot=self.onehots[clase]
        with Image.open(caso) as pil:
            #print("pilt.type",type(pil))
            image= self.transform(pil)    # Siempre existe transformacion
        #print("image.type",type(image))
        return image,onehot,caso # pixeles, onehot, (ruta_al_archivo, vista_id)
    
    def get_target(self, index: int) -> Any:
        return self.clases_dict[get_clase(self.lista_ficheros[index])]
    
    def __len__(self) -> int:
        return len(self.lista_ficheros)
