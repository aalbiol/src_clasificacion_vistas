
import numpy as np
import zlib
import os
from PIL import Image
import torch
import json

typesDict={'float':'float32' ,'double':'float64',
'unsigned_short':'uint16','unsigned_char':'uint8',
'int':'int32', 'short':'int16'}

def __cimgread( filename ):
    """ USAGE: a= cimgread(filename)
    For CImg Images:
        * returns a npy array in the case of cimg
        * Supports compression
        * It squeezes singleton dimensions. If a CImg image has dimensions (w,h,1,c)
            the returned python object will have shape
                a.shape --> (h,w,c)
        * a(y,x,z,c) to access one element
    For CImgList:
        * returns a list of npy arrays
        * if original CImgList has nimages, then
             len(a) --> nimages
        * To access one pixel of the j-th image use a[j](y,x,z,c)

        """

    basename, file_extension = os.path.splitext(filename)
    fa = open(filename, 'rb')

    out =[]
    line0 = fa.readline() #Endiannes
    tiposdato=line0.split()
    number_of_images=int(tiposdato[0])
    datatypecimg=tiposdato[1].decode()
    endiannes = tiposdato[2]

    datatype = typesDict[datatypecimg];

    for n in range(number_of_images):
        line1 = fa.readline() # Dimensions
        dimensiones = line1.split()
        width = int(dimensiones[0]);
        height = int(dimensiones[1]);
        depth = int(dimensiones[2]);
        spectrum = int(dimensiones[3]);
        if width==0:
            continue
        if file_extension == '.cimgz':
            csize= int(dimensiones[4].decode()[1:])
            data = fa.read(csize)
            data = zlib.decompress(data)
        else:
            data = fa.read(width*height*depth*spectrum*np.dtype(datatype).itemsize)

        flattened = np.frombuffer(data,dtype=datatype)

        cimg=flattened.reshape((spectrum,depth,height,width))
        cimg=np.squeeze(np.transpose(cimg,(2,3,1,0)))
        out.append(cimg)

    fa.close()
    if len(out)==1:
        return out[0]
    return out

def cimgread(filename): 
    '''Devuelve lista de PILS'''
    a=__cimgread(filename)
    apil=[]

    
    for aa in a:
        pil=Image.fromarray(aa[:,:,:3])
        apil.append(pil)
    
    return apil

def cimgread_np(filename):
    a=__cimgread(filename)  
    return a   

def __cimglistread( filename ):
    """ USAGE: a= cimgread(filename)
    For CImg Images:
        * returns a npy array in the case of cimg
        * Supports compression
        * It squeezes singleton dimensions. If a CImg image has dimensions (w,h,1,c)
            the returned python object will have shape
                a.shape --> (h,w,c)
        * a(y,x,z,c) to access one element
    For CImgList:
        * returns a list of npy arrays
        * if original CImgList has nimages, then
             len(a) --> nimages
        * To access one pixel of the j-th image use a[j](y,x,z,c)

        """

    basename, file_extension = os.path.splitext(filename)
    fa = open(filename, 'rb')

    out =[]
    line0 = fa.readline() #Endiannes
    tiposdato=line0.split()
    number_of_images=int(tiposdato[0])
    datatypecimg=tiposdato[1].decode()
    endiannes = tiposdato[2]
    # print('Filename:',filename)
    # print('Number of images:',number_of_images)
    # print('CimgDataType:',datatypecimg)

    datatype = typesDict[datatypecimg];
    # print('DataType:',datatype)

    for n in range(number_of_images):
        line1 = fa.readline() # Dimensionz
        dimensiones = line1.split()
        # print(line1)
        # print(dimensiones)
        width = int(dimensiones[0]);
        height = int(dimensiones[1]);
        depth = int(dimensiones[2]);
        spectrum = int(dimensiones[3]);
        if width==0:
            # print("skipping empty image")
            continue


        #if file_extension == '.cimgz':
        if len(dimensiones) > 4:
            csize= int(dimensiones[4].decode()[1:])
            data = fa.read(csize)
            data = zlib.decompress(data)
        else:
            data = fa.read(width*height*depth*spectrum*np.dtype(datatype).itemsize)

        flattened = np.frombuffer(data,dtype=datatype)

        cimg=flattened.reshape((spectrum,depth,height,width))
        cimg=np.squeeze(np.transpose(cimg,(2,3,1,0)))
        out.append(cimg)
    fa.close()
    return out


def cimglistread_torch(filename,max_value,channel_list=None):
    '''
    A partir de una lista de numpys de cualquier tipo de dato
    Devuelve una lista de tensores
    de tipo float32
    normalizado por max_value
    Los ejes en el orden (color,fila,columna)
    
    SE QUEDA SOLO CON EL RGB
    '''
    numpys =__cimglistread(filename) # Es una lista de npys

    #print('Channel list cimgRead: ',channel_list)

    tensores=[]
    if channel_list is None:
        for a in numpys:
            t=torch.permute( torch.from_numpy(a[:,:,:3].copy()).to(torch.float32)/max_value, (2,0,1)).clip(0.0,1.0)
            tensores.append(t)
    else:
        for a in numpys:
            t=torch.permute( torch.from_numpy(a[:,:,channel_list].copy()).to(torch.float32)/max_value, (2,0,1)).clip(0.0,1.0)
            tensores.append(t)

    
    return tensores


def npzread_torch(nombre_img,nombre_json=None,maxval=None,channel_list=None):
    ''' Lee un npz y devuelve una lista de tensores
    de tipo float32
     normalizado por max_value
    Los ejes en el orden (color,fila,columna)'''
    npz_file = np.load(nombre_img)
    maxvalue = None
    channels = None
    if nombre_json is not None:
        with open(nombre_json) as f:
            data=json.load(f)
        print(data)
        channels=data['channels'].split(';')
        maxvalue=data['maxValChannels']
    else:
        maxvalue=maxval
        channels=channel_list
    assert channels is not None
    assert maxvalue is not None
    #print(channels)
    channels_dict={}
    for c,v in enumerate(channels):
        channels_dict[v]=c
    print(channels_dict)
    
    


# List the keys (names of the arrays inside the .npz file)
    #print("Arrays in the .npz file:", len(npz_file.files), npz_file.files)

    num_vistas = len(npz_file.files)
    num_planos = npz_file[npz_file.files[0]].shape[0]

    vistas=[]

    for view in npz_file:
        vista_allchannels=npz_file[view]
        lista_canales=[]
        for channel in channel_list:
            channel_idx=channels_dict[channel]
            canal=vista_allchannels[channel_idx]
            lista_canales.append(canal)
        vista=np.stack(lista_canales,axis=0)
        vista=vista.astype(np.float32)/maxvalue
        vista=torch.from_numpy(vista)
        
        vistas.append(vista)
    return vistas
    

def npzread_auxB1(nombre_img,nombre_json):
    ''' Lee un npz y devuelve una lista de tensores
    de tipo in16
     
    con la mascara auxB1'''
    npz_file = np.load(nombre_img)

    with open(nombre_json) as f:
        data=json.load(f)

    channels=data['channels'].split(';')
    #print(channels)
    channels_dict={}
    for c,v in enumerate(channels):
        channels_dict[v]=c
    
    auxb1idx=channels_dict['AuxB1']
    
    

    vistas=[]

    for view in npz_file:
        vista_allchannels=npz_file[view]
        lista_canales=[]

        canal=vista_allchannels[auxb1idx]        
        vista=torch.from_numpy(canal)
        
        vistas.append(vista)
    return vistas


def torch2npz(lista_tensores,nombrenpz,maxval=1023):
    ''' Guarda una lista de tensores en un npz
    normalizando por maxval'''
    lista_npys=[]
    for tensor in lista_tensores:
        tensor *=maxval
        npy=tensor.numpy().astype(np.uint16)
        lista_npys.append(npy)
    np.savez(nombrenpz,*lista_npys)
    
