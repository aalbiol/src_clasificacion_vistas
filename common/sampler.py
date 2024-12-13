import sys
import os
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)


import torch


import pl_datamodule

import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data  import Sampler, BatchSampler, SubsetRandomSampler
from tqdm import tqdm   

#########################################
## Funciones para manejar datasets multilabel posiblemente bastante desbalanceados
########################################

def get_matriz_casos(dataset):
    ''' matriz de casos de un CImgFruitsViewsDataSet
    '''
    matriz_casos=[]
    print(">>>> Get_matriz_casos")
    print(">>>> Dataset type:", type(dataset))
    for k in tqdm(range(len(dataset))):
        target=dataset.__get_target__(k)
        #print(">>>> Get_matriz_casos", target)
        matriz_casos.append(target)
    matriz_casos=torch.stack(matriz_casos)       
    return matriz_casos

def get_class_items(matriz_casos,clase):
    ''' 
    dataset: CImgFruitsViewsDataSet
    if clase >=0 lista de elementos en los que clase esta a 1
    if clase < 0 lista de elementos en los que todas las clases estan a 0
    '''

    if clase >=0:
        indices=torch.argwhere(matriz_casos[:,clase])
    else:
        suma=torch.sum(matriz_casos,axis=1)
        indices=torch.argwhere(suma==0)   #tensor
    
    indices=[x.item() for x in indices] # Lista de enteros
    return indices

def get_class_distribution(matriz_casos):
    '''
    dataset: CImgFruitsViewsDataSet
    Devuelve las probabilidades de cada label
    '''

    distribution=torch.sum(matriz_casos,dim=0)
    return distribution

# class Balanced_BatchSampler(Sampler):
#     '''
#     Dado un CImgFruitsViewsDataSet multilabel
#     devuelve batches donde se asegura la misma cantidad de etiquetas positivas de todas las clases
    
#     Util para clases muy desbalanceadas
#     '''
#     def __init__(self,dataset):
#         estadistica_clase=get_class_distribution(dataset)
#         num_clases=len(estadistica_clase) 
        
        
#         self.listas=[]
#         self.lengths=[] 
#         for k in range(num_clases): 
#             lista= get_class_items(dataset,k)
#             self.listas.append(lista)
#             self.lengths.append(len(lista))
         
#         self.dataset = dataset
#         self.len = len(dataset)

                
   
    
#     def barajarListas(self):
#         for lista in self.listas:
#             random.shuffle(lista)
        
        
#     def __iter__(self):
#         ''' Devuelve un epoch balanceado'''
#         iteration = 0
#         self.barajarListas()
        
#         #batch=[]
#         n=0
        
#         while n <= self.len:
#             iteration += 1
#             # Coger secuencialemente un elemento de cada lista
#             # Cada lista se recorre ciclicamente
#             for count,lista in enumerate(self.listas):
#                 pos=iteration % self.lengths[count]
#                 n+=1
#                 yield lista[pos]
#                 # batch.append( lista[pos])
#                 # if len(batch)==self.batch_size:
#                 #     out=batch
#                 #     batch=[]
#                 #     yield out             
         
#     def __len__(self) -> int:
#         return self.len
    

class Balanced_BatchSamplerMultiLabel(Sampler):
    '''
    Dado un CImgFruitsViewsDataSet multilabel
    devuelve batches donde se asegura la misma cantidad de etiquetas positivas de todas las clases
    
    Util para clases muy desbalanceadas
    '''
    def __init__(self,dataset):
        print ('>>>>>>>>>>>>>>>>< Sampler init Type Dataset:', type(dataset))
        matriz_casos=get_matriz_casos(dataset)
        estadistica_clase=get_class_distribution(matriz_casos)
        num_clases=len(estadistica_clase) 
        
        print('Sampler Numclases=',num_clases)
        self.listas=[]
        self.lengths=[] 
        lista= get_class_items(matriz_casos,-1)
        self.listas.append(lista)
        self.lengths.append(len(lista))
        
        for k in tqdm(range(num_clases)): 
            #print('Populating class:',k)
            lista= get_class_items(matriz_casos,k)
            self.listas.append(lista)
            self.lengths.append(len(lista))
        
        print('Sampler Numlistas=',len(self.listas)) 
        self.dataset = dataset
        self.len =  2*len(dataset)
        print('Sampler len=',self.len) 

        for count,l in enumerate(self.listas):
            print(f'  ** Length Lista {count}: {len(l)}')

                
   
    
    def barajarListas(self):
        for lista in self.listas:
            random.shuffle(lista)
        
        
    def __iter__(self):
        ''' Devuelve un minibatch balanceado'''
        iteration = 0
        self.barajarListas()
        
        #batch=[]
        n=0
        
        while n <= self.len:
            iteration += 1
            # Coger secuencialemente un elemento de cada lista
            # Cada lista se recorre ciclicamente
            for count,lista in enumerate(self.listas):
                pos=iteration % self.lengths[count]
                n+=1
                yield lista[pos]
                #print('Yield:',lista[pos])
                # batch.append( lista[pos])
                # if len(batch)==self.batch_size:
                #     out=batch
                #     batch=[]
                #     yield out             
         
    def __len__(self) -> int:
        return self.len