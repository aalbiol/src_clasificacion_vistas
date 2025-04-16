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
    #print(">>>> Get_matriz_casos")
    #print(">>>> Dataset type:", type(dataset))
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

class Balanced_BatchSampler(Sampler):
    '''
    Dado un dataset multiclass
    devuelve batches donde se asegura la misma cantidad de etiquetas positivas de todas las clases
    
    Util para clases muy desbalanceadas
    '''
    def __init__(self,dataset,clases):


        
        
        self.listas=[]
        for c in clases: # Crear tantas listas vacías como clases
            self.listas.append([])
        
        nitems=len(dataset)
        for k in range(nitems):
            clase=dataset.get_target(k)
            #print('Populating class:',clase)
            self.listas[clase].append(k)
                
        self.lengths=[]
        for k in range(len(clases)): 
             self.lengths.append(len(self.listas[k]))
         
        self.dataset = dataset

        copia_lengths=self.lengths.copy()
        copia_lengths.sort(reverse=True)
        max_length=copia_lengths[1]

        self.len = max_length*len(clases) # Un epoch es cuando la segunda lista más larga da una vuelta completa

                
   
    
    def barajarListas(self):
        for lista in self.listas:
            random.shuffle(lista)
        
        
    def __iter__(self):
        ''' Devuelve un epoch balanceado'''
        iteration = 0
        self.barajarListas()
        

        n=0
        
        while n <= self.len:
            iteration += 1
            # Coger secuencialemente un elemento de cada lista
            # Cada lista se recorre ciclicamente
            for count,lista in enumerate(self.listas):
                pos=iteration % self.lengths[count]
                n+=1
                yield lista[pos]
          
         
    def __len__(self) -> int:
        return self.len
    

class Balanced_BatchSamplerMultiLabel(Sampler):
    '''
    Dado un CImgFruitsViewsDataSet multilabel
    devuelve batches donde se asegura la misma cantidad de etiquetas positivas de todas las clases
    
    Util para clases muy desbalanceadas
    '''
    def __init__(self,dataset,class_names):
        print ('>>>>>>>>>>>>>>>>< Sampler init Type Dataset <<<<<<<<<<<<<<<< ')#, type(dataset))
        matriz_casos=get_matriz_casos(dataset)
        estadistica_clase=get_class_distribution(matriz_casos)
        num_clases=len(estadistica_clase) 
        self.matriz_casos=matriz_casos
        
        print('Sampler Numclases=',num_clases)
        self.listas=[]
        self.lengths=[] 
        lista= get_class_items(matriz_casos,-1)
        self.listas.append(lista)
        self.lengths.append(len(lista))
        self.class_names=class_names
        
        for k in tqdm(range(num_clases)): 
            #print('Populating class:',k)
            lista= get_class_items(matriz_casos,k)
            self.listas.append(lista)
            self.lengths.append(len(lista))
        
        max_length=max(self.lengths)
        print('Sampler Numlistas=',len(self.listas)) 
        self.dataset = dataset
        self.len =  len(dataset)
        print('Sampler len=',max_length*len(self.listas)) 

        for count,l in enumerate(self.listas):
            if count ==0:
                print(f'  ** Length Lista sin defecto : {len(l)}')
            else:
                print(f'  ** Length Lista {self.class_names[count-1]} : {len(l)}')

                
   
    
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
                # print("Lista:",count,"Pos:",pos,"Elemento",lista[pos])
                # print("matriz_casos:",self.matriz_casos[lista[pos]])
                yield lista[pos]
                #print('Yield:',lista[pos])
                # batch.append( lista[pos])
                # if len(batch)==self.batch_size:
                #     out=batch
                #     batch=[]
                #     yield out             
         
    def __len__(self) -> int:
        return self.len