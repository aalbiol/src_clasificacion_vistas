import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
warnings.filterwarnings('ignore')

# torch and lightning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pytorch_lightning as pl
from torchvision import transforms


from pytorch_lightning.loggers import WandbLogger

from pl_modulo import PatchMILClassifier



import pycimg
import piltools

import time



# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument("--resnetmodel", default = 18,
                        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
                        type=int)

    
    # parser.add_argument("-b", "--batch_size", help="""Manually determine batch size. Defaults to 16.""",
    #                      type=int, default=10)
    # Optional arguments
    parser.add_argument("-m", "--model", required=True,help="""Model""")
    parser.add_argument("-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None)

    parser.add_argument("-n", "--num_classes", default=4, help="""Number of classes""")
    parser.add_argument("-f","--cimgfilename", default=None, required=True, help="""Fichero cimglist""")
    args = parser.parse_args()

    # datamodule = FruitDataModule(batch_size=args.batch_size,
    # train_set_folder=None,
    # test_set_folder=None,                             
    # predict_set_folder=args.directory,
    # num_clases=3)
   
    model = PatchMILClassifier(num_classes = args.num_classes, 
                            num_channels_in=3)
                                                           
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    
    

    
    trainer_args = {'gpus': args.gpus}

#Cargar fichero --> lista de PILS    
    vistas=pycimg.cimgread(args.cimgfilename)

    probs,batch = model.predice_fruto(vistas)

    mosaicprobs=piltools.createMosaic(probs)
    mosaicimgs=piltools.createMosaicRGB(batch)
    
    print('ProbsSize:',mosaicprobs.size)
    print('RGB Size: ',mosaicimgs.size)
    
    mosaicimgs.save('imags.png')
    mosaicprobs.save('probs.png')
    print('imags.png and probs.png saved')
    mosaicimgs.show("Vistas")
    mosaicprobs.show("Probs")
    
    
    
    

 
    


