import sys
import os
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)
import torch

from torchmetrics import AUROC

def calculate_auc_multilabel(preds,targets,clases):

    f_auroc=AUROC(task='binary')
    aucs={}
    for i,clase in enumerate(clases):
        p=preds[:,i]
        t=targets[:,i]
        notnan_mask = torch.logical_not(torch.isnan(t))
        if torch.sum(notnan_mask) == 0:
            print(f"All values for class {clase} are NaN.")
            aucs[clase]=0
        else:
            aucs[clase]=f_auroc(p[notnan_mask],t[notnan_mask].int()).item()


    
    return aucs
