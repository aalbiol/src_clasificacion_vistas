import sys
import os
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_dir)

from torchmetrics import AUROC

def calculate_auc_multilabel(preds,targets,clases):
    f_auroc=AUROC(task='multilabel',num_labels=len(clases),average='none')

    res=f_auroc(preds,targets.int())
    res=res.tolist()

    aucs={}
    print('res',res)
    for c,auc in zip(clases,res):
        print(type(c), c)
        print('c',c)
        aucs[c]=auc
        print(f'AUC({c}) : {auc:.3f}')
    
    return aucs
