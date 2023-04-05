import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from torch.utils import data
import math
import random
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import glob
import os
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
print(f"Torch Version : {torch.__version__}")

from datasets.sleep_edf import split_data, SleepEDF_MultiChan_Dataset, get_dataset
from models.epoch_cmt import Epoch_Cross_Transformer_Network, train_epoch_cmt, eval_epoch_cmt
from models.sequence_cmt import Epoch_Cross_Transformer, Seq_Cross_Transformer_Network, train_seq_cmt, eval_seq_cmt
from models.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward
from utils.metrics import accuracy, kappa, g_mean, plot_confusion_matrix, confusion_matrix


def parse_option():
    parser = argparse.ArgumentParser('Argument for training')


    parser.add_argument('--project_path', type=str, default='./results', help='Path to store project results')
    parser.add_argument('--data_path', type=str, help='Path to the dataset file')
    parser.add_argument('--train_data_list', nargs="+", default = [] ,  help='Folds in the dataset for training')
    parser.add_argument('--val_data_list', nargs="+", default = [4] ,  help='Folds in the dataset for validation')
    parser.add_argument('--is_interpret',  type=bool, default = False,  help='To get interpretations')
    parser.add_argument('--is_best_kappa',  type=bool, default = True,  help='True to read checkpoint with best kappa, else best acc')

    #model parameters
    parser.add_argument('--model_type', type=str, default = 'Epoch'  ,choices=['Epoch', 'Seq'],  help='Model type')
    parser.add_argument('--num_seq', type=int, default = 15,  help='Number of epochs in a PSG sequence')
    parser.add_argument('--batch_size', type=int, default = 32 ,  help='Batch Size')
    
    opt = parser.parse_args()
    
    return opt



def main():
    
    args = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation    Arguements ====================================>")
    
    for arg in vars(args):
        print(f"    {arg} :  {getattr(args, arg)}")
        
    #Get Dataset
    print("Getting Dataset ===================================>")
    data_loader = get_dataset(device,args,only_val = True)
    
    #Get Results
    print("Getting Results ===================================>")
    if args.model_type == "Epoch":
        eval_epoch_cmt(data_loader, device, args)
    else:
        eval_seq_cmt(data_loader, device, args)
    
if __name__ == '__main__':
    main()       
    
    
    
#python cmt_evaluate.py --project_path "testing" --data_path "/home/mmsm/Sleep_EDF_Dataset" --val_data_list [4] --model_type "Epoch" --batch_size 1

#python cmt_evaluate.py --project_path "testing_seq" --data_path "/home/mmsm/Sleep_EDF_Dataset" --val_data_list [4] --model_type "Seq" --batch_size 1 


#python cmt_evaluate.py --project_path "testing_seq" --data_path "/home/mmsm/Sleep_EDF_Dataset" --val_data_list [4] --model_type "Seq" --batch_size 1 --is_interpret True
