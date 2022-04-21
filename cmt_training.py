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
import scipy.signal
import os
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
print(f"Torch Version : {torch.__version__}")

from datasets.sleep_edf import split_data, SleepEDF_MultiChan_Dataset, get_dataset
from models.epoch_cmt import Epoch_Cross_Transformer_Network
from models.sequence_cmt import Seq_Cross_Transformer_Network 
from utils.metrics import accuracy, kappa, g_mean, plot_confusion_matrix, confusion_matrix, AverageMeter 


def parse_option():
    parser = argparse.ArgumentParser('Argument for training')


    parser.add_argument('--project_path', type=str, default='./results', help='Path to store project results')
    parser.add_argument('--data_path', type=str, help='Path to the dataset file')
    parser.add_argument('--train_data_list', type=list, default = [0,1,2,3] ,  help='Folds in the dataset for training')
    parser.add_argument('--val_data_list', type=list, default = [0,1,2,3] ,  help='Folds in the dataset for validation')
    parser.add_argument('--save_model_freq', type=int, default = 50 ,  help='Frequency of saving the model checkpoint')

    #model parameters
    parser.add_argument('--model_type', type=str, default = 'Epoch'  ,choices=['Epoch', 'Sequence'],  help='Model type epoch or sequence cross modal transformer')
    parser.add_argument('--d_model ', type=int, default = 256  ,  help='Embedding size of the CMT')
    parser.add_argument('--dim_feedforward', type=int, default = 1024  ,  help='No of neurons in the hidden layer of feed forward block')
    parser.add_argument('--window_size ', type=int, default = 50 ,  help='Size of non-overlapping window')
    
    #training parameters
    parser.add_argument('--batch_size', type=int, default = 32 ,  help='Batch Size')
    
    #For weighted loss
    parser.add_argument('--weigths', type=list, default = [1., 2., 1., 2., 2.] ,  help='Weights for cross entropy loss')
    
    #For Optimizer
    parser.add_argument('--lr', type=float, default = 0.001 ,  help='Learning rate')
    parser.add_argument('--beta_1', type=float, default = 0.9 ,  help='beta 1 for adam optimizer')
    parser.add_argument('--beta_2', type=float, default = 0.999 ,  help='beta 2 for adam optimizer')
    parser.add_argument('--eps', type=float, default = 1e-9 ,  help='eps for adam optimizer')
    parser.add_argument('--weight_decay ', type=float, default = 0.0001 ,  help='weight_decay  for adam optimizer')
    parser.add_argument('--n_epochs', type=int, default = 200 ,  help='No of training epochs')
    
    #For scheduler
    parser.add_argument('--step_size', type=float, default = 30 ,  help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default = 0.5,  help='Gamma for LR scheduler')
    
    #Neptune
    parser.add_argument('--is_neptune', type=bool, default=False, help='Is neptune used to track experiments')
    parser.add_argument('--nep_project', type=str, default='', help='Neptune Project Name')
    parser.add_argument('--nep_api', type=str, default='', help='Neptune API Token')
        
    opt = parser.parse_args()
    
    return opt


def main():
    
    args = parse_option()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(args)
    
    if args.is_neptune:   # Initiate Neptune
        import neptune.new as neptune
        run = neptune.init(project= args.nep_project, api_token=args.nep_api)
    
    if not os.path.isdir(args.project_path):
        os.makedirs(args.project_path)
        print(f"Project directory created at {args.project_path}")
    else:
        print(f"Project directory already available at {args.project_path}")
    
    
    
    
    
    
    
    

        