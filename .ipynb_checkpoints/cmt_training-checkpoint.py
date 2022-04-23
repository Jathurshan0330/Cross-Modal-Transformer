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
from models.epoch_cmt import Epoch_Cross_Transformer_Network,train_epoch_cmt
from models.sequence_cmt import Seq_Cross_Transformer_Network, train_seq_cmt 
from utils.metrics import accuracy, kappa, g_mean, plot_confusion_matrix, confusion_matrix, AverageMeter 


def parse_option():
    parser = argparse.ArgumentParser('Argument for training')


    parser.add_argument('--project_path', type=str, default='./results', help='Path to store project results')
    parser.add_argument('--data_path', type=str, help='Path to the dataset file')
    parser.add_argument('--train_data_list', nargs="+", default = [0,1,2,3] ,  help='Folds in the dataset for training')
    parser.add_argument('--val_data_list', nargs="+", default = [4] ,  help='Folds in the dataset for validation')
    parser.add_argument('--is_retrain', type=bool, default=False,   help='To retrain a from saved checkpoint')
    parser.add_argument('--model_path', type=str, default="",   help='Path to saved checkpoint for retraining')
    parser.add_argument('--save_model_freq', type=int, default = 50 ,  help='Frequency of saving the model checkpoint')

    #model parameters
    parser.add_argument('--model_type', type=str, default = 'Epoch'  ,choices=['Epoch', 'Seq'],  help='Model type')
    parser.add_argument('--d_model', type=int, default = 256,  help='Embedding size of the CMT')
    parser.add_argument('--dim_feedforward', type=int, default = 1024,  help='No of neurons feed forward block')
    parser.add_argument('--window_size', type=int, default = 50,  help='Size of non-overlapping window')
    parser.add_argument('--num_seq', type=int, default = 5,  help='Number of epochs in a PSG sequence')
    #training parameters
    parser.add_argument('--batch_size', type=int, default = 32 ,  help='Batch Size')
    
    #For weighted loss
    parser.add_argument('--weigths', type=list, default = [1., 2., 1., 2., 2.] ,  help='Weights for cross entropy loss')
    
    #For Optimizer
    parser.add_argument('--lr', type=float, default = 0.001 ,  help='Learning rate')
    parser.add_argument('--beta_1', type=float, default = 0.9 ,  help='beta 1 for adam optimizer')
    parser.add_argument('--beta_2', type=float, default = 0.999 ,  help='beta 2 for adam optimizer')
    parser.add_argument('--eps', type=float, default = 1e-9 ,  help='eps for adam optimizer')
    parser.add_argument('--weight_decay', type=float, default = 0.0001 ,  help='weight_decay  for adam optimizer')
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
    print("Training    Arguements ====================================>")
    for arg in vars(args):
        print(f"    {arg} :  {getattr(args, arg)}")
    
    if args.is_neptune:   # Initiate Neptune
        import neptune.new as neptune
        run = neptune.init(project= args.nep_project, api_token=args.nep_api)
    
    if not os.path.isdir(args.project_path):
        os.makedirs(args.project_path)
        print(f"Project directory created at {args.project_path}")
    else:
        print(f"Project directory already available at {args.project_path}")
    
    
    #Get Dataset
    print("Getting Dataset ===================================>")
    train_data_loader, val_data_loader = get_dataset(device,args)
    
    
    ##Load Model
    if args.model_type == "Epoch":   # Initialize epoch cross-modal transformer
        if args.is_retrain:
            print(f"Loading previous checkpoint from {args.model_path}")
            Net = torch.load(f"{args.model_path}")
        else:
            print(f"Initializing Epoch Cross Modal Transformer ==================>")
            Net = Epoch_Cross_Transformer_Network(d_model = args.d_model, dim_feedforward = args.dim_feedforward,
                                                  window_size = args.window_size ).to(device)
    
    if args.model_type == "Seq":   # Initialize sequence cross-modal transformer
        if args.is_retrain:
            print(f"Loading previous checkpoint from {args.model_path}")
            Net = torch.load(f"{args.model_path}")
        else:
            print(f"Initializing Sequence Cross Modal Transformer ==================>")
            Net = Seq_Cross_Transformer_Network(d_model = args.d_model, dim_feedforward = args.dim_feedforward,
                                window_size = args.window_size ).to(device)

    weights = torch.tensor(args.weigths)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(Net.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2),
                                 eps = args.eps, weight_decay = args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma) 
        
    
    if args.is_neptune:
        parameters = {
        "Experiment" : "Training test",
        'Model Type' : f"{args.model_type} Cross-Modal Transformer",
        'd_model' : args.d_model,
        'dim_feedforward' : args.dim_feedforward,
        'window_size ':args.window_size ,
        'Batch Size': args.batch_size,
        'Loss': f"Weighted Categorical Loss,{args.weights}",  # Check this every time
        'Optimizer' : "Adam",        # Check this every time   
        'Learning Rate': args.lr,
        'eps' : args.eps,
        "LR Schduler": "StepLR",
        'Beta 1': args.beta_1,
        'Beta 2': args.beta_2,
        'n_epochs': args.n_epochs,
        'val_set' : args.val_data_list[0]+1
        }
        run['model/parameters'] = parameters
        run['model/model_architecture'] = Net
    
    
    if not os.path.isdir(os.path.join(args.project_path,"model_check_points")):
        os.makedirs(os.path.join(args.project_path,"model_check_points"))
    
    
    # Train Epoch Cross-Modal Transformer
    if args.model_type == "Epoch":  
        train_epoch_cmt(Net, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler, device, args)
        
    # Train Seq Cross-Modal Transformer
    if args.model_type == "Seq":  
        train_seq_cmt(Net, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler, device, args)  
        
if __name__ == '__main__':
    main()
        

        
# Training Epoch CMT        
#python cmt_training.py --project_path "testing" --data_path "/home/mmsm/Sleep_EDF_Dataset" --train_data_list [0,1,2,3] --val_data_list [4] --model_type "Epoch" --is_neptune True --nep_project "jathurshan0330/V2-Cros" --nep_api "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYmRmNjE0Zi0xMDRkLTRlNzUtYmIxNi03NzM2ODBlZDc5NTMifQ=="
