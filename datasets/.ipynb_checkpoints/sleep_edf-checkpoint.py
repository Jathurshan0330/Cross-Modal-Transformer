"""
Sleepedf Dataset Library
"""
import torch
from torchvision import transforms, datasets
import h5py
import numpy as np
from pathlib import Path
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import os
import glob
import os
import matplotlib.pyplot as plt
def split_data(data_list,train_list,val_list):
    data_list = np.array(data_list)
    train_data_list = data_list[train_list]
    val_data_list = data_list[val_list]
    return train_data_list, val_data_list



def read_h5py(path):
    with h5py.File(path, "r") as f:
        print(f"Reading from {path} ====================================================")
        print("Keys in the h5py file : %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data1 = np.array((f[a_group_key]))
        print(f"Number of samples : {len(data1)}")
        print(f"Shape of each data : {data1.shape}")
    return data1



# For One-to-one Classification
class SleepEDF_MultiChan_Dataset(Dataset):
    def __init__(self, eeg_file, eog_file, label_file, device, mean_eeg_l = None, sd_eeg_l = None, 
                 mean_eog_l = None, sd_eog_l = None,transform=None, 
                 target_transform=None, sub_wise_norm = False):
        """
      
        """
        # Get the data
        for i in range(len(eeg_file)):
          if i == 0:
            self.eeg = read_h5py(eeg_file[i])
            self.eog = read_h5py(eog_file[i])

            self.labels = read_h5py(label_file[i])
          else:
            self.eeg = np.concatenate((self.eeg, read_h5py(eeg_file[i])),axis = 0)
            self.eog = np.concatenate((self.eog, read_h5py(eog_file[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

        self.labels = torch.from_numpy(self.labels)
        
        bin_labels = np.bincount(self.labels)
        print(f"Labels count: {bin_labels}")
        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}")
        print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_eeg_l)):
            if i == 0:
              self.mean_eeg  = read_h5py(mean_eeg_l[i])
              self.sd_eeg = read_h5py(sd_eeg_l[i])
              self.mean_eog  = read_h5py(mean_eog_l[i])
              self.sd_eog = read_h5py(sd_eog_l[i])
            else:
              self.mean_eeg = np.concatenate((self.mean_eeg, read_h5py(mean_eeg_l[i])),axis = 0)
              self.sd_eeg = np.concatenate((self.sd_eeg, read_h5py(sd_eeg_l[i])),axis = 0)
              self.mean_eog = np.concatenate((self.mean_eog, read_h5py(mean_eog_l[i])),axis = 0)
              self.sd_eog = np.concatenate((self.sd_eog, read_h5py(sd_eog_l[i])),axis = 0)
          
          print(f"Shapes of Mean  : EEG: {self.mean_eeg.shape}, EOG : {self.mean_eog.shape}")#, EMG : {self.mean_eeg2.shape}")
          print(f"Shapes of Sd  : EEG: {self.sd_eeg.shape}, EOG : {self.sd_eog.shape}")#, EMG : {self.sd_eeg2.shape}")
        else:     
          self.mean = None
          self.sd = None 
          print(f"Mean : {self.mean} and SD {self.sd}")  

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_data = self.eeg[idx]    
        eog_data = self.eog[idx]

        label = self.labels[idx,]
        
        if self.sub_wise_norm ==True:
          eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
          eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
        elif self.mean and self.sd:
          eeg_data = (eeg_data-self.mean[0])/self.sd[0]
          eog_data = (eog_data-self.mean[1])/self.sd[1]

        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
        if self.target_transform:
            label = self.target_transform(label)
        return eeg_data, eog_data, label
    
    
    
# For Many-to-many Classification

class SleepEDF_Seq_MultiChan_Dataset(Dataset):
    def __init__(self, eeg_file, eog_file, label_file, device, mean_eeg_l = None, sd_eeg_l = None, 
                 mean_eog_l = None, sd_eog_l = None,transform=None, target_transform=None, 
                 sub_wise_norm = False, data_type = None, num_seq = 5):
        """
      
        """
        # Get the data
        for i in range(len(eeg_file)):
          if i == 0:
            self.eeg = read_h5py(eeg_file[i])
            self.eog = read_h5py(eog_file[i])
        

            self.labels = read_h5py(label_file[i])
          else:
            self.eeg = np.concatenate((self.eeg, read_h5py(eeg_file[i])),axis = 0)
            self.eog = np.concatenate((self.eog, read_h5py(eog_file[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

        self.labels = torch.from_numpy(self.labels)
        

        bin_labels = np.bincount(self.labels)
        print(f"Labels count: {bin_labels}")
        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}")
        print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_eeg_l)):
            if i == 0:
              self.mean_eeg  = read_h5py(mean_eeg_l[i])
              self.sd_eeg = read_h5py(sd_eeg_l[i])
              self.mean_eog  = read_h5py(mean_eog_l[i])
              self.sd_eog = read_h5py(sd_eog_l[i])
            else:
              self.mean_eeg = np.concatenate((self.mean_eeg, read_h5py(mean_eeg_l[i])),axis = 0)
              self.sd_eeg = np.concatenate((self.sd_eeg, read_h5py(sd_eeg_l[i])),axis = 0)
              self.mean_eog = np.concatenate((self.mean_eog, read_h5py(mean_eog_l[i])),axis = 0)
              self.sd_eog = np.concatenate((self.sd_eog, read_h5py(sd_eog_l[i])),axis = 0)
          
          print(f"Shapes of Mean  : EEG: {self.mean_eeg.shape}, EOG : {self.mean_eog.shape}")
          print(f"Shapes of Sd  : EEG: {self.sd_eeg.shape}, EOG : {self.sd_eog.shape}")
        else:     
          self.mean = mean_l
          self.sd = sd_l
          print(f"Mean : {self.mean} and SD {self.sd}")  

        self.sub_wise_norm = sub_wise_norm
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.num_seq = num_seq

    def __len__(self):
        return len(self.labels) - self.num_seq

    def __getitem__(self, idx):
        eeg_data = self.eeg[idx:idx+self.num_seq].squeeze()   
        eog_data = self.eog[idx:idx+self.num_seq].squeeze() 
        label = self.labels[idx:idx+self.num_seq,]   #######
      
        if self.sub_wise_norm ==True:
          eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
          eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
        elif self.mean and self.sd:
          eeg_data = (eeg_data-self.mean[0])/self.sd[0]
          eog_data = (eog_data-self.mean[1])/self.sd[1]
        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
        if self.target_transform:
            label = self.target_transform(label)
        return eeg_data, eog_data, label
    
    
    

def get_dataset(device,args):
    args.train_data_list = list(args.train_data_list[0])
    args.train_data_list = [ int(x) for x in args.train_data_list if x.isdigit() ]
    args.val_data_list = list(args.val_data_list[0])
    args.val_data_list = [ int(x) for x in args.val_data_list if x.isdigit() ]
    
    eeg_list = glob.glob(f'{args.data_path}/x*.h5')
    eeg_list.sort()
    [train_eeg_list, val_eeg_list] = split_data(eeg_list,args.train_data_list,args.val_data_list)

    mean_eeg_list = glob.glob(f'{args.data_path}/mean*.h5')
    mean_eeg_list.sort()
    [train_mean_eeg_list, val_mean_eeg_list] = split_data(mean_eeg_list,args.train_data_list,args.val_data_list)

    sd_eeg_list = glob.glob(f'{args.data_path}/std*.h5')
    sd_eeg_list.sort()
    [train_sd_eeg_list, val_sd_eeg_list] = split_data(sd_eeg_list,args.train_data_list,args.val_data_list)

    eog_list = glob.glob(f'{args.data_path}/eog*.h5')
    eog_list.sort()
    [train_eog_list, val_eog_list] = split_data(eog_list,args.train_data_list,args.val_data_list)

    mean_eog_list = glob.glob(f'{args.data_path}/eog_m*.h5')
    mean_eog_list.sort()
    [train_mean_eog_list, val_mean_eog_list] = split_data(mean_eog_list,args.train_data_list,args.val_data_list)

    sd_eog_list = glob.glob(f'{args.data_path}/eog_s*.h5')
    sd_eog_list.sort()
    [train_sd_eog_list, val_sd_eog_list] = split_data(sd_eog_list,args.train_data_list,args.val_data_list)

    label_list = glob.glob(f'{args.data_path}/y*.h5')
    label_list.sort()
    [train_label_list, val_label_list] = split_data(label_list,args.train_data_list,args.val_data_list)
    
    print("Training Data Files: ===========================>")
    print(train_eeg_list)
    print(train_eog_list)
    print(train_label_list)
    print(train_mean_eeg_list)
    print(train_sd_eeg_list)
    print(train_mean_eog_list)
    print(train_sd_eog_list)
    
    print("Validation Data Files: ===========================>")
    print(val_eeg_list)
    print(val_eog_list)
    print(val_label_list)
    print(val_mean_eeg_list)
    print(val_sd_eeg_list)
    print(val_mean_eog_list)
    print(val_sd_eog_list)
    
    
    if args.model_type == "Epoch":   # Dataset to train epoch transformer
           
        train_dataset = SleepEDF_MultiChan_Dataset(eeg_file = train_eeg_list , 
                                       eog_file = train_eog_list, 
                                       label_file = train_label_list, 
                                       device = device, mean_eeg_l = train_mean_eeg_list, sd_eeg_l = train_sd_eeg_list, 
                                       mean_eog_l = train_mean_eog_list, sd_eog_l = train_sd_eog_list, 
                                       sub_wise_norm = True, 
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                            ]) )

        val_dataset = SleepEDF_MultiChan_Dataset(eeg_file = val_eeg_list ,
                                         eog_file = val_eog_list, 
                                         label_file = val_label_list, 
                                         device = device, mean_eeg_l = val_mean_eeg_list, sd_eeg_l = val_sd_eeg_list,
                                         mean_eog_l = val_mean_eog_list, sd_eog_l = val_sd_eog_list,
                                         sub_wise_norm = True,
                                         transform=transforms.Compose([
                                               transforms.ToTensor(),
                                                ]) )
        
        train_data_loader = data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
        val_data_loader = data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)
        
        eeg_data, eog_data, label = next(iter(train_data_loader))
        print(f"EEG batch shape: {eeg_data.size()}")
        print(f"EOG batch shape: {eog_data.size()}")
        print(f"Labels batch shape: {label.size()}")

        t = np.arange(0,30,1/100)
        fig = plt.figure(figsize = (15,5))
        plt.plot(t,eeg_data[0].squeeze(),label="EEG1")
        plt.plot(t,eog_data[0].squeeze()+5,label="E0G")
        plt.title(f"Label {label[0].squeeze()}")
        plt.legend()
        plt.show()
        fig.savefig(os.path.join(args.project_path,"train_sample.png"))


        eeg_data, eog_data, label = next(iter(val_data_loader))
        print(f"EEG batch shape: {eeg_data.size()}")
        print(f"EOG batch shape: {eog_data.size()}")
        print(f"Labels batch shape: {label.size()}")

        t = np.arange(0,30,1/100)
        fig = plt.figure(figsize = (10,10))
        plt.plot(t,eeg_data[0].squeeze())
        plt.plot(t,eog_data[0].squeeze()+5)
        plt.title(f"Label {label[0].squeeze()}")
        plt.show()
        fig.savefig(os.path.join(args.project_path,"val_sample.png"))
    return train_data_loader, val_data_loader

