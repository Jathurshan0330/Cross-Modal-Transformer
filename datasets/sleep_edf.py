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