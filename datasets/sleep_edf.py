"""
Dataset Library
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


class SleepEDF_MultiChan_Dataset(Dataset):
    def __init__(self, eeg_file, eog_file, eeg2_file, label_file, device, mean_eeg_l = None, sd_eeg_l = None, 
                 mean_eog_l = None, sd_eog_l = None, mean_eeg2_l = None, sd_eeg2_l = None,transform=None, 
                 target_transform=None, sub_wise_norm = False, data_type = None):
        """
      
        """
        # Get the data
        for i in range(len(eeg_file)):
          if i == 0:
            self.eeg = read_h5py(eeg_file[i])
            self.eog = read_h5py(eog_file[i])
            # self.eeg2 = read_h5py(eeg2_file[i])

            self.labels = read_h5py(label_file[i])
          else:
            self.eeg = np.concatenate((self.eeg, read_h5py(eeg_file[i])),axis = 0)
            self.eog = np.concatenate((self.eog, read_h5py(eog_file[i])),axis = 0)
            # self.eeg2 = np.concatenate((self.eeg2, read_h5py(eeg2_file[i])),axis = 0)
            self.labels = np.concatenate((self.labels, read_h5py(label_file[i])),axis = 0)

        self.labels = torch.from_numpy(self.labels)
        if data_type == 'train':   # Removing wake epochs
          wake = np.array(np.where(self.labels == 0))
          print(wake.shape)
          wake = random.choices(wake[0], k = int(wake.shape[1]*0.65))
          print(len(wake))
          self.labels = np.delete(self.labels,obj = wake,axis = 0)
          self.eeg = np.delete(self.eeg,obj = wake,axis = 0)
          self.eog = np.delete(self.eog,obj = wake,axis = 0)

        bin_labels = np.bincount(self.labels)
        print(f"Labels count: {bin_labels}")
        print(f"Labels count weights: {1/bin_labels}")
        print(f"Shape of EEG : {self.eeg.shape} , EOG : {self.eog.shape}")#, EMG: {self.eeg2.shape}")
        print(f"Shape of Labels : {self.labels.shape}")

        if sub_wise_norm == True:
          print(f"Reading Subject wise mean and sd")
          for i in range(len(mean_eeg_l)):
            if i == 0:
              self.mean_eeg  = read_h5py(mean_eeg_l[i])
              self.sd_eeg = read_h5py(sd_eeg_l[i])
              self.mean_eog  = read_h5py(mean_eog_l[i])
              self.sd_eog = read_h5py(sd_eog_l[i])
              # self.mean_eeg2  = read_h5py(mean_eeg2_l[i])
              # self.sd_eeg2 = read_h5py(sd_eeg2_l[i])
            else:
              self.mean_eeg = np.concatenate((self.mean_eeg, read_h5py(mean_eeg_l[i])),axis = 0)
              self.sd_eeg = np.concatenate((self.sd_eeg, read_h5py(sd_eeg_l[i])),axis = 0)
              self.mean_eog = np.concatenate((self.mean_eog, read_h5py(mean_eog_l[i])),axis = 0)
              self.sd_eog = np.concatenate((self.sd_eog, read_h5py(sd_eog_l[i])),axis = 0)
              # self.mean_eeg2 = np.concatenate((self.mean_eeg2, read_h5py(mean_eeg2_l[i])),axis = 0)
              # self.sd_eeg2 = np.concatenate((self.sd_eeg2, read_h5py(sd_eeg2_l[i])),axis = 0)
          if data_type == 'train':   # Removing wake epochs
            self.mean_eeg = np.delete(self.mean_eeg,obj = wake,axis = 0)
            self.sd_eeg = np.delete(self.sd_eeg,obj = wake,axis = 0)
            self.mean_eog = np.delete(self.mean_eog,obj = wake,axis = 0)
            self.sd_eog = np.delete(self.sd_eog,obj = wake,axis = 0)
          
          print(f"Shapes of Mean  : EEG: {self.mean_eeg.shape}, EOG : {self.mean_eog.shape}")#, EMG : {self.mean_eeg2.shape}")
          print(f"Shapes of Sd  : EEG: {self.sd_eeg.shape}, EOG : {self.sd_eog.shape}")#, EMG : {self.sd_eeg2.shape}")
        else:     
          self.mean = None#mean_l
          self.sd = None #sd_l
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

        # eeg_data = (eeg_data - np.mean(eeg_data))/np.std(eeg_data)
        # eog_data = (eog_data - np.mean(eog_data))/np.std(eog_data)
        # eeg2_data = self.eeg2[idx]            
        # print(data.shape)
        label = self.labels[idx,]
        
        if self.sub_wise_norm ==True:
          eeg_data = (eeg_data - self.mean_eeg[idx]) / self.sd_eeg[idx]
          eog_data = (eog_data - self.mean_eog[idx]) / self.sd_eog[idx]
          # eeg2_data = (eeg2_data - self.mean_eeg2[idx]) / self.sd_eeg2[idx]
        elif self.mean and self.sd:
          eeg_data = (eeg_data-self.mean[0])/self.sd[0]
          eog_data = (eog_data-self.mean[1])/self.sd[1]
          # eeg2_data = (eeg2_data-self.mean[2])/self.sd[2]

        #SG filtering
        # eeg_data = scipy.signal.savgol_filter(eeg_data, 51, 4)
        # eog_data = scipy.signal.savgol_filter(eog_data, 51, 4)

        if self.transform:
            eeg_data = self.transform(eeg_data)
            eog_data = self.transform(eog_data)
            # eeg2_data = self.transform(eeg2_data)
        if self.target_transform:
            label = self.target_transform(label)
        return eeg_data, eog_data, label