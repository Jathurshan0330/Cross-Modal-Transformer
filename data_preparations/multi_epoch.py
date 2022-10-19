# !pip install mne
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
import h5py


def signal_extract_sequential(subjects, days, channel = 'eeg1', filter = True, freq = [0.2,40], stride = 3):

  ignore_data = [[13,2],[36,1],[39,1],[39,2],[52,1],[68,1],[68,2], [69,1],[69,2],[78,1],[78,2],[79,1],[79,2]]
  all_channels = ('EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental', 'Resp oro-nasal', 'Temp rectal', 'Event marker')
  
  first_sub_flag = 0
  for sub in subjects:
    for day_ in days:
      if [sub,day_] in ignore_data:
        continue  
      print()
      [data] = fetch_data(subjects = [sub], recording = [day_])
      signal2idx = {"eeg1": 0, "eeg2": 1, "eog": 2, "emg": 3}

      all_channels_list = list(all_channels)
      all_channels_list.remove(all_channels[signal2idx[channel]])
      exclude_channels = tuple(all_channels_list)

      sleep_signals = mne.io.read_raw_edf(data[0], verbose =True, exclude = exclude_channels, preload = True)


      annot = mne.read_annotations(data[1])

      ann2label = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 4,
    "Sleep stage R": 5, "Sleep stage ?": 6, "Movement time": 7}

      ann2label_without_unknown_stages = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 4,
    "Sleep stage R": 5}

      annot.crop(annot[1]['onset'] - 30 * 60,
                  annot[-2]['onset'] + 30 * 60)

      sleep_signals.set_annotations(annot, emit_warning=False)

      events, _ = mne.events_from_annotations(
          sleep_signals, event_id=ann2label, chunk_duration=30.)
      
      #Filtering
      tmax = 30. - 1. / sleep_signals.info['sfreq']

      if filter==True:
        sleep_signals = sleep_signals.copy().filter(l_freq = freq[0], h_freq = freq[1])

      #Breaking into Epochs
      epochs_data = mne.Epochs(raw = sleep_signals, events = events,
                                  event_id = ann2label, tmin = 0., tmax = tmax, baseline = None, preload = True, on_missing = 'warn' )
      
      epochs_data_without_unknown_stages = mne.Epochs(raw = sleep_signals, events = events,
                                  event_id = ann2label_without_unknown_stages, tmin = 0., tmax = tmax, baseline = None, preload = True, on_missing = 'warn' )

      print('===================================================================================================================================')
      print(f"                    Shape of Extracted Raw Signal for Subject {sub} and Day {day_}                          ")
      print(f"                    Shape of Extracted Label for Subject {sub} and Day {day_}                             ")
      # print('===================================================================================================================================')


      sig_epochs = []
      label_epochs = []

      mean_epochs = []
      std_epochs = []

      signal_mean = np.mean(np.array([epochs_data_without_unknown_stages]))
      signal_std = np.std(np.array([epochs_data_without_unknown_stages]))
      

      for ep in range(len(epochs_data)-(stride-1)):
        
        existing_labels = epochs_data[ep:ep+stride].event_id

        if (not('Sleep stage ?' in existing_labels or 'Movement time' in existing_labels)):

          temp_epochs = []
          temp_labels = []

          for i in range(ep,ep+stride):

            for sig in epochs_data[i]:
              temp_epochs.append(sig[0])

              sleep_stage = epochs_data[i].event_id
              if sleep_stage == {"Sleep stage W": 0}:
                temp_labels.append(0)
              if sleep_stage == {"Sleep stage 1": 1}:
                temp_labels.append(1)
              if sleep_stage == {"Sleep stage 2": 2}:
                temp_labels.append(2)
              if sleep_stage == {"Sleep stage 3": 3}:
                temp_labels.append(3)
              if sleep_stage == {"Sleep stage 4": 4}:
                temp_labels.append(3)
              if sleep_stage == {"Sleep stage R": 5}:
                temp_labels.append(4)
            
          sig_epochs.append(temp_epochs)
          label_epochs.append(temp_labels)         
          mean_epochs.append(signal_mean)
          std_epochs.append(signal_std)
      
      sig_epochs = np.array(sig_epochs)
      mean_epochs = np.array(mean_epochs)
      std_epochs = np.array(std_epochs)
      label_epochs = np.array(label_epochs)
            
      if first_sub_flag == 0:
        main_ext_raw_data = sig_epochs
        main_labels = label_epochs
        main_sub_len = np.array([len(label_epochs)])
        main_mean = mean_epochs
        main_std = std_epochs        
        first_sub_flag = 1
      else:
        main_ext_raw_data = np.concatenate((main_ext_raw_data,sig_epochs), axis = 0)
        main_labels = np.concatenate((main_labels,label_epochs), axis = 0)
        main_sub_len =  np.concatenate((main_sub_len,np.array([len(label_epochs)])), axis = 0)
        main_mean = np.concatenate((main_mean, mean_epochs), axis = 0)
        main_std = np.concatenate((main_std, std_epochs), axis = 0)

  return main_ext_raw_data, main_labels, main_sub_len, main_mean, main_std

#Separate Subjects into 5 groups
from sklearn.model_selection import KFold
days = np.arange(1,3) 
subjects = np.arange(0,83) 
print(f"Subjects : {subjects}")
print(f"Days : {days}")

fivefold_list = []
kf = KFold(n_splits=5, shuffle = True # 5, 2
           , random_state = 2
           )


sub_1, sub_2, sub_3, sub_4, sub_5 = kf.split(subjects)
sub_1 = sub_1[1]
sub_2 = sub_2[1]
sub_3 = sub_3[1]
sub_4 = sub_4[1]
sub_5 = sub_5[1]


print(f"Subjects Group 1 : {sub_1}")
print(f"Subjects Group 2 : {sub_2}")
print(f"Subjects Group 3 : {sub_3}")
print(f"Subjects Group 4 : {sub_4}")
print(f"Subjects Group 5 : {sub_5}")

for i in sub_1:
  if i in subjects:
    subjects[i] = 0 
  else:
    print("Error")

for i in sub_2:
  if i in subjects:
    subjects[i] = 0 
  else:
    print("Error")
    
for i in sub_3:
  if i in subjects:
    subjects[i] = 0 
  else:
    print("Error")

for i in sub_4:
  if i in subjects:
    subjects[i] = 0 
  else:
    print("Error")

for i in sub_5:
  if i in subjects:
    subjects[i] = 0 
  else:
    print("Error")

print(subjects)



# ==============================================================>
# Change Channels to extract data for other PSG channels 'eeg1', ''eog', 'eeg2'
# ==============================================================>

eeg1_1, labels_1, len_1, eeg1_m1, eeg1_std1 = signal_extract_sequential(sub_1, days, channel = 'eeg1', filter = True, freq = [0.2,40], stride = 5)
print(f"Train data shape : {eeg1_1.shape}, Train label shape : {labels_1.shape}")

eeg1_2, labels_2, len_2, eeg1_m2, eeg1_std2 =  signal_extract_sequential(sub_2, days, channel = 'eeg1', filter = True, freq = [0.2,40], stride = 5)
print(f"Train data shape : {eeg1_2.shape}, Train label shape : {labels_2.shape}")

eeg1_3, labels_3, len_3, eeg1_m3, eeg1_std3 =  signal_extract_sequential(sub_3, days, channel = 'eeg1', filter = True, freq = [0.2,40], stride = 5)
print(f"Train data shape : {eeg1_3.shape}, Train label shape : {labels_3.shape}")

eeg1_4, labels_4, len_4, eeg1_m4, eeg1_std4 =  signal_extract_sequential(sub_4, days, channel = 'eeg1', filter = True, freq = [0.2,40], stride = 5)
print(f"Train data shape : {eeg1_4.shape}, Train label shape : {labels_4.shape}")

eeg1_5, labels_5, len_5, eeg1_m5, eeg1_std5 =  signal_extract_sequential(sub_5, days, channel = 'eeg1', filter = True, freq = [0.2,40], stride = 5)
print(f"Train data shape : {eeg1_5.shape}, Train label shape : {labels_5.shape}")