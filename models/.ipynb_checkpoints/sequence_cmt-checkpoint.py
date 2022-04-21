import copy
from typing import Optional, Any
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d

from models.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward

class Epoch_Block(nn.Module):
    def __init__(self,d_model = 64, dim_feedforward=512,window_size = 25): #  filt_ch = 4
        super(Epoch_Block, self).__init__()
        
        self.eeg_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1,
                                            window_size =window_size, First = True )
        self.eog_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, 
                                            window_size =window_size, First = True )
        
        self.cross_atten = Cross_modal_atten(d_model=d_model, nhead=8, dropout=0.1, First = True )
        
        

    def forward(self, eeg: Tensor,eog: Tensor):#,finetune = False): 
        self_eeg = self.eeg_atten(eeg)
        self_eog = self.eog_atten(eog)

        cross = self.cross_atten(self_eeg[:,0,:],self_eog[:,0,:])

        cross_cls = cross[:,0,:].unsqueeze(dim=1)
        cross_eeg = cross[:,1,:].unsqueeze(dim=1)
        cross_eog = cross[:,2,:].unsqueeze(dim=1)

        feat_list = [self_eeg,self_eog]  
          
        return cross_cls,feat_list
    
    
class Seq_Cross_Transformer_Network(nn.Module):
    def __init__(self,d_model = 64, dim_feedforward=512,window_size = 25): #  filt_ch = 4
        super(Seq_Cross_Transformer_Network, self).__init__()
        
        self.epoch_1 = Epoch_Block(d_model = d_model, dim_feedforward=dim_feedforward,
                                                window_size = window_size)
        self.epoch_2 = Epoch_Block(d_model = d_model, dim_feedforward=dim_feedforward,
                                                window_size = window_size)
        self.epoch_3 = Epoch_Block(d_model = d_model, dim_feedforward=dim_feedforward,
                                                window_size = window_size)
        self.epoch_4 = Epoch_Block(d_model = d_model, dim_feedforward=dim_feedforward,
                                                window_size = window_size)
        self.epoch_5 = Epoch_Block(d_model = d_model, dim_feedforward=dim_feedforward,
                                                window_size = window_size)
        
        self.seq_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, 
                                            window_size =window_size, First = False )

        self.ff_net = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)


        self.mlp_1    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))  ##################
        self.mlp_2    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_3    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_4    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))
        self.mlp_5    = nn.Sequential(nn.Flatten(),nn.Linear(d_model,5))   
        # 

    def forward(self, eeg: Tensor,eog: Tensor,num_seg = 5): 
        epoch_1 = self.epoch_1(eeg[:,:,0,:],eog[:,:,0,:])[0]
        epoch_2 = self.epoch_2(eeg[:,:,1,:],eog[:,:,1,:])[0]
        epoch_3 = self.epoch_3(eeg[:,:,2,:],eog[:,:,2,:])[0]
        epoch_4 = self.epoch_4(eeg[:,:,3,:],eog[:,:,3,:])[0]
        epoch_5 = self.epoch_5(eeg[:,:,4,:],eog[:,:,4,:])[0]

        seq =  torch.cat([epoch_1, epoch_2,epoch_3,epoch_4,epoch_5], dim=1)
        seq = self.seq_atten(seq)
        seq = self.ff_net(seq)
        out_1 = self.mlp_1(seq[:,0,:])
        out_2 = self.mlp_2(seq[:,1,:])
        out_3 = self.mlp_3(seq[:,2,:])
        out_4 = self.mlp_4(seq[:,3,:])
        out_5 = self.mlp_5(seq[:,4,:])

        return [out_1,out_2,out_3,out_4,out_5]
        