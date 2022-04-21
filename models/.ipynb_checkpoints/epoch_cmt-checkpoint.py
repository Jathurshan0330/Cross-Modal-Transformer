class Epoch_Cross_Transformer_Network(nn.Module):
    def __init__(self,d_model = 64, dim_feedforward=512,window_size = 25): #  filt_ch = 4
        super(Cross_Transformer_Network, self).__init__()
        
        self.eeg_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1,
                                            window_size =window_size, First = True )
        self.eog_atten = Intra_modal_atten(d_model=d_model, nhead=8, dropout=0.1, 
                                            window_size =window_size, First = True )
        
        self.cross_atten = Cross_modal_atten(d_model=d_model, nhead=8, dropout=0.1, First = True )
        
        self.eeg_ff = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)
        self.eog_ff = Feed_forward(d_model = d_model,dropout=0.1,dim_feedforward = dim_feedforward)


        self.mlp    = nn.Sequential(nn.Flatten(),
                                    nn.Linear(d_model*2,5))  ##################
        # 

    def forward(self, eeg: Tensor,eog: Tensor,finetune = False): 
        self_eeg = self.eeg_atten(eeg)
        self_eog = self.eog_atten(eog)

        cross = self.cross_atten(self_eeg[:,0,:],self_eog[:,0,:])

        cross_cls = cross[:,0,:].unsqueeze(dim=1)
        cross_eeg = cross[:,1,:].unsqueeze(dim=1)
        cross_eog = cross[:,2,:].unsqueeze(dim=1)

        eeg_new =  torch.cat([cross_cls, self_eeg[:,1:,:]], dim=1)
        eog_new =  torch.cat([cross_cls, self_eog[:,1:,:]], dim=1)

        ff_eeg = self.eeg_ff(eeg_new)
        ff_eog = self.eog_ff(eog_new)

        

        cls_out = torch.cat([ff_eeg[:,0,:], ff_eog[:,0,:]], dim=1).unsqueeze(dim=1) 

        feat_list = [cross_cls,ff_eeg,ff_eog]
        if finetune == True:
            out = self.mlp(cls_out)  #########
            return out,cls_out,feat_list
        else:
            return cls_out