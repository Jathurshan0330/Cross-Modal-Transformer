import copy
from typing import Optional, Any
import torch
import torch.nn as nn
import time
import argparse
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from models.model_blocks import PositionalEncoding, Window_Embedding, Intra_modal_atten, Cross_modal_atten, Feed_forward
from utils.metrics import accuracy, kappa, g_mean, plot_confusion_matrix, confusion_matrix, AverageMeter

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
        
        
        
def train_seq_cmt(Net, train_data_loader, val_data_loader, criterion,optimizer, lr_scheduler,device, args):
    # Training the model
    best_val_acc = 0
    best_val_kappa = 0
    for epoch_idx in range(args.n_epochs):  # loop over the dataset multiple times
        if args.is_neptune:
            run['train/epoch/learning_Rate'].log(optimizer.param_groups[0]["lr"]) 
        Net.train()
        print(f'============================= Training Epoch : [{epoch_idx+1}/{args.n_epochs}]===============================>')
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        val_losses = AverageMeter()

        train_accuracy = AverageMeter()
        val_accuracy = AverageMeter()

        train_sensitivity = AverageMeter()
        val_sensitivity = AverageMeter()

        train_specificity = AverageMeter()
        val_specificity = AverageMeter()

        train_gmean = AverageMeter()
        val_gmean = AverageMeter()

        train_kappa = AverageMeter()
        val_kappa = AverageMeter()

        train_f1_score = AverageMeter()
        val_f1_score = AverageMeter()

        train_precision = AverageMeter()
        val_precision = AverageMeter()

        class1_sens = AverageMeter()
        class2_sens = AverageMeter()
        class3_sens = AverageMeter()
        class4_sens = AverageMeter()
        class5_sens = AverageMeter()

        class1_spec = AverageMeter()
        class2_spec = AverageMeter()
        class3_spec = AverageMeter()
        class4_spec = AverageMeter()
        class5_spec = AverageMeter()

        class1_f1 = AverageMeter()
        class2_f1 = AverageMeter()
        class3_f1 = AverageMeter()
        class4_f1 = AverageMeter()
        class5_f1 = AverageMeter()

        end = time.time()

        for batch_idx, data_input in enumerate(train_data_loader):
            data_time.update(time.time() - end)
            eeg,eog, labels = data_input
            cur_batch_size = len(eeg)
            optimizer.zero_grad()

            outputs = Net(eeg.float().to(device), eog.float().to(device))

            loss = 0
            for ep in range(args.num_seq):
                loss += criterion(outputs[ep].cpu(), labels[:,ep])
                train_accuracy.update(accuracy(outputs[ep].cpu(), labels[:,ep]))
                _,_,_,_,sens,spec,f1, prec = confusion_matrix(outputs[ep].cpu(), labels[:,ep], 5, cur_batch_size)
                train_sensitivity.update(sens)
                train_specificity.update(spec)
                train_f1_score.update(f1)
                train_precision.update(prec)
                train_gmean.update(g_mean(sens, spec))
                train_kappa.update(kappa(outputs[ep].cpu(), labels[:,ep]))

            loss.backward()
            optimizer.step()
            # scheduler.step()
            losses.update(loss)

            if args.is_neptune:
                run['train/epoch/batch_loss'].log(losses.val)     
                run['train/epoch/batch_accuracy'].log(train_accuracy.val)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if batch_idx % 1000 == 0:

                msg = 'Epoch: [{0}/{3}][{1}/{2}]\t' \
                      'Train_Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                      'Train_Acc {train_acc.val:.5f} ({train_acc.avg:.5f})\t'\
                      'Train_G-Mean {train_gmean.val:.5f}({train_gmean.avg:.5f})\t'\
                      'Train_Kappa {train_kap.val:.5f}({train_kap.avg:.5f})\t'\
                      'Train_MF1 {train_mf1.val:.5f}({train_mf1.avg:.5f})\t'\
                      'Train_Precision {train_prec.val:.5f}({train_prec.avg:.5f})\t'\
                      'Train_Sensitivity {train_sens.val:.5f}({train_sens.avg:.5f})\t'\
                      'Train_Specificity {train_spec.val:.5f}({train_spec.avg:.5f})\t'\
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'.format(
                          epoch_idx+1, batch_idx, len(train_data_loader),args.n_epochs, batch_time=batch_time,
                          speed=data_input[0].size(0)/batch_time.val,
                          data_time=data_time, loss=losses, train_acc = train_accuracy,
                          train_sens =train_sensitivity, train_spec = train_specificity, train_gmean = train_gmean,
                          train_kap = train_kappa, train_mf1 = train_f1_score, train_prec = train_precision)
                print(msg)


        #evaluation
        with torch.no_grad():
          Net.eval()
          for batch_val_idx, data_val in enumerate(val_data_loader):
            val_eeg,val_eog, val_labels = data_val
            cur_val_batch_size = len(val_eeg)
            pred = Net(val_eeg.float().to(device), val_eog.float().to(device))

            for ep in range(args.num_seq):
                  val_loss += criterion(pred[ep].cpu(), val_labels[:,ep])

                  val_accuracy.update(accuracy(pred[ep].cpu(), val_labels[:,ep]))
                  sens_list,spec_list,f1_list,prec_list, sens,spec,f1,prec = confusion_matrix(pred[ep].cpu(), val_labels[:,ep],  5, cur_val_batch_size)
                  val_sensitivity.update(sens)
                  val_specificity.update(spec)
                  val_f1_score.update(f1)
                  val_precision.update(prec)
                  val_gmean.update(g_mean(sens, spec))
                  val_kappa.update(kappa(pred[ep].cpu(), val_labels[:,ep]))

                  class1_sens.update(sens_list[0])
                  class2_sens.update(sens_list[1])
                  class3_sens.update(sens_list[2])
                  class4_sens.update(sens_list[3])
                  class5_sens.update(sens_list[4])

                  class1_spec.update(spec_list[0])
                  class2_spec.update(spec_list[1])
                  class3_spec.update(spec_list[2])
                  class4_spec.update(spec_list[3])
                  class5_spec.update(spec_list[4])

                  class1_f1.update(f1_list[0])
                  class2_f1.update(f1_list[1])
                  class3_f1.update(f1_list[2])
                  class4_f1.update(f1_list[3])
                  class5_f1.update(f1_list[4])

        
          val_losses.update(val_loss)
        

      # print(batch_val_idx)



        print(f'===========================================================Epoch : [{epoch_idx+1}/{args.n_epochs}]  Evaluation ===========================================================================================================>')
        print("Training Results : ")
        print(f"Training Loss     : {losses.avg}, Training Accuracy      : {train_accuracy.avg}, Training G-Mean      : {train_gmean.avg}") 
        print(f"Training Kappa      : {train_kappa.avg},Training MF1     : {train_f1_score.avg}, Training Precision      : {train_precision.avg}, Training Sensitivity      : {train_sensitivity.avg}, Training Specificity      : {train_specificity.avg}")

        print("Validation Results : ")
        print(f"Validation Loss   : {val_losses.avg}, Validation Accuracy : {val_accuracy.avg}, Validation G-Mean      : {val_gmean.avg}") 
        print(f"Validation Kappa     : {val_kappa.avg}, Validation MF1      : {val_f1_score.avg}, Validation Precision      : {val_precision.avg},  Validation Sensitivity      : {val_sensitivity.avg}, Validation Specificity      : {val_specificity.avg}")


        print(f"Class wise sensitivity W: {class1_sens.avg}, S1: {class2_sens.avg}, S2: {class3_sens.avg}, S3: {class4_sens.avg}, R: {class5_sens.avg}")
        print(f"Class wise specificity W: {class1_spec.avg}, S1: {class2_spec.avg}, S2: {class3_spec.avg}, S3: {class4_spec.avg}, R: {class5_spec.avg}")
        print(f"Class wise F1  W: {class1_f1.avg}, S1: {class2_f1.avg}, S2: {class3_f1.avg}, S3: {class4_f1.avg}, R: {class5_f1.avg}")

        if args.is_neptune:
            run['train/epoch/epoch_train_loss'].log(losses.avg)
            run['train/epoch/epoch_val_loss'].log(val_losses.avg)

            run['train/epoch/epoch_train_accuracy'].log(train_accuracy.avg)
            run['train/epoch/epoch_val_accuracy'].log(val_accuracy.avg)

            run['train/epoch/epoch_train_sensitivity'].log(train_sensitivity.avg)
            run['train/epoch/epoch_val_sensitivity'].log(val_sensitivity.avg)

            run['train/epoch/epoch_train_specificity'].log(train_specificity.avg)
            run['train/epoch/epoch_val_specificity'].log(val_specificity.avg)

            run['train/epoch/epoch_train_G-Mean'].log(train_gmean.avg)
            run['train/epoch/epoch_val_G-Mean'].log(val_gmean.avg)

            run['train/epoch/epoch_train_Kappa'].log(train_kappa.avg)
            run['train/epoch/epoch_val_Kappa'].log(val_kappa.avg)

            run['train/epoch/epoch_train_MF1 Score'].log(train_f1_score.avg)
            run['train/epoch/epoch_val_MF1 Score'].log(val_f1_score.avg)

            run['train/epoch/epoch_train_Precision'].log(train_precision.avg)
            run['train/epoch/epoch_val_Precision'].log(val_precision.avg)

            #################################

            run['train/epoch/epoch_val_Class wise sensitivity W'].log(class1_sens.avg)
            run['train/epoch/epoch_val_Class wise sensitivity S1'].log(class2_sens.avg)
            run['train/epoch/epoch_val_Class wise sensitivity S2'].log(class3_sens.avg)
            run['train/epoch/epoch_val_Class wise sensitivity S3'].log(class4_sens.avg)
            run['train/epoch/epoch_val_Class wise sensitivity R'].log(class5_sens.avg)

            run['train/epoch/epoch_val_Class wise specificity W'].log(class1_spec.avg)
            run['train/epoch/epoch_val_Class wise specificity S1'].log(class2_spec.avg)
            run['train/epoch/epoch_val_Class wise specificity S2'].log(class3_spec.avg)
            run['train/epoch/epoch_val_Class wise specificity S3'].log(class4_spec.avg)
            run['train/epoch/epoch_val_Class wise specificity R'].log(class5_spec.avg)

            run['train/epoch/epoch_val_Class wise F1 Score W'].log(class1_f1.avg)
            run['train/epoch/epoch_val_Class wise F1 Score S1'].log(class2_f1.avg)
            run['train/epoch/epoch_val_Class wise F1 Score S2'].log(class3_f1.avg)
            run['train/epoch/epoch_val_Class wise F1 Score S3'].log(class4_f1.avg)
            run['train/epoch/epoch_val_Class wise F1 Score R'].log(class5_f1.avg)

        if val_accuracy.avg > best_val_acc or (epoch_idx+1)%100==0 or val_kappa.avg > best_val_kappa:
              if val_accuracy.avg > best_val_acc:
                best_val_acc = val_accuracy.avg
                print("================================================================================================")
                print("                                          Saving Best Model (ACC)                                     ")
                print("================================================================================================")
                torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_best_acc.pth.tar')
              if val_kappa.avg > best_val_kappa:
                best_val_kappa = val_kappa.avg
                print("================================================================================================")
                print("                                          Saving Best Model (Kappa)                                    ")
                print("================================================================================================")
                torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_best_kappa.pth.tar')
              if (epoch_idx+1)%args.save_model_freq==0:
                torch.save(Net, f'{args.project_path}/model_check_points/checkpoint_model_epoch_{epoch_idx+1}.pth.tar')
        lr_scheduler.step()

    print(f"========================================= Training Completed =================================================")