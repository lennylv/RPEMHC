import torch
from torch import nn
import math
import numpy as np
import random
import torch.nn.functional as F


class ablation_model(nn.Layer):
    def __init__(self,args):
        super(ablation_model, self).__init__()

        self.mhc_emb_aa = nn.Embedding(21+1,128)
        self.pep_emb_aa = nn.Embedding(21+1,128)
      
        self.mult_id_step1 =nn.Sequential(
    nn.Conv2D(in_channels=128,out_channels=256,kernel_size=(34,1),padding=0),nn.BatchNorm2D(256),nn.ReLU(),
)

        self.mult_id_step2_1 = nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1,padding='same'),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.2)
        ) 
        self.mult_id_step2_2 =  nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,padding='same'),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0.35)
        ) 
        self.mult_id_step2_3 = nn.Sequential(
            nn.Conv1d(in_channels=128,out_channels=256,kernel_size=5,padding='same'),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.5)
        ) 

        self.id_step2_max = nn.Sequential(
        nn.MaxPool1d(2,2),
         nn.Dropout(0.42)
        ) 

        self.mult_id_step3_1 = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=128,kernel_size=1,padding='same'),nn.BatchNorm1d(128),nn.ReLU()
        ) 
        self.mult_id_step3_2 = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=3,padding='same'),nn.BatchNorm1d(256),nn.ReLU()
        ) 
        self.mult_id_step3_3 =nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=512,kernel_size=5,padding='same'),nn.BatchNorm1d(512),nn.ReLU()
        ) 
        self.mult_id_step3_4 = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=1024,kernel_size=7,padding='same'),nn.BatchNorm1d(1024),nn.ReLU()
        ) 
        
        self.id_step3_max = nn.Sequential(
            nn.MaxPool1d(3,3),
            nn.Dropout(0.54)
        ) 
        self.id_step3_fusion = nn.Sequential(
                    nn.Conv1d(in_channels=128*5,out_channels=128,kernel_size=1,padding = 'valid'),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
        )

        self.id_rnn_step3_1 = nn.LSTM(128, 128, 1, batch_first=True, bidirectional=False)
        self.id_rnn_step3_2 = nn.LSTM(128, 128, 2, batch_first=True, bidirectional=False)
        self.id_mult_conv = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=9,padding='valid'),nn.BatchNorm1d(256),nn.ReLU()
            )
        
        self.fc = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
 

    def forward(self,pep_id,mhc_id,pep_mask_len,targets):

        pep_id = self.pep_emb_aa(pep_id)
        mhc_id = self.mhc_emb_aa(mhc_id)
        mult_id = self.cross_attn(pep_id,mhc_id,mhc_id).permute([0,2,1])

        mult_id = torch.concat([self.mult_id_step2_1(mult_id),self.mult_id_step2_2(mult_id),self.mult_id_step2_3(mult_id)],dim=1)
        mult_id = self.id_step2_max(mult_id.permute([0,2,1])).permute([0,2,1])

        mult_id = torch.concat([self.mult_id_step3_1(mult_id),self.mult_id_step3_2(mult_id),self.mult_id_step3_3(mult_id),self.mult_id_step3_4(mult_id)],dim=1)
        mult_id = self.id_step3_max(mult_id.permute([0,2,1])).permute([0,2,1])
        mult_id = self.id_step3_fusion(mult_id).permute([0,2,1])

        mult_id_1,_ = self.id_rnn_step3_1(mult_id)
        mult_id_2,_ =  self.id_rnn_step3_2(mult_id)
        mult_id = torch.concat([mult_id_1,mult_id_2],dim=-1)
        # mult_id = torch.mean(mult_id,dim=1)
        mult_id = self.id_mult_conv(mult_id.permute([0, 2, 1])).permute([0, 2, 1])
        masks_id = pep_mask_len[:, -mult_id.shape[1]:, None].bool()
        mult_id, _ = mult_id.masked_fill(~masks_id, -0.0000000000000000000000000000000001).max(dim=1)
        output = self.sigmoid(self.fc(mult_id))

        return output,targets



    
    def cross_attn(self,q,k,v):
        weight = torch.matmul(q,k.permute([0,2,1]))
        weight = F.softmax(weight,dim=2)
        out = torch.matmul(weight,v)
        return out

