import paddle
from paddle import nn
import math
import numpy as np
import random
import paddle.nn.functional as F


class ablation_model(nn.Layer):
    def __init__(self,args):
        super(ablation_model, self).__init__()

        self.mhc_emb_aa = nn.Embedding(21+1,128)
        self.pep_emb_aa = nn.Embedding(21+1,128)
      
        self.mult_id_step1 =nn.Sequential(
    nn.Conv2D(in_channels=128,out_channels=256,kernel_size=(34,1),padding=0),nn.BatchNorm2D(256),nn.ReLU(),
)

        self.mult_id_step2_1 = nn.Sequential(
            nn.Conv1D(in_channels=128,out_channels=128,kernel_size=1,padding='same'),nn.BatchNorm1D(128),nn.ReLU(),nn.Dropout(0.2)
        ) 
        self.mult_id_step2_2 =  nn.Sequential(
            nn.Conv1D(in_channels=128,out_channels=128,kernel_size=3,padding='same'),nn.BatchNorm1D(128),nn.ReLU(),nn.Dropout(0.35)
        ) 
        self.mult_id_step2_3 = nn.Sequential(
            nn.Conv1D(in_channels=128,out_channels=256,kernel_size=5,padding='same'),nn.BatchNorm1D(256),nn.ReLU(),nn.Dropout(0.5)
        ) 

        self.id_step2_max = nn.Sequential(
        nn.MaxPool1D(2,2),
         nn.Dropout(0.42)
        ) 

        self.mult_id_step3_1 = nn.Sequential(
            nn.Conv1D(in_channels=256,out_channels=128,kernel_size=1,padding='same'),nn.BatchNorm1D(128),nn.ReLU()
        ) 
        self.mult_id_step3_2 = nn.Sequential(
            nn.Conv1D(in_channels=256,out_channels=256,kernel_size=3,padding='same'),nn.BatchNorm1D(256),nn.ReLU()
        ) 
        self.mult_id_step3_3 =nn.Sequential(
            nn.Conv1D(in_channels=256,out_channels=512,kernel_size=5,padding='same'),nn.BatchNorm1D(512),nn.ReLU()
        ) 
        self.mult_id_step3_4 = nn.Sequential(
            nn.Conv1D(in_channels=256,out_channels=1024,kernel_size=7,padding='same'),nn.BatchNorm1D(1024),nn.ReLU()
        ) 
        
        self.id_step3_max = nn.Sequential(
            nn.MaxPool1D(3,3),
            nn.Dropout(0.54)
        ) 
        self.id_step3_fusion = nn.Sequential(
                    nn.Conv1D(in_channels=128*5,out_channels=128,kernel_size=1,padding = 'valid'),
                    nn.BatchNorm1D(128),
                    nn.ReLU(),
        ) 

        self.id_rnn_step3_1 = nn.LSTM(128,128,1,direction='forward') 
        self.id_rnn_step3_2 =  nn.LSTM(128,128,2,direction='forward')
        self.id_mult_conv = nn.Sequential(
            nn.Conv1D(in_channels=256,out_channels=256,kernel_size=9,padding='valid'),nn.BatchNorm1D(256),nn.ReLU()
            )
        
        self.fc = nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
 

    def forward(self,pep_id,mhc_id,pep_mask_len,targets):

        pep_id = self.pep_emb_aa(pep_id)
        mhc_id = self.mhc_emb_aa(mhc_id)
        mult_id = self.cross_attn(pep_id,mhc_id,mhc_id).transpose([0,2,1])

        mult_id = paddle.concat([self.mult_id_step2_1(mult_id),self.mult_id_step2_2(mult_id),self.mult_id_step2_3(mult_id)],axis=1)
        mult_id = self.id_step2_max(mult_id.transpose([0,2,1])).transpose([0,2,1])

        mult_id = paddle.concat([self.mult_id_step3_1(mult_id),self.mult_id_step3_2(mult_id),self.mult_id_step3_3(mult_id),self.mult_id_step3_4(mult_id)],axis=1)
        mult_id = self.id_step3_max(mult_id.transpose([0,2,1])).transpose([0,2,1])
        mult_id = self.id_step3_fusion(mult_id).transpose([0,2,1])

        mult_id_1,_ = self.id_rnn_step3_1(mult_id)
        mult_id_2,_ =  self.id_rnn_step3_2(mult_id)
        mult_id = paddle.concat([mult_id_1,mult_id_2],axis=-1)
        # mult_id = paddle.mean(mult_id,axis=1)
        mult_id = self.id_mult_conv(mult_id.transpose([0,2,1])).transpose([0,2,1])
        masks_id = pep_mask_len[:,-mult_id.shape[1]:,None]
        masks_id = paddle.repeat_interleave(masks_id,mult_id.shape[2],axis=2)
        masks_id = paddle.cast(masks_id, dtype='bool')
        mult_id = self.masked_fill(mult_id, masks_id, -0.00000000000000001)
        mult_id = mult_id.max(axis=1)
        output = self.sigmoid(self.fc(mult_id))

        return output,targets

    def masked_fill(self,x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, x, y)

    
    def cross_attn(self,q,k,v):
        weight = paddle.matmul(q,k.transpose([0,2,1]))
        weight = F.softmax(weight,axis=2)
        out = paddle.matmul(weight,v)
        return out

