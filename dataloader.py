import numpy as np
from paddle.io import Dataset
from typing import Dict
import pandas as pd
import paddle
import random

def pep_cat(prot,max_seq_len):
    seq_voc = '0-ACDEFGHIKLMNPQRSTVWY'
    seq_dict = {v:(i) for i,v in enumerate(seq_voc)}
    x = np.zeros(max_seq_len,dtype = 'int32')
    for i, ch in enumerate(prot):
        if ch in seq_voc:
            x[i+3] = seq_dict[ch]
        else:
            x[i+3] = seq_dict['-']
    return x
def lenpep_feature(pep,expected_pep_len):
    lenpep = len(pep)
    f1 = 1.0/(1.0 + np.exp((lenpep-expected_pep_len)/2.0))
    len_fea = np.array((f1,1.0-f1),dtype= 'float32')
    return len_fea
def seq_cat(prot,max_seq_len):
    seq_voc = '0-ACDEFGHIKLMNPQRSTVWY'
    seq_dict = {v:int(i) for i,v in enumerate(seq_voc)}
    x = np.zeros(max_seq_len,dtype = 'int32')
    for i, ch in enumerate(prot):
        if ch in seq_voc:
            x[i] = seq_dict[ch]
        else:x[i] = seq_dict['-']
    return x
class CSVDataset(Dataset):
    def __init__(self,file,compound,cut_pep=22):

        f = pd.read_csv(file)
        # self.emb = np.load(pep_embed,allow_pickle=True)
        self.mult_com = np.load(compound,allow_pickle=True)
        self.pep = f['pep'].values
        self.mhc = f['mhc'].values
        self.allele = f['allele'].values
        self.logic = f['logic'].values
        self.cut_pep = cut_pep
        # self.SIdata = Tensordataset

    def __len__(self) -> int:
        return len(self.pep)

    def __getitem__(self, index: int):
        pep = self.pep[index]
        len_fea = lenpep_feature(pep, 15)
        pep = pep[:self.cut_pep]
        pep_len = len(pep)
        mult = self.mult_com[index]
        pep_mask = np.zeros(self.cut_pep,dtype = 'int32')
        pep_mask[:pep_len] = 1
        logic = self.logic[index]

        return {
            'mult':mult,
            # 'pep_emb':pep_vec,
            'length':len_fea,
            # 'pep_id': pep_input,
            # 'mhc_id': mhc_input,
            # 'pep_len':pep_len,
            'pep_mask':pep_mask,
            # 'mhc_len':mhc_len,
            'logic':logic
        }

class myDataset(Dataset):
    def __init__(self,all_data):

        self.data_list = all_data

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        pep = self.data_list[index][1]
        compound_id = self.data_list[index][4]
        logic = self.data_list[index][3]
        pep = pep[:20]
        pep_len = len(pep)
        pep_mask = np.zeros(20,dtype = 'int32')
        pep_mask[:pep_len] = 1
        return {
            'mult_id':compound_id,
            'pep_mask':pep_mask,
            'logic':logic
        }



def collate_fn(batch) -> Dict[str, paddle.Tensor]:
    elem = batch[0]
    batch = {key: [d[key] for d in batch] for key in elem}
    pep_mask = paddle.to_tensor(np.stack(batch['pep_mask']), dtype=paddle.int32)
    mult_id = paddle.to_tensor(np.stack(batch['mult_id']), dtype=paddle.int32)

    logic = paddle.to_tensor(batch['logic'], dtype=paddle.float32)

    return {
        'mult_id':mult_id,
        'pep_mask_len':pep_mask,
        'targets':logic,
    }

class abla_Dataset(Dataset):
    def __init__(self,all_data):

        self.data_list = all_data

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        mhc = self.data_list[index][0]
        pep = self.data_list[index][1][:20]
        logic = self.data_list[index][2]   
        
        pep_len = len(pep)
        pep_mask = np.zeros(20, dtype='int32')
        pep_mask[:pep_len] = 1
        mhc_id = seq_cat(mhc,34)
        pep_id = seq_cat(pep,20)

        return {
            'pep_mask':pep_mask,
            'mhc_id':mhc_id,
            'pep_id':pep_id,
            'logic':logic
        }
def abla_collate_fn(batch) -> Dict[str, paddle.Tensor]:
    elem = batch[0]
    batch = {key: [d[key] for d in batch] for key in elem}
    pep_id = paddle.to_tensor(np.stack(batch['pep_id']), dtype=paddle.int32)
    mhc_id = paddle.to_tensor(np.stack(batch['mhc_id']), dtype=paddle.int32)
    logic = paddle.to_tensor(batch['logic'], dtype=paddle.float32)
    pep_mask = paddle.to_tensor(np.stack(batch['pep_mask']), dtype=paddle.int32)
    return {
        'pep_id':pep_id,
        'mhc_id':mhc_id,
        'pep_mask_len':pep_mask,
        'targets':logic
    }

ramino = 'ARNDCQEGHILKMFPSTWYV'
amino = 'ARNDCQEGHILKMFPSTWYVX'
aminos = []
for i in amino:
    for j in amino:
        aminos.append(i+j)
seq_dict ={j:(i+1) for i,j in enumerate(aminos)}
def encode(mhc,pep):
    pep = pep[:20]
    arr = np.zeros((len(mhc),20))
    for a,m in enumerate(mhc):
        for b,p in enumerate(pep):
            if m in ramino and p in ramino:
                arr[a][b] = seq_dict[m+p]
            if m in ramino and p not in ramino:
                arr[a][b] = seq_dict[m+'X']
            if m not in ramino and p in ramino:
                arr[a][b] = seq_dict['X'+p]
    return arr

class tcellDataset(Dataset):
    def __init__(self,data_list,cut_pep=20):

        self.data_list = data_list
        self.cut_pep = cut_pep
  
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        compound_id = encode(data[0],data[1][:20])
        # compound = dis_encode(data[0],data[1][:20])
        logic = data[2]

        pep =data[1][:20]
        pep_len = len(pep)
        pep_mask = np.zeros(20,dtype = 'int32')
        pep_mask[:pep_len] = 1

        return {
            'mult_id':compound_id,
            'pep_mask':pep_mask,
            'logic':logic
        }


class NetMHCDataset(Dataset):
    def __init__(self, data_list, cut_pep=20):
        self.data_list = data_list
        self.cut_pep = cut_pep

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        data = self.data_list[index]
        compound_id = encode(data[0], data[1][:self.cut_pep], cut_pep=self.cut_pep)
        # compound = dis_encode(data[0],data[1][:20])
        logic = data[2]

        pep = data[1][:self.cut_pep]
        pep_len = len(pep)
        pep_mask = np.zeros(self.cut_pep, dtype='int32')
        pep_mask[:pep_len] = 1

        return {
            'mult_id': compound_id,
            'pep_mask': pep_mask,
            'logic': logic
        }

class bind_Dataset(Dataset):
    def __init__(self,all_data,cut_pep=20):

        self.data_list = all_data
        self.cut_pep = cut_pep

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        mhc = self.data_list[index][2]
        pep = self.data_list[index][1]
        compound_id = encode(mhc,pep[:self.cut_pep],self.cut_pep)
        logic = 1
        pep = pep[:self.cut_pep]
        pep_len = len(pep)
        pep_mask = np.zeros(self.cut_pep,dtype = 'int32')
        pep_mask[:pep_len] = 1
        return {
            'mult_id':compound_id,
            'pep_mask':pep_mask,
            'logic':logic
        }



def bind_collate_fn(batch) -> Dict[str, paddle.Tensor]:
    elem = batch[0]
    batch = {key: [d[key] for d in batch] for key in elem}
    pep_mask = paddle.to_tensor(np.stack(batch['pep_mask']), dtype=paddle.int32)
    mult_id = paddle.to_tensor(np.stack(batch['mult_id']), dtype=paddle.int32)

    logic = paddle.to_tensor(batch['logic'], dtype=paddle.int32)

    return {
        'mult_id':mult_id,
        'pep_mask_len':pep_mask,
        'targets':logic,
    }