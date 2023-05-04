import numpy as np
from torch.utils.data.dataset import Dataset
from typing import Dict
import pandas as pd
import torch
from typing import Dict

class myDataset(Dataset):
    def __init__(self, all_data, cut_pep=20):
        self.data_list = all_data
        self.cut_pep = cut_pep

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int):
        allele = self.data_list[index][0]
        pep = self.data_list[index][1]
        # len_fea = lenpep_feature(pep, 15)
        # compound = self.data_list[index][4][:,:self.cut_pep]
        # mhc = self.data_list[index][4]
        compound_id = self.data_list[index][3][:, :self.cut_pep]
        # pep_compound_id = self.data_list[index][4]
        # tri = self.data_list[index][5]
        logic = self.data_list[index][2]

        pep = pep[:self.cut_pep]
        pep_len = len(pep)
        pep_mask = np.zeros(self.cut_pep, dtype='int32')
        pep_mask[:pep_len] = 1
        # compound = aa_encode(mhc,pep)
        # pep_id = seq_cat(pep,20)
        # mhc_id = seq_cat(mhc,34)

        return {
            # 'pep_len':pep_len,
            # 'tri':tri,
            # 'mult':compound,
            'mult_id': compound_id,
            # 'pep_id':pep_id,
            # 'mhc_id':mhc_id,
            # 'length':len_fea,
            # 'pep_emb':pep_emb,
            'pep_mask': pep_mask,
            'logic': logic
        }




def collate_fn(batch) -> Dict[str, torch.Tensor]:
    elem = batch[0]
    batch = {key: [d[key] for d in batch] for key in elem}
    # pep_id = torch.to_tensor(np.stack(batch['pep_id']), dtype=torch.int32)
    pep_mask = torch.LongTensor(np.stack(batch['pep_mask']))
    # mhc_id = torch.to_tensor(np.stack(batch['mhc_id']), dtype=torch.int32)
    # allele_input = torch.Tensor(np.stack(batch['allele_id'], axis=0))
    # pep_emb = torch.to_tensor(np.stack(batch['pep_emb']), dtype=torch.float32)
    # mult = torch.to_tensor(np.stack(batch['mult']), dtype=torch.float32)
    mult_id = torch.LongTensor(np.stack(batch['mult_id']))
    # tri_mult = torch.to_tensor(np.stack(batch['tri']), dtype=torch.int32)
    # pep_mult_id = torch.to_tensor(np.stack(batch['pep_mult_id']), dtype=torch.int32)
    # mhc_mult_id = torch.to_tensor(np.stack(batch['mhc_mult_id']), dtype=torch.int32)
    # length =  torch.to_tensor(np.stack(batch['length']), dtype=torch.float32)
    # pep_len = torch.to_tensor(batch['pep_len'], dtype=torch.int32)
    # mhc_len = torch.to_tensor(batch['mhc_len'], dtype=torch.int32)
    logic = torch.LongTensor(batch['logic'])

    return {
        # 'pep_id':pep_id,
        # 'mhc_id':mhc_id,
        # 'pep_len':pep_len,
        # 'tri_mult':tri_mult,
        # 'mult':mult,
        'mult_id': mult_id,
        # 'pep_mult_id':pep_mult_id,
        # 'mhc_mult_id':mhc_mult_id,
        # 'length':length,
        # 'pep_emb':pep_emb,
        'pep_mask_len': pep_mask,
        'targets': logic
    }