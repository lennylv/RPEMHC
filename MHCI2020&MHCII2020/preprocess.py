import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations_with_replacement,permutations
ramino = 'ARNDCQEGHILKMFPSTWYV'
amino = 'ARNDCQEGHILKMFPSTWYVX'
aminos = []
for i in amino:
    for j in amino:
        aminos.append(i+j)
# aminos = list(permutations(amino,2))+list(combinations_with_replacement(amino,2))
# aminos = sorted(set(aminos))
seq_dict ={j:(i+1) for i,j in enumerate(aminos)}
print(seq_dict)

def encode(mhc,pep):
    pep = pep[:20]
    arr = np.zeros([34,20])
    for a,m in enumerate(mhc):
        for b,p in enumerate(pep):
            if m in ramino and p in ramino:
                arr[a][b] = seq_dict[m+p]
            if m in ramino and p not in ramino:
                arr[a][b] = seq_dict[m+'X']
            if m not in ramino and p in ramino:
                arr[a][b] = seq_dict['X'+p]
    return arr


f = pd.read_csv('dataset/HLA_II_data.csv')
pep = f['pep'].values
mhc = f['mhc'].values

compound = []
for index in tqdm(range(len(pep))):
    compound.append(encode(mhc[index],pep[index]))
np.save('dataset/HLA_II_compound_id.npy',np.asarray(compound).astype(float))
f = pd.read_csv('dataset/HLA_I_data.csv')
pep = f['pep'].values
mhc = f['mhc'].values
compound = []
for index in tqdm(range(len(pep))):
    compound.append(encode(mhc[index],pep[index]))
np.save('dataset/HLA_I_compound_id.npy',np.asarray(compound).astype(float))