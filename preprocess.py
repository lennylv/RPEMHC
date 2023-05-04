#数据预处理
import pandas as pd
mhc_name_dict={}
mhc_allele = open('dataset/pseudosequence.2016.all.X.dat').readlines()
for ma in mhc_allele:
    ma_ = ma.strip().split()
    mhc_name_dict[ma_[0]] = ma_[1]
lines = open('dataset/data.txt').readlines()
# len(lines)
# pep = pd.read_csv('dataset/data.csv')['pep'].values
# len(pep)
fw = open('dataset/data.csv','w')
fw.write('pep,logic,allele,mhc'+'\n')
for line in lines:
    line = line.strip().split()
    fw.write(line[0]+','+line[1]+','+line[2]+','+mhc_name_dict[line[2]]+'\n')
fw.close()

#################################################
#compound id
#################################################
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

f = pd.read_csv('dataset/data.csv')
pep = f['pep'].values
mhc = f['mhc'].values
compound = []
for index in tqdm(range(len(pep))):
    compound.append(encode(mhc[index],pep[index]))
np.save('dataset/all_compound_id.npy',np.asarray(compound).astype(float))

mhc_name_dict={}
mhc_allele = open('dataset/pseudosequence.2016.all.X.dat').readlines()
for ma in mhc_allele:
    ma_ = ma.strip().split()
    mhc_name_dict[ma_[0]] = ma_[1]

lines = open('dataset/test_binary.txt').readlines()
pep = [line.strip().split()[0] for line in lines]
mhc = [mhc_name_dict[line.strip().split()[2]] for line in lines]
compound = []
for index in tqdm(range(len(pep))):
    compound.append(encode(mhc[index],pep[index]))
np.save('dataset/test_compound_id.npy',np.asarray(compound).astype(float))

f = pd.read_csv('dataset/indep_test.csv')
pep = f['pep'].values
mhc = f['mhc'].values
compound = []
for index in tqdm(range(len(pep))):
    compound.append(encode(mhc[index],pep[index]))
np.save('dataset/indep_compound_id.npy',np.asarray(compound).astype(float))

