import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#MHCAttn
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix

# def evalue(true, pred):
#     acc = accuracy_score(true, pred)
#     f1 = f1_score(true,pred)
#     roc_auc = roc_auc_score(true,pred)
#     prc_auc = average_precision_score(true,pred)
#     pcc, p = pearsonr(true,pred)
#     srcc, p = spearmanr(true,pred)
#     tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
#     sensitivity = float(tp)/(tp+fn)
#     PPV = float(tp)/(tp+fp)
#     return roc_auc,pcc,PPV,f1,sensitivity
#
# f = pd.read_csv('HLA_I_5cv_pred.csv')
# allele = f['allele'].values
# group = sorted(set(allele))
# pred = f['pred'].values
# true = f['true'].values
#
# fw = open('HLAI_group.csv','w')
# fw.write('allele,sum,positive,AUC,PCC,PPV,F1-score,Sensitivity'+'\n')
#
# for g in group:
#     true_group = true[allele==g]
#     pred_group = pred[allele==g]
#     # print(len(true_group[true_group==0]))
#     if len(true_group[true_group==0])>=1 and len(true_group[true_group==1])>=10 and  len(true_group) >=100 :
#         roc_auc,pcc,PPV,f1,sensitivity = evalue(true_group,pred_group)
#         fw.write(g+','+str(len(true_group))+','+str(len(true_group[true_group==1]))+ ',' +str(roc_auc)+ ','+str(pcc)+','+str(PPV)+ ','+str(f1)+ ','+str(sensitivity)+'\n')
#
# fw.close()

# plt.figure(figsize=(35,2))
plt.figure(figsize=(20,3))
# Create a dataset
HLAII = ['AUC' ,'PCC' ,'PPV' ,'Sensitivity','F1-score', ]
HLAI = ['AUC' ,'PCC'  ,'PPV'  ,'Sensitivity','F1-score' , ]
f = pd.read_csv('HLAII_group.csv')
data = f[HLAII].values
allele = list(f['allele'].values)

p1 = sns.heatmap(data.T,cmap='Paired',vmax=1,vmin = 0.,xticklabels = allele,yticklabels = HLAII)
p1.set_xticklabels(p1.get_xticklabels(), rotation=30,fontsize =10)
plt.savefig('HLAII_group.eps',bbox_inches = 'tight')
plt.show()