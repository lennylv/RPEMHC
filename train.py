import math
import sys
import datetime
from math import log
import pandas as pd
from dataloader import CSVDataset, collate_fn, myDataset, abla_collate_fn, abla_Dataset, tcellDataset
import os
import numpy as np
import paddle
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import KFold
from paddle import nn
from tqdm import tqdm
from model import mult_cnn
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from ablation import *
import warnings
warnings.filterwarnings("ignore")


parser = ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='cv', type=str, help='mode')
parser.add_argument('--cuda', default=0, type=int, help='cuda')
parser.add_argument('--epoch', default=20, type=int, help='epoch')
parser.add_argument('--model_id_start', default=1, type=int, help='model_id_start')
parser.add_argument('--batch_size', default=256, type=int, help='this is the batch size of training samples')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weigth_decay')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
parser.add_argument('--alpha', default=0.8, type=float, help='mixup lambda')
parser.add_argument('--kl_beta', default=0.3, type=float, help='R-drop beta')
parser.add_argument('--seed', default=9876, type=int, help='seed')
parser.add_argument('--cut_pep', default=20, type=int, help='cut_pep')

args = parser.parse_args()
# paddle.device.set_device(paddle.get_device())
CUTOFF = 1.0 - math.log(500, 50000)
mhc_name_dict={}
mhc_allele = open('dataset/pseudosequence.2016.all.X.dat').readlines()
for ma in mhc_allele:
    ma_ = ma.strip().split()
    mhc_name_dict[ma_[0]] = ma_[1]

def train(model,train_loader, optimizer, epoch):
    train_loss = 0.
    total_preds =[]
    total_labels = []
    model.train()
    for batch in tqdm(train_loader,desc=f'Epoch {epoch+1}',leave=False, dynamic_ncols=True):
        optimizer.clear_grad()
        output, labels_train= model(**batch)
        loss = loss_fn(output, labels_train.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        total_preds.append(output)
        total_labels.append(labels_train)
    train_loss /= len(train_loader.dataset)
    total_labels = paddle.concat(total_labels,0)
    total_preds = paddle.concat(total_preds,0)
    total_labels, total_preds = total_labels.detach().numpy().flatten(), total_preds.detach().numpy().flatten()

    auroc = roc_auc_score(total_labels >= CUTOFF, total_preds)
    R = pearsonr(total_labels, total_preds)[0]
    return auroc, R, train_loss

def evaluate(model,  epoch ,loader):
    model.eval()
    test_loss=0
    
    total_preds = []
    total_labels = []

    with paddle.no_grad():
        for batch in loader:
            output,labels_test  = model(**batch)
            loss = loss_fn(output, labels_test.unsqueeze(1))
            test_loss += loss.item()
            total_preds.append(output)
            total_labels.append(labels_test)
        test_loss /= len(loader.dataset)
        total_labels = paddle.concat(total_labels,0)
        total_preds = paddle.concat(total_preds,0)
        total_labels,total_preds = total_labels.numpy().flatten(), total_preds.numpy().flatten()
        auroc = roc_auc_score(total_labels >= CUTOFF, total_preds)
        R = pearsonr(total_labels, total_preds)[0]

    return auroc,R,test_loss

def predict(model,  loader):
    model.eval()
    # test_loss = 0
    total_preds = []
    total_labels = []
    with paddle.no_grad():
        for batch in loader:

            output,labels_test = model(**batch)
            total_preds.append(output)
            total_labels.append(labels_test)

        total_labels = paddle.concat(total_labels,0)
        total_preds = paddle.concat(total_preds,0)
        total_labels, total_preds = total_labels.squeeze(-1).numpy(), total_preds.squeeze(-1).numpy()

    return total_labels,total_preds


if args.mode =='cv':
    data = pd.read_csv('dataset/data.csv')
    mhc_name = data['allele'].values
    mhc = data['mhc'].values
    logic = data['logic'].values
    compounds_id = np.load('dataset/all_compound_id.npy')
    pep = data['pep'].values
    all_data = np.asarray(list(zip(mhc_name,pep,mhc,logic,compounds_id)),dtype=object)
    lines = open('dataset/cv_id.txt').readlines()
    cv_id = np.asarray([int(line) for line in lines])
    assert len(all_data) == len(cv_id)
    pred_list = []
    for i in range(args.model_id_start,args.model_id_start+20):
        true = np.empty(len(all_data), dtype=np.float32)
        pred = np.empty(len(all_data), dtype=np.float32)
        for cv in range(5):
            print('model_id',str(i),'flod', cv+1)
            train_data_list, test_data_list = all_data[cv_id != cv], all_data[cv_id == cv]
            train_data = myDataset(train_data_list)
            test_data = myDataset(test_data_list)

            train_loader = paddle.io.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
            test_loader = paddle.io.DataLoader(test_data,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn)

            loss_fn = nn.MSELoss(reduction='sum')
            model = mult_cnn(args)
            optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=0.001,weight_decay=args.weight_decay)
            save_path = 'result/5cv/models/best_model_model_id_'+str(i)+'_CV'+str(cv)+'.pt' #当前目录下
           
            best_auroc = 0.5
            best_auc_epoch = 1
            best_R = 0.5
            # for epoch in range(args.epoch):
            #     train_auroc,train_R,train_loss=train(model,train_loader, optimizer, epoch)
            #     # scheduler.step() 
            #     val_auroc, val_R, val_loss = evaluate(model,epoch,test_loader)
            #     # test_auroc, test_R, test_loss = evaluate(model,epoch,test_loader)
            #     print('epoch:',epoch+1,'|train_loss:','{:.3f}'.format(train_loss),'|train auroc:','{:.3f}'.format(train_auroc),'|train R:','{:.3f}'.format(train_R),
            #     '|val auroc:','{:.3f}'.format(val_auroc),'|val R:','{:.3f}'.format(val_R),
            #     # '|test auroc:','{:.3f}'.format(test_auroc),'|test R:','{:.3f}'.format(test_R),
            #     )
            #     # if val_R>best_R:
            #     paddle.save(model.state_dict(), save_path)
            #         best_R = val_R
            model.set_state_dict(paddle.load('result/5cv/models/best_model_'+'model_id_'+str(i)+'_CV'+str(cv)+'.pt'))
            label,preds = predict(model,test_loader)
            true[cv_id==cv] = label
            pred[cv_id==cv] = preds

        pred_list.append(pred)
        auroc = roc_auc_score(true>=CUTOFF,np.mean(pred_list,axis=0))
        R = pearsonr(true, np.mean(pred_list,axis=0))[0]
        print(i,":",'auroc',auroc,'R',R)
        scores = np.mean(pred_list,axis=0)
        group_auc,group_R,mhc_name_ = [],[],[]
        for allele in sorted(set(mhc_name)):
            true, scores, mhc_name = np.asarray(true), np.asarray(scores), np.asarray(mhc_name)
            t, p = true[mhc_name == allele], scores[mhc_name == allele]
            if len(t) > 40 and len(t[t >= CUTOFF]) >= 3:
                mhc_name_.append(allele)
                group_auc.append(roc_auc_score(np.asarray(t)>=CUTOFF,p))
                group_R.append(pearsonr(t, p)[0])
        mhc_name_.append('avg')
        group_auc.append(np.mean(group_auc,axis=0))
        group_R.append(np.mean(group_R,axis=0))
        group = pd.DataFrame({'allele': mhc_name_, 'auc': group_auc,'pcc':group_R})
       
        if not os.path.exists('result/5cv'):
            os.makedirs('result/5cv')

        group.to_csv('result/5cv/group_evalue.csv', index=False)
        print('group_auc:',np.mean(group_auc,axis=0),'group_pcc:',np.mean(group_R,axis=0))
        out = pd.DataFrame({'allele':list(mhc_name),'true': list(true), 'pred': list(pred)})
        if not os.path.exists('result/5cv/pred'):
            os.makedirs('result/5cv/pred')
        out.to_csv('result/5cv/pred/model_id_' + str(i) +'_5cv_pred.csv')


if args.mode =='lomo':
    data = pd.read_csv('dataset/data.csv')
    mhc_name = data['allele'].values
    logic = data['logic'].values
    compounds_id = np.load('dataset/all_compound_id.npy')
    pep = data['pep'].values
    mhc = data['mhc'].values
    all_data = np.asarray(list(zip(mhc_name, pep, mhc, logic, compounds_id)), dtype=object)
    lines = open('dataset/cv_id.txt').readlines()
    cv_id = np.asarray([int(line) for line in lines])
    for i in range(args.model_id_start, args.model_id_start + 20):
        for name in sorted(set(mhc_name)):
            train_data,train_cv_id = all_data[mhc_name!=name],cv_id[mhc_name!=name]
            test_data,test_cv_id = all_data[mhc_name==name],cv_id[mhc_name==name]
            if len(test_data) > 40 and len([x[-2] for x in test_data if x[-2] >= CUTOFF]) >= 3:
                pred_list = []
                if not os.path.exists('result/lomo/'+name+'/pred/model_id_' + str(i) +'lomo_pred.csv'):
                    true = []
                    pred = []
                    for cv in range(5):
                        print(name,'flod', cv+1)
                        train_data_list = train_data[train_cv_id!=cv]
                        test_data_list = test_data[test_cv_id==cv]

                        train_dataset = myDataset(train_data_list)
                        test_dataset = myDataset(test_data_list)

                        train_dataset, valid_dataset = train_test_split(train_dataset, test_size=1000,random_state=args.seed)

                        train_loader = paddle.io.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
                        val_loader = paddle.io.DataLoader(valid_dataset,batch_size=512,shuffle=False,collate_fn=collate_fn)
                        test_loader = paddle.io.DataLoader(test_dataset,batch_size=512,shuffle=False,collate_fn=collate_fn)
                        loss_fn = nn.MSELoss(reduction='sum')

                        model = mult_cnn(args)
                        optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr,weight_decay=args.weight_decay)

                        path = 'result/lomo/'+name+'/models'
                        if not os.path.exists(path):
                            os.makedirs(path)

                        save_path =  path+'/best_model_model_id_'+str(i)+'_CV'+str(cv)+'.pt' #当前目录下

                        best_auroc = 0.5
                        best_auc_epoch = 1
                        best_R = 0.5
                        # for epoch in range(args.epoch):
                        #     train_auroc,train_R,train_loss=train(model,train_loader, optimizer, epoch)
                        #     val_auroc, val_R, val_loss = evaluate(model,epoch,val_loader)
                        #     print('epoch:',epoch+1,'|train_loss:','{:.3f}'.format(train_loss),'|train auroc:','{:.3f}'.format(train_auroc),'|train R:','{:.3f}'.format(train_R),'|val_loss:','{:.3f}'.format(val_loss),'|val auroc:','{:.3f}'.format(val_auroc),'|val R:','{:.3f}'.format(val_R))
                        #     if val_R > best_R:
                        #         paddle.save(model.state_dict(), save_path)
                        #         best_R = val_R
                        model.set_state_dict(paddle.load(save_path))
                        label,preds = predict(model,test_loader)
                        true += list(label)
                        pred += list(preds)

                        
                    pred_list.append(np.asarray(pred))
                    scores = np.mean(pred_list,axis=0)
                    auroc = roc_auc_score(np.asarray(true)>=CUTOFF,np.asarray(scores))
                    R = pearsonr(np.asarray(true), np.asarray(scores))[0]
                    print(name,'auroc',auroc,'R',R)

                    temp = pd.DataFrame({'true':list(true),'pred':list(pred)})
                    if not os.path.exists('result/lomo/'+name+'/pred'):
                            os.makedirs('result/lomo/'+name+'/pred')
                    temp.to_csv('result/lomo/'+name+'/pred/model_id_' + str(i) +'lomo_pred.csv')
                    out = pd.DataFrame({'true':list(true),'pred':list(scores)})
                    out.to_csv('result/lomo/'+name+'/lomo_pred.csv')

if args.mode =='ic50_test':
    data = pd.read_csv('dataset/indep_test.csv')
    mhc_name = data['allele'].values
    logic = data['logic'].values
    for j,i in enumerate(logic):
        if str(i) == 'nan':logic[j] = 0.
    compounds_id = np.load('dataset/indep_compound_id.npy')
    pep = data['pep'].values
    mhc = data['mhc'].values
    test_data = np.asarray(list(zip(mhc_name,pep,mhc,logic,compounds_id)),dtype=object)

    test_data = myDataset(test_data)
    test_loader = paddle.io.DataLoader(test_data,batch_size=128,shuffle=False,collate_fn=collate_fn)

    pred_list = []
    for i in range(1, 21):
        print('model_id', i)
        model = mult_cnn(args)
        save_path = 'result/5cv/models/best_model_' + 'model_id_' + str(i) + '_CV' + str(0) + '.pt'
        model.set_state_dict(paddle.load(save_path))
        label, preds = predict(model, test_loader)
        pred_list.append(preds)
    scores = np.mean(np.asarray(pred_list), axis=0)
    auroc = roc_auc_score(np.asarray(logic) >= CUTOFF, np.asarray(scores))
    R = pearsonr(logic, np.asarray(scores))[0]
    print(auroc, R)
    #0.7070779200352323 0.42937391672842024
    group_auc, group_R, mhc_name_ = [], [], []
    for allele in sorted(set(mhc_name)):
        true, scores, mhc_name = np.asarray(logic), np.asarray(scores), np.asarray(mhc_name)
        t, p = logic[mhc_name == allele], scores[mhc_name == allele]
        if len(t) > 40 and len(t[t >= CUTOFF]) >= 3:
            mhc_name_.append(allele)
            group_auc.append(roc_auc_score(np.asarray(t) >= CUTOFF, p))
            group_R.append(pearsonr(t, p)[0])
    group_auc.append(np.mean(group_auc, axis=0))
    group_R.append(np.mean(group_R, axis=0))
    print('group_auc:', np.mean(group_auc, axis=0), 'group_pcc:', np.mean(group_R, axis=0))


if args.mode == 'binary_test':
    print('test')
   
    lines = open('dataset/test_binary.txt').readlines()
    mhc_name = [line.strip().split()[2] for line in lines]
    logic = [line.strip().split()[1] for line in lines]
    compounds_id = np.load('dataset/test_compound_id.npy')
    mhc = [mhc_name_dict[i] for i in mhc_name]
    pep = [line.strip().split()[0] for line in lines]
    test_data = np.asarray(list(zip(mhc_name,pep,mhc,logic,compounds_id)),dtype=object)

    test_data = myDataset(test_data)
    test_loader = paddle.io.DataLoader(test_data,batch_size=512,shuffle=False,collate_fn=collate_fn)

    pred_list = []
    for i in range(1,21):
        model = mult_cnn(args)
        print('model_id',i)
        best_auc = 0.
        pred = []
        # for cv in range(5):
        save_path = 'result/5cv/models/best_model_'+'model_id_'+str(i)+'_CV'+str(0)+'.pt'
        model.set_state_dict(paddle.load(save_path))
        label,preds = predict(model,test_loader)
        auroc =  roc_auc_score(np.asarray(label,dtype='int32'),preds)
        print(auroc)
        pred_list.append(preds)
    
    print(len(pred_list))
    scores = np.mean(np.asarray(pred_list),axis=0)
    auc = roc_auc_score(np.asarray(logic),scores)
    print(auc)
    group_auc,mhc_name_ = [],[]
    for allele in sorted(set(mhc_name)):
        true, scores, mhc_name = np.asarray(logic,dtype='int32'), np.asarray(scores), np.asarray(mhc_name)
        t, p = true[mhc_name == allele], scores[mhc_name == allele]
        if len(t) > 40 and np.sum(t,axis=0) >= 3:
            mhc_name_.append(allele)
            group_auc.append(roc_auc_score(t,p))
    print(np.mean(group_auc))

if args.mode == 'ablation':
    data = pd.read_csv('dataset/data.csv')
    mhc_name = data['allele'].values
    mhc = data['mhc'].values
    logic = data['logic'].values
    pep = data['pep'].values
    all_data = np.asarray(list(zip(mhc,pep,logic)),dtype=object)
    lines = open('dataset/cv_id.txt').readlines()
    cv_id = np.asarray([int(line) for line in lines])
    assert len(all_data) == len(cv_id)
    pred_list = []
    for i in range(1,21):
        true = np.empty(len(all_data), dtype=np.float32)
        pred = np.empty(len(all_data), dtype=np.float32)
        for cv in range(5):
            print('model_id',str(i),'flod', cv+1)

            train_data_list, test_data_list = all_data[cv_id != cv], all_data[cv_id == cv]
            train_data = abla_Dataset(train_data_list)
            test_data = abla_Dataset(test_data_list)

            train_loader = paddle.io.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,collate_fn=abla_collate_fn)
            test_loader = paddle.io.DataLoader(test_data,batch_size=args.batch_size,shuffle=False,collate_fn=abla_collate_fn)

            loss_fn = nn.MSELoss(reduction='sum')
            model = ablation_model(args)
            optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=0.001,weight_decay=args.weight_decay)
            if not os.path.exists('result/ablation/'+'/models'):
                os.makedirs('result/ablation/'+'/models')
            save_path = 'result/ablation/'+'/models/best_model_model_id_'+str(i)+'_CV'+str(cv)+'.pt' #当前目录下
           
            best_auroc = 0.5
            best_auc_epoch = 1
            best_R = 0.5
            # for epoch in range(args.epoch):
            #     train_auroc,train_R,train_loss=train(model,train_loader, optimizer, epoch)
            #     val_auroc, val_R, val_loss = evaluate(model,epoch,test_loader)
            #     # test_auroc, test_R, test_loss = evaluate(model,epoch,test_loader)
            #     print('epoch:',epoch+1,'|train_loss:','{:.3f}'.format(train_loss),'|train auroc:','{:.3f}'.format(train_auroc),'|train R:','{:.3f}'.format(train_R),
            #     '|val auroc:','{:.3f}'.format(val_auroc),'|val R:','{:.3f}'.format(val_R),
            #     # '|test auroc:','{:.3f}'.format(test_auroc),'|test R:','{:.3f}'.format(test_R),
            #     )
            #     # if val_R>best_R:
            #     paddle.save(model.state_dict(), save_path)
            #         # best_R = val_R
            model.set_state_dict(paddle.load('result/ablation/models/best_model_model_id_'+str(i)+'_CV'+str(cv)+'.pt'))
            label,preds = predict(model,test_loader)
            true[cv_id==cv] = label
            pred[cv_id==cv] = preds

        pred_list.append(pred)
        auroc = roc_auc_score(true>=CUTOFF,np.mean(pred_list,axis=0))
        R = pearsonr(true, np.mean(pred_list,axis=0))[0]
        print(i,":",'auroc',auroc,'R',R)
        scores = np.mean(pred_list,axis=0)
        group_auc,group_R,mhc_name_ = [],[],[]
        for allele in sorted(set(mhc_name)):
            true, scores, mhc_name = np.asarray(true), np.asarray(scores), np.asarray(mhc_name)
            t, p = true[mhc_name == allele], scores[mhc_name == allele]
            if len(t) > 40 and len(t[t >= CUTOFF]) >= 3:
                mhc_name_.append(allele)
                group_auc.append(roc_auc_score(np.asarray(t)>=CUTOFF,p))
                group_R.append(pearsonr(t, p)[0])
        mhc_name_.append('avg')
        group_auc.append(np.mean(group_auc,axis=0))
        group_R.append(np.mean(group_R,axis=0))
        group = pd.DataFrame({'allele': mhc_name_, 'auc': group_auc,'pcc':group_R})
        group.to_csv('result/ablation/group_evalue.csv', index=False)
        print('group_auc:',np.mean(group_auc,axis=0),'group_pcc:',np.mean(group_R,axis=0))
        out = pd.DataFrame({'allele':list(mhc_name),'true': list(true), 'pred': list(pred)})
        if not os.path.exists('result/ablation/pred'):
            os.makedirs('result/ablation/pred')
        out.to_csv('result/ablation/pred/model_id_' + str(i) +'_5cv_pred.csv')

if args.mode == 'T_cell_test':
    lines = open('dataset/T_cell.fa.txt').readlines()
    fw = open('result/T_cell/t_cell_test.txt', 'w')
    for i, line in enumerate(lines):
        if i >= 0:
            if line.startswith('>'):
                line = line.strip().split('=')
                s = line[2].replace('HLA-', '').replace('/', '-').replace('DRA01:01-', '').replace(':', '')
                if s[:3] != 'DRB' and s[:3] != 'H2-': s = 'HLA-' + s.replace('_', '')
                if s[:3] == 'DRB': s = s[:4] + '_' + s[4:]
                if s[:3] == 'H2-': s = 'H-2-' + s[3:]
                # print(s)
                print('t_cell', i, s)
                seq = lines[i + 1]
                hit = line[1]
                seq_list = np.asarray([seq[j:j + len(hit)] for j in range(len(seq) - len(hit) + 1)])
                logic = np.zeros(len(seq_list), dtype='int32')
                logic[seq_list == hit] = 1
                mhc_name = np.asarray([s] * len(seq_list))
                mhc = [mhc_name_dict[j] for j in mhc_name]
                test_data = np.asarray(list(zip(mhc, seq_list, logic)), dtype=object)
                test_data = tcellDataset(test_data)
                test_loader = paddle.io.DataLoader(test_data, batch_size=256, shuffle=False, collate_fn=collate_fn)
                pred_list = []
                # print(i)
                for m in range(args.model_id_start, args.model_id_start + 20):
                    best_auc = 0.
                    pred = []
                    best_fpr = 1.
                    # for cv in range(5):
                    model = mult_cnn(args)
                    save_path = 'result/5cv/models/best_model_' + 'model_id_' + str(m) + '_CV' + str(0) + '.pt'
                    model.set_state_dict(paddle.load(save_path))
                    label, preds = predict(model, test_loader)
                    pred_list.append(preds)
                    # print(len(pred_list))
                scores = np.mean(np.asarray(pred_list), axis=0)
                posi_score = scores[logic == 1]
                frank = np.sum(scores > posi_score) / len(seq_list)
                auc = roc_auc_score(logic, scores)
                print(s, frank, auc)
                fw.write(s + ' ' + str(frank) + ' ' + str(auc) + '\n')
    fw.close()

    ######T_cell_group_test
    # lines = open('result/T_cell/t_cell_test.txt').readlines()
    # mhc = []
    # frank = []
    # auc = []
    # for line in lines:
    #     line = line.strip().split()
    #     mhc.append(line[0])
    #     frank.append(line[1])
    #     auc.append(line[2])
    # mhc = np.asarray(mhc)
    # frank = np.asarray(frank).astype('float32')
    # auc = np.asarray(auc).astype('float32')
    # avg1 = []
    # avg2 = []
    #
    # fw = open('result/T_cell/t_cell_group_test.txt', 'w')
    # for m in sorted(set(mhc)):
    #     avg_frank = np.around(np.mean(frank[mhc == m]), decimals=3)
    #     avg1.append(avg_frank)
    #     avg_auc = np.around(np.mean(auc[mhc == m]), decimals=3)
    #     avg2.append(avg_auc)
    #     fw.write(m + ' ' + str(avg_frank) + ' ' + str(avg_auc) + '\n')
    # print(len(avg1))
    # fw.write('avg' + ' ' + str(np.around(np.mean(avg1), decimals=3)) + ' ' + str(
    #     np.around(np.mean(avg2), decimals=3)) + '\n')
    # fw.close()

