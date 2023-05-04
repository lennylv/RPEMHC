import math

import pandas as pd
from dataloader import collate_fn,myDataset
import os
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, confusion_matrix
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

parser = ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='cv', type=str, help='mode')
parser.add_argument('--cuda', default=0, type=int, help='cuda')
parser.add_argument('--epoch', default=20, type=int, help='epoch')
parser.add_argument('--model_id_start', default=1, type=int, help='model_id_start')
parser.add_argument('--batch_size', default=256, type=int, help='this is the batch size of training samples')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0.001, type=float, help='weigth_decay')
parser.add_argument('--t', default=0.1, type=float, help='temperature')
parser.add_argument('--seed', default=9876, type=int, help='seed')
parser.add_argument('--cut_pep', default=20, type=int, help='cut_pep')

args = parser.parse_args()
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
CUTOFF = 1.0 - math.log(500, 50000)
if args.mode == 'HLA_II':
    #python train.py --mode HLA_II --lr 0.0002 --batch_size 64 --epoch 50

    same_seeds(args.seed)
    from sklearn.model_selection import KFold
    from hla_II_model import hla_ii_mult_cnn

    data = pd.read_csv('dataset/HLA_II_data.csv')
    mhc_name = data['allele'].values
    mhc = data['mhc'].values
    logic = data['logic'].values
    compounds_id = np.load('dataset/HLA_II_compound_id.npy')
    pep = data['pep'].values
    all_data = np.asarray(list(zip(mhc_name, pep, logic, compounds_id)), dtype=object)
    lines = open('dataset/HLA_II_cv_id.txt').readlines()
    cv_id = np.asarray([int(line) for line in lines])
    true = []
    pred = []
    for cv in range(1, 6):
        print('flod', str(cv))
        train_data_list, test_data_list = all_data[cv_id != cv], all_data[cv_id == cv]

        train_data = myDataset(train_data_list)
        test_data = myDataset(test_data_list)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        model = hla_ii_mult_cnn(args).to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        save_path = 'result/HLA_II/best_model' + '_CV' + str(cv) + '.pt'  # 当前目录下
        if not os.path.exists('result/HLA_II'):
            os.makedirs('result/HLA_II')
        best_auroc = 0.5
        best_auc_epoch = 1
        best_R = 0.5
        # for epoch in range(args.epoch):
        #     train_loss = 0
        #     total_preds = []
        #     total_labels = []
        #     model.train()
        #     for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False, dynamic_ncols=True):
        #         batch = {name: tensor.to(device) for name, tensor in batch.items()}
        #         optimizer.zero_grad()
        #         output, labels_train = model(**batch)
        #         loss = loss_fn(output, labels_train)
        #         loss.backward()
        #         optimizer.step()
        #         train_loss += loss.item()
        #         output_idx = output.argmax(axis=1)
        #         # print(output_idx)
        #         total_preds.append(output_idx)
        #         total_labels.append(labels_train)
        #     train_loss /= len(train_loader.dataset)
        #     total_labels = torch.concat(total_labels, 0)
        #     total_preds = torch.concat(total_preds, 0)
        #     total_labels, total_preds = total_labels.cpu().detach().numpy().flatten(), total_preds.cpu().detach().numpy().flatten()
        #
        #     train_f1 = f1_score(total_labels, total_preds)
        #     train_roc_auc = roc_auc_score(total_labels, total_preds)
        #     train_prc_auc = average_precision_score(total_labels, total_preds)
        #     train_pcc, p = pearsonr(total_labels, total_preds)
        #     train_srcc, p = spearmanr(total_labels, total_preds)
        #     tn, fp, fn, tp = confusion_matrix(total_labels, total_preds).ravel()
        #     train_sensitivity = float(tp) / (tp + fn)
        #     train_PPV = float(tp) / (tp + fp)
        #
        #     model.eval()
        #     test_preds = []
        #     test_labels = []
        #
        #     with torch.no_grad():
        #         for batch in test_loader:
        #             batch = {name: tensor.to(device) for name, tensor in batch.items()}
        #             output_test, labels_test = model(**batch)
        #             output_idx_test = output_test.argmax(axis=1)
        #             test_preds.append(output_idx_test)
        #             test_labels.append(labels_test)
        #         test_labels = torch.concat(test_labels, 0)
        #         test_preds = torch.concat(test_preds, 0)
        #         test_labels, test_preds = test_labels.cpu().numpy().flatten(), test_preds.cpu().numpy().flatten()
        #
        #     test_f1 = f1_score(test_labels, test_preds)
        #     test_roc_auc = roc_auc_score(test_labels, test_preds)
        #     test_prc_auc = average_precision_score(test_labels, test_preds)
        #     test_pcc, p = pearsonr(test_labels, test_preds)
        #     test_srcc, p = spearmanr(test_labels, test_preds)
        #     tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_labels, test_preds).ravel()
        #     test_sensitivity = float(tp_test) / (tp_test + fn_test)
        #     test_PPV = float(tp_test) / (tp_test + fp_test)
        #     print('epoch:', epoch + 1, '|train_loss:', '{:.3f}'.format(train_loss), '|train auroc:',
        #           '{:.3f}'.format(train_roc_auc), '|train prc_auc:', '{:.3f}'.format(train_prc_auc),
        #           '|train pcc:', '{:.3f}'.format(train_pcc), '|train srcc:', '{:.3f}'.format(train_srcc),
        #           '|train_sensitivity:', '{:.3f}'.format(train_sensitivity),
        #           '|train_PPV:', '{:.3f}'.format(train_PPV),
        #           '|test auroc:', '{:.3f}'.format(test_roc_auc), '|test prc_auc:', '{:.3f}'.format(test_prc_auc),
        #           '|test pcc:', '{:.3f}'.format(test_pcc), '|test srcc:', '{:.3f}'.format(test_srcc),
        #           '|test_sensitivity:', '{:.3f}'.format(test_sensitivity),
        #           '|test_PPV:', '{:.3f}'.format(test_PPV),
        #
        #           )
        #
        #     torch.save(model.state_dict(), save_path)

        model.load_state_dict(torch.load('result/HLA_II/best_model'+'_CV'+str(cv)+'.pt'))
        model.eval()
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                output,labels_test = model(**batch)
                output_idx = output.argmax(axis=1)
                test_preds.append(output_idx)
                test_labels.append(labels_test)
            test_labels = torch.concat(test_labels,0)
            test_preds = torch.concat(test_preds,0)

            test_labels,test_preds = test_labels.cpu().numpy().flatten(), test_preds.cpu().numpy().flatten()
            # print(test_preds.shape)

        true.append(test_labels)
        pred.append(test_preds)
    true = np.concatenate(true,axis=0)
    pred = np.concatenate(pred,axis=0)

    out = pd.DataFrame({'allele': list(mhc_name), 'true': list(true), 'pred': list(pred)})
    out.to_csv('result/HLA_II_5cv_pred.csv')

    test_acc = accuracy_score(true, pred)
    test_f1 = f1_score(true,pred)
    test_roc_auc = roc_auc_score(true,pred)
    test_prc_auc = average_precision_score(true,pred)
    test_pcc, p = pearsonr(true,pred)
    test_srcc, p = spearmanr(true,pred)
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    test_sensitivity = float(tp)/(tp+fn)
    test_PPV = float(tp)/(tp+fp)
    print('test f1:','{:.3f}'.format(test_f1),
        '|test auroc:','{:.3f}'.format(test_roc_auc),'|test prc_auc:','{:.3f}'.format(test_prc_auc),
            '|test pcc:','{:.3f}'.format(test_pcc),'|test srcc:','{:.3f}'.format(test_srcc),'|test_sensitivity:','{:.3f}'.format(test_sensitivity),
            '|test_PPV:','{:.3f}'.format(test_PPV),'|test_acc','{:.3f}'.format(test_acc)
            )



if args.mode == 'HLA_I':
    #python train.py --mode HLA_I  --cuda 1 --batch_size 128

    same_seeds(args.seed)
    # from sklearn.model_selection import KFold
    from hla_I_model import hla_i_mult_cnn

    data = pd.read_csv('dataset/HLA_I_data.csv')
    mhc_name = data['allele'].values
    mhc = data['mhc'].values
    logic = data['logic'].values
    compounds_id = np.load('dataset/HLA_I_compound_id.npy')
    pep = data['pep'].values
    all_data = np.asarray(list(zip(mhc_name, pep, logic, compounds_id)), dtype=object)
    lines = open('dataset/HLA_I_cv_id.txt').readlines()
    cv_id = np.asarray([int(line) for line in lines])
    true = []
    pred = []
    for cv in range(1, 6):
        print('flod', str(cv))
        train_data_list, test_data_list = all_data[cv_id != cv], all_data[cv_id == cv]

        train_data = myDataset(train_data_list,cut_pep=15)
        test_data = myDataset(test_data_list,cut_pep=15)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


        device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        model = hla_i_mult_cnn(args).to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        save_path = 'result/HLA_I/best_model' + '_CV' + str(cv) + '.pt'  # 当前目录下
        if not os.path.exists('result/HLA_I'):
            os.makedirs('result/HLA_I')
        best_auroc = 0.5
        best_auc_epoch = 1
        best_R = 0.5
        # for epoch in range(args.epoch):
        #     train_loss = 0
        #     total_preds = []
        #     total_labels = []
        #     model.train()
        #     for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False, dynamic_ncols=True):
        #         batch = {name: tensor.to(device) for name, tensor in batch.items()}
        #         optimizer.zero_grad()
        #         output, labels_train = model(**batch)
        #         loss = loss_fn(output, labels_train)
        #         loss.backward()
        #         optimizer.step()
        #         train_loss += loss.item()
        #         output_idx = output.argmax(axis=1)
        #         # print(output_idx)
        #         total_preds.append(output_idx)
        #         total_labels.append(labels_train)
        #     train_loss /= len(train_loader.dataset)
        #     total_labels = torch.concat(total_labels, 0)
        #     total_preds = torch.concat(total_preds, 0)
        #     total_labels, total_preds = total_labels.cpu().detach().numpy().flatten(), total_preds.cpu().detach().numpy().flatten()
        #
        #     train_f1 = f1_score(total_labels, total_preds)
        #     train_roc_auc = roc_auc_score(total_labels, total_preds)
        #     train_prc_auc = average_precision_score(total_labels, total_preds)
        #     train_pcc, p = pearsonr(total_labels, total_preds)
        #     train_srcc, p = spearmanr(total_labels, total_preds)
        #     tn, fp, fn, tp = confusion_matrix(total_labels, total_preds).ravel()
        #     train_sensitivity = float(tp) / (tp + fn)
        #     train_PPV = float(tp) / (tp + fp)
        #
        #     model.eval()
        #     test_preds = []
        #     test_labels = []
        #
        #     with torch.no_grad():
        #         for batch in test_loader:
        #             batch = {name: tensor.to(device) for name, tensor in batch.items()}
        #             output_test, labels_test = model(**batch)
        #             output_idx_test = output_test.argmax(axis=1)
        #             test_preds.append(output_idx_test)
        #             test_labels.append(labels_test)
        #         test_labels = torch.concat(test_labels, 0)
        #         test_preds = torch.concat(test_preds, 0)
        #         test_labels, test_preds = test_labels.cpu().numpy().flatten(), test_preds.cpu().numpy().flatten()
        #
        #     test_f1 = f1_score(test_labels, test_preds)
        #     test_roc_auc = roc_auc_score(test_labels, test_preds)
        #     test_prc_auc = average_precision_score(test_labels, test_preds)
        #     test_pcc, p = pearsonr(test_labels, test_preds)
        #     test_srcc, p = spearmanr(test_labels, test_preds)
        #     tn_test, fp_test, fn_test, tp_test = confusion_matrix(test_labels, test_preds).ravel()
        #     test_sensitivity = float(tp_test) / (tp_test + fn_test)
        #     test_PPV = float(tp_test) / (tp_test + fp_test)
        #     print('epoch:', epoch + 1, '|train_loss:', '{:.3f}'.format(train_loss), '|train auroc:',
        #           '{:.3f}'.format(train_roc_auc), '|train prc_auc:', '{:.3f}'.format(train_prc_auc),
        #           '|train pcc:', '{:.3f}'.format(train_pcc), '|train srcc:', '{:.3f}'.format(train_srcc),
        #           '|train_sensitivity:', '{:.3f}'.format(train_sensitivity),
        #           '|train_PPV:', '{:.3f}'.format(train_PPV),
        #           '|test auroc:', '{:.3f}'.format(test_roc_auc), '|test prc_auc:', '{:.3f}'.format(test_prc_auc),
        #           '|test pcc:', '{:.3f}'.format(test_pcc), '|test srcc:', '{:.3f}'.format(test_srcc),
        #           '|test_sensitivity:', '{:.3f}'.format(test_sensitivity),
        #           '|test_PPV:', '{:.3f}'.format(test_PPV),
        #
        #           )
        #
        #     torch.save(model.state_dict(), save_path)

        model.load_state_dict(torch.load('result/HLA_I/best_model'+'_CV'+str(cv)+'.pt'))
        model.eval()
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {name: tensor.to(device) for name, tensor in batch.items()}
                output,labels_test = model(**batch)
                output_idx = output.argmax(axis=1)
                test_preds.append(output_idx)
                test_labels.append(labels_test)
            test_labels = torch.concat(test_labels,0)
            test_preds = torch.concat(test_preds,0)

            test_labels,test_preds = test_labels.cpu().numpy().flatten(), test_preds.cpu().numpy().flatten()
            # print(test_preds.shape)

        true.append(test_labels)
        pred.append(test_preds)
    true = np.concatenate(true,axis=0)
    pred = np.concatenate(pred,axis=0)

    out = pd.DataFrame({'allele': list(mhc_name), 'true': list(true), 'pred': list(pred)})
    out.to_csv('result/HLA_I_5cv_pred.csv')

    test_acc = accuracy_score(true, pred)
    test_f1 = f1_score(true,pred)
    test_roc_auc = roc_auc_score(true,pred)
    test_prc_auc = average_precision_score(true,pred)
    test_pcc, p = pearsonr(true,pred)
    test_srcc, p = spearmanr(true,pred)
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    test_sensitivity = float(tp)/(tp+fn)
    test_PPV = float(tp)/(tp+fp)
    print('test f1:','{:.3f}'.format(test_f1),
        '|test auroc:','{:.3f}'.format(test_roc_auc),'|test prc_auc:','{:.3f}'.format(test_prc_auc),
            '|test pcc:','{:.3f}'.format(test_pcc),'|test srcc:','{:.3f}'.format(test_srcc),'|test_sensitivity:','{:.3f}'.format(test_sensitivity),
            '|test_PPV:','{:.3f}'.format(test_PPV),'|test_acc','{:.3f}'.format(test_acc)
            )

