import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
from termcolor import colored
import pandas as pd

pep2label = list()

'''要改一改'''
import sys
print(sys.path)
sys.path.append('/mnt/sdb/home/yxt/CBC_and_DATA')
from preprocess.get_data import collate, get_prelabel, MyDataSet
from preprocess.original_data import get_original_data, get_original_data_AAPred, get_original_data_ACPred, get_original_data_Anti, get_original_data_MPMABP
from MyUtils.util_eval import evaluate_accuracy
from MyUtils.util_cal import caculate_metric
from model.ContrastModel import newModel, ContrastiveLoss
from configuration import config as cf
config = cf.get_train_config()
config.lr = 1e-5
config.epoch = 3000
train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0_Main/internal")
test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Main/validation")
print(train_data.shape, train_label.shape)
print(test_data.shape, test_label.shape)
'''
Main:
torch.Size([1378, 50]) torch.Size([1378])
torch.Size([344, 50]) torch.Size([344])
'''
train_dataset = MyDataSet(train_data, train_label, train_seq)
test_dataset = MyDataSet(test_data, test_label, test_seq)
batch_size = 32# 但是用64时loss下不去，所以这里直接改一下range了，改range也不行，试图吧最后的弄掉
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate, drop_last=True)

device = config.device

output5 = []
label3 = []
for num_model in range(10):
# for num_model in range(8):
    # torch.cuda.empty_cache()
    net = newModel().to(device)
    lr = config.lr
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = ContrastiveLoss()

    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    best_acc = 0
    EPOCH = config.epoch

    for epoch in range(EPOCH):
        loss_ls = []
        loss1_ls = []
        loss2_3_ls = []

        t0 = time.time()
        net.train()

        for seq1, seq2, label, label1, label2, pep1, pep2 in train_iter_cont:

            output1 = net(seq1)
            output2 = net(seq2)
            output3 = net.trainModel(seq1)
            output4 = net.trainModel(seq2)
            loss1 = criterion(output1, output2, label)
            loss2 = criterion_model(output3, label1)
            loss3 = criterion_model(output4, label2)
            loss = loss1 + loss2 + loss3
            # loss = loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
            loss1_ls.append(loss1.item())
            loss2_3_ls.append((loss2 + loss3).item())
            output5.extend([output1, output2])
            label3.extend([label1, label2])
        net.eval()
        with torch.no_grad():
#             print(2)
            train_acc = evaluate_accuracy(train_iter, net)
#             print(1)
            test_acc = evaluate_accuracy(test_iter, net)
            A, B = get_prelabel(test_iter, net)
            A = [np.concatenate(A)]
            B = [np.concatenate(B)]
            A = np.array(A)
            B = np.array(B)
            A = A.reshape(-1, 1)
            B = B.reshape(-1, 1)

            df1 = pd.DataFrame(A, columns=['prelabel'])
            df2 = pd.DataFrame(B, columns=['realabel'])
            df4 = pd.concat([df1, df2], axis=1)

            acc_sum, n = 0.0, 0
            outputs = []
            for x, y, z in test_iter:
                x, y = x.to(device), y.to(device)
                output = torch.softmax(net.trainModel(x), dim=1)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            pre_pro = outputs[:, 1]
            pre_pro = np.array(pre_pro.cpu().detach().numpy())
            pre_pro = pre_pro.reshape(-1)
            df3 = pd.DataFrame(pre_pro, columns=['pre_pro'])
            df5 = pd.concat([df4, df3], axis=1)
            real1 = df5['realabel']
            pre1 = df5['prelabel']
            pred_pro1 = df5['pre_pro']
            metric1, roc_data1, prc_data1 = caculate_metric(pre1, real1, pred_pro1)

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, loss1: {np.mean(loss1_ls):.5f}, loss2_3: {np.mean(loss2_3_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        print(results)

        if test_acc > best_acc:
            best_acc = test_acc
            if best_acc > 0.77:
                torch.save(net, '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main.pkl'.format(best_acc))
                # torch.save(net.state_dict(), '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main_params.pkl'.format(best_acc))  # 只保存模型参数
                torch.save(metric1,  '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main_metrics.pt'.format(best_acc))
                np.savetxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main_roc.txt'.format(best_acc), roc_data1[:-1], delimiter=' ', fmt = '%s')
                np.savetxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main_roc2.txt'.format(best_acc), roc_data1, delimiter=' ', fmt='%s')
                np.savetxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main_prc.txt'.format(best_acc), prc_data1[:-1], delimiter=' ', fmt = '%s')
                np.savetxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc{}_model_Main_prc2.txt'.format(best_acc), prc_data1, delimiter=' ', fmt='%s')
                print('roc_data:\n', roc_data1)
                print('\nprc_data\n', prc_data1)
            # torch.save({"best_acc": best_acc,"metric":metric1, "model": net.state_dict()}, f'./{num_model}.pl')
            print(f"best_acc: {best_acc},metric:{metric1}")

'''
nohup python Save_M.py > /mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/Save_M_lr=1e-4_epoch=300.log 2>&1 &
nohup python Save_M.py > /mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/Save_M_lr=1e-5_epoch=3000.log 2>&1 &
'''