import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
from transformers import BertModel, BertTokenizer
import random
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
# torch.cuda.set_device(2)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
pep2label = list()

'''
def genData(file, max_len):
#     aa_dict = {'A': 1, 'C': 2, 'T': 3, 'G': 4}
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    long_pep_counter = 0
    pep_codes = []
    labels = []
    pep_seq = []
    max_seq_len = 70
    for pep in lines:
        if line[0] == '>':
                key = line[1:].split('\n')[0]
            else:
                enhancers.append({key: line.split('\n')[0]})
        pep, label = pep.split(",")
        labels.append(int(label))
        input_seq = ' '.join(pep)
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        pep_seq.append(input_seq)
        if not len(pep) > max_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
        else:
            long_pep_counter += 1
    print("length > 200:", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)

    return data, torch.tensor(labels), pep_seq
'''

'''要改一改'''

def get_first_layer_data(path):# ../benchmark dataset & ../independent dataset
    '''

    :param path:
    :return:
    1 represents enhancer
    0 represents non-enhancer
    '''
    sequences = []
    labels = []
    datas = []
    path_enhancer = os.path.join(path, 'first layer/enhancers.txt')
    path_non_enhancer = os.path.join(path, 'first layer/non-enhancers.txt')
    aa_dict = {'A': 1, 'C': 2, 'T': 3, 'G': 4}
    with open(path_enhancer, encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            if line[0] == '>':
                continue
            else:
                sequences.append(line.split('\n')[0][:50])
                labels.append(1)
                current_DNA = []
                for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                datas.append(current_DNA)
    with open(path_non_enhancer, encoding='utf-8') as f:
        for line in f.readlines():
            if line[0] == '>':
                continue
            else:
                sequences.append(line.split('\n')[0][:50])
                labels.append(0)
                current_DNA = []
                for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                datas.append(current_DNA)

    return torch.tensor(datas), torch.tensor(labels), sequences

def get_prelabel(data_iter, net):
    prelabel, relabel = [], []
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
#         for i in range(len(z)):
#             if i == 0:
#                 vec = torch.tensor(seq2vec[z[0]]).to(device)
#             else:
#                 vec = torch.cat((vec, torch.tensor(seq2vec[z[i]]).to(device)), dim=0)
        outputs = net.trainModel(x)
        prelabel.append(outputs.argmax(dim=1).cpu().numpy())
        relabel.append(y.cpu().numpy())
        # print()
    return prelabel, relabel

import sys
print(sys.path)


def caculate_metric(pred_y, labels, pred_prob):

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if int(labels[index]) == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1


    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC and AUC
    labels = list(map(int, labels))
    pred_prob = list(map(float, pred_prob))
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data
'''这里要改'''
train_data, train_label, train_seq = get_first_layer_data("./dataset/benchmark dataset")
test_data, test_label, test_seq = get_first_layer_data("./dataset/independent dataset")

print(train_data.shape, train_label.shape)
print(test_data.shape,test_label.shape)

class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]

train_dataset = MyDataSet(train_data, train_label, train_seq)
test_dataset = MyDataSet(test_data, test_label, test_seq)

batch_size = 128
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# seq2vec = json.load(open('../seq2vec_CPP.emb'))

class newModel(nn.Module):
    def __init__(self, vocab_size=4):
        super().__init__()

        # self.filter_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.filter_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.embedding_dim = 100  # the MGF process dim
        dim_cnn_out = 128
        filter_num = 64

        # self.filter_sizes = [int(fsz) for fsz in self.filter_sizes.split(',')]
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, self.embedding_dim)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)

        # self.linear = nn.Linear(len(self.filter_sizes) * filter_num, dim_cnn_out)
        # self.classification = nn.Linear(dim_cnn_out, 2)  # label_num: 28
        # 已经2分类了，不用改
        # self.classification = nn.Sequential(
        #     nn.Linear(len(self.filter_sizes) * filter_num, 256),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(64, 2)
        # )
        self.block1 = nn.Sequential(nn.Linear(len(self.filter_sizes) * filter_num, 256),
                                    nn.BatchNorm1d(256),
                                    nn.LeakyReLU(),
                                    nn.Linear(256, 64),
                                    )
        self.classification = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.cuda(deviceno)
        # print(x.shape)
        # # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        # print('raw x', x.size())
        # input_ids = x
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)
        # print('embedding x', x.size())

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        # print('view x', x.size())

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]
        # print(x)
        # print('conv x', len(x), [x_item.size() for x_item in x])

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        # print('max_pool2d x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # print('flatten x', len(x), [x_item.size() for x_item in x])

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # print('concat x', x.size()) torch.Size([320, 1024])

        # dropout层
        x = self.dropout(x)

        # 全连接层
        # representation = self.linear(x)
        output = self.block1(x)

        return output

    def trainModel(self, x):
        with torch.no_grad():
            output = self.forward(x)

        return self.block1(output)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # euclidean_distance: [128]
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def collate(batch):
    seq1_ls = []
    seq2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    pep1_ls = []
    pep2_ls = []
    batch_size = len(batch)
    for i in range(int(batch_size / 2)):
        seq1, label1, pep_seq1 = batch[i][0], batch[i][1], batch[i][2]
        seq2, label2, pep_seq2 = batch[i + int(batch_size / 2)][0], batch[i + int(batch_size / 2)][1], batch[i + int(batch_size / 2)][2]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        pep1_ls.append(pep_seq1)
        pep2_ls.append(pep_seq2)
        label = (label1 ^ label2)
        seq1_ls.append(seq1.unsqueeze(0))
        seq2_ls.append(seq2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    seq1 = torch.cat(seq1_ls).to(device)
    seq2 = torch.cat(seq2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return seq1, seq2, label, label1, label2, pep1_ls, pep2_ls


train_iter_cont = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, collate_fn=collate)

deviceno = 3
device = torch.device("cuda", deviceno)


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
#         for i in range(len(z)):
#             if i == 0:
#                 vec = torch.tensor(seq2vec[z[0]]).to(device)
#             else:
#                 vec = torch.cat((vec, torch.tensor(seq2vec[z[i]]).to(device)), dim=0)
        outputs = net.trainModel(x)

        acc_sum += (outputs.argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

output5 = []
label3 = []
for num_model in range(10):
    net = newModel().to(device)
    lr = 0.00001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion = ContrastiveLoss()

    criterion_model = nn.CrossEntropyLoss(reduction='sum')
    best_acc = 0
    EPOCH = 2000

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
#             loss = loss1 + loss2 + loss3
            loss = loss2 + loss3
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
#                 for i in range(len(z)):
#                     if i == 0:
#                         vec = torch.tensor(seq2vec[z[0]]).to(device)
#                     else:
#                         vec = torch.cat((vec, torch.tensor(seq2vec[z[i]]).to(device)), dim=0)
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


            torch.save({"best_acc": best_acc,"metric":metric1, "model": net.state_dict()}, f'./{num_model}.pl')
            print(f"best_acc: {best_acc},metric:{metric1}")

