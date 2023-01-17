'''
只在Alternate上面做了实验，包括正样本和负样本的
这个是随机将样本中的氨基酸替换为CDE
'''
from collections import defaultdict, OrderedDict
import random
random.seed(50)
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

from tqdm import tqdm

sys.path.append('/mnt/sdb/home/yxt/CBC_and_DATA')
from configuration import config as cf
from preprocess.original_data import get_original_data_Anti
from preprocess.get_data import collate, get_prelabel, MyDataSet


def get_score(x, y, net):
    config = cf.get_train_config()
    device = config.device
    x, y = x.to(device), y.to(device)
    ''' X: torch.Size([1, 50])   Y: torch.Size([1]) '''
    outputs = net.trainModel(x)
    # print(outputs)
    outputs = F.softmax(outputs)[0].tolist()
    # print(outputs)
    if y == 1:  # 正样本看正对应位置
        return outputs[1]
    else:  # 负样本看负对应位置
        return outputs[0]

def get_data(idx):
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/validation")
    test_dataset = MyDataSet(test_data, test_label, test_seq)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    cnt = 0
    for x, y, z in test_iter:
        if cnt == idx:
            return x, y
        cnt += 1

def draw_pos_scores(pos_dict, neg_dict, name):
    # plt.figure(figsize=(15, 5))
    plt.figure(figsize=(15, 12))
    plt.plot(list(pos_dict.keys()), list(pos_dict.values()), color='#cd979b', label='ACP')
    plt.plot(list(neg_dict.keys()), list(neg_dict.values()), color='#6f989c', label='non-ACP')
    plt.xlabel('probability')
    plt.ylabel('score')
    plt.legend(loc='lower left')
    plt.xticks(sorted(list(pos_dict.keys()) + list(neg_dict.keys())))
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/prob_change_{}_scores.png'.format(name))

def draw_pos_scores2(pos_dict_list, neg_dict_list, target_list):
    # colors = ['#ffe5b4', '#f5e5c1', '#eae5cd', '#dfe5da', '#d2e5e6', '#c4e5f3', '#b4e5ff',
    #           '#a6d6f0', '#c4e2f3', '#d2dfe7', '#dedcda', '#e9d8ce', '#f4d5c2', '#fdd2b6',
    #           '#e0b69a', '#efd7b5', '#e2dbba', '#d8ddc3', '#d3ddcc', '#d4dcd5', '#dadada',
    #           ]
    colors = ['#518ef7', '#46b4f7', '#87cefa', '#52cbc0', '#65dacf', '#89f8ed',
              '#6cd89b', '#7ce6a9', '#8cf5b8', '#66c75f', '#76d66f', '#94f38e',
              '#99c560', '#b6e280', '#c5f190', '#c1bf61', '#dfdd83', '#eeec93',
              '#c49a6d', '#d1a67a', '#dfb388', '#ecc095']
    plt.figure(figsize=(15, 12))
    for i in range(len(target_list)):
        pos_dict = pos_dict_list[i]
        label = target_list[i]
        plt.plot(list(pos_dict.keys()), list(pos_dict.values()), color=colors[i], label=label)
        xtick = list(pos_dict.keys())
    plt.xlabel('probability')
    plt.ylabel('score')
    plt.legend(loc='lower left')
    plt.xticks(xtick)
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/prob_change_neg_scores2.svg')

    plt.figure(figsize=(15, 12))
    for i in range(len(target_list)):
        neg_dict = neg_dict_list[i]
        label = target_list[i]
        plt.plot(list(neg_dict.keys()), list(neg_dict.values()), color=colors[i], label=label)
        xtick = list(neg_dict.keys())
    plt.xlabel('probability')
    plt.ylabel('score')
    plt.legend(loc='lower left')
    plt.xticks(xtick)
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/prob_change_pos_scores2.svg')

def change_and_test(x, y, model, Target):  #  'C': 5, 'D': 4, 'E': 7
    # 初始化替换目标
    taget_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'T': 19,
                'W': 20, 'Y': 21, 'V': 22}
    Target = taget_dict[Target]
    # 初始化长度
    L = getLen(x[0].tolist())
    # 初始化重要度字典
    importance = OrderedDict()
    # cnt_prob = OrderedDict()
    # 初始化概率
    probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 进行真正的替换
    for prob in probs:
        new_x = x.clone().detach()
        # cnt = 0
        for idx in range(L):  # 读取每一个位置
            if random.random() < prob:
                new_x[0, idx] = Target
                # cnt += 1
        other_score = get_score(new_x, y, model)  # 测试
        importance[prob] = other_score  # 记录
        # cnt_prob[prob] = cnt
    return importance#, cnt_prob

def getLen(l):
    cnt = 0
    for item in l:
        if item == 0:
            return cnt
        cnt += 1
    return cnt

def changeTargets(Target):
    # 读取模型
    model = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter.pkl')  # 加载整个神经网络的模型结构以及参数
    model.eval()
    # 初始化统计内容
    probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    pos_cnt0, neg_cnt0 = 0, 0
    pos_scores, neg_scores = OrderedDict(), OrderedDict()
    for i in probs:
        pos_scores[i] = 0
        neg_scores[i] = 0
    for i in range(388):
        x, y = get_data(i)
        if y == 1:
            outputs = model.trainModel(x)[0].tolist()
            if outputs[1] > outputs[0]:
                new_pos_scores = change_and_test(x, y, model, Target=Target)
                for idx in probs:
                    pos_scores[idx] += new_pos_scores[idx]
                pos_cnt0 += 1
        if y == 0:
            outputs = model.trainModel(x)[0].tolist()
            if outputs[0] > outputs[1]:
                new_neg_scores = change_and_test(x, y, model, Target=Target)
                for idx in probs:
                    neg_scores[idx] += new_neg_scores[idx]
                neg_cnt0 += 1
    for i in probs:
        pos_scores[i] /= pos_cnt0
        neg_scores[i] /= neg_cnt0
    # draw_pos_scores(pos_scores, neg_scores, Target)
    return pos_scores, neg_scores
    # print(pos_cnt0, neg_cnt0)

if __name__ == '__main__':
    a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'T': 19,
                'W': 20, 'Y': 21, 'V': 22}
    # U X Z本来就不存在于数据集中所以这里扰动也不应该添加
    pos_draw, neg_draw, targets = [], [], []
    for t in tqdm(a2n_dict.keys()):
        pos_scores, neg_scores = changeTargets(t)  # 191 176
        pos_draw.append(pos_scores)
        neg_draw.append(neg_scores)
        targets.append(t)
    # 都是所有长度都换，只是换去的内容不一样
    draw_pos_scores2(pos_draw, neg_draw, targets)

