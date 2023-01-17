'''
只在Alternate上面做了实验，包括正样本和负样本的
'''
from collections import defaultdict, OrderedDict
import seaborn as sns
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
    outputs = F.softmax(net.trainModel(x))[0].tolist()
    if y[0] == 1:  # 正样本看正对应位置
        return outputs[1]
    else:  # 负样本看负对应位置
        return outputs[0]

def get_data(idx):
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/validation")
    # print(test_data.shape, test_label.shape)
    test_dataset = MyDataSet(test_data, test_label, test_seq)
    # batch_size 为 total
    # test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=388, shuffle=False)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    cnt = 0
    for x, y, z in test_iter:
        if cnt == idx:
            return x, y
        cnt += 1

def draw_importance(importance_dict, name):
    map = []
    a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'T': 19,
                'W': 20, 'Y': 21, 'V': 22}
    for key in importance_dict:
        acid_importance_dict = importance_dict[key]
        acid_importance_dict = {key: acid_importance_dict[key] for key in sorted(acid_importance_dict.keys())}
        # print(acid_importance_dict)
        map.append(list(acid_importance_dict.values()))
    map = np.array(map).transpose()# 行是sequence，列是acid
    # print(map.shape)
    plt.figure()
    sns.heatmap(map, cmap="YlGnBu", xticklabels=list(importance_dict.keys()), yticklabels=sorted(list(a2n_dict.keys())))
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/{} importance distribution.svg'.format(name))
    plt.show()

# def change_and_test(x, y, model, name):
#     # 替换数据并进行测试
#     # 初始化替换需要用到的数据
#     new_x = x.clone().detach()
#     a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
#                 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'T': 19,
#                 'W': 20, 'Y': 21, 'V': 22}
#     n2a_dict = {}
#     for key, val in a2n_dict.items():  # items获取所有的键值对,先取出来的为关键字，后取出来的为键值
#         n2a_dict[val] = key
#     # 初始化重要度字典
#     importance = OrderedDict()
#     idx = 0
#     for original_acid in x[0].tolist():  # 读取每一个位置
#         if original_acid == 0:
#             break
#         importance[idx] = OrderedDict()
#         idx += 1
#     # 进行真正的替换
#     idx = 0
#     for original_acid in x[0].tolist():  # 读取每一个位置
#         if original_acid == 0:
#             break
#         for other_acid in n2a_dict.keys():  # 替换成每一种氨基酸
#             new_x.copy_(x)
#             new_x[0, idx] = other_acid
#             # 测试
#             other_score = get_score(new_x, y, model)
#             importance[idx][n2a_dict[other_acid]] = other_score
#         idx += 1
#     # print(importance[1]['A'])
#     # draw_importance(importance, name)
#     return importance
def change_and_test2(x, y, model):
    # 替换数据并进行测试
    # 初始化替换需要用到的数据
    a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'T': 19,
                'W': 20, 'Y': 21, 'V': 22}
    n2a_dict = {}
    for key, val in a2n_dict.items():  # items获取所有的键值对,先取出来的为关键字，后取出来的为键值
        n2a_dict[val] = key
    # 初始化重要度字典
    importance = OrderedDict()
    for idx in range(getLen(x[0].tolist())):
        importance[idx] = OrderedDict()
    # 进行真正的替换
    for idx in range(getLen(x[0].tolist())):  # 读取每一个位置
        for other_acid in n2a_dict.keys():  # 替换成每一种氨基酸
            new_x = x.clone()
            new_x[0, idx] = other_acid
            # 测试
            other_score = get_score(new_x, y, model)
            importance[idx][n2a_dict[other_acid]] = other_score
    # print(importance[1]['A'])
    # draw_importance(importance, name)
    return importance

def getLen(l):
    cnt = 0
    for item in l:
        if item == 0:
            return cnt
        cnt += 1
    return cnt

def MyPlot(dict_pos, dict_neg):
    amino_acids = dict_pos.keys()
    pos_vals = dict_pos.values()
    neg_vals = dict_neg.values()
    x = np.arange(0, 2.1 * len(amino_acids), 2.1)  # 标签位置
    width = 0.37  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, pos_vals, 2 * width, color='#cde79b', label='ACP')
    rects2 = ax.bar(x + width, neg_vals, 2 * width, color='#6fd89c', label='non-ACP')
    ax.set_xticks(x)
    ax.set_xticklabels(amino_acids)
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/pred_statistics.svg')

if __name__ == '__main__':
    # 读取模型
    model = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter.pkl')  # 加载整个神经网络的模型结构以及参数
    # model.load_state_dict(torch.load('net_params.pkl'))# 仅加载参数，重复效果，没必要
    model.eval()
    # 初始化统计内容
    a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                  'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'T': 19,
                  'W': 20, 'Y': 21, 'V': 22}
    # a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
    #             'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
    #             'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z': 24}
    amino_acids = sorted(list(a2n_dict.keys()))
    LEN = 20
    pos_cnt, neg_cnt, pos_scores, neg_scores = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for i in range(LEN):
        pos_scores[i] = OrderedDict()
        neg_scores[i] = OrderedDict()
        pos_cnt[i] = OrderedDict()
        neg_cnt[i] = OrderedDict()
        for amino_acid in amino_acids:
            pos_scores[i][amino_acid] = 0
            neg_scores[i][amino_acid] = 0
            pos_cnt[i][amino_acid] = 0
            neg_cnt[i][amino_acid] = 0

    # 获取所有数据
    for i in tqdm(range(388)):
        x, y = get_data(i)
        if y == 0 and getLen(x[0].tolist()) == LEN:
            outputs = model.trainModel(x)[0].tolist()
            if outputs[0] > outputs[1]:
                new_neg_scores = change_and_test2(x, y, model)
                for idx in new_neg_scores.keys():
                    for amino_acid in amino_acids:
                        neg_scores[idx][amino_acid] += new_neg_scores[idx][amino_acid]
                        neg_cnt[idx][amino_acid] += 1
        if y == 1 and getLen(x[0].tolist()) == LEN:
            outputs = model.trainModel(x)[0].tolist()
            if outputs[0] < outputs[1]:
                new_pos_scores = change_and_test2(x, y, model)
                for idx in new_pos_scores.keys():
                    for amino_acid in amino_acids:
                        pos_scores[idx][amino_acid] += new_pos_scores[idx][amino_acid]
                        pos_cnt[idx][amino_acid] += 1

    for i in range(LEN):
        for amino_acid in amino_acids:
            pos_scores[i][amino_acid] /= pos_cnt[i][amino_acid]
            neg_scores[i][amino_acid] /= neg_cnt[i][amino_acid]
    print(pos_cnt)
    print(pos_scores)
    draw_importance(pos_scores, 'ACP_len={}'.format(LEN))
    draw_importance(neg_scores, 'non-ACP_len={}'.format(LEN))


    # # 统计每一行的平均值
    # pos_acid_cnt = OrderedDict()
    # neg_acid_cnt = OrderedDict()
    # for a in amino_acids:
    #     pos_acid_cnt[a], neg_acid_cnt[a] = 0, 0
    # for idx in pos_scores.keys():
    #     for amino_acid in amino_acids:
    #         pos_acid_cnt[amino_acid] += pos_scores[idx][amino_acid]
    #         neg_acid_cnt[amino_acid] += neg_scores[idx][amino_acid]
    # for a in amino_acids:
    #     pos_acid_cnt[a] /= float(LEN)
    #     neg_acid_cnt[a] /= float(LEN)
    # MyPlot(pos_acid_cnt, neg_acid_cnt)
