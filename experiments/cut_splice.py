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
    test_dataset = MyDataSet(test_data, test_label, test_seq)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    cnt = 0
    for x, y, z in test_iter:
        if cnt == idx:
            return x, y
        cnt += 1

def draw_pos_scores(pos_dict, neg_dict, name):
    plt.figure(figsize=(10, 8))
    plt.plot(list(pos_dict.keys()), list(pos_dict.values()), color='#cd979b', label='ACP')
    plt.plot(list(neg_dict.keys()), list(neg_dict.values()), color='#6f989c', label='non-ACP')
    plt.plot(list(neg_dict.keys()), [0] * len(list(neg_dict.keys())), color='#d3d3d3', linestyle='--')
    plt.xlabel('position')
    plt.ylabel('score')
    plt.legend(loc='upper left')
    # plt.xticks(sorted(list(pos_dict.keys()) + list(neg_dict.keys())))
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/{}_position_scores.svg'.format(name))

def draw_pos_importance(pos_dict, neg_dict, name):
    plt.figure(figsize=(10, 8))
    plt.plot(list(pos_dict.keys()), list(pos_dict.values()), color='#cd979b', label='ACP')
    plt.plot(list(neg_dict.keys()), list(neg_dict.values()), color='#6f989c', label='non-ACP')
    plt.plot(list(neg_dict.keys()), [0] * len(list(neg_dict.keys())), color='#d3d3d3', linestyle='--')
    plt.xlabel('most important position')
    plt.ylabel('count')
    plt.legend(loc='upper left')
    # plt.xticks(sorted(list(pos_dict.keys()) + list(neg_dict.keys())))
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics2/{}_position_importance.svg'.format(name))

def change_and_test(x, y, model, L_min=20):
    # 初始化重要度字典
    max_idx, min_idx, maxx, minn = None, None, 0, 0
    importance = OrderedDict()
    origin_score = get_score(x, y, model)
    # 进行真正的替换
    for idx in range(L_min):  # 读取每一个位置
        new_x = torch.cat([x[0, :idx], x[0, idx+1:]]).unsqueeze(0)  # 剪切
        other_score = get_score(new_x, y, model)  # 测试
        importance[idx] = other_score - origin_score  # 记录
        if importance[idx] > maxx:
            max_idx, maxx = idx, importance[idx]
        if importance[idx] < minn:
            min_idx, minn = idx, importance[idx]

    return importance, max_idx, min_idx

def getLen(l):
    cnt = 0
    for item in l:
        if item == 0:
            return cnt
        cnt += 1
    return cnt

if __name__ == '__main__':
    # 读取模型
    model = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter.pkl')  # 加载整个神经网络的模型结构以及参数
    # model.load_state_dict(torch.load('net_params.pkl'))# 仅加载参数，重复效果，没必要
    model.eval()
    # 初始化统计内容
    Lmin, LEN = 1, 10
    pos_cnt, neg_cnt = dict(), dict()
    pos_scores, neg_scores, pos_importance, neg_importance = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    for i in range(LEN):
        pos_scores[i] = 0
        neg_scores[i] = 0
    for i in range(LEN):
        pos_importance[i] = 0
        neg_importance[i] = 0
    for i in range(LEN):
        pos_cnt[i] = 0
        neg_cnt[i] = 0
    for i in tqdm(range(388)):
        x, y = get_data(i)
        Lx = getLen(x[0].tolist())
        if Lx >= Lmin and Lx <= LEN:
            if y == 0:
                outputs = model.trainModel(x)[0].tolist()
                if outputs[0] > outputs[1]:
                    new_neg_scores, max_idx, min_idx = change_and_test(x, y, model, L_min=getLen(x[0].tolist()))
                    for idx in range(50):
                        if idx in new_neg_scores:
                            neg_scores[idx] += new_neg_scores[idx]
                            neg_cnt[idx] += 1
                    if max_idx is not None:
                        neg_importance[max_idx] += 1
                    if min_idx is not None:
                        neg_importance[min_idx] += 1
            if y == 1:
                outputs = model.trainModel(x)[0].tolist()
                if outputs[0] < outputs[1]:
                    new_pos_scores, max_idx, min_idx = change_and_test(x, y, model, L_min=getLen(x[0].tolist()))
                    for idx in range(50):
                        if idx in new_pos_scores:
                            pos_scores[idx] += new_pos_scores[idx]
                            pos_cnt[idx] += 1  # 实际位置和长度是不同的，长度2的序列的最后一个位置是1
                    if max_idx is not None:
                        pos_importance[max_idx] += 1
                    if min_idx is not None:
                        pos_importance[min_idx] += 1

    for i in range(LEN):
        if pos_scores[i] != 0:
            pos_scores[i] /= pos_cnt[i]
        if neg_scores[i] != 0:
            neg_scores[i] /= neg_cnt[i]
    draw_pos_scores(pos_scores, neg_scores, 'L{}_{}'.format(Lmin, LEN))
    draw_pos_importance(pos_importance, neg_importance, 'L{}_{}'.format(Lmin, LEN))
    print(pos_cnt, neg_cnt)

    # pos_cnt0, neg_cnt0 = 0, 0
    # pos_scores, neg_scores = OrderedDict(), OrderedDict()
    # for i in range(20):
    #     pos_scores[i] = 0
    #     neg_scores[i] = 0
    # for i in range(50):
    #     pos_importance[i] = 0
    #     neg_importance[i] = 0
    # for i in tqdm(range(388)):
    #     x, y = get_data(i)
    #     if getLen(x[0].tolist()) < 40:
    #         if y == 0 and getLen(x[0].tolist()) >= 20:
    #             outputs = model.trainModel(x)[0].tolist()
    #             if outputs[0] > outputs[1]:
    #                 new_pos_scores, max_idx, min_idx = change_and_test(x, y, model, L_min=20)
    #                 for idx in range(20):
    #                     pos_scores[idx] += new_pos_scores[idx]
    #                 pos_cnt0 += 1
    #                 if max_idx is not None:
    #                     pos_importance[max_idx] += 1
    #                 if min_idx is not None:
    #                     pos_importance[min_idx] += 1
    #         if y == 1 and getLen(x[0].tolist()) >= 20:
    #             outputs = model.trainModel(x)[0].tolist()
    #             if outputs[0] < outputs[1]:
    #                 new_neg_scores, max_idx, min_idx = change_and_test(x, y, model, L_min=20)
    #                 for idx in range(20):
    #                     neg_scores[idx] += new_neg_scores[idx]
    #                 neg_cnt0 += 1
    #                 if max_idx is not None:
    #                     neg_importance[max_idx] += 1
    #                 if min_idx is not None:
    #                     neg_importance[min_idx] += 1
    # for i in range(20):
    #     pos_scores[i] /= pos_cnt0
    #     neg_scores[i] /= neg_cnt0
    # draw_pos_scores(pos_scores, neg_scores, 'L<40')
    # draw_pos_importance(pos_importance, neg_importance, 'L<40')
    # print(pos_cnt0, neg_cnt0)  # 79 75
    #
    # pos_cnt0, neg_cnt0 = 0, 0
    # pos_scores, neg_scores = OrderedDict(), OrderedDict()
    # for i in range(40):
    #     pos_scores[i] = 0
    #     neg_scores[i] = 0
    # for i in range(50):
    #     pos_importance[i] = 0
    #     neg_importance[i] = 0
    # for i in tqdm(range(388)):
    #     x, y = get_data(i)
    #     if getLen(x[0].tolist()) >= 40:
    #         if y == 0:
    #             outputs = model.trainModel(x)[0].tolist()
    #             if outputs[0] > outputs[1]:
    #                 new_pos_scores, max_idx, min_idx = change_and_test(x, y, model, L_min=40)
    #                 for idx in range(40):
    #                     pos_scores[idx] += new_pos_scores[idx]
    #                 pos_cnt0 += 1
    #                 if max_idx is not None:
    #                     pos_importance[max_idx] += 1
    #                 if min_idx is not None:
    #                     pos_importance[min_idx] += 1
    #         if y == 1:
    #             outputs = model.trainModel(x)[0].tolist()
    #             if outputs[0] < outputs[1]:
    #                 new_neg_scores, max_idx, min_idx = change_and_test(x, y, model, L_min=40)
    #                 for idx in range(40):
    #                     neg_scores[idx] += new_neg_scores[idx]
    #                 neg_cnt0 += 1
    #                 if max_idx is not None:
    #                     neg_importance[max_idx] += 1
    #                 if min_idx is not None:
    #                     neg_importance[min_idx] += 1
    # for i in range(40):
    #     pos_scores[i] /= pos_cnt0
    #     neg_scores[i] /= neg_cnt0
    # draw_pos_scores(pos_scores, neg_scores, 'L>=40')
    # draw_pos_importance(pos_importance, neg_importance, 'L>=40')
    # print(pos_cnt0, neg_cnt0)  # 50 14


