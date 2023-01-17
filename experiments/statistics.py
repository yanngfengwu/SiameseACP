from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

def MyPlot(dict_pos, dict_neg):
    amino_acids = dict_pos.keys()
    pos_vals = dict_pos.values()
    neg_vals = dict_neg.values()
    plt.figure(figsize=(15, 12))
    x = np.arange(0, 2.1 * len(amino_acids), 2.1)  # 标签位置
    width = 0.37  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, pos_vals, 2 * width, color='#cde79b', label='ACP')
    rects2 = ax.bar(x + width, neg_vals, 2 * width, color='#6fd89c', label='non-ACP')
    ax.set_xticks(x)
    ax.set_xticklabels(amino_acids)
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig('/mnt/sdb/home/yxt/CBC_and_DATA/tmpPics/statistics.svg')


# a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
#                 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
#                 'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z': 24}
a2n_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 17, 'T': 19,
                'W': 20, 'Y': 21, 'V': 22}
amino_acids = sorted(list(a2n_dict.keys()))
pos_path = r'/mnt/sdb/home/yxt/CBC_and_DATA/dataset/AntiACP2.0_Alternate/validation/positive.txt'
neg_path = r'/mnt/sdb/home/yxt/CBC_and_DATA/dataset/AntiACP2.0_Alternate/validation/negative.txt'
pos_cnt, neg_cnt = OrderedDict(), OrderedDict()
with open(pos_path, 'r') as f:
    pos_data = f.read().replace('\n', '')
    for key in amino_acids:
        pos_cnt[key] = pos_data.count(key) / float(len(pos_data))
    f.close()
with open(neg_path, 'r') as f:
    neg_data = f.read().replace('\n', '')
    for key in amino_acids:
        neg_cnt[key] = neg_data.count(key) / float(len(neg_data))
    f.close()
MyPlot(pos_cnt, neg_cnt)
