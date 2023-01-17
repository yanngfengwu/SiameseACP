import numpy as np
import os
import torch

# iEnhancer
def get_original_data(path):# ../benchmark dataset & ../independent dataset
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
                sequences.append(line.split('\n')[0])# sequences.append(line.split('\n')[0][:50])
                labels.append(1)
                current_DNA = []
                for aa in line.split('\n')[0]:# for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                datas.append(current_DNA)
    with open(path_non_enhancer, encoding='utf-8') as f:
        for line in f.readlines():
            if line[0] == '>':
                continue
            else:
                sequences.append(line.split('\n')[0])# sequences.append(line.split('\n')[0][:50])
                labels.append(0)
                current_DNA = []
                for aa in line.split('\n')[0]:#for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                datas.append(current_DNA)

    return torch.tensor(datas), torch.tensor(labels), sequences

# AAPred-CNN
def get_original_data_AAPred(path):# ../benchmark dataset & ../independent dataset
    '''

    :param path:
    :return:
    1 represents enhancer
    0 represents non-enhancer
    '''
    sequences = []
    labels = []
    datas = []
    path_pos = os.path.join(path, 'positive.txt')
    path_neg = os.path.join(path, 'negative.txt')
    paths = [path_neg, path_pos]
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z':24}
    maxlen = 0
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                maxlen = max(maxlen, len(line.split('\n')[0]))
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                sequences.append(line.split('\n')[0])# sequences.append(line.split('\n')[0][:50])
                labels.append(i)
                current_DNA = []
                for aa in line.split('\n')[0]:# for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                for _ in range(maxlen - len(line.split('\n')[0])):
                    current_DNA.append(0)
                datas.append(current_DNA)


    # print(torch.tensor(datas).shape, torch.tensor(labels).shape)
    return torch.tensor(datas), torch.tensor(labels), sequences

# ACPred-FL
def get_original_data_ACPred(path):# ../benchmark dataset & ../independent dataset
    '''

    :param path:
    :return:
    1 represents enhancer
    0 represents non-enhancer
    '''
    sequences = []
    labels = []
    datas = []
    path_pos = os.path.join(path, 'positive.txt')
    path_neg = os.path.join(path, 'negative.txt')
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z':24}
    paths = [path_neg, path_pos]
    maxlen = 0
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                maxlen = max(maxlen, len(line.split('\n')[0]))
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                # print(line)
                if line[0] == '>':
                    continue
                else:
                    sequences.append(line.split('\n')[0])  # sequences.append(line.split('\n')[0][:50])
                    labels.append(i)
                    current_DNA = []
                    for aa in line.split('\n')[0]:  # for aa in line.split('\n')[0][:50]:
                        current_DNA.append(aa_dict[aa])
                    for _ in range(maxlen - len(line.split('\n')[0])):
                        current_DNA.append(0)
                    datas.append(current_DNA)

    return torch.tensor(datas), torch.tensor(labels), sequences

# AntiACP2.0
def get_original_data_Anti(path):# ../benchmark dataset & ../independent dataset
    '''

    :param path:
    :return:
    1 represents enhancer
    0 represents non-enhancer
    '''
    sequences = []
    labels = []
    datas = []
    path_pos = os.path.join(path, 'positive.txt')
    path_neg = os.path.join(path, 'negative.txt')
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z':24}
    paths = [path_neg, path_pos]
    maxlen = 0
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                maxlen = max(maxlen, len(line.split('\n')[0]))
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                sequences.append(line.split('\n')[0])# sequences.append(line.split('\n')[0][:50])
                labels.append(i)
                current_DNA = []
                for aa in line.split('\n')[0]:# for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                for _ in range(maxlen - len(line.split('\n')[0])):
                    current_DNA.append(0)
                datas.append(current_DNA)

    return torch.tensor(datas), torch.tensor(labels), sequences
def get_original_data_Anti_new(path):# ../benchmark dataset & ../independent dataset
    '''

    :param path:
    :return:
    1 represents enhancer
    0 represents non-enhancer
    '''
    sequences = []
    labels = []
    datas = []
    path_pos = os.path.join(path, 'new_positive.txt')
    path_neg = os.path.join(path, 'new_negative.txt')
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z':24}
    paths = [path_neg, path_pos]
    maxlen = 0
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                maxlen = max(maxlen, len(line.split('\n')[0]))
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                sequences.append(line.split('\n')[0])# sequences.append(line.split('\n')[0][:50])
                labels.append(i)
                current_DNA = []
                for aa in line.split('\n')[0]:# for aa in line.split('\n')[0][:50]:
                    current_DNA.append(aa_dict[aa])
                for _ in range(maxlen - len(line.split('\n')[0])):
                    current_DNA.append(0)
                datas.append(current_DNA)

    return torch.tensor(datas), torch.tensor(labels), sequences
# MPMABP
def get_original_data_MPMABP(path):# ../benchmark dataset & ../independent dataset
    '''

    :param path:
    :return:
    1 represents enhancer
    0 represents non-enhancer
    '''
    sequences = []
    labels = []
    datas = []
    path_1 = os.path.join(path, 'ACP.txt')
    path_2 = os.path.join(path, 'ADP.txt')
    path_3 = os.path.join(path, 'AHP.txt')
    path_4 = os.path.join(path, 'AIP.txt')
    path_5 = os.path.join(path, 'AMP.txt')
    paths = [path_1, path_2, path_3, path_4, path_5]
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z': 24}
    maxlen = 0
    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                maxlen = max(maxlen, len(line.split('\n')[0]))

    for i, path in enumerate(paths):
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                # print(line)
                if line[0] == '>':
                    continue
                else:
                    sequences.append(line.split('\n')[0])  # sequences.append(line.split('\n')[0][:50])
                    labels.append(i)
                    current_DNA = []
                    for aa in line.split('\n')[0]:  # for aa in line.split('\n')[0][:50]:
                        current_DNA.append(aa_dict[aa])
                    for _ in range(maxlen - len(line.split('\n')[0])):
                        current_DNA.append(0)
                    datas.append(current_DNA)
    return torch.tensor(datas), torch.tensor(labels), sequences

# 这边一律用了0填补后面不满足max_len的部分，max_len直接用了这个集合中的最大长度
if __name__ == '__main__':
    # a = [[1,2,3],[1,2],[3,2,1,2]]
    # print(torch.tensor(a))
    # train_data, train_label, train_seq = get_original_data("../dataset/iEnhancer/benchmark dataset")
    # test_data, test_label, test_seq = get_original_data("../dataset/iEnhancer/independent dataset")
    # train_data, train_label, train_seq = get_original_data_AAPred("../dataset/AAPred-CNN/benchmark")
    # test_data, test_label, test_seq = get_original_data_AAPred("../dataset/AAPred-CNN/independent")
    # train_data, train_label, train_seq = get_original_data_ACPred("../dataset/ACPred-FL/ACP500（train）")
    # test_data, test_label, test_seq = get_original_data_ACPred("../dataset/ACPred-FL/ACP164（test）")
    # train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0/internal")
    # test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0/validation")
    data, label, seq = get_original_data_MPMABP('../dataset/MPMABP')

    # print(train_data.shape, train_label.shape)
    # print(test_data.shape, test_label.shape)
    print(data, label)
    print(data.shape, label.shape)
    '''
    iEnhancer:
        torch.Size([2968, 200]) torch.Size([2968])
        torch.Size([400, 200]) torch.Size([400])
    AAPred-CNN:
        torch.Size([270, 67]) torch.Size([270])
        torch.Size([28, 43]) torch.Size([28])
    ACPred-FL:
        torch.Size([500, 97]) torch.Size([500])
        torch.Size([164, 207]) torch.Size([164])
    AntiACP2.0:
        torch.Size([1552, 50]) torch.Size([1552])
        torch.Size([388, 50]) torch.Size([388])
    MPMABP:
        torch.Size([6115, 517]) torch.Size([6115])
    '''