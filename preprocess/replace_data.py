# AntiACP2.0
import os
import torch

def get_original_data_Anti(path, previous, replace):# ../benchmark dataset & ../independent dataset
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