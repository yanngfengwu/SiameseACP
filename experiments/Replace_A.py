import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
colorslist = ['#557B83', '#A2D5AB']
myCmap = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=800)
import sys
print(sys.path)
sys.path.append('/mnt/sdb/home/yxt/CBC_and_DATA')
from configuration import config as cf
from preprocess.original_data import get_original_data_Anti
from preprocess.get_data import collate, get_prelabel, MyDataSet

def replace_predict(data_iter, net):
    config = cf.get_train_config()
    device = config.device
    prelabel, relabel = [], []
    imp0, imp1 = {}, {}
    cnt0, cnt1 = {}, {}
    for i in range(1, 25):# 里面有1~24，是根据我们的编码来的
        imp0[i], imp1[i] = 0, 0
        cnt0[i], cnt1[i] = 0, 0
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)# torch.Size([1552, 50])    # torch.Size([1552])
        outputs = net.trainModel(x)
        # print(outputs[:, 0])# 选0的倾向（不是0~1，可以负，可以大于1）
        # print(outputs[:, 1])# 选1的倾向（不是0~1，可以负，可以大于1）
        # print(y)
        for i in range(x.shape[0]):
            xi = x[i, :].unsqueeze(0)
            output0 = net.trainModel(xi)
            # print(xi, '\n', '*'*20, '\n', output0[:,0].item(), output0[:,1].item())
            for j in range(x.shape[1]):
                if xi[0][j] == 0:
                    # print(j)
                    break
                for num in range(1, 25):
                    if xi[0][j] == num:
                        continue
                    x_replace = xi.clone()
                    x_replace[0][j] = num
                    output1 = net.trainModel(x_replace)
                    imp0[xi[0][j].item()] += abs(output0[:, 0].item() - output1[:, 0].item())
                    imp1[xi[0][j].item()] += abs(output0[:, 1].item() - output1[:, 1].item())
                    cnt0[xi[0][j].item()] += 1
                    cnt1[xi[0][j].item()] += 1
            break
    for i in range(1, 25):
        if cnt0[i] != 0:
            imp0[i] = imp0[i] / cnt0[i]
            imp1[i] = imp1[i] / cnt1[i]
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23, 'Z': 24}
    new_imp0, new_imp1 = {}, {}
    for key in aa_dict.keys():
        new_imp0[key] = imp0[aa_dict[key]]
        new_imp1[key] = imp1[aa_dict[key]]
    print(imp0, imp1)
    print(new_imp0, new_imp1)
    return new_imp0, new_imp1

def get_data():
    train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/internal")
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/validation")
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)
    train_dataset = MyDataSet(train_data, train_label, train_seq)
    test_dataset = MyDataSet(test_data, test_label, test_seq)

    # batch_size 为 total
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=1552, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=388, shuffle=False)

    return train_iter, test_iter


if __name__ == '__main__':
    train_iter, test_iter = get_data()
    model = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter.pkl')  # 加载整个神经网络的模型结构以及参数
    # model.load_state_dict(torch.load('net_params.pkl'))# 仅加载参数，重复效果，没必要
    model.eval()
    with torch.no_grad():
        train_imp0, train_imp1 = replace_predict(train_iter, model)
        test_imp0, test_imp1 = replace_predict(test_iter, model)
