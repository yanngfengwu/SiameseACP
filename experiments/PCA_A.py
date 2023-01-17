import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
colorslist = ['#bdf85c', '#55d9ff']
myCmap = colors.LinearSegmentedColormap.from_list('mylist',colorslist,N=800)
import sys
print(sys.path)
sys.path.append('/mnt/sdb/home/yxt/CBC_and_DATA')
from configuration import config as cf
from preprocess.original_data import get_original_data_Anti
from preprocess.get_data import collate, get_prelabel, MyDataSet
from model.ContrastModel import newModel

def get_PCA(data_iter, net):
    config = cf.get_train_config()
    device = config.device
    prelabel, relabel = [], []
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
        print('X:', x.shape)# torch.Size([1552, 50])    # torch.Size([388, 50])
        print('Y:', y.shape)# torch.Size([1552])        # torch.Size([388])
        outputs = net.trainModel(x)
        prelabel.append(outputs.argmax(dim=1).cpu().numpy())
        relabel.append(y.cpu().numpy())
        pca = PCA(n_components=2)
        newX = pca.fit_transform(net.forward(x).cpu().numpy())
    return newX, prelabel, relabel

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

def plot_PCA(datas, name):
    i = 1
    titles = ['train dataset', 'test dataset']
    f, ax = plt.subplots(1, 2)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
    for i in range(2):
            ax[i].set_title(titles[i])
    for data in datas:
        x = data[0][:, 0]
        y = data[0][:, 1]
        label = data[1]
        plt.subplot(1,2,i)
        plt.scatter(x, y, c=label, alpha=0.5, cmap=myCmap)#'Spectral')
        i += 1
    plt.savefig('../tmpPics2/{}_PCA_Plot_A.svg'.format(name))
    plt.show()

if __name__ == '__main__':
    train_iter, test_iter = get_data()
    model = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter.pkl')  # 加载整个神经网络的模型结构以及参数
    # model.load_state_dict(torch.load('net_params.pkl'))# 仅加载参数，重复效果，没必要
    model.eval()
    with torch.no_grad():
        train_x, train_pre, train_true = get_PCA(train_iter, model)
        test_x, test_pre, test_true = get_PCA(test_iter, model)
    plot_PCA([[train_x, train_true],
              #[train_x, train_pre],
              [test_x, test_true],
              #[test_x, test_pre]
              ],
             'trained model')
    model = newModel()
    config = cf.get_train_config()
    model.to(config.device)
    model.eval()
    with torch.no_grad():
        train_x, train_pre, train_true = get_PCA(train_iter, model)
        test_x, test_pre, test_true = get_PCA(test_iter, model)
    plot_PCA([[train_x, train_true],
              # [train_x, train_pre],
              [test_x, test_true],
              # [test_x, test_pre]
              ],
             'original model')

