import torch
import numpy as np
import sys
sys.path.append('/mnt/sdb/home/yxt/CBC_and_DATA')
from configuration import config as cf
from preprocess.original_data import get_original_data_Anti
from preprocess.get_data import collate, get_prelabel, MyDataSet

def test_X_on_net(data_iter, net):
    config = cf.get_train_config()
    device = config.device
    prelabel, relabel = [], []
    for x, y, z in data_iter:
        x, y = x.to(device), y.to(device)
        # print('X:', x.shape)# torch.Size([1552, 50])    # torch.Size([388, 50])
        # print('Y:', y.shape)# torch.Size([1552])        # torch.Size([388])
        outputs = net.trainModel(x)
        prelabel.append(outputs.argmax(dim=1).cpu().numpy())
        relabel.append(y.cpu().numpy())
    return prelabel[0], relabel[0]

def get_data_A():
    train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/internal")
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/validation")
    train_dataset = MyDataSet(train_data, train_label, train_seq)
    test_dataset = MyDataSet(test_data, test_label, test_seq)

    # batch_size 为 total
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=1552, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=388, shuffle=False)

    return train_iter, test_iter

def get_data_M():
    train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0_Main/internal")
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Main/validation")
    train_dataset = MyDataSet(train_data, train_label, train_seq)
    test_dataset = MyDataSet(test_data, test_label, test_seq)

    # batch_size 为 total
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=1552, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=388, shuffle=False)

    return train_iter, test_iter

def get_ACC(pred, true):
    cnt = np.sum(np.where(pred-true, 0, 1))
    return float(cnt) / pred.shape[0]

if __name__ == '__main__':
    A_train_iter, A_test_iter = get_data_A()
    M_train_iter, M_test_iter = get_data_M()
    model_A = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter.pkl')  # 加载整个神经网络的模型结构以及参数
    model_M = torch.load(
        '/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.784375_model_Main.pkl')  # 加载整个神经网络的模型结构以及参数
    model_A.eval()
    model_M.eval()
    with torch.no_grad():
        Aed_on_A_test_pred, Aed_on_A_test_true = test_X_on_net(A_test_iter, model_A)
        Aed_on_M_test_pred, Aed_on_M_test_true = test_X_on_net(M_test_iter, model_A)
        Med_on_A_test_pred, Med_on_A_test_true = test_X_on_net(A_test_iter, model_M)
        Med_on_M_test_pred, Med_on_M_test_true = test_X_on_net(M_test_iter, model_M)
        print(get_ACC(Aed_on_A_test_pred, Aed_on_A_test_true))  # 0.9458762886597938
        print(get_ACC(Aed_on_M_test_pred, Aed_on_M_test_true))  # 0.6104651162790697
        print(get_ACC(Med_on_A_test_pred, Med_on_A_test_true))  # 0.7963917525773195
        print(get_ACC(Med_on_M_test_pred, Med_on_M_test_true))  # 0.7819767441860465
