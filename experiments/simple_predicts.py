import sys
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, matthews_corrcoef
print(sys.path)
sys.path.append('/mnt/sdb/home/yxt/CBC_and_DATA')
from preprocess.get_data import collate, get_prelabel, MyDataSet
from preprocess.original_data import get_original_data, get_original_data_AAPred, get_original_data_ACPred, get_original_data_Anti, get_original_data_MPMABP
from MyUtils.util_eval import evaluate_accuracy
from MyUtils.util_cal import caculate_metric
from model.ContrastModel import newModel, ContrastiveLoss
from configuration import config as cf
import numpy as np
config = cf.get_train_config()
config.lr = 1e-4
config.epoch = 300
def sensitivityCalc(Predictions, Labels):
    MCM = confusion_matrix(Labels, Predictions)
    # MCM此处是 5 * 2 * 2的混淆矩阵（ndarray格式），5表示的是5分类

    # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
    tn_sum = MCM[0, 0] # True Negative
    fp_sum = MCM[0, 1] # False Positive

    tp_sum = MCM[1, 1] # True Positive
    fn_sum = MCM[1, 0] # False Negative

    # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
    Condition_negative = tp_sum + fn_sum + 1e-6

    sensitivity = tp_sum / Condition_negative
    macro_sensitivity = np.average(sensitivity, weights=None)

    micro_sensitivity = np.sum(tp_sum) / np.sum(tp_sum+fn_sum)

    return macro_sensitivity, micro_sensitivity

def specificityCalc(Predictions, Labels):
    MCM = confusion_matrix(Labels, Predictions)
    tn_sum = MCM[0, 0]
    fp_sum = MCM[0, 1]

    tp_sum = MCM[1, 1]
    fn_sum = MCM[1, 0]

    Condition_negative = tn_sum + fp_sum + 1e-6

    Specificity = tn_sum / Condition_negative
    macro_specificity = np.average(Specificity, weights=None)

    micro_specificity = np.sum(tn_sum) / np.sum(tn_sum+fp_sum)

    return macro_specificity, micro_specificity
'''
macro指对单个类别计算F1值,再用其算数平均值作为最终结果;而micro将全部类别当作一个整体,只计算1次F1值。
'''
def get_A_Data():
    train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/internal")
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Alternate/validation")
    # print(train_data.shape, train_label.shape)
    # print(test_data.shape, test_label.shape)
    # print(train_data, train_label, train_seq, test_data, test_label, test_seq)
    return train_data.tolist(), train_label.tolist(), test_data.tolist(), test_label.tolist()
'''
Alternate:
    torch.Size([1552, 50]) torch.Size([1552])
    torch.Size([388, 50]) torch.Size([388])
'''
def get_M_Data():
    train_data, train_label, train_seq = get_original_data_Anti("../dataset/AntiACP2.0_Main/internal")
    test_data, test_label, test_seq = get_original_data_Anti("../dataset/AntiACP2.0_Main/validation")
    # print(train_data.shape, train_label.shape)
    # print(test_data.shape, test_label.shape)
    return train_data.tolist(), train_label.tolist(), test_data.tolist(), test_label.tolist()

if __name__ == '__main__':
    # SVM A
    train_data, train_label, test_data, test_label = get_A_Data()
    model = svm.SVC()
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    print(accuracy_score(y_pred, test_label), sensitivityCalc(y_pred, test_label)[0], specificityCalc(y_pred, test_label)[0], matthews_corrcoef(y_pred, test_label))
    # SVM M
    train_data, train_label, test_data, test_label = get_M_Data()
    model = svm.SVC()
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    print(accuracy_score(y_pred, test_label), sensitivityCalc(y_pred, test_label)[0], specificityCalc(y_pred, test_label)[0], matthews_corrcoef(y_pred, test_label))
    # KNN A
    train_data, train_label, test_data, test_label = get_A_Data()
    model = knn()
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    print(accuracy_score(y_pred, test_label), sensitivityCalc(y_pred, test_label)[0],
          specificityCalc(y_pred, test_label)[0], matthews_corrcoef(y_pred, test_label))
    # KNN M
    train_data, train_label, test_data, test_label = get_M_Data()
    model = knn()
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    print(accuracy_score(y_pred, test_label), sensitivityCalc(y_pred, test_label)[0],
          specificityCalc(y_pred, test_label)[0], matthews_corrcoef(y_pred, test_label))
    # DecisionTree A
    train_data, train_label, test_data, test_label = get_A_Data()
    model = DecisionTreeClassifier()
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    print(accuracy_score(y_pred, test_label), sensitivityCalc(y_pred, test_label)[0],
          specificityCalc(y_pred, test_label)[0], matthews_corrcoef(y_pred, test_label))
    # DecisionTree M
    train_data, train_label, test_data, test_label = get_M_Data()
    model = DecisionTreeClassifier()
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    print(accuracy_score(y_pred, test_label), sensitivityCalc(y_pred, test_label)[0],
          specificityCalc(y_pred, test_label)[0], matthews_corrcoef(y_pred, test_label))

'''
python simple_predicts.py
acc sensitivity specificity mcc

SVM
A
0.7422680412371134 0.783505150600489 0.70103092422149 0.48619244395904365
M
0.6540697674418605 0.6744186007301244 0.6337209265481342 0.3083950381844256

KNN
A
0.6855670103092784 0.8350515420873632 0.5360824714634924 0.3889222341312986
M
0.6831395348837209 0.7906976698215252 0.5755813920024337 0.3750598038437739

DecisionTree
A
0.7164948453608248 0.7525773157083644 0.6804123676267404 0.4341215710622296
M
0.7005813953488372 0.7034883680029747 0.6976744145484046 0.40116957093295497
'''