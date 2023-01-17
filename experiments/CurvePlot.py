import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
def plot_ROC(fpr1, tpr1, fpr2, tpr2, name):
    plt.figure()
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, color='#1aadff', label='Ablation Model ROC curve (area = %0.2f)' % auc2)
    plt.plot(fpr1, tpr1, color='#e3b636', label='SiameseACP ROC curve (area = %0.2f)' % auc1)
    plt.plot([0, 1], [0, 1], color='#c9dcaf', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('../tmpPics2/{}_ROC.svg'.format(name), dpi=800)
    plt.show()
def plot_PR(rec1, pre1, AP1, rec2, pre2, AP2, name):
    plt.figure()
    plt.plot(rec2, pre2, color='#1aadff', label='Ablation Model PR curve (area = %0.2f)' % AP2)
    plt.plot(rec1, pre1, color='#e3b636', label='SiameseACP PR curve (area = %0.2f)' % AP1)
    # plt.plot([0, 0.5, 1], [1, 0.5, 0], color='#A2D5AB', lw=2, linestyle='--')
    plt.xlim([0.5, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.legend(loc="lower right")
    plt.savefig('../tmpPics2/{}_PR.svg'.format(name), dpi=800)
    plt.show()

A_prc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter_prc.txt')
A_ab_prc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc0.9201030927835051_absorb_A_prc.txt')
A_roc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.9458762886597938_model_Alter_roc.txt')
A_ab_roc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc0.9201030927835051_absorb_A_roc.txt')
M_prc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.784375_model_Main_prc.txt')
M_ab_prc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc0.7034883720930233_absorb_M_prc.txt')
M_roc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_normal_results/acc0.784375_model_Main_roc.txt')
M_ab_roc = np.loadtxt('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc0.7034883720930233_absorb_M_roc.txt')
A_fpr, A_tpr = A_roc[0], A_roc[1]
A_ab_fpr, A_ab_tpr = A_ab_roc[0], A_ab_roc[1]
M_fpr, M_tpr = M_roc[0], M_roc[1]
M_ab_fpr, M_ab_tpr = M_ab_roc[0], M_ab_roc[1]
A_pre, A_rec = A_prc[0], A_prc[1]
A_ab_pre, A_ab_rec = A_ab_prc[0], A_ab_prc[1]
M_pre, M_rec = M_prc[0], M_prc[1]
M_ab_pre, M_ab_rec = M_ab_prc[0], M_ab_prc[1]
plot_ROC(A_fpr, A_tpr, A_ab_fpr, A_ab_tpr, 'A')
plot_PR(A_rec, A_pre, 0.9753358633444352, A_ab_rec, A_ab_pre, 0.9700771325003635, 'A')
plot_ROC(M_fpr, M_tpr, M_ab_fpr, M_ab_tpr, 'M')
plot_PR(M_rec, M_pre, 0.838738411155087, M_ab_rec, M_ab_pre, 0.8580894327701221, 'M')