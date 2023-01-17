# import torch
# mine = torch.load('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc0.771875_model_Main_metrics.pt')
# print(mine)
# tensor([0.7844, 0.7310, 0.8446, 0.7326, 0.7837, 0.8456, 0.5769],
#        dtype=torch.float64)
#torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])
'''AntiCP = torch.tensor([0.7746, 0.7341, 0.7543, 0.51])
# Sensitivity, Specificity, ACC, MCC'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['ACC', 'Sensitivity', 'Specificity', 'MCC']
ContrastCNN = [0.7844, 0.8446, 0.7326, 0.5769]
AntiMF = [0.7594,	0.6534,	0.8663,	0.5324]
AntiCP = [0.7543, 0.7746, 0.7341, 0.51]
textCNN = [0.7471,	0.6105,	0.8837,	0.5137]
SVM = [0.6540697674418605, 0.6744186007301244, 0.6337209265481342, 0.3083950381844256]
KNN = [0.6831395348837209, 0.7906976698215252, 0.5755813920024337, 0.3750598038437739]
DT = [0.7005813953488372, 0.7034883680029747, 0.6976744145484046, 0.40116957093295497]
# ACC, Sensitivity, Specificity, MCC

x = np.arange(0, 3 * len(labels), 3)  # 标签位置
width = 0.37  # 柱状图的宽度，可以根据自己的需求和审美来改

fig, ax = plt.subplots()
rects1 = ax.bar(x - width * 3, ContrastCNN, width, color='#87ceeb', label='SiameseACP')
rects2 = ax.bar(x - width * 2, AntiMF, width, color='#88ead7', label='AntiMF')
rects3 = ax.bar(x - width * 1, AntiCP, width, color='#88eaa6', label='AntiCP 2.0')
rects4 = ax.bar(x, textCNN, width, color='#a0e092', label='TextCNN')
rects5 = ax.bar(x + width * 1, SVM, width, color='#c9ea88', label='SVM')
rects6 = ax.bar(x + width * 2, KNN, width, color='#e9da89', label='KNN')
rects7 = ax.bar(x + width * 3, DT, width, color='#e9ab89', label='DecisionTree')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
fig.tight_layout()
plt.savefig('../tmpPics2/Metrics_M.svg')
plt.show()
