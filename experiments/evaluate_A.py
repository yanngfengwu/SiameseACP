# import torch
# mine = torch.load('/mnt/sdb/home/yxt/CBC_and_DATA/experiment_results/acc0.9458762886597938_model_Alter_metrics.pt')
# print(mine)
# tensor([0.9459, 0.9832, 0.9072, 0.9845, 0.9437, 0.9650, 0.8944],
#        dtype=torch.float64)
#torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])
'''AntiCP = torch.tensor([0.9227, 0.9175, 0.9201, 0.84])
# Sensitivity, Specificity, ACC, MCC'''
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

labels = ['ACC', 'Sensitivity', 'Specificity', 'MCC']
ContrastCNN = [0.9459, 0.9072, 0.9845, 0.8944]
AntiMF = [0.9103, 0.8824, 0.9378, 0.8256]
AntiCP = [0.9201, 0.9227, 0.9175, 0.84]
textCNN = [0.9046, 0.8918, 0.9175, 0.8095]
SVM = [0.7422680412371134, 0.783505150600489, 0.70103092422149, 0.48619244395904365]
KNN = [0.6855670103092784, 0.8350515420873632, 0.5360824714634924, 0.3889222341312986]
DT = [0.7164948453608248, 0.7525773157083644, 0.6804123676267404, 0.4341215710622296]

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
ax.legend(loc='lower right')
# num1 = 1.01
# num2 = 0.5
# num3 = 3
# num4 = 0
# ax.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
fig.tight_layout()
plt.savefig('../tmpPics2/Metrics_A.svg')
plt.show()
