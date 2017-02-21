"""
calculate the mic of two result

"""


import pandas as pd
import numpy as np
from minepy import MINE

fs = ['discret_5','R_7199','rank','discret_10','raw_rank','Py_717','Py_725','svm_6938']

#对预测结果进行最大信息系数（MIC）计算  直观的观察单模型之间的差异性
res = []
res.append(pd.read_csv('./avg_xgbs_discret_feature_5.csv').score.values)
res.append(pd.read_csv('./R_7199.csv').score.values)
res.append(pd.read_csv('./rank_feature_xgb_ensemble.csv').score.values)
res.append(pd.read_csv('./avg_xgbs_discret_feature_10.csv').score.values)
res.append(pd.read_csv('./based_on_select_rank_feature.csv').score.values)
res.append(pd.read_csv('./xgb717.csv').score.values)
res.append(pd.read_csv('./725.csv').score.values)
res.append(pd.read_csv('./svm6938.csv').score.values)

cm = []
for i in range(8):
    tmp = []
    for j in range(8):
        m = MINE()
        m.compute_score(res[i], res[j])
        tmp.append(m.mic())
    cm.append(tmp)


import numpy as np
import matplotlib.pyplot as plt
#混淆矩阵的形式
#画出（颜色越浅，表示相关性越小）
def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(8)
    plt.xticks(tick_marks, fs, rotation=45)
    plt.yticks(tick_marks, fs)
    plt.tight_layout()

plot_confusion_matrix(cm, title='mic')
plt.show()

