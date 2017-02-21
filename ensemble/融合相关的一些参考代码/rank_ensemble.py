#coding=utf-8

"""
模型融合示例，实现简单的线性加权融合，但对单模型的结果先进行了rank，再加权。
通过validation set选取最佳的融合方案。

"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

val = pd.read_csv('../data/validation/validation_set.csv')

xgb_7844 = pd.read_csv('xgb_7844.csv')
svm_771 = pd.read_csv('svm_771.csv')
xgb_787 = pd.read_csv('xgb_787.csv')

xgb_7844.score = xgb_7844.score.rank()
svm_771.score = svm_771.score.rank()
xgb_787.score = xgb_787.score.rank()

Idx = xgb_7844.Idx

pred = 0.7*xgb_787.score + 0.2*xgb_7844.score + 0.1*svm_771.score

auc = int(roc_auc_score(val.target.values,pred.values)*10000)
print auc