# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:24:03 2017

@author: mirdar

两层的stacking
"""

from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
import os
os.chdir('H:\\ML\\DC\\user_loan_risk_predict')
import mergeFeature as mf
import pandas as pd
import ks

def get_train_data():
    dataset== pd.read_csv("H:\DC\个人征信\zudui",encoding="gb2312") # 注意自己数据路径
    X=1
    y=2
    return X,y
    
def get_test_data():
    
    
def run():
    np.random.seed(0)  # seed to shuffle the train set
    n_folds = 4
#    verbose = True
    shuffle = False

    X,y = get_train_data()
    X_submission = mf.get_test_data()

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))
# 这里可以改变参数生成多个模型

    clfs = [RandomForestClassifier(n_estimators=500,
                                   max_features=0.8,
                                   bootstrap=True,
                                   min_samples_leaf=50,
                                   oob_score=True,
                                   criterion='gini',
                                   n_jobs=-1),
            RandomForestClassifier(n_estimators=500,
                                   max_features=0.5,
                                   bootstrap=True,
                                   min_samples_leaf=50,
                                   oob_score=True,
                                   criterion='entropy',
                                   n_jobs=-1),
            ExtraTreesClassifier(n_estimators=500,
                                   min_samples_leaf=50,
                                   criterion='gini',
                                   n_jobs=-1),
            ExtraTreesClassifier(n_estimators=500,
                                   min_samples_leaf=50,
                                   criterion='entropy',
                                   n_jobs=-1),
            GradientBoostingClassifier(learning_rate=0.05, 
                                       n_estimators=500,
                                       max_depth=3, 
                                       max_features=0.65, 
                                       subsample=0.7,
                                       random_state=10,
                                       min_samples_split=350,
                                       min_samples_leaf=70),
            GradientBoostingClassifier(learning_rate=0.01, 
                                       n_estimators=1000,
                                       max_depth=4, 
                                       max_features=0.7, 
                                       subsample=0.8,
                                       random_state=10,
                                       min_samples_split=350,
                                       min_samples_leaf=70),
            XGBClassifier(learning_rate=0.05,
                                      n_estimators=350,
                                      gamma=0,
                                      min_child_weight=5,
                                      max_depth=5,
                                      subsample=0.8,
                                      scale_pos_weight=1,
                                      colsample_bytree=0.8,
                                      objective='binary:logistic',
                                      nthread=8,
                                      eval_metric= 'auc',
                                      seed=10),
            XGBClassifier(learning_rate=0.02,
                                      n_estimators=500,
                                      gamma=0,
                                      min_child_weight=5,
                                      max_depth=5,
                                      subsample=0.7,
                                      scale_pos_weight=1,
                                      colsample_bytree=0.7,
                                      objective='binary:logistic',
                                      nthread=8,
                                      eval_metric= 'auc',
                                      seed=10) 
                                      ]

    print ("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print (j, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print ("Fold", i)
            X_train = X.ix[train,:]
            y_train = y[train]
            X_test = X.ix[test,:]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            print ("train ks_score: ",ks.ks_score(y_submission,y_test))
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

    print
    print ("Blending.")
    clf = LogisticRegression()
    
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    dataset_blend_train.to_csv('H:\\ML\\DC\\user_loan_risk_predict\\predict/dataset_blend_train.csv',index=False)
    y.to_csv('H:\\ML\\DC\\user_loan_risk_predict\\predict/y.csv',index=False)
    y_submission.to_csv('H:\\ML\\DC\\user_loan_risk_predict\\predict/y_submission.csv',index=False)
    X_user_id.to_csv('H:\\ML\\DC\\user_loan_risk_predict\\predict/X_user_id.csv',index=False)
	
    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    test_pre = pd.DataFrame({u'userid':X_user_id,u'probability':y_submission})
    test_pre = test_pre[['userid','probability']]    
    print (test_pre.head())
    test_pre.to_csv('H:\\ML\\DC\\user_loan_risk_predict\\predict/pre_blending.csv',index=False)
               
if __name__ == '__main__':
    run()  
    print ('run end')
     