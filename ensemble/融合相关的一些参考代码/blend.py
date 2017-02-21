from __future__ import division
import numpy as np
#import load_data
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import pandas as pd

    
def loadData():
    """Conveninence function to load all data as numpy arrays.
    """
    print ("Loading data...")
    features=['zan', 'answer', 'perfect',
       'perf_ans', 'unperf_ans', 'q_index', 'q_inviteNum', 'q_answerNum',
       'q_answerRate', 'q_unanswerRate', 'q_perfectRate', 'q_unperfectRate',
       'qlabel_rate', 'u_index', 'u_inviteTimes',
       'u_invitelabel', 'u_answerlabel', 'u_answerTimes', 'u_answerRate',
        'u_noAnswerTimes', 'u_labelNum', 'common_word',
       'common_alpha', 'common_label']

    train = pd.read_csv("data/train_data_features.csv")
    X_train=train[features]
    y_train=train['label']
    val = pd.read_csv("data/val_data_features.csv")
    X_test=val[features]
    y_test=val['label']
    test = pd.read_csv("data/test_features.csv")
    return X_train,y_train,X_test,y_test,test[features]


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set

    X_train,y_train,X_test,y_test,X_submission = loadData()


    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print ("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X_test.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))


    for j, clf in enumerate(clfs):
        print (j, clf)
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[:, j] = y_submission
        dataset_blend_test[:, j] = clf.predict_proba(X_submission)[:, 1]
 

    print
    print ("Blending.")
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y_test)
    y_submission = clf.predict_proba(dataset_blend_test)[:, 1]

    print ("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print ("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission)+1), y_submission]).T
    np.savetxt(fname='submission.csv', X=tmp, fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability', comments='')

    to_sub = pd.read_csv("data/test_nolabel.txt")
    to_sub['label']=y_submission
    to_sub.to_csv('submission_blend.csv',index=False)