""" Amazon Access Challenge Code for ensemble

Marios Michaildis script for Amazon .

xgboost on input data

based on Paul Duan's Script.

"""
from __future__ import division
import numpy as np
from sklearn import  preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from xgboost import XGBClassifier
import pandas as pd
import sys,random


SEED = 42  # always use a seed for randomized procedures


def load_data(filename, use_labels=True):
    """
    Load data from CSV files and return them as numpy arrays
    The use_labels parameter indicates whether one should
    read the first column (containing class labels). If false,
    return all 0s. 
    """

    # load column 1 to 8 (ignore last one)
    data = np.loadtxt(open( filename), delimiter=',',
                      usecols=range(1, 9), skiprows=1)
    if use_labels:
        labels = np.loadtxt(open( filename), delimiter=',',
                            usecols=[0], skiprows=1)
    else:
        labels = np.zeros(data.shape[0])
    return labels, data


def save_results(predictions, filename):
    """Given a vector of predictions, save results in CSV format."""
    with open(filename, 'w') as f:
        f.write("id,ACTION\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


def bagged_set(X_t,y_c, estimators, xt):
    
   # create array object to hold predictions 
   baggedpred=[ 0.0  for d in range(0, (xt.shape[0]))]
   #loop for as many times as we want bags
   for n in range (0, estimators):
        #shuff;e first, aids in increasing variance and forces different results
        #X_t,y_c=shuffle(Xs,ys, random_state=seed+n)
        print '{0} model'.format(n)          
        model = xgb_param(n)
        model.fit(X_t,y_c) # fit model0.0917411475506
        preds=model.predict_proba(xt)[:,1] # predict probabilities
        # update bag's array
        for j in range (0, (xt.shape[0])):           
                baggedpred[j]+=preds[j]
   # divide with number of bags to create an average estimate            
   for j in range (0, len(baggedpred)): 
                baggedpred[j]/=float(estimators)
   # return probabilities            
   return np.array(baggedpred) 
   
   
# using numpy to print results
def printfilcsve(X, filename):

    np.savetxt(filename,X) 

def xgb_param(k):
    random_seed = range(2017)
    gamma = [i/1000.0 for i in range(50,300,10)]
    max_depth = [5,6,7,8]
    lambd = range(200,400,5)
    subsample = [i/1000.0 for i in range(500,800,10)]
    colsample_bytree = [i/1000.0 for i in range(250,450,10)]
    min_child_weight = [i/1000.0 for i in range(1000,4000,30)]
    scale_pos_weight = [i for i in range(2,6)]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    random.shuffle(scale_pos_weight)

    model = xgb_model(random_seed[k],gamma[k],max_depth[k%4],lambd[k],subsample[k],colsample_bytree[k],min_child_weight[k],scale_pos_weight[k%4])
    return model

def xgb_model(seed=1,gamma=0.1,max_depth=12,lamb=300,subsample=0.7,colsample_bytree=0.5,min_child_weight=1,scale_pos_weight=1):
    model=XGBClassifier(n_estimators=350 ,nthread=8,  learning_rate=0.08, gamma=gamma,max_depth=max_depth, min_child_weight=min_child_weight, subsample=subsample, 
    colsample_bytree=colsample_bytree,objective='binary:logistic',seed=seed,reg_lambda=lamb,scale_pos_weight=scale_pos_weight) 
    return model
def main():
    
    filename="H:\\ET/model/main_xgb_2" # nam prefix
    #model = linear_model.LogisticRegression(C=3)  # the classifier we'll use
    
    # === load data in memory === #
    print "loading data"

    train = pd.read_csv("H:\\ET/data/train_flit.csv")
    test = pd.read_csv("H:\\ET/data/test_flit.csv")
    train[train>10000000000] = 0
    test[test>10000000000] = 0

    overdue_train = pd.read_csv("H:\\ET/data/overdue_train.csv") 
    overdue_train = overdue_train.rename(columns={'user_id':'userid'})
    train = pd.merge(train,overdue_train,on='userid',how='left')
    scoreDir = 'H:\\ET/feature_score'
    feature_score_xgb = pd.read_csv('H:\\ET/feature_score/rank_feature_score.csv')
    fea_set = feature_score_xgb.feature.values

    y, X = train.label.values,train[fea_set[range(1,len(fea_set),2)]].values
    #y_test, X_test = valid.label.values,valid.drop(['userid','label'],axis=1).values
    X_test = test[fea_set[range(1,len(fea_set),2)]].values

    #create arrays to hold cv an dtest predictions
    train_stacker=[ 0.0  for k in range (0,(X.shape[0])) ] 

    # === training & metrics === #
    mean_auc = 0.0
    mean_ks = 0.0
    bagging=3 # number of models trained with different seeds
    n = 4  # number of folds in strattified cv
    kfolder=StratifiedKFold(y, n_folds= n,shuffle=True, random_state=1)     
    i=0
    for train_index, test_index in kfolder: # for each train and test pair of indices in the kfolder object
        # creaning and validation sets
        X_train, X_cv = X[train_index], X[test_index]
        y_train, y_cv = np.array(y)[train_index], np.array(y)[test_index]
        print (" train size: %d. test size: %d, cols: %d " % ((X_train.shape[0]) ,(X_cv.shape[0]) ,(X_train.shape[1]) ))

        # train model and make predictions 
        preds=bagged_set(X_train,y_train, bagging, X_cv)   
        
        # compute AUC metric for this CV fold
        roc_auc = roc_auc_score(y_cv, preds)
        ks = ks_score(preds,y_cv)
        print "AUC (fold %d/%d): %f" % (i + 1, n, roc_auc)
        print "KS (fold %d/%d): %f" % (i + 1, n, ks)
        mean_auc += roc_auc
        mean_ks += ks
        no=0
        for real_index in test_index:
                 train_stacker[real_index]=(preds[no])
                 no+=1
        i+=1
        
    mean_auc/=n
    mean_ks/=n
    print (" Average AUC: %f" % (mean_auc) )
    print (" Average ks: %f" % (mean_ks) )
    print (" printing train datasets ")
    #printfilcsve(np.array(train_stacker), filename + ".train.csv")          
    save_results(np.array(train_stacker), filename + ".train.csv")

    # === Predictions === #
    # When making predictions, retrain the model on the whole training set
    preds=bagged_set(X, y, bagging, X_test)  

    #create submission file 
    #printfilcsve(np.array(preds), filename+ ".test.csv")  
    #save_results(preds, filename +str(mean_auc) + ".valid.csv")
    save_results(preds, filename+"_submission_" +str(mean_ks) + ".test.csv")

def ks_score(pred,real):
    from sklearn.metrics import roc_curve  
    fpr,tpr,thres = roc_curve(real,pred,pos_label=1)
    ks = abs(fpr-tpr).max()
    print 'KS value: ',ks
    return float(ks)

if __name__ == '__main__':
    
	main()
