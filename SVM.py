
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)


class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         
         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist) 
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )
         
         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)         
         self.clf.fit(self.kernels_,self.y_)
         
         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred

def main():

    adult = pd.read_csv('data/adult_parsed.csv')

    adult_income_X = adult.drop('income',1).copy().values
    adult_income_Y = adult['income'].copy().values

    # wine_data = pd.read_csv('data/winequality_white.csv')
    # wine_data['category'] = wine_data['quality'] >= 7
    #
    # wineX = wine_data[wine_data.columns[0:11]].values
    # wineY = wine_data['category'].values.astype(np.int)

    adult_income_trgX, adult_income_tstX, adult_income_trgY, adult_income_tstY = ms.train_test_split(adult_income_X, adult_income_Y, test_size=0.3, random_state=0,stratify=adult_income_Y)
    # wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)

    N_adult_income = adult_income_trgX.shape[0]
    # N_wine = wine_trgX.shape[0]

    # alphas = [10**-x for x in np.arange(1,9.01,1/2)]


    #Linear SVM
    pipeM = Pipeline([('Scale',StandardScaler()),
                    ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                    ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                    ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])
    pipeA = Pipeline([('Scale',StandardScaler()),
                    ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

    params_adult_income = {'SVM__alpha':[100,10,1,0.1,0.001,0.0001],'SVM__n_iter':np.arange(0.1,1,10)}
    # params_wine = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_wine)/.8)+1]}

    adult_income_clf = basicResults(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,params_adult_income,'SVM_Lin','adult_income')
    # wine_clf = basicResults(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,params_wine,'SVM_Lin','wine')


    #wine_final_params = {'SVM__alpha': 0.031622776601683791, 'SVM__n_iter': 687.25}
    # wine_final_params = wine_clf.best_params_
    # wine_OF_params = {'SVM__n_iter': 1303, 'SVM__alpha': 1e-16}
    #adult_income_final_params ={'SVM__alpha': 0.0001, 'SVM__n_iter': 428}
    adult_income_final_params =adult_income_clf.best_params_
    adult_income_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}


    # pipeM.set_params(**wine_final_params)
    # makeTimingCurve(wineX,wineY,pipeM,'SVM_Lin','wine')
    pipeA.set_params(**adult_income_final_params)
    makeTimingCurve(adult_income_X,adult_income_Y,pipeA,'SVM_Lin','adult_income')

    # pipeM.set_params(**wine_final_params)
    # iterationLC(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_Lin','wine')
    pipeA.set_params(**adult_income_final_params)
    iterationLC(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','adult_income')

    pipeA.set_params(**adult_income_OF_params)
    iterationLC(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','adult_income')
    # pipeM.set_params(**wine_OF_params)
    # iterationLC(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_LinOF','wine')






    #RBF SVM
    gamma_fracsA = np.arange(0.2,2.1,0.2)
    gamma_fracsM = np.arange(0.05,1.01,0.1)

    #
    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                     ('SVM',primalSVM_RBF())])

    pipeA = Pipeline([('Scale',StandardScaler()),
                     ('SVM',primalSVM_RBF())])


    params_adult_income = {'SVM__alpha':[100,10,1,0.1,0.001,0.0001],'SVM__n_iter':[int((1e6/N_adult_income)/.8)+1],'SVM__gamma_frac':gamma_fracsA}
    # params_wine = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_wine)/.8)+1],'SVM__gamma_frac':gamma_fracsM}
    #
    # wine_clf = basicResults(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,params_wine,'SVM_RBF','wine')
    adult_income_clf = basicResults(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,params_adult_income,'SVM_RBF','adult_income')

    # wine_final_params = wine_clf.best_params_
    # wine_OF_params = wine_final_params.copy()
    # wine_OF_params['SVM__alpha'] = 1e-16
    adult_income_final_params =adult_income_clf.best_params_
    adult_income_OF_params = adult_income_final_params.copy()
    adult_income_OF_params['SVM__alpha'] = 1e-16

    # pipeM.set_params(**wine_final_params)
    # makeTimingCurve(wineX,wineY,pipeM,'SVM_RBF','wine')
    pipeA.set_params(**adult_income_final_params)
    makeTimingCurve(adult_income_X,adult_income_Y,pipeM,'SVM_RBF','adult_income')


    # pipeM.set_params(**wine_final_params)
    # iterationLC(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_RBF','wine')
    pipeA.set_params(**adult_income_final_params)
    iterationLC(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','adult_income')

    pipeA.set_params(**adult_income_OF_params)
    iterationLC(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','adult_income')
    # pipeM.set_params(**wine_OF_params)
    # iterationLC(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_RBF_OF','wine')


if __name__ == "__main__":
    main()
