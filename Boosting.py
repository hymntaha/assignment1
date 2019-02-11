# -*- coding: utf-8 -*-
import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
from helpers import dtclf_pruned
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)

def main():

    # adult = pd.read_csv('data/adult_parsed.csv')
    # adult['net_capital'] = adult['capital-gain']-adult['capital-loss']
    # adult = adult.drop(["fnlwgt","capital-gain","capital-loss","workclass","native-country"],axis=1)
    #
    # adult['income']=adult['income'].map({'<=50K': 0, '>50K': 1})
    # adult['gender'] = adult['gender'].map({'Male': 0, 'Female': 1}).astype(int)
    # adult['race'] = adult['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3,
    #                                    'Amer-Indian-Eskimo': 4}).astype(int)
    # adult['marital-status'] = adult['marital-status'].map({'Never-married':0,'Widowed':1,'Divorced':2, 'Separated':3,
    #                                                        'Married-spouse-absent':4, 'Married-civ-spouse':5, 'Married-AF-spouse':6})
    # adult['education'] = adult['education'].map({'Preschool':0,'1st-4th':1,'5th-6th':2, '7th-8th':3,
    #                                              '9th':4, '10th':5, '11th':6, '12th':7, 'Prof-school':8,
    #                                              'HS-grad':9, 'Some-college':10, 'Assoc-voc':11, 'Assoc-acdm':12,
    #                                              'Bachelors':13, 'Masters':14, 'Doctorate':15})
    #
    # adult['occupation'] = adult['occupation'].map({'Priv-house-serv':0,'Protective-serv':1,'Handlers-cleaners':2, 'Machine-op-inspct':3,
    #                                                'Adm-clerical':4, 'Farming-fishing':5, 'Transport-moving':6, 'Craft-repair':7, 'Other-service':8,
    #                                                'Tech-support':9, 'Sales':10, 'Exec-managerial':11, 'Prof-specialty':12, 'Armed-Forces':13 })
    #
    # adult['relationship'] = adult['relationship'].map({'Unmarried':0,'Other-relative':1, 'Not-in-family':2,
    #                                                    'Wife':3, 'Husband':4,'Own-child':5})
    #
    # adult = pd.get_dummies(adult)
    # adult_income_X = adult.drop('income',1).copy().values
    # adult_income_Y = adult['income'].copy().values

    wine_data = pd.read_csv('data/winequality_white.csv')
    wine_data['category'] = wine_data['quality'] >= 7

    wineX = wine_data[wine_data.columns[0:11]].values
    wineY = wine_data['category'].values.astype(np.int)

    alphas = np.append(np.arange(0.001, 0.05, 0.001), 0)



    # adult_income_trgX, adult_income_tstX, adult_income_trgY, adult_income_tstY = ms.train_test_split(adult_income_X, adult_income_Y, test_size=0.3, random_state=0,stratify=adult_income_Y)
    wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)


    # adult_income_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
    wine_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)

    OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
    #paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
    paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
              'Boost__base_estimator__alpha':alphas}
    #paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
    #           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

    paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
               'Boost__base_estimator__alpha':alphas}


    # adult_income_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_income_base,random_state=55)
    wine_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=wine_base,random_state=55)
    OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                     ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                     ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                     ('Boost',wine_booster)])

    pipeA = Pipeline([('Scale',StandardScaler()),
                     ('Boost',wine_booster)])

    #
    # adult_income_clf = basicResults(pipeM,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,paramsM,'Boost','adult_income')
    wine_clf = basicResults(pipeA,wine_trgX,wine_trgY,wine_tstX,wine_tstY,paramsA,'Boost','wine')

    #
    #

    # adult_income_final_params = adult_income_clf.best_params_
    wine_final_params = wine_clf.best_params_
    OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

    ##
    # pipeM.set_params(**adult_income_final_params)
    pipeA.set_params(**wine_final_params)
    # makeTimingCurve(adult_income_X,adult_income_Y,pipeM,'Boost','adult_income')
    makeTimingCurve(wineX,wineY,pipeA,'Boost','wine')

    # pipeM.set_params(**adult_income_final_params)
    # iterationLC(pipeM,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','adult_income')
    pipeM.set_params(**wine_final_params)
    iterationLC(pipeA,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','wine')
    # pipeM.set_params(**OF_params)
    # iterationLC(pipeM,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','adult_income')
    pipeA.set_params(**OF_params)
    iterationLC(pipeA,wine_trgX,wine_trgY,wine_tstX,wine_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','wine')

if __name__ == "__main__":
    main()
