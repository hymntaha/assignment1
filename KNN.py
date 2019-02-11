# -*- coding: utf-8 -*-


import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)



def main():
    adult = pd.read_csv('data/adult_parsed.csv')
    # plt.figure(figsize=(15,12))
    # cor_map = adult.corr()
    # sns.heatmap(cor_map, annot=True, fmt='.3f', cmap='YlGnBu')
    # plt.show()

    adult['net_capital'] = adult['capital-gain']-adult['capital-loss']
    adult = adult.drop(["fnlwgt","capital-gain","capital-loss","workclass"],axis=1)

    adult['income']=adult['income'].map({'<=50K': 0, '>50K': 1})
    adult['gender'] = adult['gender'].map({'Male': 0, 'Female': 1}).astype(int)
    adult['race'] = adult['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3,
                                       'Amer-Indian-Eskimo': 4}).astype(int)
    adult['marital-status'] = adult['marital-status'].map({'Never-married':0,'Widowed':1,'Divorced':2, 'Separated':3,
                                                           'Married-spouse-absent':4, 'Married-civ-spouse':5, 'Married-AF-spouse':6})
    adult['education'] = adult['education'].map({'Preschool':0,'1st-4th':1,'5th-6th':2, '7th-8th':3,
                                                 '9th':4, '10th':5, '11th':6, '12th':7, 'Prof-school':8,
                                                 'HS-grad':9, 'Some-college':10, 'Assoc-voc':11, 'Assoc-acdm':12,
                                                 'Bachelors':13, 'Masters':14, 'Doctorate':15})

    adult['occupation'] = adult['occupation'].map({'Priv-house-serv':0,'Protective-serv':1,'Handlers-cleaners':2, 'Machine-op-inspct':3,
                                                   'Adm-clerical':4, 'Farming-fishing':5, 'Transport-moving':6, 'Craft-repair':7, 'Other-service':8,
                                                   'Tech-support':9, 'Sales':10, 'Exec-managerial':11, 'Prof-specialty':12, 'Armed-Forces':13 })
    adult['native-country'] = adult['native-country'].map({'?':-1,'Puerto-Rico':0,'Haiti':1,'Cuba':2, 'Iran':3,
                                                           'Honduras':4, 'Jamaica':5, 'Vietnam':6, 'Mexico':7, 'Dominican-Republic':8,
                                                           'Laos':9, 'Ecuador':10, 'El-Salvador':11, 'Cambodia':12, 'Columbia':13,
                                                           'Guatemala':14, 'South':15, 'India':16, 'Nicaragua':17, 'Yugoslavia':18,
                                                           'Philippines':19, 'Thailand':20, 'Trinadad&Tobago':21, 'Peru':22, 'Poland':23,
                                                           'China':24, 'Hungary':25, 'Greece':26, 'Taiwan':27, 'Italy':28, 'Portugal':29,
                                                           'France':30, 'Hong':31, 'England':32, 'Scotland':33, 'Ireland':34,
                                                           'Holand-Netherlands':35, 'Canada':36, 'Germany':37, 'Japan':38,
                                                           'Outlying-US(Guam-USVI-etc)':39, 'United-States':40
                                                           })

    adult['relationship'] = adult['relationship'].map({'Unmarried':0,'Other-relative':1, 'Not-in-family':2,
                                                       'Wife':3, 'Husband':4,'Own-child':5})

    adult = pd.get_dummies(adult)
    adult_income_X = adult.drop('income',1).copy().values
    adult_income_Y = adult['income'].copy().values

    # wine_data = pd.read_csv('data/wine-red-white-merge.csv')

    # wineX = wine_data.drop('quality',1).copy().values
    # wineY = wine_data['quality'].copy().values





    adult_income_trgX, adult_income_tstX, adult_income_trgY, adult_income_tstY = ms.train_test_split(adult_income_X, adult_income_Y, test_size=0.3, random_state=0,stratify=adult_income_Y)
    # wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)


    d = adult_income_X.shape[1]
    hiddens_adult_income = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
    alphas = [10**-x for x in np.arange(1,9.01,1/2)]
    # d = wineX.shape[1]
    # hiddens_wine = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]


    pipeM = Pipeline([('Scale',StandardScaler()),
                     ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                     ('KNN',knnC())])

    pipeA = Pipeline([('Scale',StandardScaler()),
                     ('KNN',knnC())])



    params_adult_income= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
    # params_wine= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}


    adult_income_clf = basicResults(pipeA,adult_income_trgX,adult_income_trgY,adult_income_tstX,adult_income_tstY,params_adult_income,'KNN','adult_income')
    # wine_clf = basicResults(pipeM,wine_trgX,wine_trgY,wine_tstX,wine_tstY,params_wine,'KNN','wine')


    # wine_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
    #adult_income_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
    # wine_final_params=wine_clf.best_params_
    adult_income_final_params=adult_income_clf.best_params_



    # pipeM.set_params(**wine_final_params)
    # makeTimingCurve(wineX,wineY,pipeM,'KNN','wine')
    pipeA.set_params(**adult_income_final_params)
    makeTimingCurve(adult_income_X,adult_income_Y,pipeA,'KNN','adult_income')

if __name__ == "__main__":
    main()
