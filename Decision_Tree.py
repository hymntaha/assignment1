# -*- coding: utf-8 -*-
"""

Script for full tests, decision tree (pruned)

"""
import sklearn.model_selection as ms
import numpy as np
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    # rows = []
    # nodes = []
    # for a in alphas[:-1]:
    #     clf.set_params(**{'DT__alpha':a})
    #     clf.fit(trgX,trgY)
    #     node_count = clf.steps[-1][-1].num_nodes()
    #     rows.append([a, node_count])
    #     nodes.append(node_count)
    # out = pd.DataFrame(columns=['alpha', 'nodes'], data=rows)
    # out.to_csv('reports/output/DT_{}_nodecounts.csv'.format(dataset))
    # return
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('reports/output/DT_{}_nodecounts.csv'.format(dataset))

    return


def main():
    # adult = pd.read_csv('data/adult_parsed.csv')
    # plt.figure(figsize=(15,12))
    # cor_map = adult.corr()
    # sns.heatmap(cor_map, annot=True, fmt='.3f', cmap='YlGnBu')
    # plt.show()

    # adult['net_capital'] = adult['capital-gain']-adult['capital-loss']
    # adult = adult.drop(["fnlwgt","capital-gain","capital-loss","workclass"],axis=1)
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
    # adult['native-country'] = adult['native-country'].map({'?':-1,'Puerto-Rico':0,'Haiti':1,'Cuba':2, 'Iran':3,
    #                                                        'Honduras':4, 'Jamaica':5, 'Vietnam':6, 'Mexico':7, 'Dominican-Republic':8,
    #                                                        'Laos':9, 'Ecuador':10, 'El-Salvador':11, 'Cambodia':12, 'Columbia':13,
    #                                                        'Guatemala':14, 'South':15, 'India':16, 'Nicaragua':17, 'Yugoslavia':18,
    #                                                        'Philippines':19, 'Thailand':20, 'Trinadad&Tobago':21, 'Peru':22, 'Poland':23,
    #                                                        'China':24, 'Hungary':25, 'Greece':26, 'Taiwan':27, 'Italy':28, 'Portugal':29,
    #                                                        'France':30, 'Hong':31, 'England':32, 'Scotland':33, 'Ireland':34,
    #                                                        'Holand-Netherlands':35, 'Canada':36, 'Germany':37, 'Japan':38,
    #                                                        'Outlying-US(Guam-USVI-etc)':39, 'United-States':40
    #                                                        })
    #
    # adult['relationship'] = adult['relationship'].map({'Unmarried':0,'Other-relative':1, 'Not-in-family':2,
    #                                                    'Wife':3, 'Husband':4,'Own-child':5})
    #
    # adult = pd.get_dummies(adult)
    # adult_income_X = adult.drop('income',1).copy().values
    # adult_income_Y = adult['income'].copy().values
    #
    #
    #
    #
    #
    # adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adult_income_X, adult_income_Y, test_size=0.3, random_state=0,stratify=adult_income_Y)
    # # alphas = [0.00005, 0.0001, 0.0002,0.00025, 0.0003, 0.0004,0.0005, 0.0006,0.0007, 0.0008, 0.001, 0.0015, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5]
    alphas = np.append(np.arange(0.001, 0.05, 0.001), 0)
    pipeA = Pipeline([('Scale',StandardScaler()),
                      ('DT',dtclf_pruned(random_state=55))])
    #
    params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}
    # adult_income_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params,'DT','adult_income')
    # adult_final_params = adult_income_clf.best_params_
    # pipeA.set_params(**adult_final_params)
    # makeTimingCurve(adult_income_X,adult_income_Y,pipeA,'DT','adult_income')
    # DTpruningVSnodes(pipeA,alphas,adult_trgX,adult_trgY,'adult_income')

    #wine_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
    #adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}

    # Data Parsing for wine quality dataset
    wine_data = pd.read_csv('data/winequality_white.csv')
    wine_data['category'] = wine_data['quality'] >= 7

    wineX = wine_data[wine_data.columns[0:11]].values
    wineY = wine_data['category'].values.astype(np.int)
    # plt.figure(figsize=(12,6))
    # sns.heatmap(wine_data.corr(),annot=True)
    # plt.show()


    wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)
    wine_clf = basicResults(pipeA,wine_trgX,wine_trgY,wine_tstX,wine_tstY,params,'DT','wine')
    wine_final_params = wine_clf.best_params_
    pipeA.set_params(**wine_final_params)
    makeTimingCurve(wineX,wineY,pipeA,'DT','wine')

    DTpruningVSnodes(pipeA,alphas,wine_trgX,wine_trgY,'wine')

if __name__ == "__main__":
    main()
