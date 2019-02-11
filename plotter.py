import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def drawTimingCurve(timingData):
    plt.plot(timingData['Unnamed: 0'], timingData['test'], 'go-', label = 'Test')
    plt.plot(timingData['Unnamed: 0'], timingData['train'], 'bo-', label = 'Train')

    plt.legend()
    plt.xlabel('Amount of Data Used to Train')
    plt.ylabel('Execution Time (seconds)')

    plt.show()

plt.figure(figsize = (40, 20))
plt.subplot(221)

timingData = pd.read_csv('reports/output/DT_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'Decision Tree')

timingData = pd.read_csv('reports/output/ANN_adult_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'ANN')

timingData = pd.read_csv('reports/output/KNN_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'KNN')

timingData = pd.read_csv('reports/output/Boost_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'Boost')
plt.grid(linestyle='dotted')
plt.title("Testing Times - Adult Income Choices")
plt.ylabel("Execution Time (in seconds)")
plt.xlabel("Fraction of Data Used to Test")
plt.legend()

plt.subplot(222)

timingData = pd.read_csv('reports/output/DT_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'Decision Tree')

timingData = pd.read_csv('reports/output/ANN_adult_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'ANN')

timingData = pd.read_csv('reports/output/KNN_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'KNN')

timingData = pd.read_csv('reports/output/Boost_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'Boost')
plt.grid(linestyle='dotted')
plt.title("Training Times - Adult Income Choices")
plt.ylabel("Execution Time (in seconds)")
plt.xlabel("Fraction of Data Used to Test")
plt.legend()

plt.subplot(223)
timingData = pd.read_csv('reports/output/SVM_RBF_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'SVM')
plt.grid(linestyle='dotted')
plt.title("Testing Times - Adult Income Choices")
plt.ylabel("Execution Time (in seconds)")
plt.xlabel("Fraction of Data Used to Test")
plt.legend()

plt.subplot(224)
timingData = pd.read_csv('reports/output/SVM_RBF_adult_income_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'SVM')
plt.grid(linestyle='dotted')
plt.title("Training Times - Adult Income Choices")
plt.ylabel("Execution Time (in seconds)")
plt.xlabel("Fraction of Data Used to Test")
plt.legend()

plt.savefig('reports/figures/Timing_curves_adult_income.png')

plt.figure(figsize = (40, 10))
plt.subplot(121)

timingData = pd.read_csv('reports/output/DT_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'Decision Tree')

timingData = pd.read_csv('reports/output/ANN_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'ANN')

timingData = pd.read_csv('reports/output/KNN_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'KNN')

timingData = pd.read_csv('reports/output/Boost_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'Boost')

timingData = pd.read_csv('reports/output/SVM_RBF_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['test'], label = 'SVM')

plt.title("Testing Times")
plt.ylabel("Execution Time (in seconds)")
plt.xlabel("Fraction of Data Used to Test")
plt.legend()

plt.subplot(122)

timingData = pd.read_csv('reports/output/DT_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'Decision Tree')

timingData = pd.read_csv('reports/output/ANN_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'ANN')

timingData = pd.read_csv('reports/output/KNN_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'KNN')

timingData = pd.read_csv('reports/output/Boost_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'Boost')

timingData = pd.read_csv('reports/output/SVM_RBF_wine_timing.csv')
plt.plot(timingData['Unnamed: 0'], timingData['train'], label = 'SVM')

plt.title("Training Times")
plt.ylabel("Execution Time (in seconds)")
plt.xlabel("Fraction of Data Used to Test")
plt.legend()

def plotLC(train, test, name = None):
    test_means = test.drop('Unnamed: 0', axis = 1).mean(axis = 1)
    train_means = train.drop('Unnamed: 0', axis = 1).mean(axis = 1)
    plt.grid(linestyle='dotted')
    plt.plot(train['Unnamed: 0'], train_means, label = 'Training', marker='o')
    plt.plot(test['Unnamed: 0'], test_means, label = 'Prediction', marker='o')

    plt.legend()
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.title('{} Best Parameters Learning Curve'.format(name))

def plotIterLC(iter_data):
    plt.plot(iter_data.param_SVM__n_iter, iter_data.mean_train_score)
    plt.plot(iter_data.param_SVM__n_iter, iter_data.mean_test_score)

ann_params = pd.read_csv('reports/output/ann_adult_reg.csv')

ann_layers = ann_params.param_MLP__hidden_layer_sizes.unique()
ann_activations = ann_params.param_MLP__activation.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']

plt.figure(figsize = (40, 10))
plt.subplot(121)

for act in ann_activations:
    line_ind += 1
    color_ind = -1
    for layer in ann_layers:
        color_ind += 1
        plt.plot(ann_params[(ann_params.param_MLP__hidden_layer_sizes == layer) & (ann_params.param_MLP__activation == act)].param_MLP__alpha,
                 ann_params[(ann_params.param_MLP__hidden_layer_sizes == layer) & (ann_params.param_MLP__activation == act)].mean_test_score,
                 label = layer + ' - '+ act,
                 marker='o',
                 linestyle = linetypes[line_ind],
                 color = colors[color_ind])

#line_ind = -1
line_ind = 1
plt.grid(linestyle='dotted')
plt.legend(bbox_to_anchor = [1, 1], title = 'Parameter Combos')

plt.xlabel('Alpha Value')
plt.ylabel('CV Accuracy')
plt.title('ANN - Adult Income Choice - Parameter Selection')

plt.subplot(122)

ann_iter = pd.read_csv('reports/output/ITER_base_ANN_adult.csv')

# Since the accuracies don't change after ~200 we'll limit the chart
#ann_iter = ann_iter[ann_iter.param_MLP__max_iter <= 500]

plt.plot(ann_iter.param_MLP__max_iter, ann_iter.mean_train_score,marker='o')
plt.plot(ann_iter.param_MLP__max_iter, ann_iter.mean_test_score,marker='o')
plt.grid(linestyle='dotted')
plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('ANN - Adult Income Choice - Best Parameters Learning Curve')

plt.savefig('reports/figures/ann_adult_income_figures.png')

# Moving on to KNN

knn_params = pd.read_csv('reports/output/KNN_adult_income_reg.csv')

knn_metric = knn_params.param_KNN__metric.unique()
knn_weights = knn_params.param_KNN__weights.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
plt.grid(linestyle='dotted')
plt.figure(figsize = (40, 10))
plt.subplot(121)

for weight in knn_weights:
    line_ind += 1
    color_ind = -1
    for metric in knn_metric:
        color_ind += 1

        plt.plot(knn_params[(knn_params.param_KNN__metric == metric) & (knn_params.param_KNN__weights == weight)].param_KNN__n_neighbors,
                 knn_params[(knn_params.param_KNN__metric == metric) & (knn_params.param_KNN__weights == weight)].mean_test_score,
                 label = weight + ' - '+ metric,
                 marker='o',
                 linestyle = linetypes[line_ind],
                 color = colors[color_ind])

#plt.legend(bbox_to_anchor = [1, 1])
plt.legend(title = 'Parameter Combos')
plt.xlabel('Number of Neighbors')
plt.ylabel('CV Accuracy')
plt.title('KNN - Adult Income Choice - Parameter Selection')

plt.subplot(122)

knn_test = pd.read_csv('reports/output/knn_adult_income_LC_test.csv')
knn_train = pd.read_csv('reports/output/knn_adult_income_LC_train.csv')

plotLC(knn_train, knn_test, 'KNN - Adult Income Choice - ')

plt.savefig('reports/figures/KNN_adult_income_figures.png')

svm_params = pd.read_csv('reports/output/SVM_RBF_adult_income_reg.csv')
svm_params.param_SVM__alpha = svm_params.param_SVM__alpha.astype(str)
svm_params.sort_values(by = ['param_SVM__alpha', 'param_SVM__gamma_frac'], inplace=True)

svm_gamma = svm_params.param_SVM__gamma_frac.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
color_ind = -1

plt.figure(figsize = (40, 10))
plt.subplot(121)
plt.grid(linestyle='dotted')
for gamma in svm_gamma:
    color_ind += 1
    plt.plot(#np.log(svm_params[(svm_params.param_SVM__gamma_frac == gamma)].param_SVM__alpha),
        svm_params[(svm_params.param_SVM__gamma_frac == gamma)].param_SVM__alpha.astype(str),
        svm_params[(svm_params.param_SVM__gamma_frac == gamma)].mean_test_score,
        label = gamma,marker='o')

plt.xticks(rotation = 90)
#axis_ticks = svm_params.param_SVM__alpha.unique()
#plt.semilogx(axis_ticks, np.exp(axis_ticks))
#plt.xscale('exp')
plt.legend(title = 'Gamma')
plt.grid(linestyle='dotted')
plt.xlabel('Alpha')
plt.ylabel('CV Accuracy')
plt.title('SVM - Adult Income Choice - Parameter Selection')

plt.subplot(122)

svm_iter = pd.read_csv('reports/output/ITER_base_SVM_RBF_adult_income.csv')

#svm_iter = svm_iter[svm_iter.param_MLP__max_iter <= 500]

plt.plot(svm_iter.param_SVM__n_iter, svm_iter.mean_train_score,marker='o')
plt.plot(svm_iter.param_SVM__n_iter, svm_iter.mean_test_score,marker='o')
plt.grid(linestyle='dotted')
plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('SVM - Adult Income Choice - Best Parameters Learning Curve')

plt.savefig('reports/figures/SVM_adult_income_figures.png')

dt_params = pd.read_csv('reports/output/DT_adult_income_reg.csv')
dt_params.sort_values(by = 'param_DT__alpha', inplace=True)
dt_params = dt_params[dt_params.param_DT__alpha > -1]

dt_criterion = dt_params.param_DT__criterion.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
color_ind = -1

plt.figure(figsize = (40, 10))
plt.subplot(121)
plt.grid(linestyle='dotted')
for criterion in dt_criterion:
    color_ind += 1
    plt.plot(dt_params[(dt_params.param_DT__criterion == criterion)].param_DT__alpha,
             dt_params[(dt_params.param_DT__criterion == criterion)].mean_test_score,
             marker='o',
             label = criterion,
             color = colors[color_ind])

plt.legend(title = 'Splitting Criterion')

plt.xlabel('Alpha')
plt.ylabel('CV Accuracy')
plt.title('Decision Tree - Adult Income Choice - Parameter Selection')

plt.subplot(122)

DT_test = pd.read_csv('reports/output/DT_adult_income_LC_test.csv')
DT_train = pd.read_csv('reports/output/DT_adult_income_LC_train.csv')

plotLC(DT_train, DT_test, 'Decision Tree - Adult Income Choice - ')

plt.savefig('reports/figures/DT_adult_income_figures.png')

boost_params = pd.read_csv('reports/output/Boost_adult_income_reg.csv')
boost_params.sort_values(by = ['param_Boost__base_estimator__alpha', 'param_Boost__n_estimators'], inplace=True)
boost_params = boost_params[boost_params.param_Boost__base_estimator__alpha > -1]

boost_estimator = boost_params.param_Boost__n_estimators.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
color_ind = -1
plt.grid(linestyle='dotted')
plt.figure(figsize = (40, 10))
plt.subplot(121)

for estimator in boost_estimator:
    color_ind += 1
    plt.plot(boost_params[(boost_params.param_Boost__n_estimators == estimator)].param_Boost__base_estimator__alpha,
             boost_params[(boost_params.param_Boost__n_estimators == estimator)].mean_test_score,
             label = estimator,marker='o')
plt.grid(linestyle='dotted')
plt.legend(title = 'Maximum Estimators')

plt.xlabel('Alpha')
plt.ylabel('CV Accuracy')
plt.title('Boosting - Adult Income Choice - Parameter Selection')

plt.subplot(122)

boost_iter = pd.read_csv('reports/output/ITER_base_Boost_adult_income.csv')

plt.plot(boost_iter.param_Boost__n_estimators, boost_iter.mean_train_score,marker='o')
plt.plot(boost_iter.param_Boost__n_estimators, boost_iter.mean_test_score,marker='o',)
plt.grid(linestyle='dotted')
plt.legend()
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Boosting - Adult Income Choice - Best Parameters Learning Curve')

plt.savefig('reports/figures/Boost_adult_income_figures.png')


# Let's work on ann parameter selection plots

ann_params = pd.read_csv('reports/output/ann_wine_reg.csv')
#print(ann_params.head())

ann_layers = ann_params.param_MLP__hidden_layer_sizes.unique()
ann_activations = ann_params.param_MLP__activation.unique()
#print(len(ann_layers))
#print(len(ann_activations))

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']

plt.figure(figsize = (40, 10))
plt.subplot(121)

for act in ann_activations:
    line_ind += 1
    color_ind = -1
    for layer in ann_layers:
        color_ind += 1
        plt.grid(linestyle='dotted')
        plt.plot(ann_params[(ann_params.param_MLP__hidden_layer_sizes == layer) & (ann_params.param_MLP__activation == act)].param_MLP__alpha,
                 ann_params[(ann_params.param_MLP__hidden_layer_sizes == layer) & (ann_params.param_MLP__activation == act)].mean_test_score,
                 label = layer + ' - '+ act,
                 marker='o',
                 linestyle = linetypes[line_ind],
                 color = colors[color_ind])

#line_ind = -1
line_ind = 1

plt.legend(bbox_to_anchor = [1, 1], title = 'Parameter Combos')

#for act in ann_activations:
#    #line_ind += 1
#    #color_ind = -1
#    for layer in ann_layers:
#        color_ind += 1
#        plt.plot(ann_params[(ann_params.param_MLP__hidden_layer_sizes == layer) & (ann_params.param_MLP__activation == act)].param_MLP__alpha,
#                 ann_params[(ann_params.param_MLP__hidden_layer_sizes == layer) & (ann_params.param_MLP__activation == act)].mean_train_score,
#                 label = layer + ' - '+ act,
#                 linestyle = linetypes[line_ind])#,
#                 #color = colors[color_ind])

#plt.legend(bbox_to_anchor = [1, 1])
plt.xlabel('Alpha Value')
plt.ylabel('CV Accuracy')
plt.title('ANN - Wine Quality - Parameter Selection')
#plt.show()
##plt.plot(ann_params.param_MLP__alpha, ann_params.mean_train_score)


plt.subplot(122)

ann_iter = pd.read_csv('reports/output/ITER_base_ANN_wine.csv')

# Since the accuracies don't change after ~200 we'll limit the chart
ann_iter = ann_iter[ann_iter.param_MLP__max_iter <= 500]
plt.grid(linestyle='dotted')
plt.plot(ann_iter.param_MLP__max_iter, ann_iter.mean_train_score,marker='o')
plt.plot(ann_iter.param_MLP__max_iter, ann_iter.mean_test_score,marker='o')

plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('ANN - Wine Quality - Best Parameters Learning Curve')

plt.savefig('reports/figures/ann_wine_figures.png')

# Moving on to KNN

knn_params = pd.read_csv('reports/output/KNN_wine_reg.csv')

knn_metric = knn_params.param_KNN__metric.unique()
knn_weights = knn_params.param_KNN__weights.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']

plt.figure(figsize = (40, 10))
plt.subplot(121)
plt.grid(linestyle='dotted')
for weight in knn_weights:
    line_ind += 1
    color_ind = -1
    for metric in knn_metric:
        color_ind += 1
        plt.grid(linestyle='dotted')
        plt.plot(knn_params[(knn_params.param_KNN__metric == metric) & (knn_params.param_KNN__weights == weight)].param_KNN__n_neighbors,
                 knn_params[(knn_params.param_KNN__metric == metric) & (knn_params.param_KNN__weights == weight)].mean_test_score,
                 label = weight + ' - '+ metric,
                 marker='o',
                 linestyle = linetypes[line_ind],
                 color = colors[color_ind])

#plt.legend(bbox_to_anchor = [1, 1])
plt.legend(title = 'Parameter Combos')
plt.grid(linestyle='dotted')
plt.xlabel('Number of Neighbors')
plt.ylabel('CV Accuracy')
plt.title('KNN - Wine Quality - Parameter Selection')

plt.subplot(122)
plt.grid(linestyle='dotted')
knn_test = pd.read_csv('reports/output/knn_wine_LC_test.csv')
knn_train = pd.read_csv('reports/output/knn_wine_LC_train.csv')

plotLC(knn_train, knn_test, 'KNN - Wine Quality - ')

plt.savefig('reports/figures/KNN_wine_figures.png')

svm_params = pd.read_csv('reports/output/SVM_RBF_wine_reg.csv')
svm_params.param_SVM__alpha = svm_params.param_SVM__alpha.astype(str)
svm_params.sort_values(by = ['param_SVM__alpha', 'param_SVM__gamma_frac'], inplace=True)

svm_gamma = svm_params.param_SVM__gamma_frac.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
color_ind = -1

plt.figure(figsize = (40, 10))
plt.subplot(121)
plt.grid(linestyle='dotted')
for gamma in svm_gamma:
    color_ind += 1
    plt.plot(svm_params[(svm_params.param_SVM__gamma_frac == gamma)].param_SVM__alpha,
             svm_params[(svm_params.param_SVM__gamma_frac == gamma)].mean_test_score,
             label = gamma,marker='o')

plt.xticks(rotation = 90)
#axis_ticks = svm_params.param_SVM__alpha.unique()
#plt.semilogx(axis_ticks, np.exp(axis_ticks))
#plt.xscale('exp')
plt.legend(title = 'Gamma')
plt.grid(linestyle='dotted')
plt.xlabel('Alpha')
plt.ylabel('CV Accuracy')
plt.title('SVM - Wine Quality - Parameter Selection')

plt.subplot(122)

svm_iter = pd.read_csv('reports/output/ITER_base_SVM_RBF_wine.csv')

#svm_iter = svm_iter[svm_iter.param_MLP__max_iter <= 500]

plt.plot(svm_iter.param_SVM__n_iter, svm_iter.mean_train_score,marker='o',)
plt.plot(svm_iter.param_SVM__n_iter, svm_iter.mean_test_score,marker='o',)
plt.grid(linestyle='dotted')
plt.legend()
plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('SVM - Wine Quality - Best Parameters Learning Curve')

plt.savefig('reports/figures/SVM_wine_figures.png')

dt_params = pd.read_csv('reports/output/DT_wine_reg.csv')
dt_params.sort_values(by = 'param_DT__alpha', inplace=True)
dt_params = dt_params[dt_params.param_DT__alpha > -1]

dt_criterion = dt_params.param_DT__criterion.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
color_ind = -1

plt.figure(figsize = (40, 10))
plt.subplot(121)

for criterion in dt_criterion:
    color_ind += 1
    plt.plot(dt_params[(dt_params.param_DT__criterion == criterion)].param_DT__alpha,
             dt_params[(dt_params.param_DT__criterion == criterion)].mean_test_score,
             label = criterion,
             marker='o',
             color = colors[color_ind])

plt.legend(title = 'Splitting Criterion')

plt.xlabel('Alpha')
plt.ylabel('CV Accuracy')
plt.title('Decision Tree - Wine Quality - Parameter Selection')

plt.subplot(122)

DT_test = pd.read_csv('reports/output/DT_wine_LC_test.csv')
DT_train = pd.read_csv('reports/output/DT_wine_LC_train.csv')

plotLC(DT_train, DT_test, 'Decision Tree - Wine Quality - ')

plt.savefig('reports/figures/DT_wine_figures.png')

boost_params = pd.read_csv('reports/output/Boost_wine_reg.csv')
boost_params.sort_values(by = ['param_Boost__base_estimator__alpha', 'param_Boost__n_estimators'], inplace=True)
boost_params = boost_params[boost_params.param_Boost__base_estimator__alpha > -1]

boost_estimator = boost_params.param_Boost__n_estimators.unique()

linetypes = ['-', '--']
line_ind = -1

colors = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'gray', 'cyan', 'magenta']
color_ind = -1
plt.grid(linestyle='dotted')
plt.figure(figsize = (40, 10))
plt.subplot(121)

for estimator in boost_estimator:
    color_ind += 1
    plt.plot(boost_params[(boost_params.param_Boost__n_estimators == estimator)].param_Boost__base_estimator__alpha,
             boost_params[(boost_params.param_Boost__n_estimators == estimator)].mean_test_score,
             label = estimator,marker='o')

plt.legend(title = 'Maximum Estimators')
plt.grid(linestyle='dotted')
plt.xlabel('Alpha')
plt.ylabel('CV Accuracy')
plt.title('Boosting - Wine Quality - Parameter Selection')

plt.subplot(122)

boost_iter = pd.read_csv('reports/output/ITER_base_Boost_wine.csv')

plt.plot(boost_iter.param_Boost__n_estimators, boost_iter.mean_train_score,marker='o')
plt.plot(boost_iter.param_Boost__n_estimators, boost_iter.mean_test_score,marker='o')
plt.grid(linestyle='dotted')
plt.legend()
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Boosting - Wine Quality - Best Parameters Learning Curve')

plt.savefig('reports/figures/Boost_wine_figures.png')

ANN_test = pd.read_csv('reports/output/ANN_wine_LC_test.csv')
ANN_train = pd.read_csv('reports/output/ANN_wine_LC_train.csv')

plt.figure(figsize = (40, 10))
plt.grid(linestyle='dotted')
plt.subplot(121)
plotLC(ANN_train, ANN_test, 'ANN - Wine Quality - ')
plt.legend()

ANN_test = pd.read_csv('reports/output/ANN_adult_LC_test.csv')
ANN_train = pd.read_csv('reports/output/ANN_adult_LC_train.csv')
plt.grid(linestyle='dotted')
plt.subplot(122)

plotLC(ANN_train, ANN_test, 'ANN - Adult Income Choice - ')
plt.legend()

plt.savefig('reports/figures/ANN_Curves_by_Training_Samples')

# nodeData = pd.read_csv('reports/output/DT_adult_income_nodecounts.csv')
# plt.plot(nodeData['alpha'], nodeData['nodes'] ,marker='o', color='green' )
# plt.grid(linestyle='dotted')
# plt.title('Weighted accuracy of post-pruning')
# plt.xlabel('Alpha')
# plt.ylabel('Nodes Count')
# plt.savefig('reports/figures/DT_adult_nodecount.png')

# nodeData = pd.read_csv('reports/output/DT_wine_nodecounts.csv')
# plt.plot(nodeData['alpha'], nodeData['nodes'] ,marker='o', color='green' )
# plt.grid(linestyle='dotted')
# plt.title('Weighted accuracy of post-pruning')
# plt.xlabel('Alpha')
# plt.ylabel('Nodes Count')
# plt.savefig('reports/figures/DT_wine_nodecount.png')
