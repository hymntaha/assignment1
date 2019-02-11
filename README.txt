Author: Taha UYGUN(tuygun3) with code adapted from Jonathan Tay
Date: 2/5/2019

This repo locates in https://github.com/hymntaha/assignment1

The assignment code is written in Python 3.6.1. Library dependencies are:
scikit-learn 0.18.1
numpy 1.12.1
pandas 0.20.1
matplotlib 2.0.2

## Example
Sample execution of any algorithm:
python3 KNN.py

This will parse both datasets and execute specified algorithm for both datasets. When the execution is completed results will appear in reports/output.
Once you run all the algorithms and get all the results for different algorithms, you can run:
python3 plotter.py

This will run plotter and you will see all the identical plots in the report. If you use pycharm as a default IDE, you can also right click on an algorithm and click 'Run'
to execute specific algorithm.

The main folder contains the following files:
1. data/ holds adult income and wine quality datasets
3. readme.txt -> This file
4. reports directory holds 2 subdirectories figures and output. Output holds results as a csv file. Figure gets all the plots after running plotter.py
5. helpers.py -> Essential methods for post pruning, iterations, accuracy, results, performance( time )
6. plotter.py -> experimental plotting file
7. ANN.py -> Code for the Neural Network Experiments
8. Boosting.py -> Code for the Boosted Tree experiments
9. Decision_Tree.py -> Code for the Decision Tree experiments
10. KNN.py -> Code for the K-nearest Neighbours experiments
11. SVM.py -> Code for the Support Vector Machine (SVM) experiments


There is also a reports/output. This folder contains the results.
Here, I use DT/ANN/BT/KNN/SVM_Lin/SVM_RBF to refer to decision trees, artificial neural networks, boosted trees, K-nearest neighbours, linear and RBF kernel SVMs respectively.
There files in the output folder. They come the following types:
1. <Algorithm>_<dataset>_reg.csv -> The validation curve tests for <algorithm> on <dataset>
2. <Algorithn>_<dataset>_LC_train.scv -> Table of # of examples vs. CV training accuracy (for 5 folds) for <algorithm> on <dataset>. Used for learning curves.
3. <Algorithn>_<dataset>_LC_test.csv -> Table of # of examples vs. CV testing accuracy (for 5 folds) for <algorithm> on <dataset>. Used for learning curves.
4. <Algorithm>_<dataset>_timing.csv -> Table of fraction of training set vs. training and evaluation times. If the fulll training set is of size T and a fraction f are used for training, then the evaluation set is of size (T-fT)= (1-f)T
5. ITER_base_<Algorithm>_<dataset>.csv -> Table of results for learning curves based on number of iterations/epochs.
6. ITERtestSET_<Algorithm>_<dataset>.csv -> Table showing training and test set accuracy as number of iterations/epochs is varied. NOT USED in report.
7. "test results.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set.
##########################################################################################################################################################
https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229
Adult Income Dataset: https://archive.ics.uci.edu/ml/datasets/adult
White Wine Quality Dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality
Code inheritance: https://github.com/JonathanTay/CS-7641-assignment-1
Alpha pruning: https://stats.stackexchange.com/questions/193538/how-to-choose-alpha-in-cost-complexity-pruning
