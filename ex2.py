#!/usr/bin/env python3
#%%
'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
import matplotlib.pyplot as plt


def compute_information_gain(X, y, attribute_index, theta):
    return compute_information_content(y) - compute_information_a(X, y, attribute_index, theta)

def split_data(X, y, attribute_index, theta):
    idx = X[:,attribute_index] >= theta
    return X[idx,:], y[idx], X[~idx,:], y[~idx]

def compute_information_content(y):

    p = [np.sum(y == i) / len(y) for i in set(y)]
    return -np.sum(p*np.log2(p))

def compute_information_a(X, y, attribute_index, theta):
    X1, y1, X2, y2 = split_data(X, y, attribute_index, theta)
    y_ = [y1, y2]
    X_ = [X1, X2]

    s = 0
    for i in range(2):
        s += len(X_[i]) / len(X) *compute_information_content(y_[i])
    
    return s

#%%
#if __name__ == '__main__':
if True:

    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))

    ####################################################################
    # Your code goes here.
    ####################################################################

    print('Exercise 2.b')
    print('------------')

    splits = [(0, 5.5), (1, 3.0), (2, 2.0), (3, 1.0)]
    gains = [compute_information_gain(X, y, i[0], i[1]) for i in splits]
    z = list(zip(feature_names, [i[1] for i in splits]))
    for idx, value in enumerate(z):
        print('Split (',value[0],'<',value[1], '): Information gain =', np.round(gains[idx], 2))
    print('\n')
    print('Exercise 2.c')
    print('------------')
    print('Based on those gains I would select either (petal length < 2.0), or (petal width < 1.0) because for those the gain is maximal.')

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)
    print('\n')
    print('Exercise 2.d')
    print('------------\n')
    print('Accuracy score using cross-validation')
    print('-------------------------------------\n')

    kf = KFold(n_splits=5, shuffle=True)
    clf = DecisionTreeClassifier()
    
    #CV maually 
    score = []
    feature_importance = np.full((len(feature_names),),-1)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score.append(accuracy_score(y_test,y_pred))
        feature_importance = np.vstack((feature_importance,clf.feature_importances_))
        
    feature_importance_mean =  np.mean(feature_importance[1:,:], axis=0)
    
    #sklearn library
    cv_score = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
    print('The mean accuracy is:', np.mean(score).round(2), '(manual)', np.mean(score).round(2), '(sklearn lib)')


    print('')
    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')

    print('-petal length with mean importance of', feature_importance_mean[2].round(2))
    print('-petal width with mean importance of', feature_importance_mean[3].round(2))


    print('')
    print('Feature importances for _reduced_ data set')
    print('------------------------------------------\n')

    X = X[y != 2]
    y = y[y != 2]

    #CV maually 
    score = []
    feature_importance = np.full((len(feature_names),),-1)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score.append(accuracy_score(y_test,y_pred))
        feature_importance = np.vstack((feature_importance,clf.feature_importances_))
        
    feature_importance_mean =  np.mean(feature_importance[1:,:], axis=0)

    print('-petal width with mean importance of', feature_importance_mean[3].round(2))
    print('This means the data set only consits of two classes that can be separeted only based on petal width.')

#%%
