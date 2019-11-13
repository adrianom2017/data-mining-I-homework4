#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''

import numpy as np
import sklearn.datasets


if __name__ == '__main__':

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

    print('')

    print('Exercise 2.c')
    print('------------')

    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)

    print('Accuracy score using cross-validation')
    print('-------------------------------------\n')


    print('')
    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')


    print('')
    print('Feature importances for _reduced_ data set')
    print('------------------------------------------\n')
