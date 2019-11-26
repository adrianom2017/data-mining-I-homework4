#%%
'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('Exercise 1.a')
    print('------------')
    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))
    #print('Balanced Accuracy: {0:.3f}'.format(balanced_accuracy_score(y_true, y_pred)))
    

#%%

if __name__ == "__main__":

    ###################################################################
    # Your code goes here.
    ###################################################################

    #Read in files
    train = pd.read_csv('./data/diabetes_train.csv')
    X_train = train.drop(columns = 'type').values
    y_train = train['type'].values

    test = pd.read_csv('./data/diabetes_test.csv')
    X_test = test.drop(columns = 'type').values
    y_test = test['type'].values

    #Scaler
    scaler = StandardScaler()

    #Classifier
    clf = LogisticRegression(C = 1)

    #Fit classifeir
    XX_train = scaler.fit_transform(X_train)
    clf.fit(XX_train, y_train)

    #Predict
    XX_test = scaler.transform(X_test)
    y_pred = clf.predict(XX_test)

    #Evaluate predictions
    compute_metrics(y_test, y_pred)

    print('\n\n')
    print('Exercise 1.b')
    print('------------\n')
    print('I would prefere LDA because this method detects more TP and misses less positive (FN) which allows to treat more patiants that actually have the disease - even though the method is worse in terms of accuracy (bad measure for unbalanced data).\n\n')

    print('Exercise 1.c')
    print('------------\n')
    print('I would still choose LDA since I do not have any additional information to conclude that LDA performs worse compared to logisitc regression on an other data set (in terms of detection of relevant patients).\n\n')

    print('Exercise 1.d')
    print('------------\n')
    #print('The coefficiants are\n', clf.coef_)
    print('The features contributing the moste are glu and ped.')
    print('The coefficient for npreg is {:.2f}. Calculating the exponential function results in {:.2f}, which amounts to an {} in diabetes risk of {:.2f} percent per additional pregnancy.'.format(clf.coef_[0][0], np.exp(clf.coef_[0][0]), 'increase', (np.exp(clf.coef_[0][0]/np.std(train['npreg']))-1)*100))
    print('\n\n')

#%%