# -*- coding: utf-8 -*-
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix


X = pickle.load(open("X.pkl","rb"))
y = pickle.load(open("y.pkl","rb"))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 17, stratify = y)

lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions_train = lr.predict(X_train)
accuracy_score(y_train, predictions_train)

predictions_test = lr.predict(X_test)
accuracy_score(y_test, predictions_test)

cm = confusion_matrix(y_test, predictions_test)



###Oversample!!!
import imblearn.over_sampling

# randomly oversample by telling it the number of samples to have in each class
ROS = imblearn.over_sampling.RandomOverSampler(\
                                               ratio={0:70056,1:18069*3}, \
                                               random_state=42) 
    
X_train_oversample, y_train_oversample = ROS.fit_sample(X_train, y_train)



###GridSearch
solver = ['liblinear', 'newton-cg']

C = [int(x) for x in np.linspace(1800, 2100, num = 3)]

penalty = ['l2']

class_weight = [{1:1.075, 0:1}, {1:1.07, 0:1}]

max_iter = [350]

random_grid = {'solver': solver,
               'C': C,
               'penalty': penalty,
               'class_weight' : class_weight,
               'max_iter': max_iter}

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations, and use all available cores
lr_random = GridSearchCV(estimator = LogisticRegression(), scoring = 'roc_auc', verbose = 2, param_grid = random_grid, cv = 2, n_jobs = -1)

# Fit the random search model
lr_random.fit(X_train, y_train)

lr_random.best_score_
lr_random.best_params_
lr_random.best_estimator_


lr = LogisticRegression(C = 2100, class_weight = {1:1.07, 0:1}, penalty = 'l2', solver = 'liblinear')
lr.fit(X_train_oversample, y_train_oversample)
predictions_train = lr.predict(X_train)
roc_auc_score(y_train, predictions_train)

predictions_test = lr.predict(X_test)
roc_auc_score(y_test, predictions_test)

cm = confusion_matrix(y_test, predictions_test)
import itertools

plt.figure(dpi=150)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xticks([0,1],["Liberal", "Conservative"])
plt.yticks([0,1],["Liberal", "Conservative"])
plt.title("Journal Outlets Tweet Prediction", y = 1.05)
plt.ylabel("Reality", rotation = 0)
plt.xlabel("Prediction")
fmt = ',d'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.savefig("Predictions.jpg")