import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

import numpy as np

# Ensure there are no negative values for MultinomialNB
def remove_negative_values(X):
    return np.maximum(X, 0)

def logistic_test(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'Logistic Regression', acc

def svm_test(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', C=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'SVM', acc

def nb_test(X_train, X_test, y_train, y_test):
    X_train_non_neg = remove_negative_values(X_train)
    X_test_non_neg = remove_negative_values(X_test)
    model = MultinomialNB()
    model.fit(X_train_non_neg, y_train)
    y_pred = model.predict(X_test_non_neg)
    acc = accuracy_score(y_test, y_pred)
    return 'Naive Bayes', acc

def rf_test(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'Random Forest', acc

def gb_test(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'Gradient Boosting', acc

def ensemble_test(X_train, X_test, y_train, y_test):
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('svc', SVC(kernel='linear', probability=True)),
        ('rf', RandomForestClassifier())
    ]
    ensemble = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier())
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'Ensemble', acc

def stacking_test(X_train, X_test, y_train, y_test):
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('svc', SVC(kernel='linear', probability=True)),
        ('rf', RandomForestClassifier())
    ]
    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'Stacking', acc

def run_model(model_func, X_train, X_test, y_train, y_test):
    return model_func(X_train, X_test, y_train, y_test)
#%%
def ensemble_test(X_train, X_test, y_train, y_test):
    estimators = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('svc', SVC(kernel='linear', probability=True)),
        ('rf', RandomForestClassifier())
    ]
    ensemble = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier())
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return 'Ensemble', acc