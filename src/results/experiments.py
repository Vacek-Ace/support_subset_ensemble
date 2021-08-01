import pandas as pd
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report

import pickle

from src.model.overfitted_ensemble import *
from src.utils import *

import warnings
warnings.filterwarnings('ignore')


# Experiment paths
experiment = 'haberman'

print(f'Experiment: {experiment}\n')

results_folder = f'results/{experiment}'

os.makedirs(results_folder, exist_ok=True)

data = pd.read_parquet(f'data/prep_real_data/{experiment}.parquet')


# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=['y']))
y = data.y.values


# Random Forest hyperparameters gridsearch
grid_params = {'max_features': [None, 'sqrt',
                                'log2'], 'n_estimators': [100, 300, 500]}

rf = gridsearch(RandomForestClassifier, grid_params,
                scaled_mcc, cv=10, n_jobs=-1)

print('Random Forest: \n')

rf.fit(X, y)

print(f'params: {rf.best_params_}, score: {rf.best_score_}\n')

with open(f'results/{experiment}/randomforest.p', 'wb') as fout:
    pickle.dump(rf, fout, protocol=pickle.HIGHEST_PROTOCOL)


# SVM hyperparameters gridsearch

grid_params = {'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel':['rbf']}

clf = gridsearch(SVC, grid_params, scaled_mcc, cv=10, n_jobs=-1)

print('SVM: \n')

clf.fit(X, y)

print(f'params: {clf.best_params_}, score: {clf.best_score_}\n')

with open(f'results/{experiment}/svm.p', 'wb') as fout:
    pickle.dump(clf, fout, protocol=pickle.HIGHEST_PROTOCOL)

# KNN hyperparameters gridsearch

grid_params = {'n_neighbors' : list(range(1, 13, 2))}

clf = gridsearch(KNeighborsClassifier, grid_params, scaled_mcc, cv=10, n_jobs=-1)

print('KNN: \n')

clf.fit(X, y)   

print(f'params: {clf.best_params_}, score: {clf.best_score_}\n')

with open(f'results/{experiment}/knn.p', 'wb') as fout:
    pickle.dump(clf, fout, protocol=pickle.HIGHEST_PROTOCOL)


# MOE-kNN hyperparameters gridsearch
grid_params = {
    'wrab': [True, False],
    'lam': [1, 3, 5],
    'prop_sample': [0.10, 0.20, 0.30],
    'n_learners': [10, 20, 30],
    'random_state': [1234]
}

kwargs = {'method':KNeighborsClassifier, 'params':{'n_neighbors' : list(range(1, 13, 2))}}

ensemble = moe

ensemble_grid = gridsearch(ensemble, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1, kwargs=kwargs)

print('MOE knn: \n')

ensemble_grid.fit(X, y)

print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

with open(f'results/{experiment}/MOE_knn.p', 'wb') as fout:
    pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)
    


# MOE-DT hyperparameters gridsearch
grid_params = {
    'wrab': [True, False],
    'lam': [1, 3, 5],
    'prop_sample': [0.10, 0.20, 0.30],
    'n_learners': [10, 20, 30],
    'random_state': [1234]
}

kwargs = {'method':DecisionTreeClassifier, 'params':{'criterion' : {"gini", "entropy"}, 'max_depth' : list(range(1, 10, 1))}}

ensemble = moe

ensemble_grid = gridsearch(ensemble, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1, kwargs=kwargs)

print('MOE dt: \n')

ensemble_grid.fit(X, y)

print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

with open(f'results/{experiment}/MOE_dt.p', 'wb') as fout:
    pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)

# MOE-SVM hyperparameters gridsearch
grid_params = {
    'wrab': [True, False],
    'lam': [1, 3, 5],
    'prop_sample': [0.10, 0.20, 0.30],
    'n_learners': [10, 20, 30],
    'random_state': [1234]
}

kwargs = {'method':SVC, 'params':{'C': [1, 10, 100, 1000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]}}

ensemble = moe

ensemble_grid = gridsearch(ensemble, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1, kwargs=kwargs)

print('MOE SVM: \n')

ensemble_grid.fit(X, y)

print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

with open(f'results/{experiment}/moe.p', 'wb') as fout:
    pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)