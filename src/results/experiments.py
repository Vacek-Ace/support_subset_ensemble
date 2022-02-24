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

from src.model.boosted_support_subset import *
from src.utils import *

import warnings
warnings.filterwarnings('ignore')

for experiment in [
#  'mushrooms',
#  'ilpd',
#  'banknote',
#  'fourclass',
#  'svmguide3',
#  'transfusion',
 'german_numer',
 'liver-disorders',
 'heart',
 'r2',
 'haberman',
 'svmguide1',
 'breastcancer',
 'australian',
 'diabetes',
 'mammographic',
 'ionosphere',
 'colon-cancer'
 ]:
# Experiment paths
    # experiment = 'ilpd'

    print(f'Experiment: {experiment}\n')

    results_folder = f'results/{experiment}'

    os.makedirs(results_folder, exist_ok=True)

    data = pd.read_parquet(f'data/prep_real_data/{experiment}.parquet')


    # Preprocessing
    scaler = StandardScaler()
    X = scaler.fit_transform(data.drop(columns=['y']))
    y = data.y.values


    # MOE-SVM hyperparameters gridsearch
    grid_params = {
        'params': [{'C': 1, 'gamma': 1 ,'kernel': 'rbf'},
        {'C': 1, 'gamma': 0.1 ,'kernel': 'rbf'},
        {'C': 10, 'gamma': 1 ,'kernel': 'rbf'},
        {'C': 10, 'gamma': 0.1 ,'kernel': 'rbf'}], 
        'k': [3, 4, 5, 6],
        'prop_sample': [0.10, 0.30, 0.50, 0.70]
    }

    # grid_params = {
    #     'params': [{'C': 1, 'gamma': 1 ,'kernel': 'rbf'},
    #     {'C': 1, 'gamma': 0.1 ,'kernel': 'rbf'}],
    #     'k': [3, 4, 5, 6],
    #     'prop_sample': [0.10, 0.30, 0.50, 0.70]
    # }


    ensemble = BoostedSupportSubset

    ensemble_grid = gridsearch(ensemble, grid_params, scoring=scaled_mcc, cv=10, n_jobs=-1)

    print('BSS : \n')

    ensemble_grid.fit(X, y)

    print(f'params: {ensemble_grid.best_params_}, score: {ensemble_grid.best_score_}\n')

    with open(f'results/{experiment}/bss.p', 'wb') as fout:
        pickle.dump(ensemble_grid, fout, protocol=pickle.HIGHEST_PROTOCOL)